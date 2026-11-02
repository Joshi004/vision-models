"""Builds Breeze TTS 2's invocation and queues it for a warm worker.

A request has no Slurm job of its own: a few long-lived (as many as the
GPU capacity page allows) worker jobs (models/breeze_tts/runner/worker.py, with that model's own
environment) each load the model once and serve queued requests one at a
time; one idle for config.IDLE_MINUTES is stopped. common/worker_pool.py
starts, feeds and stops them; this module only writes the job's files
(args.json, output_path.txt: the same partial-path-rename convention
common/run_pipeline_job.py enforces for cold jobs, which
models/common/pool_worker.py follows) and puts the request in the queue.

args.json holds the same argv a one-shot job would run (run_synthesize.py and
its flags); the worker checks it and decodes any reference recording with that
script's own code, then generates on the model it already holds.

Text-like values are passed as `--flag=value` single argv entries (and the
request is read from args.json, never through a shell string), so user text
needs no escaping and can safely start with "-" or contain quotes/newlines.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from common import config as common_config
from common import dedup, job_store, storage
from common.slurm import resolve_partition
from common.worker_pool import PoolError, PoolRequest, WorkerPool
from services.breeze_tts import config
from services.breeze_tts.schemas import SynthesizeRequest

PIPELINE = "breeze-tts:synthesize"
# One kind of worker: every request runs on the same loaded model.
POOL_KEY = "default"


class DispatchError(RuntimeError):
    """Could not queue the request (no worker could be started, ...)."""


class CapacityError(RuntimeError):
    """Too many jobs already queued or running for this model (see common.config.MAX_QUEUED_PER_BACKEND)."""


class InputError(RuntimeError):
    """The reference recording can't be used (unsupported file extension)."""


def _check_capacity() -> None:
    if job_store.count_inflight_for(pool.pipelines) >= common_config.MAX_QUEUED_PER_BACKEND:
        raise CapacityError(
            f"This model already has {common_config.MAX_QUEUED_PER_BACKEND} job(s) queued or running. "
            "Try again shortly."
        )


def _check_reference_extension(path: Path) -> None:
    if path.suffix.lower() not in config.REFERENCE_EXTENSIONS:
        accepted = ", ".join(sorted(config.REFERENCE_EXTENSIONS))
        raise InputError(f"Unsupported file extension {path.suffix!r} (accepted: {accepted}): {path.name}")


def build_synthesize_args(req: SynthesizeRequest, job_dir: Path, output_path: Path) -> list[str]:
    """Args *after* the interpreter+script path, matching run_synthesize.py's
    own CLI. --output-path is already set to `output_path`; _write_job_files
    rewrites it to a partial path."""
    args = [
        "--model-dir", str(config.MODEL_DIR),
        "--work-dir", str(job_dir),
        "--output-path", str(output_path),
        f"--text={req.text}",
        "--seed", str(req.seed),
        "--cfg-scale", str(req.cfg_scale),
    ]
    if req.instruction and req.instruction.strip():
        args.append(f"--instruction={req.instruction}")
    if req.reference_audio_asset_id is not None:
        reference_path = storage.resolve_asset_path(req.reference_audio_asset_id)  # raises KeyError/FileNotFoundError
        _check_reference_extension(reference_path)  # raises InputError
        args += ["--ref-audio-path", str(reference_path), f"--ref-text={req.reference_text}"]
    return args


def _write_job_files(job_dir: Path, module_args: list[str], output_path: Path) -> None:
    partial_path = job_dir / f"output.partial{output_path.suffix}"
    args = list(module_args)
    idx = args.index("--output-path")
    args[idx + 1] = str(partial_path)

    full_argv = [str(config.BREEZE_TTS_PYTHON), str(config.RUN_SYNTHESIZE_SCRIPT), *args]
    (job_dir / "args.json").write_text(json.dumps(full_argv))
    (job_dir / "output_path.txt").write_text(str(output_path))


def _worker_sbatch_command(worker_id: str, worker_dir: Path, key: str, partition: str, gpus: int) -> list[str]:
    """The sbatch command for one warm worker."""
    worker_argv = [
        str(config.BREEZE_TTS_PYTHON),
        str(config.WORKER_SCRIPT),
        "--worker-dir", str(worker_dir),
        "--model-dir", str(config.MODEL_DIR),
        # A safety net only: the gateway stops an idle worker after IDLE_MINUTES.
        "--idle-exit-seconds", str(int(config.IDLE_MINUTES * 60 * 2)),
    ]
    return [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        f"--gres=gpu:{gpus}",
        f"--cpus-per-task={config.CPUS_PER_GPU}",
        f"--mem-per-gpu={config.MEM_PER_GPU}",
        f"--time={config.WORKER_TIME_LIMIT}",
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}breeze-tts-worker-{worker_id[:8]}",
        f"--output={worker_dir / 'worker.log'}",
        f"--wrap=HF_HUB_OFFLINE=1 {shlex.join(worker_argv)}",
    ]


pool = WorkerPool(
    "breeze_tts",
    label="Breeze TTS",
    pipelines=(PIPELINE,),
    default_max_jobs=4,
    idle_seconds=config.IDLE_MINUTES * 60,
    worker_time_seconds=config.worker_time_limit_seconds(),
    build_sbatch_command=_worker_sbatch_command,
)


def dispatch_synthesize(req: SynthesizeRequest) -> dedup.Submission:
    """Validate the reference asset, reuse an identical earlier request if there is one, else
    check capacity, create the job directory/DB row and queue the request, in that order, so an
    invalid request never leaves debris behind."""
    partition = resolve_partition(req.partition)
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    output_path = job_dir / "output.wav"

    # Raises KeyError/FileNotFoundError (unknown/missing asset) or InputError
    # (unsupported extension) here, before anything is created on disk or in
    # the DB.
    module_args = build_synthesize_args(req, job_dir, output_path)

    dedup_key = None if req.incognito else dedup.content_key(PIPELINE, req.model_dump(mode="json"), config.RESULT_VERSION)
    with dedup.lock:
        earlier = dedup.find_reusable(dedup_key)
        if earlier is not None:
            return dedup.reuse(earlier, req.priority)
        _check_capacity()
        job_dir.mkdir(parents=True)
        job_store.create_job(job_id, PIPELINE, job_dir, req.model_dump(mode="json"), partition, incognito=req.incognito, dedup_key=dedup_key)
    try:
        _write_job_files(job_dir, module_args, output_path)
        pool.enqueue(
            PoolRequest(
                job_id=job_id,
                key=POOL_KEY,
                partition=partition,
                gpus=1,
                limit_seconds=config.job_time_limit_seconds(),
            ),
            job_dir,
        )
    except (OSError, PoolError) as e:
        job_store.mark_failed(job_id, "Could not queue the request.")
        raise DispatchError(str(e)) from e
    return dedup.Submission(job_id, "queued")
