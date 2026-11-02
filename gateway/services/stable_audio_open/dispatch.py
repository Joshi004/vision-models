"""Builds Stable Audio Open 1.0's invocation and queues it for a warm worker.

A request has no Slurm job of its own: a few long-lived (as many as the
GPU capacity page allows) worker jobs (models/stable_audio_open/runner/worker.py, with that model's own
environment) each load the model once and serve queued requests one at a
time. common/worker_pool.py starts, feeds and stops them; this module only
writes the job's files (args.json, output_path.txt: the same
partial-path-rename convention common/run_pipeline_job.py enforces for cold
jobs, which models/common/pool_worker.py follows) and puts the request in the
queue.

Text-like values are passed as `--flag=value` single argv entries (and the
request is read from args.json, never through a shell string), so user text
needs no escaping and can safely start with "-" or contain quotes/newlines.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from common import config as common_config
from common import dedup, job_store
from common.audio_inputs import AudioInputError, resolve_audio_asset
from common.slurm import resolve_partition
from common.worker_pool import PoolError, PoolRequest, WorkerPool
from services.stable_audio_open import config
from services.stable_audio_open.schemas import StableAudioOpenGenerateRequest, StableAudioOpenTransformRequest

# One kind of worker: both endpoints run on the same loaded model.
POOL_KEY = "default"


class DispatchError(RuntimeError):
    """Could not queue the request (no worker could be started, ...)."""


class CapacityError(RuntimeError):
    """Too many jobs already queued or running for this model (see common.config.MAX_QUEUED_PER_BACKEND)."""


class InputError(RuntimeError):
    """The source recording can't be used."""


def _check_capacity() -> None:
    if job_store.count_inflight_for(pool.pipelines) >= common_config.MAX_QUEUED_PER_BACKEND:
        raise CapacityError(
            f"This model already has {common_config.MAX_QUEUED_PER_BACKEND} job(s) queued or running. "
            "Try again shortly."
        )


def build_args(req: StableAudioOpenGenerateRequest | StableAudioOpenTransformRequest, mode: str, output_path: Path) -> list[str]:
    """Args *after* the interpreter+script path, matching run_generate.py's own CLI.
    --output-path is already set to `output_path`; _write_job_files rewrites it to a partial path."""
    if isinstance(req, StableAudioOpenTransformRequest):
        try:
            source_path, source_seconds = resolve_audio_asset(req.source_audio_asset_id, config.MAX_SOURCE_SECONDS)
        except AudioInputError as e:
            raise InputError(str(e)) from e  # unknown/missing assets raise KeyError/FileNotFoundError
        duration = req.duration if req.duration is not None else min(source_seconds, config.MAX_DURATION_SECONDS)
    else:
        duration = req.duration

    args = [
        "--model-dir", str(config.MODEL_DIR),
        "--mode", mode,
        "--output-path", str(output_path),
        f"--prompt={req.prompt}",
        "--duration", str(duration),
        "--steps", str(req.num_inference_steps),
        "--cfg-scale", str(req.cfg_scale),
        "--seed", str(req.seed),
    ]
    if req.negative_prompt:
        args.append(f"--negative-prompt={req.negative_prompt}")
    if isinstance(req, StableAudioOpenTransformRequest):
        args += ["--init-audio", str(source_path), "--init-noise-level", str(req.init_noise_level)]
    return args


def _write_job_files(job_dir: Path, module_args: list[str], output_path: Path) -> None:
    """args.json (the full argv of a cold run of run_generate.py, with --output-path rewritten to the
    partial file) and output_path.txt (the final path), read by the worker."""
    partial_path = job_dir / f"output.partial{output_path.suffix}"
    args = list(module_args)
    idx = args.index("--output-path")
    args[idx + 1] = str(partial_path)

    full_argv = [str(config.STABLE_AUDIO_OPEN_PYTHON), str(config.RUN_GENERATE_SCRIPT), *args]
    (job_dir / "args.json").write_text(json.dumps(full_argv))
    (job_dir / "output_path.txt").write_text(str(output_path))


def _worker_sbatch_command(worker_id: str, worker_dir: Path, key: str, partition: str, gpus: int) -> list[str]:
    """The sbatch command for one warm worker."""
    worker_argv = [
        str(config.STABLE_AUDIO_OPEN_PYTHON),
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
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}stable-audio-open-worker-{worker_id[:8]}",
        f"--output={worker_dir / 'worker.log'}",
        f"--wrap=HF_HUB_OFFLINE=1 {shlex.join(worker_argv)}",
    ]


pool = WorkerPool(
    "stable_audio_open",
    label="Stable Audio Open 1.0",
    pipelines=("stable-audio-open:generate", "stable-audio-open:transform"),
    default_max_jobs=4,
    idle_seconds=config.IDLE_MINUTES * 60,
    worker_time_seconds=config.worker_time_limit_seconds(),
    build_sbatch_command=_worker_sbatch_command,
)


def _dispatch(req: StableAudioOpenGenerateRequest | StableAudioOpenTransformRequest, mode: str) -> dedup.Submission:
    """Validate the source audio, reuse an identical earlier request if there is one, else create the job directory/DB row and queue the request,
    in that order, so an invalid request never leaves debris behind."""
    partition = resolve_partition(req.partition)
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    output_path = job_dir / "output.flac"

    # Raises KeyError/FileNotFoundError (unknown/missing asset) or InputError (unusable audio)
    # here, before anything is created on disk or in the DB.
    module_args = build_args(req, mode, output_path)

    pipeline = f"stable-audio-open:{mode}"
    dedup_key = None if req.incognito else dedup.content_key(pipeline, req.model_dump(mode="json"), config.RESULT_VERSION)
    with dedup.lock:
        earlier = dedup.find_reusable(dedup_key)
        if earlier is not None:
            return dedup.reuse(earlier, req.priority)
        _check_capacity()
        job_dir.mkdir(parents=True)
        job_store.create_job(job_id, pipeline, job_dir, req.model_dump(mode="json"), partition, incognito=req.incognito, dedup_key=dedup_key)
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


def dispatch_generate(req: StableAudioOpenGenerateRequest) -> dedup.Submission:
    return _dispatch(req, "generate")


def dispatch_transform(req: StableAudioOpenTransformRequest) -> dedup.Submission:
    return _dispatch(req, "transform")
