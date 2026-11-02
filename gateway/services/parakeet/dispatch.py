"""Builds Parakeet's transcription invocation and queues it for a warm worker.

A request has no Slurm job of its own: a few long-lived (as many as the
GPU capacity page allows) worker jobs (models/parakeet/runner/worker.sh -> worker.py, with that model's
own environment) each load the model once and serve queued requests one at a
time; one idle for config.IDLE_MINUTES is stopped. common/worker_pool.py
starts, feeds and stops them; this module only writes the job's files
(args.json, output_path.txt: the same partial-path-rename convention
common/run_pipeline_job.py enforces for cold jobs, which
models/common/pool_worker.py follows) and puts the request in the queue.

args.json holds the same argv a one-shot job would run (run_transcribe.sh and
its flags -- see models/parakeet/runner/run_transcribe.py for what it does:
ffmpeg decode, chunking past 10 minutes of audio); the worker runs it on the
model it already holds.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from common import config as common_config
from common import dedup, job_store, storage
from common.slurm import resolve_partition
from common.worker_pool import PoolError, PoolRequest, WorkerPool
from services.parakeet import config
from services.parakeet.schemas import TranscribeRequest

PIPELINE = "parakeet:transcribe"
# One kind of worker: every request runs on the same loaded model.
POOL_KEY = "default"


class DispatchError(RuntimeError):
    """Could not queue the request (no worker could be started, ...)."""


class CapacityError(RuntimeError):
    """Too many jobs already queued or running for this model (see common.config.MAX_QUEUED_PER_BACKEND)."""


class InputError(RuntimeError):
    """The referenced asset isn't a file this deployment can actually
    transcribe -- an unsupported file extension (see
    config.INPUT_EXTENSIONS)."""


def _check_capacity() -> None:
    if job_store.count_inflight_for(pool.pipelines) >= common_config.MAX_QUEUED_PER_BACKEND:
        raise CapacityError(
            f"This model already has {common_config.MAX_QUEUED_PER_BACKEND} job(s) queued or running. "
            "Try again shortly."
        )


def _check_extension(path: Path) -> None:
    """Raises InputError if `path`'s extension isn't one run_transcribe.py's
    own ffmpeg decode step can actually read -- see config.INPUT_EXTENSIONS."""
    if path.suffix.lower() not in config.INPUT_EXTENSIONS:
        accepted = ", ".join(sorted(config.INPUT_EXTENSIONS))
        raise InputError(f"Unsupported file extension {path.suffix!r} (accepted: {accepted}): {path.name}")


def build_transcribe_args(req: TranscribeRequest, job_dir: Path, output_path: Path) -> list[str]:
    """Args *after* run_transcribe.sh's own path, matching its CLI exactly
    (--input-path/--work-dir/--model-path/--output-path -- see that
    script's own header comment). --output-path is already set to
    `output_path`; _write_job_files rewrites it to a partial path, same
    convention every other backend's dispatch module uses."""
    source_path = storage.resolve_asset_path(req.audio_asset_id)  # raises KeyError/FileNotFoundError
    _check_extension(source_path)  # raises InputError
    return [
        "--input-path", str(source_path),
        "--work-dir", str(job_dir),
        "--model-path", str(config.MODEL_PATH),
        "--output-path", str(output_path),
    ]


def _write_job_files(job_dir: Path, module_args: list[str], output_path: Path) -> None:
    partial_path = job_dir / f"output.partial{output_path.suffix}"
    args = list(module_args)
    idx = args.index("--output-path")
    args[idx + 1] = str(partial_path)

    # argv[0] is run_transcribe.sh itself, as a one-shot job would run it; the worker reads
    # the flags after it.
    full_argv = [str(config.RUN_TRANSCRIBE_SCRIPT), *args]
    (job_dir / "args.json").write_text(json.dumps(full_argv))
    (job_dir / "output_path.txt").write_text(str(output_path))


def _worker_sbatch_command(worker_id: str, worker_dir: Path, key: str, partition: str, gpus: int) -> list[str]:
    """The sbatch command for one warm worker."""
    worker_argv = [
        str(config.WORKER_SCRIPT),
        "--worker-dir", str(worker_dir),
        "--model-path", str(config.MODEL_PATH),
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
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}parakeet-worker-{worker_id[:8]}",
        f"--output={worker_dir / 'worker.log'}",
        f"--wrap=HF_HUB_OFFLINE=1 {shlex.join(worker_argv)}",
    ]


pool = WorkerPool(
    "parakeet",
    label="Parakeet transcription",
    pipelines=(PIPELINE,),
    default_max_jobs=4,
    idle_seconds=config.IDLE_MINUTES * 60,
    worker_time_seconds=config.worker_time_limit_seconds(),
    build_sbatch_command=_worker_sbatch_command,
)


def dispatch_transcribe(req: TranscribeRequest) -> dedup.Submission:
    """Validate the asset, reuse an identical earlier request if there is one, else check
    capacity, create the job directory/DB row and queue the request, in that order, so an
    invalid request never leaves debris behind."""
    partition = resolve_partition(req.partition)
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    output_path = job_dir / "output.json"

    # Raises KeyError/FileNotFoundError (unknown/missing asset) or
    # InputError (unsupported extension) here (before anything is created
    # on disk or in the DB) if audio_asset_id is invalid.
    module_args = build_transcribe_args(req, job_dir, output_path)

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
