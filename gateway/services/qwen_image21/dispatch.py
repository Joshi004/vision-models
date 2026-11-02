"""Builds Qwen-Image-2.1's invocation and queues it for a warm worker.

Unlike most backends a request has no Slurm job of its own: a few
long-lived (as many as the GPU capacity page allows) worker jobs (models/qwen_image21/runner/worker.py,
with that model's own environment) each load a checkpoint once and serve queued
requests one at a time. common/worker_pool.py starts, feeds and stops them;
this module only writes the job's files (args.json, output_path.txt: the same
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
from common import dedup, job_store, storage
from common.slurm import resolve_partition
from common.worker_pool import PoolError, PoolRequest, WorkerPool
from services.qwen_image21 import config
from services.qwen_image21.schemas import QwenImageEditRequest, QwenImageGenerateRequest


class DispatchError(RuntimeError):
    """Could not queue the request (no worker could be started, ...)."""


class CapacityError(RuntimeError):
    """Too many jobs already queued or running for this model (see common.config.MAX_QUEUED_PER_BACKEND)."""


class InputError(RuntimeError):
    """A reference image can't be used (unsupported file extension)."""


def _check_capacity() -> None:
    if job_store.count_inflight_for(pool.pipelines) >= common_config.MAX_QUEUED_PER_BACKEND:
        raise CapacityError(
            f"This model already has {common_config.MAX_QUEUED_PER_BACKEND} job(s) queued or running. "
            "Try again shortly."
        )


def _check_image_extension(path: Path) -> None:
    if path.suffix.lower() not in config.REFERENCE_IMAGE_EXTENSIONS:
        accepted = ", ".join(sorted(config.REFERENCE_IMAGE_EXTENSIONS))
        raise InputError(f"Unsupported file extension {path.suffix!r} (accepted: {accepted}): {path.name}")


def build_args(req: QwenImageGenerateRequest | QwenImageEditRequest, output_path: Path) -> list[str]:
    """Args *after* the interpreter+script path, matching run_generate.py's
    own CLI. --output-path is already set to `output_path`; _write_job_files
    rewrites it to a partial path before the request is queued."""
    args = [
        "--model-dir", str(config.MODEL_DIR_BY_MODE[req.mode]),
        "--mode", req.mode,
        "--output-path", str(output_path),
        f"--prompt={req.prompt}",
        "--seed", str(req.seed),
    ]
    if req.mode == "quality":
        args += ["--steps", str(req.num_inference_steps or config.DEFAULT_QUALITY_STEPS)]
    if req.aspect_ratio is not None:
        width, height = config.ASPECT_RATIOS[req.aspect_ratio]
        args += ["--width", str(width), "--height", str(height)]
    if isinstance(req, QwenImageEditRequest):
        for asset_id in req.reference_image_asset_ids:
            image_path = storage.resolve_asset_path(asset_id)  # raises KeyError/FileNotFoundError
            _check_image_extension(image_path)  # raises InputError
            args += ["--image", str(image_path)]
    return args


def _write_job_files(job_dir: Path, module_args: list[str], output_path: Path) -> None:
    """args.json (the full argv of a cold run of run_generate.py, with --output-path rewritten to the
    partial file) and output_path.txt (the final path), read by the worker."""
    partial_path = job_dir / f"output.partial{output_path.suffix}"
    args = list(module_args)
    idx = args.index("--output-path")
    args[idx + 1] = str(partial_path)

    full_argv = [str(config.QWEN_IMAGE21_PYTHON), str(config.RUN_GENERATE_SCRIPT), *args]
    (job_dir / "args.json").write_text(json.dumps(full_argv))
    (job_dir / "output_path.txt").write_text(str(output_path))


def _worker_sbatch_command(worker_id: str, worker_dir: Path, key: str, partition: str, gpus: int) -> list[str]:
    """The sbatch command for one warm worker. `key` is the request mode (fast or quality)."""
    worker_argv = [
        str(config.QWEN_IMAGE21_PYTHON),
        str(config.WORKER_SCRIPT),
        "--worker-dir", str(worker_dir),
        "--model-dir", str(config.MODEL_DIR_BY_MODE[key]),
        "--mode", key,
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
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}qwen-image21-worker-{worker_id[:8]}",
        f"--output={worker_dir / 'worker.log'}",
        f"--wrap=HF_HUB_OFFLINE=1 {shlex.join(worker_argv)}",
    ]


pool = WorkerPool(
    "qwen_image21",
    label="Qwen-Image-2.1",
    pipelines=("qwen-image21:generate", "qwen-image21:edit"),
    default_max_jobs=8,
    idle_seconds=config.IDLE_MINUTES * 60,
    worker_time_seconds=config.worker_time_limit_seconds(),
    build_sbatch_command=_worker_sbatch_command,
)


def _dispatch(req: QwenImageGenerateRequest | QwenImageEditRequest, pipeline: str) -> dedup.Submission:
    """Validate the reference assets, reuse an identical earlier request if there is one, else create the job directory/DB
    row and queue the request, in that order, so an invalid request never
    leaves debris behind."""
    partition = resolve_partition(req.partition)
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    output_path = job_dir / "output.png"

    # Raises KeyError/FileNotFoundError (unknown/missing asset) or InputError
    # (unsupported extension) here, before anything is created on disk or in
    # the DB.
    module_args = build_args(req, output_path)

    dedup_key = None if req.incognito else dedup.content_key(pipeline, req.model_dump(mode="json"), config.RESULT_VERSION)
    with dedup.lock:
        earlier = dedup.find_reusable(dedup_key)
        if earlier is not None:
            return dedup.reuse(earlier)
        _check_capacity()
        job_dir.mkdir(parents=True)
        job_store.create_job(job_id, pipeline, job_dir, req.model_dump(mode="json"), partition, incognito=req.incognito, dedup_key=dedup_key)
    try:
        _write_job_files(job_dir, module_args, output_path)
        pool.enqueue(
            PoolRequest(
                job_id=job_id,
                key=req.mode,
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


def dispatch_generate(req: QwenImageGenerateRequest) -> dedup.Submission:
    return _dispatch(req, "qwen-image21:generate")


def dispatch_edit(req: QwenImageEditRequest) -> dedup.Submission:
    return _dispatch(req, "qwen-image21:edit")
