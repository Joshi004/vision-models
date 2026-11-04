"""Builds the MiniMax H3 invocation and queues it for a warm worker.

Like services/ltx25/dispatch.py this builds the argv for one `models/h3/runner/run_h3.py`
run, but a request has no Slurm job of its own: a few long-lived worker (as many as the GPU
capacity page allows) jobs (models/h3/runner/worker.py) each hold 4 or 8 GPUs and run queued requests one after
another, so only a worker's first request reads the checkpoints from shared storage.
common/worker_pool.py starts, feeds and stops them; this module writes the job's files
(args.json, output_path.txt: the partial-path-rename convention of common/run_pipeline_job.py,
which models/common/pool_worker.py follows) and queues the request.
Differences from the LTX backends:

  * A worker holds 4 or 8 GPUs on one node (`--gres=gpu:N`): services/h3/sizing.py picks N
    for each request before anything is queued, and a request that does not fit is
    refused (SizingError). run_h3.py splits the text/vision encoder over two of them,
    then ALL N denoise together in ulysses mode.
  * The wrapper script, not a library module, is the entry point, so every
    flag below is the one run_h3.py itself defines (see its --help).

The two request kinds map onto diffusers' workflows: `/videos/generate` is
`fl2va` when a first and/or last frame is given and `t2va` otherwise (both read
the `transformer/` checkpoint), `/videos/audio-to-video` is `ref2va` (the
`transformer_ref/` checkpoint).
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from common import config as common_config
from common import dedup, job_store, storage
from common.slurm import resolve_partition
from common.worker_pool import PoolError, PoolRequest, WorkerPool
from services.h3 import config, sizing, video_shape
from services.h3.schemas import H3AudioToVideoRequest, H3GenerateRequest


class DispatchError(RuntimeError):
    """Could not submit the job (bad asset reference, sbatch failure, ...)."""


class CapacityError(RuntimeError):
    """Too many jobs already queued or running for this model (see common.config.MAX_QUEUED_PER_BACKEND)."""


def _check_capacity() -> None:
    if job_store.count_inflight_for(pool.pipelines) >= common_config.MAX_QUEUED_PER_BACKEND:
        raise CapacityError(
            f"This model already has {common_config.MAX_QUEUED_PER_BACKEND} job(s) queued or running. "
            "Try again shortly."
        )


# ---------------------------------------------------------------------------
# Per-recipe argv builders. Each returns the args *after* the interpreter (i.e.
# starting with the run_h3.py script path), with --output-path already set to
# `output_path` (_write_job_files rewrites it to a partial path before the
# request is queued), together with the sizing decision made for the request.
# ---------------------------------------------------------------------------


def _common_args(
    req, *, workflow: str, output_path: Path, job_dir: Path, num_frames: int | None, sized: sizing.Sizing
) -> list[str]:
    args = [
        str(config.RUN_H3_SCRIPT),
        "--model-dir", str(config.MODEL_DIR),
        "--workflow", workflow,
        "--output-path", str(output_path),
        "--work-dir", str(job_dir / "work"),
        # The runner writes its numbers (timings, GPU peaks, token counts) here for the usage log.
        "--metrics-dir", str(job_dir),
        # `--prompt=<text>` (one token) rather than two, so a prompt that
        # starts with "-" is never mistaken for another flag by argparse.
        f"--prompt={req.prompt}",
        "--seed", str(req.seed),
    ]
    if num_frames is not None:
        args += ["--num-frames", str(num_frames)]
    canvas = video_shape.resolve_canvas(req.orientation, req.height, req.width)
    if canvas is not None:
        args += ["--height", str(canvas[0]), "--width", str(canvas[1])]
    if req.num_inference_steps is not None:
        args += ["--steps", str(req.num_inference_steps)]
    if config.ATTENTION_BACKEND:
        args += ["--attention-backend", config.ATTENTION_BACKEND]
    # The GPU count is what the sizer chose for this request; the job holds exactly that many.
    args += ["--gpus", str(sized.gpus), "--parallelism", config.PARALLELISM]
    return args


def build_generate_args(req: H3GenerateRequest, output_path: Path, job_dir: Path) -> tuple[list[str], sizing.Sizing]:
    # Raises KeyError/FileNotFoundError (-> 400) for an unknown asset_id.
    first_path = storage.resolve_asset_path(req.first_frame_asset_id) if req.first_frame_asset_id else None
    last_path = storage.resolve_asset_path(req.last_frame_asset_id) if req.last_frame_asset_id else None
    workflow = "fl2va" if first_path or last_path else "t2va"
    # Raises SizingError (-> 422) when the request cannot fit.
    sized = sizing.size_generate(req, first_path, last_path)
    args = _common_args(
        req,
        workflow=workflow,
        output_path=output_path,
        job_dir=job_dir,
        num_frames=video_shape.resolve_num_frames(req.duration_seconds, req.num_frames),
        sized=sized,
    )
    if first_path:
        args += ["--first-frame", str(first_path)]
    if last_path:
        args += ["--last-frame", str(last_path)]
    return args, sized


def build_audio_to_video_args(
    req: H3AudioToVideoRequest, output_path: Path, job_dir: Path
) -> tuple[list[str], sizing.Sizing]:
    # Resolved before anything is built, so an unknown asset_id fails the request cleanly.
    image_paths = [storage.resolve_asset_path(asset_id) for asset_id in req.image_asset_ids]
    video_paths = [storage.resolve_asset_path(asset_id) for asset_id in req.video_asset_ids]
    audio_paths = [storage.resolve_asset_path(asset_id) for asset_id in req.audio_asset_ids]
    # Raises SizingError (-> 422) when the request cannot fit.
    sized = sizing.size_audio_to_video(req, image_paths, video_paths, audio_paths)
    args = _common_args(
        req,
        workflow="ref2va",
        output_path=output_path,
        job_dir=job_dir,
        # With match_audio_duration the runner sizes the clip from the audio it decodes.
        num_frames=None if req.match_audio_duration else video_shape.resolve_num_frames(req.duration_seconds, req.num_frames),
        sized=sized,
    )
    for path in image_paths:
        args += ["--reference-image", str(path)]
    for path in video_paths:
        args += ["--reference-video", str(path)]
    for path in audio_paths:
        args += ["--reference-audio", str(path)]
    # A reference video's own soundtrack is used unless its video_audio flag is false (0-based index into the videos).
    for index, use_soundtrack in enumerate(req.video_audio or []):
        if not use_soundtrack:
            args += ["--mute-reference-video", str(index)]
    if req.match_audio_duration:
        # The first audio is only decoded on the compute node, so the runner enforces the ceiling.
        args += ["--match-audio-duration", "--max-seconds", str(config.MAX_DURATION_SECONDS)]
    return args, sized


# ---------------------------------------------------------------------------
# Submission plumbing (shared by both H3 recipes above).
# ---------------------------------------------------------------------------


def _write_job_files(job_dir: Path, module_args: list[str], output_path: Path) -> None:
    """args.json (the full argv of a cold run of run_h3.py, with --output-path rewritten to the partial
    file) and output_path.txt (the final path), read by the worker."""
    partial_path = job_dir / f"output.partial{output_path.suffix}"
    args = list(module_args)
    args[args.index("--output-path") + 1] = str(partial_path)

    full_argv = [str(config.H3_PYTHON), *args]
    (job_dir / "args.json").write_text(json.dumps(full_argv))
    (job_dir / "output_path.txt").write_text(str(output_path))


def _worker_key(module_args: list[str], gpus: int) -> str:
    """Which warm workers can serve a request: the same transformer checkpoint (so the worker's page
    cache holds it) and the same number of GPUs."""
    workflow = module_args[module_args.index("--workflow") + 1]
    checkpoint = "transformer_ref" if workflow == "ref2va" else "transformer"
    return _key_for(checkpoint, gpus)


def _key_for(checkpoint: str, gpus: int) -> str:
    return f"{checkpoint}:{gpus}gpu"


def worker_key_with_gpus(key: str, gpus: int) -> str:
    """`key` (as _worker_key made it) for a worker with `gpus` GPUs instead: the same checkpoint."""
    checkpoint = key.rsplit(":", 1)[0]
    return _key_for(checkpoint, gpus)


def _worker_sbatch_command(worker_id: str, worker_dir: Path, key: str, partition: str, gpus: int) -> list[str]:
    """The sbatch command for one warm worker, holding `gpus` GPUs on one node."""
    worker_argv = [
        "python3",
        str(config.WORKER_SCRIPT),
        "--worker-dir", str(worker_dir),
        # A safety net only: the gateway stops an idle worker after IDLE_MINUTES.
        "--idle-exit-seconds", str(int(config.IDLE_MINUTES * 60 * 2)),
    ]
    return [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        "--nodes=1",
        f"--gres=gpu:{gpus}",
        f"--cpus-per-task={config.CPUS_PER_GPU * gpus}",
        f"--mem={config.MEM}",
        f"--time={config.WORKER_TIME_LIMIT}",
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}h3-worker-{worker_id[:8]}",
        f"--output={worker_dir / 'worker.log'}",
        f"--wrap=HF_HUB_OFFLINE=1 {shlex.join(worker_argv)}",
    ]


pool = WorkerPool(
    "h3",
    label="MiniMax H3",
    pipelines=("h3:generate", "h3:audio-to-video"),
    planning_gpus=4,  # a job uses 4 or 8 GPUs; limits are validated as if it used 4
    default_max_jobs=8,
    idle_seconds=config.IDLE_MINUTES * 60,
    worker_time_seconds=config.worker_time_limit_seconds(),
    build_sbatch_command=_worker_sbatch_command,
)


def _dispatch(pipeline: str, req, build_args) -> tuple[dedup.Submission, int | None, float | None]:
    """Shared submit path: validate assets + size, reuse an identical earlier request if there
    is one, else check capacity, create the job directory/DB row and queue the request, in that
    order, so an invalid request never leaves debris behind. Returns the submission and the
    sizing decision (the GPU count and the run-time estimate; those of the earlier job when it
    was reused)."""
    partition = resolve_partition(getattr(req, "partition", None))
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    output_path = job_dir / "output.mp4"

    # Raises KeyError/FileNotFoundError (-> 400) if an asset_id is invalid, and
    # sizing.SizingError (-> 422) if the request does not fit: both before
    # anything is created on disk or in the DB.
    module_args, sized = build_args(req, output_path, job_dir)

    dedup_key = None if req.incognito else dedup.content_key(pipeline, req.model_dump(mode="json"), config.RESULT_VERSION)
    with dedup.lock:
        earlier = dedup.find_reusable(dedup_key)
        if earlier is not None:
            return dedup.reuse(earlier, req.priority), earlier["gpus"], earlier["estimated_run_seconds"]
        _check_capacity()
        job_dir.mkdir(parents=True)
        sized.save(job_dir)
        job_store.create_job(
            job_id,
            pipeline,
            job_dir,
            req.model_dump(mode="json"),
            partition,
            incognito=req.incognito,
            gpus=sized.gpus,
            estimated_run_seconds=sized.estimated_run_seconds,
            dedup_key=dedup_key,
        )
    try:
        _write_job_files(job_dir, module_args, output_path)
        pool.enqueue(
            PoolRequest(
                job_id=job_id,
                key=_worker_key(module_args, sized.gpus),
                partition=partition,
                gpus=sized.gpus,
                limit_seconds=sized.time_limit_seconds,
            ),
            job_dir,
        )
    except (OSError, PoolError) as e:
        job_store.mark_failed(job_id, "Could not queue the request.")
        raise DispatchError(str(e)) from e
    return dedup.Submission(job_id, "queued"), sized.gpus, sized.estimated_run_seconds


def dispatch_generate(req: H3GenerateRequest) -> tuple[dedup.Submission, int | None, float | None]:
    return _dispatch("h3:generate", req, build_generate_args)


def dispatch_audio_to_video(req: H3AudioToVideoRequest) -> tuple[dedup.Submission, int | None, float | None]:
    return _dispatch("h3:audio-to-video", req, build_audio_to_video_args)
