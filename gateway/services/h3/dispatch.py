"""Builds the MiniMax H3 invocation and submits it as a Slurm job.

Same shape as services/ltx25/dispatch.py: build the argv for one
`models/h3/runner/run_h3.py` run, then submit it non-blocking via sbatch -- see
common/run_pipeline_job.py for what actually executes on the compute node.
Differences from the LTX backends:

  * The job holds TWO GPUs (`--gres=gpu:2`, on one node): run_h3.py puts the
    text/vision encoder on one and the transformer on the other.
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
import subprocess
from pathlib import Path

from common import config as common_config
from common import job_store, retention, storage
from common.slurm import resolve_partition
from services.h3 import config, video_shape
from services.h3.schemas import H3AudioToVideoRequest, H3GenerateRequest


class DispatchError(RuntimeError):
    """Could not submit the job (bad asset reference, sbatch failure, ...)."""


class CapacityError(RuntimeError):
    """Too many jobs already in flight (see common.config.MAX_INFLIGHT_JOBS)."""


def _check_capacity() -> None:
    if (
        common_config.MAX_INFLIGHT_JOBS is not None
        and job_store.count_inflight_jobs() >= common_config.MAX_INFLIGHT_JOBS
    ):
        raise CapacityError(
            f"At capacity: {common_config.MAX_INFLIGHT_JOBS} job(s) already in flight. Try again shortly."
        )


# ---------------------------------------------------------------------------
# Per-recipe argv builders. Each returns the args *after* the interpreter (i.e.
# starting with the run_h3.py script path), with --output-path already set to
# `output_path` (_submit_slurm_job rewrites it to a partial path before
# actually submitting).
# ---------------------------------------------------------------------------


def _common_args(req, *, workflow: str, output_path: Path, job_dir: Path, num_frames: int | None) -> list[str]:
    args = [
        str(config.RUN_H3_SCRIPT),
        "--model-dir", str(config.MODEL_DIR),
        "--workflow", workflow,
        "--output-path", str(output_path),
        "--work-dir", str(job_dir / "work"),
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
    # Always passed, so every request follows the operator's setting (run_h3.py's own default is the same).
    args += ["--parallelism", config.PARALLELISM]
    return args


def build_generate_args(req: H3GenerateRequest, output_path: Path, job_dir: Path) -> list[str]:
    # Raises KeyError/FileNotFoundError (-> 400) for an unknown asset_id.
    first_path = storage.resolve_asset_path(req.first_frame_asset_id) if req.first_frame_asset_id else None
    last_path = storage.resolve_asset_path(req.last_frame_asset_id) if req.last_frame_asset_id else None
    workflow = "fl2va" if first_path or last_path else "t2va"
    args = _common_args(
        req,
        workflow=workflow,
        output_path=output_path,
        job_dir=job_dir,
        num_frames=video_shape.resolve_num_frames(req.duration_seconds, req.num_frames),
    )
    if first_path:
        args += ["--first-frame", str(first_path)]
    if last_path:
        args += ["--last-frame", str(last_path)]
    return args


def build_audio_to_video_args(req: H3AudioToVideoRequest, output_path: Path, job_dir: Path) -> list[str]:
    # Resolved before anything is built, so an unknown asset_id fails the request cleanly.
    image_paths = [storage.resolve_asset_path(asset_id) for asset_id in req.image_asset_ids]
    video_paths = [storage.resolve_asset_path(asset_id) for asset_id in req.video_asset_ids]
    audio_paths = [storage.resolve_asset_path(asset_id) for asset_id in req.audio_asset_ids]
    args = _common_args(
        req,
        workflow="ref2va",
        output_path=output_path,
        job_dir=job_dir,
        # With match_audio_duration the runner sizes the clip from the audio it decodes.
        num_frames=None if req.match_audio_duration else video_shape.resolve_num_frames(req.duration_seconds, req.num_frames),
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
    return args


# ---------------------------------------------------------------------------
# Submission plumbing (shared by both H3 recipes above).
# ---------------------------------------------------------------------------


def _submit_slurm_job(job_id: str, job_dir: Path, module_args: list[str], output_path: Path, partition: str, incognito: bool) -> str:
    """Write args.json + submit via sbatch (non-blocking). Returns the Slurm job id."""
    partial_path = job_dir / f"output.partial{output_path.suffix}"
    args = list(module_args)
    args[args.index("--output-path") + 1] = str(partial_path)

    full_argv = [str(config.H3_PYTHON), *args]
    (job_dir / "args.json").write_text(json.dumps(full_argv))
    (job_dir / "output_path.txt").write_text(str(output_path))

    wrap_argv = ["python3", str(config.RUN_PIPELINE_JOB_SCRIPT), str(job_dir)]
    wrap_str = "HF_HUB_OFFLINE=1 " + shlex.join(wrap_argv)
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        "--nodes=1",
        f"--gres=gpu:{config.GPUS_PER_JOB}",
        f"--cpus-per-task={config.CPUS}",
        f"--mem-per-gpu={config.MEM_PER_GPU}",
        f"--time={config.JOB_TIME_LIMIT}",
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}h3-api-{job_id[:8]}",
        f"--output={retention.slurm_log_path(job_dir, incognito)}",
        f"--wrap={wrap_str}",
    ]
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DispatchError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip().split(";")[0]  # --parsable prints "<id>" or "<id>;<cluster>"


def _dispatch(pipeline: str, req, build_args) -> str:
    """Shared submit path: validate capacity + assets, then create the job
    directory/DB row/Slurm job, in that order, so an invalid request never
    leaves debris behind."""
    _check_capacity()
    partition = resolve_partition(getattr(req, "partition", None))
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    output_path = job_dir / "output.mp4"

    # Raises KeyError/FileNotFoundError here (before anything is created on
    # disk or in the DB) if an asset_id is invalid.
    module_args = build_args(req, output_path, job_dir)

    job_dir.mkdir(parents=True)
    job_store.create_job(job_id, pipeline, job_dir, req.model_dump(mode="json"), partition, incognito=req.incognito)
    try:
        slurm_job_id = _submit_slurm_job(job_id, job_dir, module_args, output_path, partition, req.incognito)
    except DispatchError:
        job_store.mark_failed(job_id, "Could not submit the Slurm job.")
        raise
    job_store.set_slurm_job_id(job_id, slurm_job_id)
    return job_id


def dispatch_generate(req: H3GenerateRequest) -> str:
    return _dispatch("h3:generate", req, build_generate_args)


def dispatch_audio_to_video(req: H3AudioToVideoRequest) -> str:
    return _dispatch("h3:audio-to-video", req, build_audio_to_video_args)
