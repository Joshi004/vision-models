"""Builds LTX-2.3's CLI invocation and submits it as a Slurm job.

Mirrors generate.sh's flag-building logic exactly, just from Python instead
of bash, and non-blocking (sbatch) instead of blocking (srun) -- see
common/run_pipeline_job.py for what actually executes on the compute node,
and ltx-2.3/ARCHITECTURE.md / the LTX-2 README for where each pipeline's
exact flags come from (verified against the real argparse definitions, not
guessed).
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from common import config as common_config
from common import job_store, storage
from common.slurm import resolve_partition
from services.ltx import config, video_shape
from services.ltx.schemas import (
    AudioToVideoRequest,
    KeyframeInterpolationRequest,
    RetakeRequest,
    TextToAudioRequest,
    TextToVideoRequest,
)


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


def _image_args(images: list) -> list[str]:
    args: list[str] = []
    for img in images:
        path = storage.resolve_asset_path(img.asset_id)  # raises KeyError/FileNotFoundError
        entry = ["--image", str(path), str(img.frame_idx), str(img.strength)]
        if img.crf is not None:
            entry.append(str(img.crf))
        args.extend(entry)
    return args


# ---------------------------------------------------------------------------
# Per-pipeline argv builders. Each returns the args *after* the interpreter
# (i.e. starting with "-m"), with --output-path already set to `output_path`
# (submit_job rewrites it to a partial path before actually submitting).
# ---------------------------------------------------------------------------


def build_text_to_video_args(req: TextToVideoRequest, output_path: Path) -> list[str]:
    height, width = video_shape.resolve_dimensions(req.orientation, req.height, req.width)
    num_frames = video_shape.resolve_num_frames(req.duration_seconds, req.num_frames, req.frame_rate)

    if req.mode == "fast":
        args = ["-m", "ltx_pipelines.distilled", "--distilled-checkpoint-path", str(config.DISTILLED_CKPT)]
    else:
        args = [
            "-m", "ltx_pipelines.ti2vid_two_stages",
            "--checkpoint-path", str(config.DEV_CKPT),
            "--distilled-lora", str(config.DISTILLED_LORA), str(config.DISTILLED_LORA_STRENGTH),
        ]
        if req.negative_prompt:
            args += ["--negative-prompt", req.negative_prompt]

    args += [
        "--spatial-upsampler-path", str(config.UPSCALER),
        "--gemma-root", str(config.GEMMA_DIR),
        "--prompt", req.prompt,
        "--output-path", str(output_path),
        "--seed", str(req.seed),
        "--height", str(height),
        "--width", str(width),
        "--num-frames", str(num_frames),
        "--frame-rate", str(req.frame_rate),
    ]
    if req.enhance_prompt:
        args.append("--enhance-prompt")
    args += _image_args(req.images)
    return args


def build_keyframe_interpolation_args(req: KeyframeInterpolationRequest, output_path: Path) -> list[str]:
    height, width = video_shape.resolve_dimensions(req.orientation, req.height, req.width)
    num_frames = video_shape.resolve_num_frames(req.duration_seconds, req.num_frames, req.frame_rate)
    args = [
        "-m", "ltx_pipelines.keyframe_interpolation",
        "--checkpoint-path", str(config.DEV_CKPT),
        "--distilled-lora", str(config.DISTILLED_LORA), str(config.DISTILLED_LORA_STRENGTH),
        "--spatial-upsampler-path", str(config.UPSCALER),
        "--gemma-root", str(config.GEMMA_DIR),
        "--prompt", req.prompt,
        "--output-path", str(output_path),
        "--seed", str(req.seed),
        "--height", str(height),
        "--width", str(width),
        "--num-frames", str(num_frames),
        "--frame-rate", str(req.frame_rate),
    ]
    if req.negative_prompt:
        args += ["--negative-prompt", req.negative_prompt]
    args += _image_args(req.keyframes)  # >= 2 entries, enforced by the schema
    return args


def build_audio_to_video_args(req: AudioToVideoRequest, output_path: Path) -> list[str]:
    height, width = video_shape.resolve_dimensions(req.orientation, req.height, req.width)
    num_frames = video_shape.resolve_num_frames(req.duration_seconds, req.num_frames, req.frame_rate)
    audio_path = storage.resolve_asset_path(req.audio_asset_id)
    args = [
        "-m", "ltx_pipelines.a2vid_two_stage",
        "--checkpoint-path", str(config.DEV_CKPT),
        "--distilled-lora", str(config.DISTILLED_LORA), str(config.DISTILLED_LORA_STRENGTH),
        "--spatial-upsampler-path", str(config.UPSCALER),
        "--gemma-root", str(config.GEMMA_DIR),
        "--prompt", req.prompt,
        "--output-path", str(output_path),
        "--seed", str(req.seed),
        "--height", str(height),
        "--width", str(width),
        "--num-frames", str(num_frames),
        "--frame-rate", str(req.frame_rate),
        "--audio-path", str(audio_path),
        "--audio-start-time", str(req.audio_start_time),
    ]
    if req.negative_prompt:
        args += ["--negative-prompt", req.negative_prompt]
    if req.audio_max_duration is not None:
        args += ["--audio-max-duration", str(req.audio_max_duration)]
    args += _image_args(req.images)
    return args


def build_retake_args(req: RetakeRequest, output_path: Path) -> list[str]:
    video_path = storage.resolve_asset_path(req.video_asset_id)
    return [
        "-m", "ltx_pipelines.retake",
        "--distilled-checkpoint-path", str(config.DISTILLED_CKPT),
        "--gemma-root", str(config.GEMMA_DIR),
        "--prompt", req.prompt,
        "--output-path", str(output_path),
        "--seed", str(req.seed),
        "--video-path", str(video_path),
        "--start-time", str(req.start_time),
        "--end-time", str(req.end_time),
    ]


def build_text_to_audio_args(req: TextToAudioRequest, output_path: Path) -> list[str]:
    num_frames = video_shape.resolve_num_frames(req.duration_seconds, req.num_frames, req.frame_rate)
    args = [
        "-m", "ltx_pipelines.t2a_one_stage",
        "--checkpoint-path", str(config.DEV_CKPT),
        "--gemma-root", str(config.GEMMA_DIR),
        "--prompt", req.prompt,
        "--output-path", str(output_path),
        "--seed", str(req.seed),
        "--num-frames", str(num_frames),
        "--frame-rate", str(req.frame_rate),
    ]
    if req.negative_prompt:
        args += ["--negative-prompt", req.negative_prompt]
    return args


# ---------------------------------------------------------------------------
# Submission plumbing (shared by every LTX pipeline recipe above).
# ---------------------------------------------------------------------------


def _submit_slurm_job(job_id: str, job_dir: Path, module_args: list[str], output_path: Path, partition: str) -> str:
    """Write args.json + submit via sbatch (non-blocking). Returns the Slurm job id."""
    partial_path = job_dir / f"output.partial{output_path.suffix}"
    args = list(module_args)
    idx = args.index("--output-path")
    args[idx + 1] = str(partial_path)

    full_argv = [str(config.PIPELINE_PYTHON), *args]
    (job_dir / "args.json").write_text(json.dumps(full_argv))
    (job_dir / "output_path.txt").write_text(str(output_path))

    job_name = f"ltx23-api-{job_id[:8]}"
    wrap_argv = ["python3", str(config.RUN_PIPELINE_JOB_SCRIPT), str(job_dir)]
    wrap_str = "HF_HUB_OFFLINE=1 " + shlex.join(wrap_argv)
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        "--gres=gpu:1",
        f"--cpus-per-task={config.CPUS_PER_GPU}",
        f"--mem-per-gpu={config.MEM_PER_GPU}",
        f"--time={config.JOB_TIME_LIMIT}",
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}{job_name}",
        f"--output={job_dir / 'slurm.log'}",
        f"--wrap={wrap_str}",
    ]
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DispatchError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip().split(";")[0]  # --parsable prints "<id>" or "<id>;<cluster>"


def _dispatch(pipeline: str, req, output_suffix: str, build_args) -> str:
    """Shared submit path: validate capacity + assets, then create the job
    directory/DB row/Slurm job, in that order, so an invalid request never
    leaves debris behind."""
    _check_capacity()
    partition = resolve_partition(getattr(req, "partition", None))
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    output_path = job_dir / f"output{output_suffix}"

    # Raises KeyError/FileNotFoundError here (before anything is created on
    # disk or in the DB) if an asset_id is invalid.
    module_args = build_args(req, output_path)

    job_dir.mkdir(parents=True)
    job_store.create_job(job_id, pipeline, job_dir, req.model_dump(mode="json"), partition)
    try:
        slurm_job_id = _submit_slurm_job(job_id, job_dir, module_args, output_path, partition)
    except DispatchError:
        job_store.mark_failed(job_id, "Could not submit the Slurm job.")
        raise
    job_store.set_slurm_job_id(job_id, slurm_job_id)
    return job_id


def dispatch_text_to_video(req: TextToVideoRequest) -> str:
    return _dispatch("ltx:text-to-video", req, ".mp4", build_text_to_video_args)


def dispatch_keyframe_interpolation(req: KeyframeInterpolationRequest) -> str:
    return _dispatch("ltx:keyframe-interpolation", req, ".mp4", build_keyframe_interpolation_args)


def dispatch_audio_to_video(req: AudioToVideoRequest) -> str:
    return _dispatch("ltx:audio-to-video", req, ".mp4", build_audio_to_video_args)


def dispatch_retake(req: RetakeRequest) -> str:
    return _dispatch("ltx:retake", req, ".mp4", build_retake_args)


def dispatch_text_to_audio(req: TextToAudioRequest) -> str:
    return _dispatch("ltx:text-to-audio", req, ".wav", build_text_to_audio_args)
