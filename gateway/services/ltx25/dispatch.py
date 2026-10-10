"""Builds LTX-2.5's CLI invocation and submits it as a Slurm job.

Same shape as services/ltx/dispatch.py (LTX-2.3): build the argv for one
`python -m ltx_pipelines.<recipe>` run, then submit it non-blocking via
sbatch -- see common/run_pipeline_job.py for what actually executes on the
compute node. The differences from 2.3 are in the argv itself: LTX-2.5's
weights are a split pack (one file per component, passed as --transformer-path,
--text-encoder-path, ...), not one checkpoint plus a Gemma directory.

Every flag used here was verified against the real argparse definitions in
the vendored ltx-2.5/LTX-2 source (and its saved --help output under
ltx-2.5/logs/), not guessed.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from common import config as common_config
from common import job_store, storage
from common.slurm import resolve_partition
from services.ltx import video_shape
from services.ltx25 import config
from services.ltx25.schemas import Ltx25InterpolateRequest, Ltx25RetakeRequest, Ltx25TextToVideoRequest


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


def _split_weight_args(*, duration_head: bool = False) -> list[str]:
    """The component flags every LTX-2.5 recipe here shares. The duration head
    is only loaded when the clip length is being predicted from the prompt."""
    args = [
        "--transformer-path", str(config.DISTILLED_TRANSFORMER),
        "--text-encoder-path", str(config.TEXT_ENCODER),
        "--video-vae-path", str(config.VIDEO_VAE),
        "--audio-vae-path", str(config.AUDIO_VAE),
    ]
    if duration_head:
        args += ["--duration-head-path", str(config.DURATION_HEAD)]
    return args


# ---------------------------------------------------------------------------
# Per-recipe argv builders. Each returns the args *after* the interpreter
# (i.e. starting with "-m"), with --output-path already set to `output_path`
# (_submit_slurm_job rewrites it to a partial path before actually submitting).
# ---------------------------------------------------------------------------


def _video_generation_args(
    *,
    mode: str,
    prompt: str,
    output_path: Path,
    seed: int,
    height: int,
    width: int,
    frame_rate: float,
    num_frames: int | None,
    enhance_prompt: bool,
) -> list[str]:
    """The argv shared by every recipe that generates a video from a prompt
    (text/image-to-video and interpolation): `mode` picks the pipeline
    (fast = distilled, quality = DFR), everything else is the same flags.
    Image-conditioning flags are left for the caller to append.

    `num_frames=None` means "let the duration head predict the length"."""
    if mode == "fast":
        module = "ltx_pipelines.distilled"
        recipe_args: list[str] = []
    else:
        module = "ltx_pipelines.dfr_pipeline"
        recipe_args = ["--detailing-lora", str(config.DETAILING_LORA)]

    args = [
        "-m", module,
        *_split_weight_args(duration_head=num_frames is None),
        *recipe_args,
        "--spatial-upsampler-path", str(config.SPATIAL_UPSCALER),
        # `--prompt=<text>` (one token) rather than two, so a prompt that
        # starts with "-" is never mistaken for another flag by argparse.
        f"--prompt={prompt}",
        "--output-path", str(output_path),
        "--seed", str(seed),
        "--height", str(height),
        "--width", str(width),
        "--frame-rate", str(frame_rate),
    ]
    if num_frames is not None:
        # Omitting --num-frames is what asks the pipeline to predict the
        # length, so it must always be passed explicitly otherwise.
        args += ["--num-frames", str(num_frames)]
    if enhance_prompt:
        # The bundled Gemma 4 text encoder can only encode, not generate, so
        # enhancement needs a separate generative instruct Gemma (see config).
        args += ["--enhance-prompt", "--prompt-enhancer-gemma-root", str(config.ENHANCER_GEMMA_DIR)]
    return args


def build_text_to_video_args(req: Ltx25TextToVideoRequest, output_path: Path) -> list[str]:
    height, width = video_shape.resolve_dimensions(req.orientation, req.height, req.width)
    num_frames = (
        None
        if req.auto_duration
        else video_shape.resolve_num_frames(req.duration_seconds, req.num_frames, req.frame_rate)
    )
    args = _video_generation_args(
        mode=req.mode,
        prompt=req.prompt,
        output_path=output_path,
        seed=req.seed,
        height=height,
        width=width,
        frame_rate=req.frame_rate,
        num_frames=num_frames,
        enhance_prompt=req.enhance_prompt,
    )
    args += _image_args(req.images)
    return args


def build_interpolate_args(req: Ltx25InterpolateRequest, output_path: Path) -> list[str]:
    """First/last-frame interpolation: the first image is placed at frame 0
    (the literal first frame) and the last image at the final frame
    (num_frames - 1), both at full strength. Frame 0 replaces the first
    latent; any later frame acts as keyframe guidance -- the same mechanism
    LTX-2.5's dedicated keyframe-interpolation pipeline uses, available on
    both the distilled and DFR pipelines without any extra weights."""
    height, width = video_shape.resolve_dimensions(req.orientation, req.height, req.width)
    num_frames = video_shape.resolve_num_frames(req.duration_seconds, req.num_frames, req.frame_rate)
    # Raises KeyError/FileNotFoundError (-> 400) for an unknown asset_id.
    first_path = storage.resolve_asset_path(req.first_frame_asset_id)
    last_path = storage.resolve_asset_path(req.last_frame_asset_id)
    args = _video_generation_args(
        mode=req.mode,
        prompt=req.prompt,
        output_path=output_path,
        seed=req.seed,
        height=height,
        width=width,
        frame_rate=req.frame_rate,
        num_frames=num_frames,
        enhance_prompt=req.enhance_prompt,
    )
    args += [
        "--image", str(first_path), "0", "1.0",
        "--image", str(last_path), str(num_frames - 1), "1.0",
    ]
    return args


def build_retake_args(req: Ltx25RetakeRequest, output_path: Path) -> list[str]:
    video_path = storage.resolve_asset_path(req.video_asset_id)
    return [
        "-m", "ltx_pipelines.retake",
        *_split_weight_args(),
        f"--prompt={req.prompt}",
        "--output-path", str(output_path),
        "--seed", str(req.seed),
        "--video-path", str(video_path),
        "--start-time", str(req.start_time),
        "--end-time", str(req.end_time),
    ]


# ---------------------------------------------------------------------------
# Submission plumbing (shared by every LTX-2.5 recipe above).
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

    job_name = f"ltx25-api-{job_id[:8]}"
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


def dispatch_text_to_video(req: Ltx25TextToVideoRequest) -> str:
    return _dispatch("ltx25:text-to-video", req, ".mp4", build_text_to_video_args)


def dispatch_interpolate(req: Ltx25InterpolateRequest) -> str:
    return _dispatch("ltx25:interpolate", req, ".mp4", build_interpolate_args)


def dispatch_retake(req: Ltx25RetakeRequest) -> str:
    return _dispatch("ltx25:retake", req, ".mp4", build_retake_args)
