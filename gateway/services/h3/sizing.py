"""Per-request sizing for MiniMax H3: how big is this request, how many GPUs does
it need, and about how long will it run.

Why: the memory H3 needs depends on much more than the video's length. The text
encoder reads the prompt AND every reference image/video as one sequence (each
reference image becomes thousands of tokens), and the transformer denoises one
packed sequence of text + references + audio + video rows. Two jobs that failed with
out-of-memory had a short video but two large reference images.

All of this is computed at submit time from the request and the headers of the
uploaded files (Pillow for images, PyAV for video/audio -- nothing is decoded), so a
request that cannot fit is refused with the reason (HTTP 422) instead of failing on a
compute node after minutes of loading weights.

The formulas below mirror what diffusers' MiniMax-H3 pipeline does (the references to
its code are in the comments). The token counts were checked against the two failed
jobs (13,673 and 16,809 encoder tokens, exact); the limits and times in
services/h3/config.py are INITIAL GUESSES, to be refitted from the usage log.

Never raises anything but SizingError for a bad request; a file that cannot be read
at all is a SizingError too (it would fail on the node as well).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from services.h3 import config, video_shape

# Bump when a formula or default below changes, so old usage-log lines can be told apart.
RULE_VERSION = 1

# Model facts (diffusers' MiniMax-H3 pipeline), not operator settings.
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
REFERENCE_IMAGE_SHORT_EDGE = 2048  # reference images are resized to this short edge, upscaling included
MIN_ASPECT_RATIO, MAX_ASPECT_RATIO = 1 / 4, 4
LATENT_FRAMES_PER_CHUNK = 5  # 17 * n + 5 video frames -> 5 * n + 2 latent frames
AUDIO_ROWS_PER_SECOND = 80  # 40 audio latents per second, two stereo channels
ENCODER_VIDEO_SAMPLE_FPS = 2  # the encoder reads a reference video at 2 fps, two sampled frames per block
VIDEO_BLOCK_OVERHEAD_TOKENS = 8  # a block's start/end markers and its "<0.2 seconds>" timestamp


class SizingError(ValueError):
    """The request cannot run on this server's H3 setup. The message says why, with the numbers."""


@dataclass
class ReferenceSize:
    """One reference (or keyframe), as numbers only."""

    kind: str  # "keyframe", "image", "video" or "audio"
    width: int | None = None  # of the uploaded image/video
    height: int | None = None
    seconds: float | None = None  # of the uploaded video/audio
    encoder_tokens: int = 0  # tokens it adds to the text encoder's input
    condition_rows: int = 0  # picture rows it adds to the denoising sequence
    audio_rows: int = 0  # soundtrack rows it adds to the denoising sequence


@dataclass
class Sizing:
    """What the gateway decided for one request. Written to the job folder as sizing.json."""

    rule_version: int
    workflow: str
    gpus: int
    encoder_tokens: int
    denoise_rows: int
    rows_per_gpu: int
    estimated_run_seconds: float
    time_limit_seconds: int
    estimate_parts_seconds: dict
    num_frames: int
    height: int
    width: int
    steps: int
    prompt_chars: int
    flash_attention_3: bool
    references: list[ReferenceSize] = field(default_factory=list)

    def save(self, job_dir: Path) -> None:
        (job_dir / "sizing.json").write_text(json.dumps(asdict(self), indent=1))


# ---------------------------------------------------------------------------
# Geometry (mirrors diffusers' resolve_canvas_size and reference-image resize).
# ---------------------------------------------------------------------------


def _round_to_multiple(value: float) -> int:
    multiple = config.SIZE_MULTIPLE
    return max(multiple, round(value / multiple) * multiple)


def _check_aspect_ratio(width: int, height: int, what: str) -> None:
    if width <= 0 or height <= 0 or not MIN_ASPECT_RATIO <= width / height <= MAX_ASPECT_RATIO:
        raise SizingError(f"{what} is {width}x{height}; H3 needs an aspect ratio between 1:4 and 4:1.")


def canvas_for_aspect(width: int, height: int, what: str = "The picture") -> tuple[int, int]:
    """(height, width) of the canvas H3 resolves for this aspect ratio: short edge 768, area at most
    768 x 1344, both sides a multiple of 32."""
    _check_aspect_ratio(width, height, what)
    ratio = width / height
    if ratio >= 1:
        canvas_width, canvas_height = CANVAS_SHORT_EDGE * ratio, float(CANVAS_SHORT_EDGE)
    else:
        canvas_width, canvas_height = float(CANVAS_SHORT_EDGE), CANVAS_SHORT_EDGE / ratio
    area = canvas_width * canvas_height
    if area > CANVAS_MAX_PIXELS:
        scale = math.sqrt(CANVAS_MAX_PIXELS / area)
        canvas_width, canvas_height = canvas_width * scale, canvas_height * scale
    return _round_to_multiple(canvas_height), _round_to_multiple(canvas_width)


def _tokens_for(height: int, width: int) -> int:
    """Rows/tokens of one picture of this size: the VAE (16x) and the patching (2x) together are 32 px per row,
    and Qwen3-VL's 16 px patches merged 2x2 are 32 px per token as well."""
    return (height // config.SIZE_MULTIPLE) * (width // config.SIZE_MULTIPLE)


def _reference_image_tokens(width: int, height: int) -> int:
    """Tokens (= condition rows) of a reference image: scaled to a 2048 px short edge, no area cap."""
    _check_aspect_ratio(width, height, "A reference image")
    scale = REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    return _tokens_for(_round_to_multiple(height * scale), _round_to_multiple(width * scale))


def _latent_frames(num_frames: int) -> int:
    """Latent frames of a 17 * n + 5 frame video: 5 * n + 2."""
    return LATENT_FRAMES_PER_CHUNK * ((num_frames - config.FRAMES_OFFSET) // config.FRAMES_PER_CHUNK) + 2


# ---------------------------------------------------------------------------
# Reading the uploaded files' headers.
# ---------------------------------------------------------------------------


def _image_size(path: Path) -> tuple[int, int]:
    """(width, height) from the image header only."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            return image.size
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise SizingError(f"An uploaded image could not be read ({type(error).__name__}).") from error


def _stream_seconds(container, stream) -> float | None:
    """The stream's length from the container's headers, or None if they do not say."""
    import av

    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return None


def _video_info(path: Path) -> tuple[int, int, float, bool]:
    """(width, height, seconds, has a soundtrack) from the video's headers only."""
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise SizingError("An uploaded reference video has no video stream.")
            stream = container.streams.video[0]
            seconds = _stream_seconds(container, stream)
            if seconds is None and stream.frames and stream.average_rate:
                seconds = float(stream.frames / stream.average_rate)
            if seconds is None:
                raise SizingError("An uploaded reference video does not say how long it is.")
            width, height = stream.codec_context.width, stream.codec_context.height
            if width <= 0 or height <= 0:
                raise SizingError("An uploaded reference video could not be read (no picture size).")
            return width, height, seconds, bool(container.streams.audio)
    except SizingError:
        raise
    except (av.error.FFmpegError, ValueError, OSError) as error:
        raise SizingError(f"An uploaded reference video could not be read ({type(error).__name__}).") from error


def _audio_seconds(path: Path) -> float:
    """Length in seconds of the file's first audio stream, from its headers only."""
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise SizingError("An uploaded audio reference has no audio stream.")
            seconds = _stream_seconds(container, container.streams.audio[0])
    except SizingError:
        raise
    except (av.error.FFmpegError, ValueError, OSError) as error:
        raise SizingError(f"An uploaded audio reference could not be read ({type(error).__name__}).") from error
    if seconds is None:
        raise SizingError("An uploaded audio reference does not say how long it is.")
    return seconds


# ---------------------------------------------------------------------------
# One reference's contribution.
# ---------------------------------------------------------------------------


def _image_reference(path: Path) -> ReferenceSize:
    width, height = _image_size(path)
    tokens = _reference_image_tokens(width, height)
    return ReferenceSize("image", width=width, height=height, encoder_tokens=tokens, condition_rows=tokens)


def _video_reference(path: Path, num_frames: int, muted: bool) -> ReferenceSize:
    """A reference video is cut to the output's length and put on the canvas of its own aspect ratio. The encoder
    reads it at 2 fps, two sampled frames per block of one canvas worth of tokens. An upper bound for clips over
    about 12 s: past 25 M pixels in all, Qwen3-VL shrinks the frames."""
    width, height, seconds, has_soundtrack = _video_info(path)
    canvas_tokens = _tokens_for(*canvas_for_aspect(width, height, "A reference video"))

    used_frames = max(1, min(int(seconds * config.FPS), num_frames))
    sampled_frames = math.ceil(used_frames / (config.FPS / ENCODER_VIDEO_SAMPLE_FPS))
    blocks = math.ceil(sampled_frames / 2)
    # The VAE encodes the frames snapped DOWN to 17 * n + 5 (at least 22).
    chunks = max(1, (used_frames - config.FRAMES_OFFSET) // config.FRAMES_PER_CHUNK)
    condition_frames = chunks * config.FRAMES_PER_CHUNK + config.FRAMES_OFFSET

    audio_seconds = min(seconds, num_frames / config.FPS)
    return ReferenceSize(
        "video",
        width=width,
        height=height,
        seconds=round(seconds, 2),
        encoder_tokens=blocks * (canvas_tokens + VIDEO_BLOCK_OVERHEAD_TOKENS),
        condition_rows=_latent_frames(condition_frames) * canvas_tokens,
        audio_rows=0 if muted or not has_soundtrack else round(AUDIO_ROWS_PER_SECOND * audio_seconds),
    )


def _audio_reference(seconds: float, output_seconds: float) -> ReferenceSize:
    """An audio reference is cut to the output's length; it only adds rows (the encoder never sees it)."""
    return ReferenceSize("audio", seconds=round(seconds, 2), audio_rows=round(AUDIO_ROWS_PER_SECOND * min(seconds, output_seconds)))


# ---------------------------------------------------------------------------
# The decision.
# ---------------------------------------------------------------------------


def _uses_flash_attention_3() -> bool:
    return "flash_3" in (config.ATTENTION_BACKEND or "")


def _estimate_parts(gpus: int, rows: int, condition_rows: int, steps: int) -> dict:
    denoise = config.BASELINE_DENOISE_SECONDS * (rows / config.BASELINE_ROWS) ** config.DENOISE_ROWS_EXPONENT
    denoise = denoise * steps / config.DEFAULT_STEPS / config.DENOISE_SPEEDUP[gpus]
    if not _uses_flash_attention_3():
        denoise *= config.NO_FA3_SLOWDOWN
    return {
        "load": config.LOAD_SECONDS,
        "copy": config.COPY_SECONDS,
        "reference_encode": round(config.REFERENCE_ENCODE_SECONDS_PER_ROW * condition_rows, 1),
        "denoise": round(denoise, 1),
        "decode": config.DECODE_SECONDS,
    }


def _time_limit_seconds(estimate: float) -> int:
    wanted = config.TIME_LIMIT_FACTOR * estimate + config.TIME_LIMIT_MARGIN_SECONDS
    return int(min(max(wanted, config.MIN_TIME_LIMIT_SECONDS), config.job_time_limit_seconds()))


def _size(
    *,
    workflow: str,
    prompt: str,
    steps: int | None,
    num_frames: int,
    canvas: tuple[int, int],
    references: list[ReferenceSize],
) -> Sizing:
    """Totals, then the smallest GPU count that fits (and finishes within the job time limit), or SizingError."""
    height, width = canvas
    steps = steps or config.DEFAULT_STEPS
    output_seconds = num_frames / config.FPS

    prompt_tokens = math.ceil(len(prompt) / config.PROMPT_CHARS_PER_TOKEN) + config.TOKENS_PER_REFERENCE_LABEL * len(references)
    encoder_tokens = prompt_tokens + sum(reference.encoder_tokens for reference in references)
    condition_rows = sum(reference.condition_rows for reference in references)
    own_rows = (
        _latent_frames(num_frames) * _tokens_for(height, width)  # the video being made
        + round(AUDIO_ROWS_PER_SECOND * output_seconds)  # and its soundtrack
    )
    rows = encoder_tokens + condition_rows + sum(reference.audio_rows for reference in references) + own_rows

    if encoder_tokens > config.MAX_ENCODER_TOKENS:
        raise SizingError(
            f"The prompt and references add up to about {encoder_tokens:,} text-encoder tokens, more than this server's "
            f"limit of {config.MAX_ENCODER_TOKENS:,}. Use fewer or smaller reference images and videos."
        )

    last_problem = ""
    for gpus in config.GPU_OPTIONS:
        rows_per_gpu = math.ceil(rows / gpus)
        if rows_per_gpu > config.ROWS_PER_GPU:
            last_problem = (
                f"The video needs about {rows:,} rows of GPU memory work; on {gpus} GPUs that is {rows_per_gpu:,} per GPU, "
                f"more than this server's limit of {config.ROWS_PER_GPU:,}. Use a shorter video, a smaller size or fewer references."
            )
            continue
        parts = _estimate_parts(gpus, rows, condition_rows, steps)
        estimate = sum(parts.values())
        if estimate > config.job_time_limit_seconds():
            last_problem = (
                f"The estimated run time on {gpus} GPUs is {estimate / 60:.0f} minutes, more than the "
                f"{config.job_time_limit_seconds() / 60:.0f} minute limit of a job. Use a shorter video or fewer steps."
            )
            continue
        return Sizing(
            rule_version=RULE_VERSION,
            workflow=workflow,
            gpus=gpus,
            encoder_tokens=encoder_tokens,
            denoise_rows=rows,
            rows_per_gpu=rows_per_gpu,
            estimated_run_seconds=round(estimate, 1),
            time_limit_seconds=_time_limit_seconds(estimate),
            estimate_parts_seconds=parts,
            num_frames=num_frames,
            height=height,
            width=width,
            steps=steps,
            prompt_chars=len(prompt),
            flash_attention_3=_uses_flash_attention_3(),
            references=references,
        )
    raise SizingError(last_problem)


def for_gpus(sizing: dict, gpus: int) -> dict:
    """A saved sizing.json (as a dict) redone for a job that runs on `gpus` GPUs instead: the per-GPU rows, the time
    estimate and the Slurm time limit change; the request's own numbers (tokens, rows, canvas, references) do not.
    Used when a 4-GPU job is run again on 8 after running out of memory (services/h3/oom_retry.py)."""
    rows = int(sizing["denoise_rows"])
    condition_rows = sum(int(reference.get("condition_rows") or 0) for reference in sizing.get("references", []))
    parts = _estimate_parts(gpus, rows, condition_rows, int(sizing["steps"]))
    estimate = sum(parts.values())
    return {
        **sizing,
        "gpus": gpus,
        "rows_per_gpu": math.ceil(rows / gpus),
        "estimated_run_seconds": round(estimate, 1),
        "time_limit_seconds": _time_limit_seconds(estimate),
        "estimate_parts_seconds": parts,
    }


# ---------------------------------------------------------------------------
# Entry points, one per endpoint. Called by dispatch.py before anything is created.
# ---------------------------------------------------------------------------


def size_generate(req, first_path: Path | None, last_path: Path | None) -> Sizing:
    """/videos/generate: text only (t2va), or with a first and/or last frame (fl2va)."""
    num_frames = video_shape.resolve_num_frames(req.duration_seconds, req.num_frames)
    keyframe_paths = [path for path in (first_path, last_path) if path is not None]
    canvas = video_shape.resolve_canvas(req.orientation, req.height, req.width)
    if canvas is None:
        # H3's own canvas: 16:9, or the aspect ratio of the first keyframe that was given.
        canvas = canvas_for_aspect(*_image_size(keyframe_paths[0]), "A keyframe") if keyframe_paths else canvas_for_aspect(16, 9)
    keyframe_tokens = _tokens_for(*canvas)
    references = [
        ReferenceSize("keyframe", encoder_tokens=keyframe_tokens, condition_rows=keyframe_tokens) for _ in keyframe_paths
    ]
    return _size(
        workflow="fl2va" if keyframe_paths else "t2va",
        prompt=req.prompt,
        steps=req.num_inference_steps,
        num_frames=num_frames,
        canvas=canvas,
        references=references,
    )


def size_audio_to_video(req, image_paths: list[Path], video_paths: list[Path], audio_paths: list[Path]) -> Sizing:
    """/videos/audio-to-video (ref2va): image, video and audio references."""
    audio_seconds = [_audio_seconds(path) for path in audio_paths]
    if req.match_audio_duration:
        # The same rule the runner applies once it has decoded the audio.
        if audio_seconds[0] > config.MAX_DURATION_SECONDS:
            raise SizingError(
                f"The first audio reference is {audio_seconds[0]:.1f} s, longer than the {config.MAX_DURATION_SECONDS:g} s "
                "maximum this server allows."
            )
        num_frames = video_shape.snap_frames(round(audio_seconds[0] * config.FPS))
    else:
        num_frames = video_shape.resolve_num_frames(req.duration_seconds, req.num_frames)
    canvas = video_shape.resolve_canvas(req.orientation, req.height, req.width) or canvas_for_aspect(16, 9)

    muted_flags = req.video_audio or [True] * len(video_paths)
    references = [_image_reference(path) for path in image_paths]
    references += [
        _video_reference(path, num_frames, muted=not use_soundtrack) for path, use_soundtrack in zip(video_paths, muted_flags)
    ]
    references += [_audio_reference(seconds, num_frames / config.FPS) for seconds in audio_seconds]
    return _size(
        workflow="ref2va",
        prompt=req.prompt,
        steps=req.num_inference_steps,
        num_frames=num_frames,
        canvas=canvas,
        references=references,
    )
