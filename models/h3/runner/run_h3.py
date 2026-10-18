#!/usr/bin/env python3
"""Runs MiniMax H3 for one video request -- the single unit of work the gateway's
Slurm job submits for `POST /v1/h3/videos/generate` and
`POST /v1/h3/videos/audio-to-video`. See `gateway/services/h3/dispatch.py`, which
builds the argv below, and `gateway/common/run_pipeline_job.py`, which invokes
this on the compute node (that wrapper owns the partial-path-rename-on-success /
`.failed`-marker-on-failure convention; this script only needs to write
`--output-path` and exit non-zero on any real failure).

Must be run with this model's own environment
(`$VISION_DATA_DIR/envs/h3/bin/python`): diffusers 0.41 (its MiniMax-H3 modular
pipeline), transformers 5.x and torch only exist there.

SEVERAL GPUS PER JOB (--gpus N: 4 or 8 for ulysses, which is what the gateway
always asks for; 2 is the older two-GPU layout). The Slurm job holds N GPUs on one
node, g0 .. g(N-1) in CUDA_VISIBLE_DEVICES order, and this script uses all of
them, splitting the pipeline the way diffusers' own MiniMax-H3 documentation does
("with two cards, nothing offloaded") and then using every GPU for denoising:

  * the conditioning half -- media preparation plus the Qwen3-VL-32B text/vision
    encoder (62 GiB of bf16 weights) -- runs in a CHILD process on g1 and g2 (with
    2 GPUs only g1): from N = 4 up the encoder's 64 decoder layers are split between
    two GPUs (layers 0-31 on g1, 32-63 on g2, see `split_text_encoder`), which leaves
    room for the tokens of many large reference images;
  * the generation half -- reference/keyframe VAE encoding, the 33B transformer
    (61.7 GiB), and the video/audio decode -- runs in THIS process on g0 (rank 0).

Both halves load their weights at the same time (two processes, so the two
sets of weights are read from disk in parallel -- threads would not be safe,
because accelerate's weight-loading context patches torch globally). When the
child is done it hands the small
conditioning result (the text embedding and the resolved canvas/frame
count/keyframes/references) over as one file in --work-dir, and exits, which
frees its GPUs.

The freed GPUs are then used, together with the others, for denoising
(--parallelism, which the gateway always sets to ulysses):

  * ulysses (default): ALL N GPUs compute every denoising step. Each holds the whole
    transformer (61.7 GiB; g0's copy is broadcast straight to the other GPUs over
    NCCL, nothing is read from disk again) and runs 1/N of the packed token sequence
    through every block; around each attention layer the GPUs exchange their parts of
    the queries/keys/values (diffusers' "Ulysses" context parallelism, see
    `enable_parallel_denoiser`). Ranks 1 .. N-1 are driven by copies of this script
    (`--stage denoise-worker --rank r`, see `run_denoise_worker_stage`), started
    together with the encoder, each running the same loop on its own part. Every rank
    ends up with the full denoised latents; rank 0 decodes them. Memory: every GPU
    holds the whole transformer, so about 15 GiB per GPU is left for activations, but
    each GPU only sees 1/N of the tokens, so N GPUs take N times the rows of one.
  * split (2 GPUs only; an operator fallback): the transformer's 50 blocks are divided,
    blocks 0-24 stay on the first GPU and blocks 25-49 move to the second (see
    `split_transformer`). Each GPU then holds about 31 GiB of weights, so a long clip
    at the full 1344x768 canvas (345 frames is ~103k tokens) has ~45 GiB per GPU for
    activations, where one GPU holding all 61.7 GiB ran out of memory. The blocks still
    run one after another, so each GPU waits while the other computes; per denoising
    step the hidden states cross between the GPUs once each way, plus a copy of the
    (~0.1 GB) rotary table for each moved block, which costs seconds over a whole job.

Output quality does not depend on any of this: the same bf16 weights run the
same computation as on one big GPU (ulysses differs from it only by floating-point
rounding in the exchange of the attention inputs).

Numbers for calibration: with --metrics-dir, every process writes `metrics.<role>.json`
there (the gateway passes the job folder): stage timings, peak GPU memory per GPU, the
encoder's token count, the packed sequence's length and, on failure, the stage that
failed (see `Metrics`). Only numbers and stage names -- never the prompt or any media.
The gateway merges these files into its usage log.

Three workflows, matching diffusers' (`pipe.blocks.available_workflows`):
  t2va  : text only                       -> `transformer/` partition ("FL2VA" checkpoint)
  fl2va : first and/or last keyframe      -> `transformer/` partition
  ref2va: image/video/audio references    -> `transformer_ref/` partition ("Ref2VA" checkpoint)

Usage:
    run_h3.py --model-dir DIR --workflow t2va|fl2va|ref2va --output-path FILE \\
        --work-dir DIR --prompt=TEXT \\
        [--first-frame FILE] [--last-frame FILE] \\
        [--reference-image FILE ...] [--reference-video FILE ...] [--mute-reference-video N ...] \\
        [--reference-audio FILE ...] \\
        [--height N --width N] [--num-frames N | --match-audio-duration] [--max-seconds S] \\
        [--steps N] [--seed N] [--attention-backend NAME] \\
        [--gpus 2|4|8] [--parallelism ulysses|split] [--metrics-dir DIR]

--prompt is passed as `--prompt=value` (not `--prompt value`) so a value starting
with "-" is never mistaken for an option. Nothing here goes through a shell, so
user text needs no escaping.

There is no prompt rewriting: the text goes to the encoder exactly as given.
(MiniMax's hosted service runs a separate "Context-IR" model that expands a
request into the structured prompt format; that model is not in the open release.
Write the structured prompt yourself -- see gateway/services/h3/GUIDE.md.)

Frames: 24 fps, always 17 * n + 5 frames. A --num-frames (or an audio-derived length)
off that grid is rounded up to it; there is no minimum or maximum. The model
is documented for 5-15 s, so anything outside that is EXPERIMENTAL and diffusers'
own 5-15 s check is switched off (see `allow_any_length`). --max-seconds, when given,
is the only ceiling: a --match-audio-duration clip longer than it fails with an
ERROR instead of holding the GPUs. --height and
--width go together and must be multiples of 32. Without them the canvas is the
model's own 768-short-edge 16:9 (1344x768), or, for fl2va, the aspect ratio of the
first keyframe.

Output is one MP4 (H.264 video + AAC stereo audio at 32 kHz), written by PyAV.

Exit code 0 only if --output-path was actually written. On failure the last line
printed is `ERROR: ...` with the root cause.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import timedelta
from pathlib import Path

FPS = 24
# The video VAE decodes 17 * n + 5 frames. The model is documented for 5-15 seconds (124 to 345 frames), but
# that window is no longer enforced here.
FRAMES_PER_CHUNK = 17
FRAMES_OFFSET = 5
DEFAULT_FRAMES = 124  # 5.17 s
# The audio VAE's sample rate, used only for a reference whose own rate is unknown.
AUDIO_SAMPLE_RATE = 32000

WORKFLOWS = ("t2va", "fl2va", "ref2va")
# ref2va reference limits, the same ones diffusers enforces (MiniMaxH3Ref2VASetupStep): per kind, and in all.
MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIOS = 3
MAX_REFERENCES = 12
HANDOFF_NAME = "h3_conditioning.pt"
HANDOFF_ERROR_NAME = "h3_conditioning_error.txt"
OUTPUTS = ["videos", "audio", "sampling_rate"]
# The transformer has 50 blocks; in --parallelism split mode the first SPLIT_AT stay on the generation GPU during
# denoising and the rest run on the conditioning GPU, which is idle once the encoder's process has exited.
SPLIT_AT = 25

# How the job's GPUs share the denoising (see the module docstring). The gateway always passes ulysses.
PARALLELISM_MODES = ("ulysses", "split")
DEFAULT_PARALLELISM = "ulysses"
# --gpus: how many of the job's GPUs this run uses. 2 is the older layout (encoder on one GPU); from 4 up the encoder is
# split across two GPUs. Ulysses needs a degree that divides the transformer's 56 attention heads: 2, 4 and 8 all do.
GPU_OPTIONS = (2, 4, 8)
DEFAULT_GPUS = 2
# The encoder (Qwen3-VL-32B) has 64 decoder layers; when it has two GPUs, layers [0, SPLIT) go to the first, the rest
# (and the final norm) to the second.
TEXT_ENCODER_SPLIT_AT = 32

# --parallelism ulysses: ranks 1 .. gpus-1 ("denoise workers", one extra process each) run the same denoising loop as
# this process (rank 0, on g0), each on cuda:<rank> = the job's GPU number <rank>.
ENCODED_NAME = "h3_encoded.pt"  # reference/keyframe latents, handed from rank 0 to the workers
WORKER_ERROR_NAME = "h3_worker_error_{rank}.txt"
DIST_PORT_ENV = "H3_DIST_PORT"  # localhost port of the rendezvous store, chosen by rank 0
# The worker is started together with the encoder and waits for rank 0 to open the store, which only happens after
# the weights are loaded, the encoder is done and the references are encoded.
CONNECT_TIMEOUT = timedelta(minutes=45)
RENDEZVOUS_TIMEOUT = timedelta(minutes=10)
# Longest any single GPU-to-GPU operation may take. A dead peer is normally noticed much sooner (see watch_worker).
COLLECTIVE_TIMEOUT = timedelta(minutes=10)

_STARTED = time.time()


def log(tag: str, message: str) -> None:
    print(f"[{time.time() - _STARTED:6.0f}s {tag}] {message}", flush=True)


_SIZE = r"([\d.]+) (MiB|GiB)"
_OOM_PATTERNS = {
    "tried_mib": r"Tried to allocate " + _SIZE,
    "total_mib": r"total capacity of " + _SIZE,
    "free_mib": r"of which " + _SIZE + r" is free",
    "in_use_mib": r"this process has " + _SIZE + r" memory in use",
    "allocated_mib": _SIZE + r" is allocated by PyTorch",
}


def oom_numbers(message: str) -> dict:
    """The figures in a CUDA out-of-memory message, all in MiB (plus which GPU), or {} for any other message."""
    if "out of memory" not in message:
        return {}
    numbers: dict = {}
    for key, pattern in _OOM_PATTERNS.items():
        found = re.search(pattern, message)
        if found:
            numbers[key] = round(float(found.group(1)) * (1024 if found.group(2) == "GiB" else 1), 1)
    gpu = re.search(r"GPU (\d+) has a total capacity", message)
    if gpu:
        numbers["gpu"] = int(gpu.group(1))
    return numbers


class Metrics:
    """Numbers about one process of a run, for the gateway's usage log: stage timings, peak GPU memory, token and row
    counts and, when it fails, the stage and (for an out-of-memory error) the sizes. Written to
    `<--metrics-dir>/metrics.<role>.json` after every change, atomically, so a process that is killed (a Slurm time
    limit, an out-of-memory kill) still leaves what it had measured, with `current_stage` saying where it was.

    Only numbers and stage names go in: never the prompt, file names or media, and not even the error message
    (only its type and the out-of-memory sizes). Without --metrics-dir nothing is written. A failure to write is
    ignored: these numbers must never fail a job."""

    def __init__(self, directory: Path | None, role: str):
        self.path = None if directory is None else Path(directory) / f"metrics.{role}.json"
        self.current: str | None = None
        self._lock = threading.RLock()
        self.data: dict = {"schema": 1, "role": role, "outcome": "running", "timings_s": {}, "peak_gib": {}}
        self.write()

    def write(self) -> None:
        if self.path is None:
            return
        with self._lock:
            try:
                partial = self.path.with_name(self.path.name + ".partial")
                partial.write_text(json.dumps(self.data))
                os.replace(partial, self.path)
            except OSError:
                pass

    def update(self, **values) -> None:
        with self._lock:
            self.data.update(values)
            self.write()

    def timing(self, name: str, seconds: float) -> None:
        with self._lock:
            self.data["timings_s"][name] = round(seconds, 1)
            self.write()

    def peak(self, name: str, gib) -> None:
        """Peak GPU memory in GiB: one number, or a list with one per GPU."""
        with self._lock:
            self.data["peak_gib"][name] = [round(g, 2) for g in gib] if isinstance(gib, (list, tuple)) else round(gib, 2)
            self.write()

    @contextlib.contextmanager
    def stage(self, name: str):
        """Times the block under `name`; if it raises, records the failure (see `fail`) against this stage."""
        with self._lock:
            self.current = name
            self.data["current_stage"] = name
            self.write()
        started = time.time()
        try:
            yield
        except BaseException as error:
            self.fail(error)
            raise
        with self._lock:
            self.current = None
            self.data.pop("current_stage", None)
            self.timing(name, time.time() - started)

    def fail(self, error, source: str | None = None) -> None:
        """Records the failure of the stage in progress (the first failure recorded wins). `error` is an exception, or the
        text of one that happened in another process (`source`: which)."""
        with self._lock:
            if self.data["outcome"] == "failed":
                return
            message = error if isinstance(error, str) else str(error)
            self.data["outcome"] = "failed"
            self.data["failed_stage"] = self.current or self.data.get("current_stage") or "unknown"
            self.data["error_type"] = message.split(":", 1)[0][:60] if isinstance(error, str) else type(error).__name__
            if source:
                self.data["failed_in"] = source
            oom = oom_numbers(message)
            if oom:
                self.data["oom"] = oom
            self.write()

    def finish(self, ok: bool) -> None:
        with self._lock:
            if self.data["outcome"] != "failed":
                self.data["outcome"] = "succeeded" if ok else "failed"
                if not ok:
                    self.data.setdefault("failed_stage", self.current or self.data.get("current_stage") or "unknown")
            self.data.pop("current_stage", None)
            self.data["total_s"] = round(time.time() - _STARTED, 1)
            self.write()


def describe_references(references) -> list[dict]:
    """Kind and size of each (normalized) reference, as numbers: an image's pixels, a video's frames and pixels, an
    audio file's seconds. Never raises: this is for the usage log only."""
    described = []
    try:
        for reference in references or []:
            entry: dict = {"kind": reference.kind}
            if reference.kind == "image":
                entry["width"], entry["height"] = reference.image.size
            elif reference.kind == "video":
                entry.update(
                    frames=int(reference.frames.shape[0]),
                    height=int(reference.frames.shape[1]),
                    width=int(reference.frames.shape[2]),
                    has_audio=bool(reference.has_audio),
                )
            else:
                entry["seconds"] = round(reference.audio.shape[-1] / (reference.sample_rate or AUDIO_SAMPLE_RATE), 2)
            described.append(entry)
    except Exception:
        return []
    return described


DEFAULT_STEPS = 50  # the pipeline's own number of denoising steps when --steps is not given


def record_run_setup(metrics: Metrics, args: argparse.Namespace) -> None:
    """The settings of this run and the machine, as numbers and short names (no prompt text, no file names). Never
    raises: this is for the usage log only."""
    try:
        import diffusers
        import torch

        properties = torch.cuda.get_device_properties(0)
        metrics.update(
            workflow=args.workflow,
            parallelism=args.parallelism,
            gpus=args.gpus,
            attention_backend=args.attention_backend,
            steps=args.steps or DEFAULT_STEPS,
            requested_height=args.height,
            requested_width=args.width,
            requested_frames=None if args.match_audio_duration else snap_num_frames(args.num_frames or DEFAULT_FRAMES),
            match_audio_duration=args.match_audio_duration,
            prompt_chars=len(args.prompt),
            requested_references={
                "images": len(args.reference_image),
                "videos": len(args.reference_video),
                "audios": len(args.reference_audio),
                "keyframes": len([path for path in (args.first_frame, args.last_frame) if path is not None]),
            },
            node=socket.gethostname(),
            gpu_model=properties.name,
            gpu_total_gib=round(properties.total_memory / 2**30, 1),
            torch_version=torch.__version__,
            diffusers_version=diffusers.__version__,
        )
    except Exception:
        pass


def record_sequence_shape(model, metrics: Metrics) -> None:
    """On the transformer's first denoising step, records the size of the packed sequence it is given: its rows in all
    (`sequence_rows`) and how many of them are text, video (reference + generated) and audio rows. Removes itself after
    that one call."""

    def record(module, args, kwargs):
        handle.remove()
        try:
            metrics.update(
                sequence_rows=int(kwargs["position_ids"].shape[0]),
                text_rows=int(kwargs["encoder_hidden_states"].shape[1]),
                video_rows=int(kwargs["hidden_states"].shape[1]),
                audio_rows=int(kwargs["audio_hidden_states"].shape[1]),
            )
        except Exception:
            pass

    handle = model.register_forward_pre_hook(record, with_kwargs=True)


def snap_num_frames(num_frames: int) -> int:
    """Round up to the next 17 * n + 5 the video VAE can decode. No minimum or
    maximum. (Same rule as gateway/services/h3/video_shape.py -- the gateway
    cannot import this file.)"""
    snapped = max(num_frames, 1)
    while snapped % FRAMES_PER_CHUNK != FRAMES_OFFSET:
        snapped += 1
    return snapped


def allow_any_length() -> None:
    """Switch off diffusers' own 5-15 s check.

    Both the ref2va conditioning step (before_encoder.py) and the denoising setup
    (before_denoise.py) raise a ValueError when the video's duration is outside
    `components.min_duration` .. `components.max_duration`, which are properties of
    MiniMaxH3ModularPipeline. They are overridden on the class, so this must run in
    every process that builds a pipeline (the conditioning child and the parent)."""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

    MiniMaxH3ModularPipeline.min_duration = property(lambda self: 0.0)
    MiniMaxH3ModularPipeline.max_duration = property(lambda self: float("inf"))
    log("run", "length limits lifted: diffusers' 5-15 s check is off (outside that range is experimental)")


def check_max_seconds(args: argparse.Namespace, seconds: float) -> None:
    """The one remaining ceiling: --max-seconds, when given. `seconds` is the
    requested length before rounding up onto the frame grid, so a request exactly
    at the ceiling is not rejected for the few frames the rounding adds."""
    if args.max_seconds is not None and seconds > args.max_seconds:
        raise ValueError(f"The video would be {seconds:.1f} s, longer than the {args.max_seconds:g} s maximum this server allows.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, type=Path, help="Local copy of MiniMaxAI/MiniMax-H3 (diffusers layout).")
    p.add_argument("--workflow", required=True, choices=WORKFLOWS)
    p.add_argument("--output-path", required=True, type=Path)
    p.add_argument("--work-dir", required=True, type=Path, help="Scratch directory inside the job folder.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--first-frame", type=Path, help="fl2va: image the video starts from.")
    p.add_argument("--last-frame", type=Path, help="fl2va: image the video ends on.")
    p.add_argument("--reference-image", action="append", default=[], type=Path, help="ref2va: repeatable, at most 9.")
    p.add_argument("--reference-video", action="append", default=[], type=Path, help="ref2va: repeatable, at most 3.")
    p.add_argument(
        "--mute-reference-video",
        action="append",
        default=[],
        type=int,
        metavar="N",
        help="ref2va: repeatable; the 0-based index (in --reference-video order) of a video whose own soundtrack is NOT "
        "given to the model. By default every video that has sound passes it in as an <Audio N> reference.",
    )
    p.add_argument(
        "--reference-audio",
        action="append",
        default=[],
        type=Path,
        help="ref2va: repeatable, at most 3; a voice/music reference. Needs an image or video reference.",
    )
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--num-frames", type=int, help=f"Default {DEFAULT_FRAMES} (5.17 s).")
    p.add_argument(
        "--match-audio-duration",
        action="store_true",
        help="ref2va: size the video to the reference audio (rounded up onto the 17n+5 frame grid; no minimum).",
    )
    p.add_argument("--max-seconds", type=float, help="Fail if the video would be longer than this. Default: no ceiling.")
    p.add_argument("--steps", type=int, help="Denoising steps. Default: the pipeline's own (50).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attention-backend", help="A diffusers attention backend name for the transformer (default: diffusers' own).")
    p.add_argument(
        "--gpus",
        type=int,
        choices=GPU_OPTIONS,
        default=DEFAULT_GPUS,
        help="How many of the job's GPUs to use (the Slurm job must hold at least that many). With 4 or more, the text "
        "encoder is split across two GPUs; every GPU takes part in the denoising (ulysses).",
    )
    p.add_argument(
        "--parallelism",
        choices=PARALLELISM_MODES,
        default=DEFAULT_PARALLELISM,
        help="How the GPUs share the denoising: 'ulysses' (the default: all of them compute every step, each on its share "
        "of the token sequence; every GPU holds the whole transformer) or 'split' (2 GPUs only: the blocks are divided "
        "between them and run one after another; less memory per GPU, slower).",
    )
    p.add_argument(
        "--metrics-dir",
        type=Path,
        help="Write metrics.<role>.json (timings, peak GPU memory, token counts; no prompt or media) into this directory.",
    )
    # Internal: the conditioning half and (in ulysses mode) the other denoising processes are this same script run as
    # child processes.
    p.add_argument("--stage", choices=["run", "condition", "denoise-worker"], default="run", help=argparse.SUPPRESS)
    p.add_argument("--handoff", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args.parallelism == "split" and args.gpus != 2:
        p.error("--parallelism split divides the transformer between exactly 2 GPUs; use --gpus 2.")

    if (args.height is None) != (args.width is None):
        p.error("--height and --width must be given together.")
    if args.height is not None and (args.height % 32 or args.width % 32):
        p.error(f"--height and --width must be multiples of 32, got {args.height}x{args.width}.")
    if not args.model_dir.is_dir():
        p.error(f"Model directory not found: {args.model_dir}")
    if args.steps is not None and args.steps < 1:
        p.error("--steps must be at least 1.")
    if args.max_seconds is not None and args.max_seconds <= 0:
        p.error("--max-seconds must be positive.")

    keyframes = [path for path in (args.first_frame, args.last_frame) if path is not None]
    references = [*args.reference_image, *args.reference_video, *args.reference_audio]
    if args.workflow == "t2va" and (keyframes or references):
        p.error("--workflow t2va takes no keyframes or references.")
    if args.workflow == "fl2va" and (not keyframes or references):
        p.error("--workflow fl2va needs --first-frame and/or --last-frame, and takes no references.")
    if args.workflow == "ref2va":
        if keyframes:
            p.error("--workflow ref2va takes references, not keyframes.")
        if not args.reference_image and not args.reference_video:
            p.error("--workflow ref2va needs at least one --reference-image or --reference-video (audio cannot be the only reference).")
        if len(args.reference_image) > MAX_IMAGES or len(args.reference_video) > MAX_VIDEOS or len(args.reference_audio) > MAX_AUDIOS:
            p.error(f"MiniMax H3 accepts at most {MAX_IMAGES} reference images, {MAX_VIDEOS} videos and {MAX_AUDIOS} audio clips.")
        if len(references) > MAX_REFERENCES:
            p.error(f"MiniMax H3 accepts at most {MAX_REFERENCES} references in all, got {len(references)}.")
    if args.mute_reference_video and args.workflow != "ref2va":
        p.error("--mute-reference-video needs --workflow ref2va.")
    for index in args.mute_reference_video:
        if not 0 <= index < len(args.reference_video):
            p.error(f"--mute-reference-video {index}: no such reference video (there are {len(args.reference_video)}, counted from 0).")
    if args.match_audio_duration and not (args.workflow == "ref2va" and args.reference_audio):
        p.error("--match-audio-duration needs --workflow ref2va and at least one --reference-audio.")
    if args.match_audio_duration and args.num_frames is not None:
        p.error("Give either --num-frames or --match-audio-duration, not both.")
    for path in [*keyframes, *references]:
        if not path.is_file():
            p.error(f"Input file not found: {path}")
    if args.stage == "condition" and args.handoff is None:
        p.error("--stage condition needs --handoff.")
    if args.stage == "denoise-worker" and (args.rank is None or not 1 <= args.rank < args.gpus):
        p.error(f"--stage denoise-worker needs --rank between 1 and {args.gpus - 1}.")
    return args


def pick_gpus(gpus: int) -> list[str]:
    """The first `gpus` GPUs as they are named in CUDA_VISIBLE_DEVICES, which Slurm sets to the job's own GPUs.
    Without it, the first `gpus` on the node. g0 is the generation GPU (rank 0); g1 (and g2 when the encoder has two
    GPUs) run the encoder first and become denoising ranks 1 (and 2) afterwards."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    ids = [item.strip() for item in visible.split(",") if item.strip()] if visible else [str(i) for i in range(gpus)]
    if len(ids) < gpus:
        raise RuntimeError(
            f"This MiniMax H3 run needs {gpus} GPUs (the text/vision encoder on the first ones after g0, the "
            f"transformer on g0), but only {len(ids)} is visible (CUDA_VISIBLE_DEVICES={visible!r})."
        )
    return ids[:gpus]


def encoder_gpu_count(gpus: int) -> int:
    """GPUs the text/vision encoder gets: two from 4 GPUs up (62 GiB of weights plus activations that grow with the
    tokens of the reference images), else one."""
    return 2 if gpus >= 4 else 1


def check_weights(model_dir: Path, workflow: str) -> None:
    transformer = "transformer_ref" if workflow == "ref2va" else "transformer"
    required = [
        "modular_model_index.json",
        f"{transformer}/diffusion_pytorch_model.safetensors.index.json",
        "text_encoder/model.safetensors.index.json",
        "vae/diffusion_pytorch_model.safetensors.index.json",
        "audio_vae/diffusion_pytorch_model.safetensors",
    ]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"MiniMax H3 weights are incomplete in {model_dir} (missing: {', '.join(missing)}). "
            "Run models/h3/download_weights.sh."
        )


def split_workflow(model_dir: Path, workflow: str):
    """The workflow's blocks cut in two, the way diffusers' two-card recipe does:
    (conditioning blocks, everything else).

    Conditioning = media preparation (`before_encode`, absent for text-only) and
    the text/vision encoder (`text_encoder`). Their outputs -- the text
    embedding, its row tags and the resolved canvas / frame count / keyframes /
    references -- are exactly the inputs the rest of the workflow reads by name,
    so they can be handed over as plain values."""
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import SequentialPipelineBlocks

    blocks = ModularPipeline.from_pretrained(str(model_dir)).blocks.get_workflow(workflow)
    parts = {}
    before_encode = blocks.sub_blocks.pop("before_encode", None)
    if before_encode is not None:
        parts["before_encode"] = before_encode
    parts["text_encoder"] = blocks.sub_blocks.pop("text_encoder")
    return SequentialPipelineBlocks.from_blocks_dict(parts), blocks


def make_pipeline(blocks, model_dir: Path, device: str):
    """A pipeline for `blocks` whose components are moved onto `device` when a
    block needs them (nothing else shares that GPU, so a component stays there
    once it is resident). Loading is lazy: the weights are memory-mapped and
    only read from disk on that first move."""
    import torch
    from diffusers import ComponentsManager

    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=device)
    pipeline = blocks.init_pipeline(str(model_dir), components_manager=manager)
    started = time.time()
    pipeline.load_components(dtype=torch.bfloat16)
    return pipeline, time.time() - started


def split_rest_blocks(rest_blocks):
    """The generation half cut into its three stages: (vae_encoder, denoise, decode).

    A workflow's blocks are a flat list whose names carry the stage: `vae_encoder` (reference/keyframe encoding; absent
    for `t2va`, so None is returned for it), `denoise.*` and `decode.*`. `rest_blocks` itself is left untouched."""
    from diffusers.modular_pipelines import SequentialPipelineBlocks

    stages: dict[str, dict] = {"vae_encoder": {}, "denoise": {}, "decode": {}}
    for name, block in rest_blocks.sub_blocks.items():
        stage = name.split(".", 1)[0]
        if stage not in stages:
            raise RuntimeError(f"Unexpected block {name!r} in the generation half of the workflow.")
        stages[stage][name] = block
    if not stages["denoise"] or not stages["decode"]:
        raise RuntimeError(f"The workflow has no denoise or no decode blocks (found {sorted(rest_blocks.sub_blocks)}).")
    groups = {stage: SequentialPipelineBlocks.from_blocks_dict(blocks) if blocks else None for stage, blocks in stages.items()}
    return groups["vae_encoder"], groups["denoise"], groups["decode"]


def sub_pipeline(blocks, source_pipeline, model_dir: Path):
    """A pipeline that runs only `blocks` and uses the components `source_pipeline` has already loaded (the same
    objects: nothing is read from disk again, and their offload hooks stay in place)."""
    pipeline = blocks.init_pipeline(str(model_dir))
    shared = {}
    for spec in blocks.expected_components:
        component = getattr(source_pipeline, spec.name, None)
        if component is not None:
            shared[spec.name] = component
    pipeline.update_components(**shared)
    return pipeline


def select_inputs(blocks, values: dict) -> dict:
    """The entries of `values` that `blocks` read by name (a pipeline warns about every other keyword it is given)."""
    names = {param.name for param in blocks.inputs if param.name}
    return {key: value for key, value in values.items() if key in names}


def output_names(blocks) -> list[str]:
    return [output.name for output in blocks.intermediate_outputs if output.name]


def denoiser_component_name(workflow: str) -> str:
    return "transformer_ref" if workflow == "ref2va" else "transformer"


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _is_cuda_tensor(value) -> bool:
    import torch

    return isinstance(value, torch.Tensor) and value.is_cuda


def pack_values(values: dict) -> dict:
    """`values` as plain CPU values (one file, readable by a process that sees another GPU), plus the names of the
    entries that were on the GPU: the rest of the pipeline expects each tensor where a single-process run would have it
    (the text embedding on the GPU, but `text_token_tags` on the CPU, where the layout step indexes CPU tensors with
    it). A list of tensors (the reference latents) is one entry and counts as on the GPU if any of them is."""
    import torch

    def to_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.cpu()
        if isinstance(value, (list, tuple)):
            return type(value)(to_cpu(item) for item in value)
        return value

    def on_gpu(value) -> bool:
        if isinstance(value, (list, tuple)):
            return any(on_gpu(item) for item in value)
        return _is_cuda_tensor(value)

    return {
        "values": {key: to_cpu(value) for key, value in values.items()},
        "on_gpu": sorted(key for key, value in values.items() if on_gpu(value)),
    }


def unpack_values(packed: dict, device) -> dict:
    """The inverse of `pack_values`: the entries that were on the GPU go to `device`, the rest stay as they are."""
    import torch

    def to_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, (list, tuple)):
            return type(value)(to_device(item) for item in value)
        return value

    values = dict(packed["values"])
    for key in packed["on_gpu"]:
        values[key] = to_device(values[key])
    return values


def build_denoise_kwargs(args: argparse.Namespace, conditioning: dict, device) -> dict:
    """Everything the generation half is called with, from the conditioning handoff and the request. Both processes of
    a ulysses job build it with this one function, so they start the denoising from identical inputs (each with its
    own generator, seeded the same: the noise they draw is identical)."""
    import torch

    kwargs = unpack_values(conditioning, device)
    if args.workflow != "ref2va":  # ref2va's frame count is resolved by the conditioning half
        kwargs["num_frames"] = snap_num_frames(args.num_frames or DEFAULT_FRAMES)
    if args.workflow == "t2va" and args.height is not None:  # fl2va/ref2va canvases come back resolved
        kwargs["height"], kwargs["width"] = args.height, args.width
    kwargs["generator"] = torch.Generator().manual_seed(args.seed)
    if args.steps is not None:
        kwargs["num_inference_steps"] = args.steps
    return kwargs


def apply_attention_backend(pipeline, backend: str | None) -> None:
    """Sets `backend` on the transformer (not on the VAEs), if one was asked for."""
    if not backend:
        return
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    default_backend = _AttentionBackendRegistry._active_backend
    for name in ("transformer", "transformer_ref"):
        model = getattr(pipeline, name, None)
        if model is not None:
            model.set_attention_backend(backend)
    # set_attention_backend also makes the backend the process-wide default, which would reach the audio VAE
    # (float32 causal attention: FlashAttention-3 only takes bf16/fp16 and failed there). The transformer's
    # attention layers keep the backend set on them above; everything else goes back to the default.
    _AttentionBackendRegistry.set_active_backend(default_backend)
    log("run", f"attention backend: {backend} (transformer only)")


def load_onto_generation_gpu(pipeline) -> None:
    """Loading is lazy (memory-mapped): the weights are only read from disk when a model first moves onto the GPU.
    Move them now, while the encoder works in the other process, so both halves really read in parallel."""
    import torch

    started = time.time()
    for name in ("transformer", "transformer_ref", "vae", "audio_vae"):
        model = getattr(pipeline, name, None)
        if isinstance(model, torch.nn.Module):
            model.to("cuda:0")
    log("run", f"transformer and VAEs on the GPU in {time.time() - started:.0f}s")


def prepare_job(args: argparse.Namespace) -> tuple[list[str], list[str], Path, Path]:
    """Checks and scratch files shared by both parallelism modes; returns (the job's GPU ids, the encoder's GPU ids,
    conditioning handoff file, its error file). Must run before torch is imported."""
    gpu_ids = pick_gpus(args.gpus)
    encoder_gpus = gpu_ids[1 : 1 + encoder_gpu_count(args.gpus)]
    check_weights(args.model_dir, args.workflow)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    handoff = args.work_dir / HANDOFF_NAME
    error_file = args.work_dir / HANDOFF_ERROR_NAME
    handoff.unlink(missing_ok=True)
    error_file.unlink(missing_ok=True)

    # This process sees all the job's GPUs: g0 as cuda:0, g1 as cuda:1 and so on (rank r of a ulysses job is cuda:r).
    # Nothing here touches cuda:1 and up until the encoder child (which sees only its own GPUs, as its cuda:0 and
    # cuda:1, see the Popen env below) has exited. This must happen before torch is imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    # Less fragmentation while 60 GB models are moved on and off the GPU.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    log("run", f"GPUs {','.join(gpu_ids)}: generation GPU {gpu_ids[0]}, text encoder on {','.join(encoder_gpus)}")
    return gpu_ids, encoder_gpus, handoff, error_file


def spawn_condition_child(handoff: Path, encoder_gpus: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--stage", "condition", "--handoff", str(handoff)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(encoder_gpus)},
    )


def wait_for_conditioning(child: subprocess.Popen, error_file: Path) -> None:
    waited = time.time()
    returncode = child.wait()
    if returncode != 0:
        reason = error_file.read_text() if error_file.is_file() else f"exit code {returncode}"
        raise RuntimeError(f"conditioning step failed: {reason}")
    log("run", f"conditioning ready (waited {time.time() - waited:.0f}s for it)")


def remove_scratch_files(args: argparse.Namespace, files: list[Path]) -> None:
    # The handoffs hold processed copies of the request's reference media; nothing needs them after this run.
    for leftover in files:
        leftover.unlink(missing_ok=True)
    try:
        args.work_dir.rmdir()
    except OSError:  # not empty (or already gone): leave it, the job folder is deleted with the job
        pass


def finish_job(args: argparse.Namespace) -> int:
    if not args.output_path.is_file() or args.output_path.stat().st_size == 0:
        raise RuntimeError(f"{args.output_path} was not written.")
    log("run", f"wrote {args.output_path} (total {time.time() - _STARTED:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# The conditioning half (child process, the GPU(s) after g0).
# ---------------------------------------------------------------------------


def load_references(args: argparse.Namespace):
    """The ref2va references, decoded: images, then videos, then the audio files, each
    kept in the order given. The pipeline labels them from this order, each kind on its own:
    <Picture N> and <Video N> count their own kind, and <Audio N> counts every audio there
    is: a video that has a soundtrack takes the next <Audio N> just before its <Video N>
    (unless it is muted, see --mute-reference-video), and the audio files follow.
    Returns (references, the audio-file references)."""
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    images = [MiniMaxH3ImageReference.from_file(path) for path in args.reference_image]
    videos = [MiniMaxH3VideoReference.from_file(path) for path in args.reference_video]
    for index in args.mute_reference_video:
        # No soundtrack, no <Audio N> label: the video is conditioned on for its picture alone.
        videos[index].audio = None
        videos[index].sample_rate = None
    audios = [MiniMaxH3AudioReference.from_file(path) for path in args.reference_audio]
    with_sound = sum(video.has_audio for video in videos)
    log("condition", f"references: {len(images)} image(s), {len(videos)} video(s) ({with_sound} with a soundtrack), {len(audios)} audio file(s)")
    return [*images, *videos, *audios], audios


def conditioning_inputs(args: argparse.Namespace) -> dict:
    """What the conditioning blocks read for this workflow (nothing else may be
    passed: the blocks choose their branch from which of these are present)."""
    from PIL import Image

    inputs: dict = {"prompt": args.prompt}
    if args.workflow == "fl2va":
        if args.first_frame:
            inputs["image"] = Image.open(args.first_frame).convert("RGB")
        if args.last_frame:
            inputs["last_image"] = Image.open(args.last_frame).convert("RGB")
    if args.workflow in ("fl2va", "ref2va") and args.height is not None:
        inputs["height"], inputs["width"] = args.height, args.width
    if args.workflow == "ref2va":
        inputs["references"], audios = load_references(args)
        if args.match_audio_duration:
            audio = audios[0]  # the first --reference-audio sets the length
            seconds = audio.audio.shape[-1] / (audio.sample_rate or AUDIO_SAMPLE_RATE)
            check_max_seconds(args, seconds)
            inputs["num_frames"] = snap_num_frames(round(seconds * FPS))
            log("condition", f"--match-audio-duration: {seconds:.2f} s of audio -> {inputs['num_frames']} frames")
        else:
            inputs["num_frames"] = snap_num_frames(args.num_frames or DEFAULT_FRAMES)
    return inputs


def record_conditioning_numbers(metrics: Metrics, values: dict) -> None:
    """What the conditioning step resolved, as numbers: the encoder's token count (the rows of the text embedding), the
    canvas and frame count, and the references' sizes. Never raises: this is for the usage log only."""
    try:
        def as_int(value):
            return None if value is None else int(value)

        embeds = values.get("prompt_embeds")
        keyframes = values.get("keyframes")
        metrics.update(
            encoder_tokens=None if embeds is None else int(embeds.shape[1]),
            num_frames=as_int(values.get("num_frames")),
            height=as_int(values.get("height")),
            width=as_int(values.get("width")),
            references=describe_references(values.get("normalized_references")),
            keyframes=0 if keyframes is None else len(keyframes),
        )
    except Exception:
        pass


def split_text_encoder(
    text_encoder, split_at: int = TEXT_ENCODER_SPLIT_AT, first_device: str = "cuda:0", second_device: str = "cuda:1"
) -> int:
    """Place the Qwen3-VL text/vision encoder on two GPUs: the vision tower, the token embeddings and decoder layers
    [0, split_at) on `first_device`, decoder layers [split_at, 64) and the final norm on `second_device`. Returns the
    number of layers on the second GPU.

    Why: the encoder reads the request's prompt and every reference image as one sequence (a 2048 px short edge makes
    4,000-8,000 tokens per image), and it keeps the output of every one of its 64 layers (about 0.63 MiB per token)
    on top of its 62 GiB of weights, so on one 80 GB GPU two portrait images already ran it out of memory. Two GPUs
    hold half the weights and half the per-layer outputs each.

    The model's top-level `device` (what the offload manager compares with its execution device, see
    `get_qwen3vl_prompt_embeds`) is that of its first parameter, the vision tower on `first_device`, so the manager
    does not try to move the encoder back onto one GPU. Placed by hand, like `split_transformer`: the layers still run
    one after another and the result is the same; a forward pre-hook on each layer from `split_at` on carries its
    arguments (the hidden states, the rotary tables, the attention mask) over to `second_device`. The embedding and the
    `lm_head` (which this pipeline never calls) are not on the second GPU; the head stays in host memory."""
    from accelerate.utils import send_to_device

    language_model = text_encoder.model.language_model
    layers = language_model.layers
    if not 0 < split_at < len(layers):
        raise ValueError(f"Cannot split the text encoder's {len(layers)} layers at {split_at}.")

    text_encoder.model.visual.to(first_device)
    language_model.embed_tokens.to(first_device)
    language_model.rotary_emb.to(first_device)
    for layer in layers[:split_at]:
        layer.to(first_device)
    for layer in layers[split_at:]:
        layer.to(second_device)
    language_model.norm.to(second_device)

    def move_inputs(module, args, kwargs):
        return send_to_device(args, second_device), send_to_device(kwargs, second_device)

    # On every layer of the second GPU, not just the first: the language model's loop hands each layer the rotary
    # tables and the attention mask from the first GPU again (only the hidden states chain from layer to layer), so
    # each of these layers has to bring them over itself. (Already-moved tensors are returned as they are.)
    for layer in layers[split_at:]:
        layer.register_forward_pre_hook(move_inputs, with_kwargs=True)

    # Everything the forward pass touches must be on a GPU now, and the model must report the first one, or the
    # offload manager would move the whole encoder onto it (and run it out of memory) when the encoder is first used.
    not_placed = [name for name, parameter in text_encoder.model.named_parameters() if parameter.device.type != "cuda"]
    if not_placed:
        raise RuntimeError(f"The text encoder was not fully placed on the GPUs; still on the CPU: {not_placed[:3]}")
    if str(text_encoder.device) != first_device:
        raise RuntimeError(f"The text encoder reports device {text_encoder.device}, expected {first_device}.")
    return len(layers) - split_at


def run_condition_stage(args: argparse.Namespace, metrics: Metrics) -> int:
    import torch

    error_file = args.handoff.with_name(HANDOFF_ERROR_NAME)
    try:
        allow_any_length()
        blocks, _ = split_workflow(args.model_dir, args.workflow)
        with metrics.stage("prepare_media"):
            inputs = conditioning_inputs(args)
        log("condition", "loading the text/vision encoder")
        pipeline, seconds = make_pipeline(blocks, args.model_dir, "cuda:0")
        metrics.timing("setup", seconds)
        gpus = torch.cuda.device_count()  # the GPUs the parent gave this child: one, or two for a 4+ GPU job
        metrics.update(encoder_gpus=gpus, gpu_total_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
        if gpus >= 2:
            # Reads the weights from disk onto the two GPUs now (this is the slow part), instead of on first use.
            with metrics.stage("load_encoder"):
                moved = split_text_encoder(pipeline.text_encoder)
            log(
                "condition",
                f"set up in {seconds:.0f}s; text encoder split across 2 GPUs (layers 0-{TEXT_ENCODER_SPLIT_AT - 1} / "
                f"{TEXT_ENCODER_SPLIT_AT}-{TEXT_ENCODER_SPLIT_AT + moved - 1}) and loaded in "
                f"{metrics.data['timings_s']['load_encoder']:.0f}s",
            )
        else:
            log("condition", f"set up in {seconds:.0f}s (the weights are read from disk when the encoder first moves onto the GPU)")

        names = [output.name for output in blocks.intermediate_outputs]
        started = time.time()
        with metrics.stage("encode"):
            values = pipeline(output=names, **inputs)
        log("condition", f"encoded in {time.time() - started:.0f}s: {sorted(values)}")
        peaks = [torch.cuda.max_memory_allocated(index) / 2**30 for index in range(gpus)]
        log("condition", "peak GPU memory " + ", ".join(f"GPU {index} {peak:.1f} GiB" for index, peak in enumerate(peaks)))
        metrics.peak("encoder", peaks)
        record_conditioning_numbers(metrics, values)

        # Plain CPU values, so the other process (a different GPU) can read them, plus which of them were on the GPU.
        partial = args.handoff.with_name(args.handoff.name + ".partial")
        torch.save(pack_values(values), partial)
        os.replace(partial, args.handoff)
        return 0
    except Exception as error:  # reported to the generation process through a file
        traceback.print_exc()
        metrics.fail(error)
        error_file.write_text(f"{type(error).__name__}: {error}")
        print(f"ERROR: conditioning step failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


# ---------------------------------------------------------------------------
# The generation half (this process, first GPU) -- also starts the child.
# ---------------------------------------------------------------------------


def split_transformer(model, second_device: str = "cuda:1", first_device: str = "cuda:0") -> int:
    """Move `model.transformer_blocks[SPLIT_AT:]` onto `second_device`, so the 61.7 GiB of weights are shared
    between two GPUs and the first one has room left for the activations of a long, full-size clip.

    Everything else (the input projections, the text refiner, `norm_out` and the output heads) stays on
    `first_device`, which is also the device `model.device` reports (that of its first parameter), so the offload
    manager does not try to move the model back. The blocks still run one after another, so the result is the same;
    forward hooks only carry the inputs of the first moved block over to `second_device` and the output of the last
    block back. Returns the number of blocks moved."""
    from accelerate.utils import send_to_device

    def move_inputs(module, args, kwargs):
        return send_to_device(args, second_device), send_to_device(kwargs, second_device)

    def move_output_back(module, args, output):
        return output.to(first_device)

    moved_blocks = model.transformer_blocks[SPLIT_AT:]
    for block in moved_blocks:
        block.to(second_device)
        block.register_forward_pre_hook(move_inputs, with_kwargs=True)
    model.transformer_blocks[-1].register_forward_hook(move_output_back)
    return len(moved_blocks)


def run_pipeline_split(args: argparse.Namespace, metrics: Metrics) -> int:
    """--parallelism split: the transformer's blocks are divided between the two GPUs and run one after another."""
    _, encoder_gpus, handoff, error_file = prepare_job(args)
    log("run", "parallelism: split (the transformer's blocks are divided between the GPUs)")

    child = spawn_condition_child(handoff, encoder_gpus)
    try:
        import torch
        from diffusers.utils.export_utils import encode_video

        record_run_setup(metrics, args)
        allow_any_length()
        _, rest_blocks = split_workflow(args.model_dir, args.workflow)
        log("run", "loading the transformer and VAEs (while the encoder loads in the other process)")
        pipeline, seconds = make_pipeline(rest_blocks, args.model_dir, "cuda:0")
        log("run", f"set up in {seconds:.0f}s")
        metrics.timing("setup", seconds)
        apply_attention_backend(pipeline, args.attention_backend)
        with metrics.stage("load_gpu"):
            load_onto_generation_gpu(pipeline)

        # The VAEs (~10 GiB, the video VAE is float32) are only used before denoising (encoding references and
        # keyframes) and after it (decoding). The offload manager makes room for a model's weights, never for
        # activations, so it would leave them here through denoising. Move them off at the first transformer
        # step; their offload hooks move them back for the decode.
        def release_vaes(module, inputs):
            release_hook.remove()
            for name in ("vae", "audio_vae"):
                model = getattr(pipeline, name, None)
                if isinstance(model, torch.nn.Module):
                    model.to("cpu")
            torch.cuda.empty_cache()
            log("run", f"VAEs moved off the GPU for denoising ({torch.cuda.memory_allocated() / 2**30:.1f} GiB in use)")

        denoiser = next(
            model
            for model in (getattr(pipeline, name, None) for name in ("transformer", "transformer_ref"))
            if isinstance(model, torch.nn.Module)
        )
        release_hook = denoiser.register_forward_pre_hook(release_vaes)
        record_sequence_shape(denoiser, metrics)

        with metrics.stage("wait_encoder"):
            wait_for_conditioning(child, error_file)

        # The child has exited, so the second GPU is free: the second half of the transformer goes there.
        started = time.time()
        with metrics.stage("split_transformer"):
            moved = split_transformer(denoiser)
        log("run", f"transformer blocks {SPLIT_AT}-{SPLIT_AT + moved - 1} moved to the second GPU in {time.time() - started:.0f}s")
        # Count peaks from here on: the first GPU held the whole transformer until just now.
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.reset_peak_memory_stats(1)

        conditioning = torch.load(handoff, weights_only=False)  # written by the child just above, in our own job folder
        kwargs = build_denoise_kwargs(args, conditioning, torch.device("cuda:0"))

        started = time.time()
        with metrics.stage("generate"):  # reference encode, denoise and decode in one call
            results = pipeline(output=OUTPUTS, **kwargs)
        frames = results["videos"][0]
        log("run", f"generated {len(frames)} frames of {frames[0].size[0]}x{frames[0].size[1]} in {time.time() - started:.0f}s")
        peaks = [torch.cuda.max_memory_allocated(index) / 2**30 for index in range(2)]
        log(
            "run",
            f"peak GPU memory since the split: generation GPU {peaks[0]:.1f} GiB, second GPU {peaks[1]:.1f} GiB",
        )
        metrics.peak("since_split", peaks)
        metrics.update(output_frames=len(frames))

        with metrics.stage("mux"):
            encode_video(
                frames,
                fps=FPS,
                output_path=str(args.output_path),
                audio=results["audio"][0],
                audio_sample_rate=results["sampling_rate"],
            )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        remove_scratch_files(args, [handoff, handoff.with_name(handoff.name + ".partial"), error_file])

    return finish_job(args)


# ---------------------------------------------------------------------------
# --parallelism ulysses: all the job's GPUs compute every denoising step.
#
# Each GPU holds the whole transformer and runs 1/N of the packed token sequence through every block; around each
# attention layer the GPUs exchange their parts of the queries, keys and values (diffusers' "Ulysses" context
# parallelism; `MiniMaxH3Transformer3DModel._cp_plan` says how the sequence is split and gathered). This process is
# rank 0 on the generation GPU. Further copies of this script (`--stage denoise-worker --rank r`) are ranks 1 .. N-1 on
# the job's other GPUs; they talk through NCCL (`torch.distributed`), joined by a rendezvous store on a localhost port.
# ---------------------------------------------------------------------------


def init_distributed(rank: int, port: int, world_size: int):
    """Joins the `world_size`-process group. Rank 0 opens the rendezvous store (and returns once every worker has
    connected to it); every other rank connects to it, waiting as long as rank 0 needs to get ready. Each rank then takes its own
    GPU (`cuda:<rank>`: rank 0 the generation GPU, rank r the job's GPU number r, in CUDA_VISIBLE_DEVICES order).

    Returns the store, which must stay referenced for as long as the group is used."""
    import torch
    import torch.distributed as dist

    store = dist.TCPStore(
        "127.0.0.1",
        port,
        world_size,
        is_master=rank == 0,
        timeout=RENDEZVOUS_TIMEOUT if rank == 0 else CONNECT_TIMEOUT,
    )
    # Only now touch the GPU: for ranks 1 and 2 (the encoder's GPUs) this point is reached after rank 0 has seen the
    # encoder exit.
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl", store=store, rank=rank, world_size=world_size, timeout=COLLECTIVE_TIMEOUT, device_id=device
    )
    return store


def transformer_tensors(model) -> list:
    """(name, tensor) for every parameter and buffer of `model`. The buffers matter: `rope.inv_freq` is computed rather
    than loaded, so a model built without weights has to be given it too."""
    return [*model.named_parameters(), *model.named_buffers()]


def send_transformer(model, device) -> None:
    """Rank 0 side of the weight copy: the shapes and dtypes first (the checkpoint is mixed precision, so the receivers
    cannot guess them), then every tensor, straight from this GPU's memory to every other rank's (a broadcast)."""
    import torch
    import torch.distributed as dist

    tensors = transformer_tensors(model)
    elsewhere = [name for name, tensor in tensors if tensor.device != device]
    if elsewhere:
        raise RuntimeError(f"The transformer must be entirely on {device} before it is copied; not there: {elsewhere[:3]}")
    specs = [(name, tuple(tensor.shape), tensor.dtype) for name, tensor in tensors]
    dist.broadcast_object_list([specs], src=0, device=device)
    for _, tensor in tensors:
        dist.broadcast(tensor.data, src=0)
    torch.cuda.synchronize(device)


def receive_transformer(model_dir: Path, denoiser_name: str, device):
    """Receiving side of the weight copy (ranks 1 and up): builds the transformer without weights (on the meta device), gives every
    parameter and buffer real, empty storage on `device` with the shape and dtype rank 0 announces, and fills it from
    rank 0, in rank 0's order. Nothing is read from disk."""
    import torch
    import torch.distributed as dist
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

    config = MiniMaxH3Transformer3DModel.load_config(str(model_dir / denoiser_name))
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel.from_config(config)

    announced = [None]
    dist.broadcast_object_list(announced, src=0, device=device)
    specs = announced[0]
    expected = {name for name, _ in transformer_tensors(model)}
    sent = {name for name, _, _ in specs}
    if expected != sent:
        raise RuntimeError(
            f"The transformer sent by the generation process does not match the one built here "
            f"(only here: {sorted(expected - sent)[:3]}, only there: {sorted(sent - expected)[:3]})."
        )
    for name, shape, dtype in specs:
        owner_path, _, attribute = name.rpartition(".")
        owner = model.get_submodule(owner_path)
        storage = torch.empty(shape, dtype=dtype, device=device)
        if attribute in owner._parameters:
            owner._parameters[attribute] = torch.nn.Parameter(storage, requires_grad=False)
        else:
            owner._buffers[attribute] = storage

    received = dict(transformer_tensors(model))
    for name, _, _ in specs:
        dist.broadcast(received[name].data, src=0)
    torch.cuda.synchronize(device)
    return model.eval()


def enable_parallel_denoiser(model, world_size: int) -> None:
    """Turns on `world_size`-way Ulysses context parallelism on `model` (the transformer, with its attention backend
    already set: the backend has to support it, and FlashAttention-3 and diffusers' default do). Every rank must call
    this at the same point: it creates process groups together.

    `ulysses_anything` because the packed sequence has whatever length the request makes it (it need not divide by the
    number of GPUs).
    """
    from diffusers import ContextParallelConfig

    model.enable_parallelism(config=ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=True))
    # enable_parallelism puts the setting on every attention layer, but the two-layer text refiner runs on the text
    # *before* the sequence is split: every rank holds all of it, so it must not be exchanged as if each held a part.
    for module in model.token_refiner.modules():
        processor = getattr(module, "processor", None)
        if processor is not None and hasattr(processor, "_parallel_config"):
            processor._parallel_config = None


def denoised_names(denoise_blocks, decode_blocks) -> list[str]:
    """What the denoising hands to the decode: the decode blocks' inputs that the denoise blocks produce."""
    produced = set(output_names(denoise_blocks))
    return [param.name for param in decode_blocks.inputs if param.name in produced]


def worker_error_file(work_dir: Path, rank: int) -> Path:
    return work_dir / WORKER_ERROR_NAME.format(rank=rank)


def gather_peak_memory(world_size: int) -> list[float]:
    """Each rank's peak GPU memory in GiB, as seen by all (also the last operation the ranks do together)."""
    import torch
    import torch.distributed as dist

    peaks: list = [None] * world_size
    dist.all_gather_object(peaks, torch.cuda.max_memory_allocated() / 2**30)
    return peaks


def watch_workers(
    workers: list[tuple[subprocess.Popen, Path]], finished: threading.Event, others: list, on_failure=None
) -> None:
    """If a denoise worker (a (process, error file) pair) dies while this process waits for the others inside a GPU
    operation, say why and stop at once (killing `others` and the remaining workers first) instead of waiting for the
    operation to time out. `on_failure(reason)`, if given, is called first (to record the failure). Stops watching
    when `finished` is set, or when every worker has exited cleanly."""

    def watch() -> None:
        while not finished.wait(2):
            for worker, error_file in workers:
                code = worker.poll()
                if code is None or code == 0:
                    continue
                reason = error_file.read_text() if error_file.is_file() else f"exit code {code}"
                if on_failure is not None:
                    on_failure(reason)
                for process in [*others, *(other for other, _ in workers)]:
                    if process.poll() is None:
                        process.kill()
                print(f"ERROR: a denoising process failed: {reason}", file=sys.stderr, flush=True)
                os._exit(1)
            if all(worker.poll() == 0 for worker, _ in workers):
                return

    threading.Thread(target=watch, name="watch-denoise-workers", daemon=True).start()


def run_pipeline_ulysses(args: argparse.Namespace, metrics: Metrics) -> int:
    """--parallelism ulysses: rank 0. Loads the generation half as in split mode, encodes the references, broadcasts the
    transformer to the other ranks' GPUs, and then all of them denoise together."""
    world_size = args.gpus
    _, encoder_gpus, handoff, error_file = prepare_job(args)
    worker_files = [worker_error_file(args.work_dir, rank) for rank in range(1, world_size)]
    encoded_file = args.work_dir / ENCODED_NAME
    for leftover in worker_files:
        leftover.unlink(missing_ok=True)
    encoded_file.unlink(missing_ok=True)
    # NVLink-SHARP buffers can take several GiB per communicator on NVSwitch nodes, and the GPUs of one node gain
    # nothing from them; every GiB counts when each GPU holds the whole transformer. (Inherited by the workers.)
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    log("run", f"parallelism: ulysses (all {world_size} GPUs compute every denoising step, each on 1/{world_size} of the token sequence)")

    port = free_local_port()
    child = spawn_condition_child(handoff, encoder_gpus)
    # Started now, so their imports are done long before they are needed. They see all the job's GPUs but touch none
    # until rank 0 opens the rendezvous store, which is after the encoder (the current user of GPUs g1 and g2) has
    # exited. Rank r takes the job's GPU number r (cuda:r).
    workers = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--stage", "denoise-worker", "--rank", str(rank)],
            env={**os.environ, DIST_PORT_ENV: str(port)},
        )
        for rank in range(1, world_size)
    ]
    worker_finished = threading.Event()
    watch_workers(list(zip(workers, worker_files)), worker_finished, [child], on_failure=lambda reason: metrics.fail(reason, "worker"))
    try:
        import torch
        import torch.distributed as dist
        from diffusers.utils.export_utils import encode_video

        record_run_setup(metrics, args)
        allow_any_length()
        _, rest_blocks = split_workflow(args.model_dir, args.workflow)
        vae_blocks, denoise_blocks, decode_blocks = split_rest_blocks(rest_blocks)
        denoiser_name = denoiser_component_name(args.workflow)
        log("run", "loading the transformer and VAEs (while the encoder loads in the other process)")
        pipeline, seconds = make_pipeline(rest_blocks, args.model_dir, "cuda:0")
        log("run", f"set up in {seconds:.0f}s")
        metrics.timing("setup", seconds)
        apply_attention_backend(pipeline, args.attention_backend)
        with metrics.stage("load_gpu"):
            load_onto_generation_gpu(pipeline)
        denoiser = getattr(pipeline, denoiser_name)

        with metrics.stage("wait_encoder"):
            wait_for_conditioning(child, error_file)
        conditioning = torch.load(handoff, weights_only=False)  # written by the child just above, in our own job folder
        kwargs = build_denoise_kwargs(args, conditioning, torch.device("cuda:0"))

        # The references and keyframes are encoded here, with the VAEs, before the VAEs leave the GPU (text-only
        # requests have nothing to encode). The worker gets the result through a file, like the conditioning.
        encoded: dict = {}
        if vae_blocks is not None:
            started = time.time()
            with metrics.stage("reference_encode"):
                vae_pipeline = sub_pipeline(vae_blocks, pipeline, args.model_dir)
                encoded = vae_pipeline(output=output_names(vae_blocks), **select_inputs(vae_blocks, kwargs))
            log("run", f"references encoded in {time.time() - started:.0f}s")
        kwargs.update(encoded)
        torch.save(pack_values(encoded), encoded_file)

        # The VAEs (~10 GiB, the video VAE is float32) are only used before and after denoising, and the whole
        # transformer now needs the room. Their offload hooks move them back for the decode.
        for name in ("vae", "audio_vae"):
            model = getattr(pipeline, name, None)
            if isinstance(model, torch.nn.Module):
                model.to("cpu")
        torch.cuda.empty_cache()
        log("run", f"VAEs moved off the GPU for denoising ({torch.cuda.memory_allocated() / 2**30:.1f} GiB in use)")

        # Open the group; from here on the workers (waiting on the store since the start) own the other GPUs.
        with metrics.stage("group_init"):
            store = init_distributed(0, port, world_size)
        log("run", f"GPU link to the {world_size - 1} other process(es) is up")
        started = time.time()
        with metrics.stage("weights_copy"):
            send_transformer(denoiser, torch.device("cuda:0"))
        log("run", f"transformer copied to the other {world_size - 1} GPU(s) in {time.time() - started:.0f}s")
        enable_parallel_denoiser(denoiser, world_size)
        record_sequence_shape(denoiser, metrics)

        denoise_pipeline = sub_pipeline(denoise_blocks, pipeline, args.model_dir)
        # What this GPU needed before denoising: the whole transformer and the VAEs, plus the reference encoding.
        metrics.peak("before_denoise", torch.cuda.max_memory_allocated() / 2**30)
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        with metrics.stage("denoise"):
            denoised = denoise_pipeline(
                output=denoised_names(denoise_blocks, decode_blocks), **select_inputs(denoise_blocks, kwargs)
            )
        denoise_seconds = time.time() - started
        peaks = gather_peak_memory(world_size)
        worker_finished.set()
        dist.destroy_process_group()
        log("run", f"denoised in {denoise_seconds:.0f}s")
        log(
            "run",
            f"peak GPU memory during denoising: generation GPU {peaks[0]:.1f} GiB, other GPUs "
            f"{', '.join(f'{peak:.1f}' for peak in peaks[1:])} GiB",
        )
        metrics.peak("denoise", peaks)
        metrics.update(denoise_seconds_per_step=round(denoise_seconds / (args.steps or DEFAULT_STEPS), 2))
        del store

        # The decode runs on this GPU alone, and needs the room the transformer is holding.
        with metrics.stage("transformer_to_host"):
            denoiser.to("cpu")
            torch.cuda.empty_cache()
        log("run", f"transformer moved off the GPU for the decode in {metrics.data['timings_s']['transformer_to_host']:.0f}s")

        started = time.time()
        torch.cuda.reset_peak_memory_stats()
        with metrics.stage("decode"):
            decode_pipeline = sub_pipeline(decode_blocks, pipeline, args.model_dir)
            results = decode_pipeline(output=OUTPUTS, **select_inputs(decode_blocks, denoised))
        frames = results["videos"][0]
        log("run", f"decoded {len(frames)} frames of {frames[0].size[0]}x{frames[0].size[1]} in {time.time() - started:.0f}s")
        metrics.peak("decode", torch.cuda.max_memory_allocated() / 2**30)
        metrics.update(output_frames=len(frames))

        with metrics.stage("mux"):
            encode_video(
                frames,
                fps=FPS,
                output_path=str(args.output_path),
                audio=results["audio"][0],
                audio_sample_rate=results["sampling_rate"],
            )
    finally:
        worker_finished.set()
        for process in (child, *workers):
            if process.poll() is None:
                process.kill()
                process.wait()
        remove_scratch_files(
            args,
            [
                handoff,
                handoff.with_name(handoff.name + ".partial"),
                error_file,
                encoded_file,
                *worker_files,
            ],
        )

    return finish_job(args)


def run_denoise_worker_stage(args: argparse.Namespace, metrics: Metrics) -> int:
    """--parallelism ulysses: rank --rank (1 .. gpus-1), started by `run_pipeline_ulysses` with this same argv. Runs the
    same denoising loop on the job's GPU number <rank>, on its share of the token sequence. Its weights come from rank 0
    (nothing is read from disk) and its result is discarded: every rank ends up with the full denoised latents, rank 0
    keeps them."""
    error_file = worker_error_file(args.work_dir, args.rank)
    rank, world_size = args.rank, args.gpus
    try:
        os.environ["TQDM_DISABLE"] = "1"  # rank 0's progress bar is the one in the log; this must precede any tqdm use
        import torch
        import torch.distributed as dist

        allow_any_length()  # the denoising setup checks the video length too
        port = int(os.environ[DIST_PORT_ENV])
        _, rest_blocks = split_workflow(args.model_dir, args.workflow)
        _, denoise_blocks, decode_blocks = split_rest_blocks(rest_blocks)
        denoiser_name = denoiser_component_name(args.workflow)
        pipeline = denoise_blocks.init_pipeline(str(args.model_dir))
        # Only the two schedulers: the transformer is not read from disk here, rank 0 sends it.
        scheduler_names = [spec.name for spec in denoise_blocks.expected_components if spec.name != denoiser_name]
        pipeline.load_components(names=scheduler_names, dtype=torch.bfloat16)
        log(f"denoise-worker {rank}", "ready, waiting for the generation process (GPUs 1 and 2 are the encoder's until then)")

        with metrics.stage("wait_for_rank0"):  # until the weights are loaded and the encoder has exited
            store = init_distributed(rank, port, world_size)
        device = torch.device(f"cuda:{rank}")
        started = time.time()
        with metrics.stage("weights_receive"):
            denoiser = receive_transformer(args.model_dir, denoiser_name, device)
        log(f"denoise-worker {rank}", f"transformer received in {time.time() - started:.0f}s")
        if args.attention_backend:
            denoiser.set_attention_backend(args.attention_backend)
        enable_parallel_denoiser(denoiser, world_size)

        # The same inputs rank 0 denoises: the conditioning handoff (still in the job folder) and the encoded
        # references rank 0 wrote before it opened the rendezvous store.
        conditioning = torch.load(args.work_dir / HANDOFF_NAME, weights_only=False)  # written by our own job
        kwargs = build_denoise_kwargs(args, conditioning, device)
        kwargs.update(unpack_values(torch.load(args.work_dir / ENCODED_NAME, weights_only=False), device))
        pipeline.update_components(**{denoiser_name: denoiser})

        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        with metrics.stage("denoise"):
            pipeline(output=denoised_names(denoise_blocks, decode_blocks), **select_inputs(denoise_blocks, kwargs))
        log(f"denoise-worker {rank}", f"denoised in {time.time() - started:.0f}s")
        gather_peak_memory(world_size)
        peak = torch.cuda.max_memory_allocated() / 2**30
        log(f"denoise-worker {rank}", f"peak GPU memory during denoising: {peak:.1f} GiB")
        metrics.peak("denoise", peak)
        dist.destroy_process_group()
        del store
        return 0
    except Exception as error:  # reported to the generation process through a file (and by its watchdog)
        traceback.print_exc()
        metrics.fail(error)
        try:
            error_file.write_text(f"{type(error).__name__}: {error}")
        except OSError:
            pass
        print(f"ERROR: denoise worker failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


def run_pipeline(args: argparse.Namespace, metrics: Metrics) -> int:
    return run_pipeline_split(args, metrics) if args.parallelism == "split" else run_pipeline_ulysses(args, metrics)


def main() -> int:
    args = parse_args()
    role = {"run": "run", "condition": "condition", "denoise-worker": f"worker{args.rank}"}[args.stage]
    metrics = Metrics(args.metrics_dir, role)
    code = 1
    try:
        if args.stage == "condition":
            code = run_condition_stage(args, metrics)
        elif args.stage == "denoise-worker":
            code = run_denoise_worker_stage(args, metrics)
        else:
            code = run_pipeline(args, metrics)
        return code
    except Exception as error:
        traceback.print_exc()
        metrics.fail(error)
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        metrics.finish(code == 0)


if __name__ == "__main__":
    sys.exit(main())
