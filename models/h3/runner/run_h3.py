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

TWO GPUS PER JOB. The Slurm job holds two GPUs and this script uses both, the
way diffusers' own MiniMax-H3 documentation splits the pipeline "with two
cards, nothing offloaded":

  * the conditioning half -- media preparation plus the Qwen3-VL-32B text/vision
    encoder (62 GiB of bf16 weights) -- runs in a CHILD process on the second GPU;
  * the generation half -- reference/keyframe VAE encoding, the 33B transformer
    (61.7 GiB), and the video/audio decode -- runs in THIS process on the first.

Both halves load their weights at the same time (two processes, so the two
sets of weights are read from disk in parallel -- threads would not be safe,
because accelerate's weight-loading context patches torch globally). When the
child is done it hands the small
conditioning result (the text embedding and the resolved canvas/frame
count/keyframes/references) over as one file in --work-dir, and exits, which
frees its GPU. Output quality does not depend on this split: the same bf16
weights run the same computation as on one big GPU.

Three workflows, matching diffusers' (`pipe.blocks.available_workflows`):
  t2va  : text only                       -> `transformer/` partition ("FL2VA" checkpoint)
  fl2va : first and/or last keyframe      -> `transformer/` partition
  ref2va: image/video/audio references    -> `transformer_ref/` partition ("Ref2VA" checkpoint)

Usage:
    run_h3.py --model-dir DIR --workflow t2va|fl2va|ref2va --output-path FILE \\
        --work-dir DIR --prompt=TEXT \\
        [--first-frame FILE] [--last-frame FILE] \\
        [--reference-image FILE ...] [--reference-video FILE ...] [--reference-audio FILE] \\
        [--height N --width N] [--num-frames N | --match-audio-duration] \\
        [--steps N] [--seed N] [--attention-backend NAME]

--prompt is passed as `--prompt=value` (not `--prompt value`) so a value starting
with "-" is never mistaken for an option. Nothing here goes through a shell, so
user text needs no escaping.

There is no prompt rewriting: the text goes to the encoder exactly as given.
(MiniMax's hosted service runs a separate "Context-IR" model that expands a
request into the structured prompt format; that model is not in the open release.
Write the structured prompt yourself -- see gateway/services/h3/GUIDE.md.)

Frames: 24 fps, always 17 * n + 5 frames, between 124 (5.17 s) and 345 (14.4 s). A
--num-frames outside that grid is rounded up to it and clamped. --height and
--width go together and must be multiples of 32. Without them the canvas is the
model's own 768-short-edge 16:9 (1344x768), or, for fl2va, the aspect ratio of the
first keyframe.

Output is one MP4 (H.264 video + AAC stereo audio at 32 kHz), written by PyAV.

Exit code 0 only if --output-path was actually written. On failure the last line
printed is `ERROR: ...` with the root cause.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

FPS = 24
# The video VAE decodes 17 * n + 5 frames; the model generates between 5 and 15 seconds.
FRAMES_PER_CHUNK = 17
FRAMES_OFFSET = 5
MIN_FRAMES = 124  # 5.17 s: the first 17 * n + 5 at or above 5 s (120 frames)
MAX_FRAMES = 345  # 14.4 s: the last 17 * n + 5 at or below 15 s (360 frames)
DEFAULT_FRAMES = 124
# The audio VAE's sample rate, used only for a reference whose own rate is unknown.
AUDIO_SAMPLE_RATE = 32000

WORKFLOWS = ("t2va", "fl2va", "ref2va")
HANDOFF_NAME = "h3_conditioning.pt"
HANDOFF_ERROR_NAME = "h3_conditioning_error.txt"
OUTPUTS = ["videos", "audio", "sampling_rate"]

_STARTED = time.time()


def log(tag: str, message: str) -> None:
    print(f"[{time.time() - _STARTED:6.0f}s {tag}] {message}", flush=True)


def snap_num_frames(num_frames: int) -> int:
    """Round up to the next 17 * n + 5 the video VAE can decode, then keep the
    result inside the 5-15 s window the model generates. (Same rule as
    gateway/services/h3/video_shape.py -- the gateway cannot import this file.)"""
    snapped = max(num_frames, 1)
    while snapped % FRAMES_PER_CHUNK != FRAMES_OFFSET:
        snapped += 1
    return min(max(snapped, MIN_FRAMES), MAX_FRAMES)


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
    p.add_argument("--reference-audio", type=Path, help="ref2va: a voice/music reference; needs an image or video reference.")
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--num-frames", type=int, help=f"Default {DEFAULT_FRAMES} (5.17 s).")
    p.add_argument(
        "--match-audio-duration",
        action="store_true",
        help="ref2va: size the video to the reference audio (rounded onto the frame grid, clamped to 5-14.4 s).",
    )
    p.add_argument("--steps", type=int, help="Denoising steps. Default: the pipeline's own (50).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attention-backend", help="A diffusers attention backend name for the transformer (default: diffusers' own).")
    # Internal: the conditioning half is this same script run as a child process.
    p.add_argument("--stage", choices=["run", "condition"], default="run", help=argparse.SUPPRESS)
    p.add_argument("--handoff", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if (args.height is None) != (args.width is None):
        p.error("--height and --width must be given together.")
    if args.height is not None and (args.height % 32 or args.width % 32):
        p.error(f"--height and --width must be multiples of 32, got {args.height}x{args.width}.")
    if not args.model_dir.is_dir():
        p.error(f"Model directory not found: {args.model_dir}")
    if args.steps is not None and args.steps < 1:
        p.error("--steps must be at least 1.")

    keyframes = [path for path in (args.first_frame, args.last_frame) if path is not None]
    references = [*args.reference_image, *args.reference_video, *([args.reference_audio] if args.reference_audio else [])]
    if args.workflow == "t2va" and (keyframes or references):
        p.error("--workflow t2va takes no keyframes or references.")
    if args.workflow == "fl2va" and (not keyframes or references):
        p.error("--workflow fl2va needs --first-frame and/or --last-frame, and takes no references.")
    if args.workflow == "ref2va":
        if keyframes:
            p.error("--workflow ref2va takes references, not keyframes.")
        if not args.reference_image and not args.reference_video:
            p.error("--workflow ref2va needs at least one --reference-image or --reference-video (audio cannot be the only reference).")
        if len(args.reference_image) > 9 or len(args.reference_video) > 3:
            p.error("MiniMax H3 accepts at most 9 reference images and 3 reference videos.")
    if args.match_audio_duration and not (args.workflow == "ref2va" and args.reference_audio):
        p.error("--match-audio-duration needs --workflow ref2va and --reference-audio.")
    if args.match_audio_duration and args.num_frames is not None:
        p.error("Give either --num-frames or --match-audio-duration, not both.")
    for path in [*keyframes, *references]:
        if not path.is_file():
            p.error(f"Input file not found: {path}")
    if args.stage == "condition" and args.handoff is None:
        p.error("--stage condition needs --handoff.")
    return args


def pick_gpus() -> tuple[str, str]:
    """(generation GPU, conditioning GPU) as they are named in CUDA_VISIBLE_DEVICES,
    which Slurm sets to the job's own GPUs. Without it, the first two on the node."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    ids = [item.strip() for item in visible.split(",") if item.strip()] if visible else ["0", "1"]
    if len(ids) < 2:
        raise RuntimeError(
            f"MiniMax H3 needs 2 GPUs per job (one for the text/vision encoder, one for the transformer), "
            f"but only {len(ids)} is visible (CUDA_VISIBLE_DEVICES={visible!r})."
        )
    return ids[0], ids[1]


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


# ---------------------------------------------------------------------------
# The conditioning half (child process, second GPU).
# ---------------------------------------------------------------------------


def load_references(args: argparse.Namespace):
    """The ref2va references, decoded: images, then videos, then the audio -- the
    order the prompt's <Picture N> / <Video N> / <Audio N> labels count in (each
    kind is numbered on its own). Returns (references, audio reference or None)."""
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    references = [MiniMaxH3ImageReference.from_file(path) for path in args.reference_image]
    references += [MiniMaxH3VideoReference.from_file(path) for path in args.reference_video]
    audio = MiniMaxH3AudioReference.from_file(args.reference_audio) if args.reference_audio else None
    if audio is not None:
        references.append(audio)
    return references, audio


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
        inputs["references"], audio = load_references(args)
        if args.match_audio_duration:
            seconds = audio.audio.shape[-1] / (audio.sample_rate or AUDIO_SAMPLE_RATE)
            inputs["num_frames"] = snap_num_frames(round(seconds * FPS))
            log("condition", f"--match-audio-duration: {seconds:.2f} s of audio -> {inputs['num_frames']} frames")
        else:
            inputs["num_frames"] = snap_num_frames(args.num_frames or DEFAULT_FRAMES)
    return inputs


def run_condition_stage(args: argparse.Namespace) -> int:
    import torch

    error_file = args.handoff.with_name(HANDOFF_ERROR_NAME)
    try:
        blocks, _ = split_workflow(args.model_dir, args.workflow)
        inputs = conditioning_inputs(args)
        log("condition", "loading the text/vision encoder")
        pipeline, seconds = make_pipeline(blocks, args.model_dir, "cuda:0")
        log("condition", f"set up in {seconds:.0f}s (the weights are read from disk when the encoder first moves onto the GPU)")

        names = [output.name for output in blocks.intermediate_outputs]
        started = time.time()
        values = pipeline(output=names, **inputs)
        log("condition", f"encoded in {time.time() - started:.0f}s: {sorted(values)}")
        log("condition", f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

        # Plain CPU values, so the other process (a different GPU) can read them, plus which of them were on the
        # GPU: the rest of the pipeline expects each tensor where a single-process run would have it (the text
        # embedding on the GPU, but `text_token_tags` on the CPU, where the layout step indexes CPU tensors with it).
        on_gpu = sorted(key for key, value in values.items() if isinstance(value, torch.Tensor) and value.is_cuda)
        values = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in values.items()}
        partial = args.handoff.with_name(args.handoff.name + ".partial")
        torch.save({"values": values, "on_gpu": on_gpu}, partial)
        os.replace(partial, args.handoff)
        return 0
    except Exception as error:  # reported to the generation process through a file
        traceback.print_exc()
        error_file.write_text(f"{type(error).__name__}: {error}")
        print(f"ERROR: conditioning step failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


# ---------------------------------------------------------------------------
# The generation half (this process, first GPU) -- also starts the child.
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> int:
    generation_gpu, conditioning_gpu = pick_gpus()
    check_weights(args.model_dir, args.workflow)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    handoff = args.work_dir / HANDOFF_NAME
    error_file = args.work_dir / HANDOFF_ERROR_NAME
    handoff.unlink(missing_ok=True)
    error_file.unlink(missing_ok=True)

    # Each process sees only its own GPU, as cuda:0. This must happen before torch is imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = generation_gpu
    # Less fragmentation while 60 GB models are moved on and off the GPU.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    log("run", f"generation GPU {generation_gpu}, conditioning GPU {conditioning_gpu}")

    child = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--stage", "condition", "--handoff", str(handoff)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": conditioning_gpu},
    )
    try:
        import torch
        from diffusers.utils.export_utils import encode_video

        _, rest_blocks = split_workflow(args.model_dir, args.workflow)
        log("run", "loading the transformer and VAEs (while the encoder loads in the other process)")
        pipeline, seconds = make_pipeline(rest_blocks, args.model_dir, "cuda:0")
        log("run", f"set up in {seconds:.0f}s")
        if args.attention_backend:
            for name in ("transformer", "transformer_ref"):
                model = getattr(pipeline, name, None)
                if model is not None:
                    model.set_attention_backend(args.attention_backend)
            log("run", f"attention backend: {args.attention_backend}")

        # Loading is lazy (memory-mapped): the weights are only read from disk when a model first moves onto the GPU.
        # Move them now, while the encoder works in the other process, so both halves really read in parallel.
        started = time.time()
        for name in ("transformer", "transformer_ref", "vae", "audio_vae"):
            model = getattr(pipeline, name, None)
            if isinstance(model, torch.nn.Module):
                model.to("cuda:0")
        log("run", f"transformer and VAEs on the GPU in {time.time() - started:.0f}s")

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

        waited = time.time()
        returncode = child.wait()
        if returncode != 0:
            reason = error_file.read_text() if error_file.is_file() else f"exit code {returncode}"
            raise RuntimeError(f"conditioning step failed: {reason}")
        log("run", f"conditioning ready (waited {time.time() - waited:.0f}s for it)")

        conditioning = torch.load(handoff, weights_only=False)  # written by the child just above, in our own job folder
        device = torch.device("cuda:0")
        kwargs = dict(conditioning["values"])
        for key in conditioning["on_gpu"]:
            kwargs[key] = kwargs[key].to(device)
        if args.workflow != "ref2va":  # ref2va's frame count is resolved by the conditioning half
            kwargs["num_frames"] = snap_num_frames(args.num_frames or DEFAULT_FRAMES)
        if args.workflow == "t2va" and args.height is not None:  # fl2va/ref2va canvases come back resolved
            kwargs["height"], kwargs["width"] = args.height, args.width
        kwargs["generator"] = torch.Generator().manual_seed(args.seed)
        if args.steps is not None:
            kwargs["num_inference_steps"] = args.steps

        started = time.time()
        results = pipeline(output=OUTPUTS, **kwargs)
        frames = results["videos"][0]
        log("run", f"generated {len(frames)} frames of {frames[0].size[0]}x{frames[0].size[1]} in {time.time() - started:.0f}s")
        log("run", f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

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
        # The handoff holds processed copies of the request's reference media; nothing needs it after this run.
        for leftover in (handoff, handoff.with_name(handoff.name + ".partial"), error_file):
            leftover.unlink(missing_ok=True)
        try:
            args.work_dir.rmdir()
        except OSError:  # not empty (or already gone): leave it, the job folder is deleted with the job
            pass

    if not args.output_path.is_file() or args.output_path.stat().st_size == 0:
        raise RuntimeError(f"{args.output_path} was not written.")
    log("run", f"wrote {args.output_path} (total {time.time() - _STARTED:.0f}s)")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run_condition_stage(args) if args.stage == "condition" else run_pipeline(args)
    except Exception as error:
        traceback.print_exc()
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
