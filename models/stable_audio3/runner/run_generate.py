#!/usr/bin/env python3
"""Runs Stable Audio 3 Medium for one request -- text-to-music, audio-to-audio
transform, or inpaint. See `gateway/services/stable_audio3/dispatch.py`, which
builds the argv below, and `models/stable_audio3/runner/worker.py`, which keeps
the model loaded and calls `parse_args` / `generate` of this file for each
request. The partial-path-rename-on-success convention is owned by the worker
loop; this script only writes `--output-path` and exits non-zero on failure.

Must be run with this model's own environment
(`$VISION_DATA_DIR/envs/stable_audio3/bin/python`).

--model-dir is `<vision-data>/weights/stable_audio3`; its `hf/` folder is a private
Hugging Face cache (see download_weights.sh). The loader is told to run offline.

Usage:
    run_generate.py --model-dir DIR --output-path FILE.flac --mode generate|transform|inpaint \\
        --prompt=TEXT --duration SECONDS [--steps N] [--seed N] \\
        [--init-audio FILE --init-noise-level X]              (transform) \\
        [--inpaint-audio FILE --mask-start S --mask-end S]    (inpaint)

--prompt is passed as `--prompt=value` so a value starting with "-" is never
mistaken for an option. Nothing here goes through a shell.

Output is always 44.1 kHz stereo 24-bit FLAC. Exit code 0 only if --output-path
was actually written.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNNER_DIR.parents[1] / "common"))  # models/common/audio_io.py

MAX_DURATION_SECONDS = 380.0  # upstream: Stable Audio 3 Medium max length
DEFAULT_STEPS = 8  # upstream default for the post-trained (distilled) Medium checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """`argv` is the flags only (no script name); None reads the real command line."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--output-path", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=["generate", "transform", "inpaint"])
    p.add_argument("--prompt", required=True)
    p.add_argument("--duration", required=True, type=float, help="Output length in seconds.")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-audio", type=Path, help="transform: the recording to start from.")
    p.add_argument("--init-noise-level", type=float, default=0.5, help="transform: 0 keeps the input, 1 ignores it.")
    p.add_argument("--inpaint-audio", type=Path, help="inpaint: the recording to repair or extend.")
    p.add_argument("--mask-start", type=float, help="inpaint: first second to regenerate.")
    p.add_argument("--mask-end", type=float, help="inpaint: last second to regenerate.")
    args = p.parse_args(argv)

    if not args.model_dir.is_dir():
        p.error(f"Model directory not found: {args.model_dir}")
    if not 0 < args.duration <= MAX_DURATION_SECONDS:
        p.error(f"--duration must be in (0, {MAX_DURATION_SECONDS:g}] seconds.")
    if not 1 <= args.steps <= 200:
        p.error("--steps must be between 1 and 200.")
    if args.seed < 0:
        p.error("--seed must be >= 0.")
    if not 0.0 <= args.init_noise_level <= 1.0:
        p.error("--init-noise-level must be between 0 and 1.")
    if args.mode == "transform":
        if args.init_audio is None or not args.init_audio.is_file():
            p.error("--mode transform needs an existing --init-audio file.")
    if args.mode == "inpaint":
        if args.inpaint_audio is None or not args.inpaint_audio.is_file():
            p.error("--mode inpaint needs an existing --inpaint-audio file.")
        if args.mask_start is None or args.mask_end is None:
            p.error("--mode inpaint needs --mask-start and --mask-end.")
        if not 0 <= args.mask_start < args.mask_end:
            p.error("--mask-start must be >= 0 and less than --mask-end.")
    return args


def configure_environment(model_dir: Path) -> None:
    """Must run before torch / huggingface_hub are imported."""
    # The upstream loader resolves its files through the Hugging Face cache; point it at the
    # private cache that download_weights.sh filled, and never reach for the network.
    os.environ["HF_HOME"] = str(model_dir / "hf")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def load_pipeline(model_dir: Path):
    """Load Stable Audio 3 Medium onto the GPU (upstream's default half precision)."""
    from stable_audio_3 import StableAudioModel

    started = time.time()
    print(f"Loading Stable Audio 3 Medium from {model_dir / 'hf'}", flush=True)
    model = StableAudioModel.from_pretrained("medium", device="cuda")
    print(f"Loaded in {time.time() - started:.0f}s", flush=True)
    return model


def generate(model, args: argparse.Namespace, started: float) -> int:
    """Run one request on a loaded model and write args.output_path (FLAC)."""
    import torch
    from audio_io import decode_audio, write_flac

    torch.cuda.reset_peak_memory_stats()

    call_kwargs: dict = {
        "prompt": args.prompt,
        "duration": args.duration,
        "steps": args.steps,
        "seed": args.seed,
        "sample_size": model.model_config["sample_size"],
    }
    if args.mode == "transform":
        sample_rate, audio = decode_audio(args.init_audio)
        call_kwargs["init_audio"] = (sample_rate, torch.from_numpy(audio))  # [channels, frames]
        call_kwargs["init_noise_level"] = args.init_noise_level
    elif args.mode == "inpaint":
        sample_rate, audio = decode_audio(args.inpaint_audio)
        call_kwargs["inpaint_audio"] = (sample_rate, torch.from_numpy(audio))
        call_kwargs["inpaint_mask_start_seconds"] = args.mask_start
        call_kwargs["inpaint_mask_end_seconds"] = args.mask_end

    generate_started = time.time()
    result = model.generate(**call_kwargs)  # [1, channels, frames] float32 in [-1, 1]
    out_rate = model.model.sample_rate
    print(f"Generated {result.shape[-1] / out_rate:.1f}s of audio in {time.time() - generate_started:.1f}s", flush=True)
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    write_flac(args.output_path, result[0].transpose(0, 1).cpu().numpy(), out_rate)
    if not args.output_path.is_file() or args.output_path.stat().st_size == 0:
        print(f"ERROR: {args.output_path} was not written.", file=sys.stderr)
        return 1
    print(f"Wrote {args.output_path} (total {time.time() - started:.0f}s)", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    configure_environment(args.model_dir)
    started = time.time()
    model = load_pipeline(args.model_dir)
    return generate(model, args, started)


if __name__ == "__main__":
    sys.exit(main())
