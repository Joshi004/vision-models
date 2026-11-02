#!/usr/bin/env python3
"""Runs YuE2 for one request -- a full song with vocals from a style description and
lyrics, either planned by the model (`generate`) or following a score the caller
supplies (`cover`). See `gateway/services/yue2/dispatch.py`, which builds the argv
below, and `models/yue2/runner/worker.py`, which keeps the model loaded and calls
`parse_args` / `generate` of this file for each request. The
partial-path-rename-on-success convention is owned by the worker loop; this script
only writes `--output-path` and exits non-zero on failure.

Must be run with this model's own environment
(`$VISION_DATA_DIR/envs/yue2/bin/python`, Python 3.12).

--model-dir is `<vision-data>/weights/yue2`, holding `YuE2-3B/` and `YuE2-Vae/`.
Everything runs in bf16 with no quantization or offload, as upstream's unquantized preset.

Usage:
    run_generate.py --model-dir DIR --output-path FILE.flac --mode generate|cover \\
        --style=TEXT --lyrics=TEXT [--cot full|melody|off] [--cfg-scale X] [--seed N] \\
        [--abc-file FILE]                                    (cover: the score, ABC text)

--style and --lyrics are passed as `--flag=value` so a value starting with "-" is
never mistaken for an option. Nothing here goes through a shell.

Output is always 48 kHz stereo 24-bit FLAC. Exit code 0 only if --output-path was
actually written.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNNER_DIR.parents[1] / "common"))  # models/common/audio_io.py

MAX_SEED = 2**63 - 1
MEMORY_BUDGET_GIB = 72  # upstream's default (24) targets small GPUs; this only raises the decoder window size


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """`argv` is the flags only (no script name); None reads the real command line."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--output-path", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=["generate", "cover"])
    p.add_argument("--style", required=True)
    p.add_argument("--lyrics", required=True)
    p.add_argument("--cot", choices=["full", "melody", "off"], default="full")
    p.add_argument("--cfg-scale", type=float, default=None, help="Text guidance, 0-20 (default: upstream's).")
    p.add_argument("--seed", type=int, default=831001)
    p.add_argument("--abc-file", type=Path, help="cover: a text file holding the ABC score to follow.")
    args = p.parse_args(argv)

    if not (args.model_dir / "YuE2-3B").is_dir() or not (args.model_dir / "YuE2-Vae").is_dir():
        p.error(f"{args.model_dir} must hold the YuE2-3B and YuE2-Vae folders.")
    if not args.style.strip() or not args.lyrics.strip():
        p.error("--style and --lyrics must not be empty.")
    if not 0 <= args.seed <= MAX_SEED:
        p.error(f"--seed must be between 0 and {MAX_SEED}.")
    if args.cfg_scale is not None and not 0 <= args.cfg_scale <= 20:
        p.error("--cfg-scale must be between 0 and 20.")
    if args.mode == "cover":
        if args.abc_file is None or not args.abc_file.is_file():
            p.error("--mode cover needs an existing --abc-file.")
        if args.cot == "off":
            p.error("--mode cover follows a score, so --cot must be full or melody.")
    elif args.abc_file is not None:
        p.error("--abc-file is only for --mode cover.")
    return args


def configure_environment(model_dir: Path) -> None:
    """Must run before torch is imported. YuE2 reads local folders only; nothing is fetched."""
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def load_pipeline(model_dir: Path):
    """Load YuE2-3B and its VAE onto the GPU."""
    from yue2 import YuE2Pipeline

    started = time.time()
    print(f"Loading YuE2 from {model_dir}", flush=True)
    pipe = YuE2Pipeline.from_pretrained(
        model_dir / "YuE2-3B",
        vae=model_dir / "YuE2-Vae",
        device="cuda",
        memory_budget_gib=MEMORY_BUDGET_GIB,
        progress=False,
    )
    # The pipeline loads its transformer on first use; do that now so a warm worker's first
    # request does not pay for it. (_load_model is private upstream API: re-check on upgrades.)
    pipe._load_model()
    print(f"Loaded in {time.time() - started:.0f}s", flush=True)
    return pipe


def generate(pipe, args: argparse.Namespace, started: float) -> int:
    """Run one request on a loaded pipeline and write args.output_path (FLAC)."""
    import torch

    torch.cuda.reset_peak_memory_stats()

    request: dict = {"style": args.style, "lyrics": args.lyrics, "cot": args.cot, "seed": args.seed}
    if args.cfg_scale is not None:
        request["cfg_scale"] = args.cfg_scale
    if args.mode == "cover":
        request["abc"] = args.abc_file.read_text(encoding="utf-8")

    generate_started = time.time()
    song = pipe(**request)
    seconds = len(song.audio) / song.sample_rate
    print(f"Generated {seconds:.1f}s of audio in {time.time() - generate_started:.1f}s (truncated={song.truncated})", flush=True)
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    song.save(args.output_path)  # upstream writes 24-bit FLAC for a .flac path
    if not args.output_path.is_file() or args.output_path.stat().st_size == 0:
        print(f"ERROR: {args.output_path} was not written.", file=sys.stderr)
        return 1
    print(f"Wrote {args.output_path} (total {time.time() - started:.0f}s)", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    configure_environment(args.model_dir)
    started = time.time()
    pipe = load_pipeline(args.model_dir)
    return generate(pipe, args, started)


if __name__ == "__main__":
    sys.exit(main())
