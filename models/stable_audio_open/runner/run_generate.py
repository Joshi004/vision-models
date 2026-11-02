#!/usr/bin/env python3
"""Runs Stable Audio Open 1.0 for one request -- text-to-audio or audio-to-audio
transform. See `gateway/services/stable_audio_open/dispatch.py`, which builds the
argv below, and `models/stable_audio_open/runner/worker.py`, which keeps the model
loaded and calls `parse_args` / `generate` of this file for each request. The
partial-path-rename-on-success convention is owned by the worker loop; this script
only writes `--output-path` and exits non-zero on failure.

Must be run with this model's own environment
(`$VISION_DATA_DIR/envs/stable_audio_open/bin/python`, Python 3.10).

--model-dir is `<vision-data>/weights/stable_audio_open`: `model/` holds
model_config.json and model.safetensors, and `hf/` is a private Hugging Face cache
with the T5 text encoder the model loads by name (see download_weights.sh). The
loader is told to run offline. The model runs in float32, as upstream's examples do.

Usage:
    run_generate.py --model-dir DIR --output-path FILE.flac --mode generate|transform \\
        --prompt=TEXT --duration SECONDS [--negative-prompt=TEXT] [--steps N] \\
        [--cfg-scale X] [--seed N] \\
        [--init-audio FILE --init-noise-level X]              (transform)

--prompt and --negative-prompt are passed as `--flag=value` so a value starting
with "-" is never mistaken for an option. Nothing here goes through a shell.

Output is always 44.1 kHz stereo 24-bit FLAC. Exit code 0 only if --output-path
was actually written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNNER_DIR.parents[1] / "common"))  # models/common/audio_io.py

MAX_DURATION_SECONDS = 47.0  # the model's fixed 2097152-sample window is 47.55 s at 44.1 kHz
DEFAULT_STEPS = 100
DEFAULT_CFG_SCALE = 7.0
# Sampler settings recommended on the model card for text-to-audio.
SAMPLER_TYPE = "dpmpp-3m-sde"
TEXT_SIGMA_MIN = 0.3
TEXT_SIGMA_MAX = 500
# Transform: the noise level replaces sigma_max (see stable_audio_tools.inference.sampling);
# sigma_min must stay below it, so the model card's 0.3 is lowered to upstream's gradio default.
TRANSFORM_SIGMA_MIN = 0.03
MIN_INIT_NOISE_LEVEL = 0.05
MAX_INIT_NOISE_LEVEL = 20.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """`argv` is the flags only (no script name); None reads the real command line."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--output-path", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=["generate", "transform"])
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--duration", required=True, type=float, help="Output length in seconds.")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-audio", type=Path, help="transform: the recording to start from.")
    p.add_argument("--init-noise-level", type=float, default=0.5, help="transform: noise added to the input; higher departs further from it.")
    args = p.parse_args(argv)

    if not args.model_dir.is_dir():
        p.error(f"Model directory not found: {args.model_dir}")
    if not 0 < args.duration <= MAX_DURATION_SECONDS:
        p.error(f"--duration must be in (0, {MAX_DURATION_SECONDS:g}] seconds.")
    if not 1 <= args.steps <= 250:
        p.error("--steps must be between 1 and 250.")
    if not 0 <= args.cfg_scale <= 30:
        p.error("--cfg-scale must be between 0 and 30.")
    if args.seed < 0:
        p.error("--seed must be >= 0.")
    if args.mode == "transform":
        if args.init_audio is None or not args.init_audio.is_file():
            p.error("--mode transform needs an existing --init-audio file.")
        if not MIN_INIT_NOISE_LEVEL <= args.init_noise_level <= MAX_INIT_NOISE_LEVEL:
            p.error(f"--init-noise-level must be between {MIN_INIT_NOISE_LEVEL:g} and {MAX_INIT_NOISE_LEVEL:g}.")
    return args


def configure_environment(model_dir: Path) -> None:
    """Must run before torch / huggingface_hub are imported."""
    # The model loads its T5 text encoder by name; serve it from the private cache, offline.
    os.environ["HF_HOME"] = str(model_dir / "hf")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def load_pipeline(model_dir: Path):
    """Load Stable Audio Open 1.0 onto the GPU. Returns (model, model_config)."""
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict

    started = time.time()
    print(f"Loading Stable Audio Open 1.0 from {model_dir / 'model'}", flush=True)
    config = json.loads((model_dir / "model" / "model_config.json").read_text())
    model = create_model_from_config(config)
    model.load_state_dict(load_ckpt_state_dict(str(model_dir / "model" / "model.safetensors")))
    model = model.to("cuda").eval().requires_grad_(False)
    print(f"Loaded in {time.time() - started:.0f}s", flush=True)
    return model, config


def generate(pipe, args: argparse.Namespace, started: float) -> int:
    """Run one request on a loaded model and write args.output_path (FLAC)."""
    import torch
    from audio_io import decode_audio, write_flac
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    model, config = pipe
    sample_rate = config["sample_rate"]
    torch.cuda.reset_peak_memory_stats()

    def conditioning(text: str) -> list[dict]:
        return [{"prompt": text, "seconds_start": 0, "seconds_total": args.duration}]

    call_kwargs: dict = {
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "conditioning": conditioning(args.prompt),
        "sample_size": config["sample_size"],
        "seed": args.seed,
        "device": "cuda",
        "sampler_type": SAMPLER_TYPE,
    }
    if args.negative_prompt:
        call_kwargs["negative_conditioning"] = conditioning(args.negative_prompt)
    if args.mode == "transform":
        in_rate, audio = decode_audio(args.init_audio)
        call_kwargs["init_audio"] = (in_rate, torch.from_numpy(audio))  # [channels, frames]
        call_kwargs["init_noise_level"] = args.init_noise_level
        call_kwargs["sigma_min"] = TRANSFORM_SIGMA_MIN
    else:
        call_kwargs["sigma_min"] = TEXT_SIGMA_MIN
        call_kwargs["sigma_max"] = TEXT_SIGMA_MAX

    generate_started = time.time()
    result = generate_diffusion_cond(model, **call_kwargs)  # [1, channels, frames]
    frames = int(args.duration * sample_rate)
    result = result[0, :, :frames].to(torch.float32)
    # Normalise to full scale only when the model overshoots; leave quiet results alone.
    peak = result.abs().max().clamp(min=1e-8)
    if peak > 1.0:
        result = result / peak
    print(f"Generated {result.shape[-1] / sample_rate:.1f}s of audio in {time.time() - generate_started:.1f}s", flush=True)
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    write_flac(args.output_path, result.transpose(0, 1).cpu().numpy(), sample_rate)
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
