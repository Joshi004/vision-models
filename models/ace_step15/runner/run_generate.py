#!/usr/bin/env python3
"""Runs ACE-Step 1.5 (XL SFT diffusion transformer + 4B language model) for one request --
text-to-music, cover, or repaint. See `gateway/services/ace_step15/dispatch.py`, which
builds the argv below, and `models/ace_step15/runner/worker.py`, which keeps the models
loaded and calls `parse_args` / `generate` of this file for each request. The
partial-path-rename-on-success convention is owned by the worker loop; this script
only writes `--output-path` and exits non-zero on failure.

Must be run with this model's own environment
(`$VISION_DATA_DIR/envs/ace_step15/bin/python`, Python 3.11).

--model-dir is `<vision-data>/weights/ace_step15`; its `checkpoints/` folder holds
acestep-v15-xl-sft, acestep-5Hz-lm-4B, vae and Qwen3-Embedding-0.6B (see
download_weights.sh). Everything runs in bf16 with no quantization and no CPU offload.

Usage:
    run_generate.py --model-dir DIR --output-path FILE.flac --mode generate|cover|repaint \\
        --caption=TEXT [--lyrics=TEXT | --instrumental] [--vocal-language CODE] \\
        [--duration SECONDS] [--bpm N] [--keyscale=TEXT] [--time-signature N] \\
        [--steps N] [--guidance-scale X] [--seed N] [--no-thinking] \\
        [--src-audio FILE --cover-strength X]                  (cover) \\
        [--src-audio FILE --repaint-start S --repaint-end S]   (repaint)

Text values are passed as `--flag=value` so a value starting with "-" is never mistaken
for an option. Nothing here goes through a shell.

Output is always 48 kHz stereo FLAC. Exit code 0 only if --output-path was actually written.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

DIT_MODEL = "acestep-v15-xl-sft"
LM_MODEL = "acestep-5Hz-lm-4B"
MIN_DURATION_SECONDS = 10.0  # upstream: 10 ~ 600
MAX_DURATION_SECONDS = 600.0
DEFAULT_STEPS = 50  # upstream: 32-100 for non-turbo models
DEFAULT_GUIDANCE_SCALE = 7.0
INSTRUMENTAL_LYRICS = "[Instrumental]"  # upstream's marker for "no vocals"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """`argv` is the flags only (no script name); None reads the real command line."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--output-path", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=["generate", "cover", "repaint"])
    p.add_argument("--caption", required=True)
    p.add_argument("--lyrics", default="")
    p.add_argument("--instrumental", action="store_true", help="No vocals, whatever the lyrics say.")
    p.add_argument("--vocal-language", default="unknown")
    p.add_argument("--duration", type=float, default=None, help="Seconds; omitted lets the model choose (generate only).")
    p.add_argument("--bpm", type=int, default=None)
    p.add_argument("--keyscale", default="")
    p.add_argument("--time-signature", default="")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True,
                   help="generate: let the language model plan metadata and audio codes first.")
    p.add_argument("--src-audio", type=Path, help="cover / repaint: the source recording.")
    p.add_argument("--cover-strength", type=float, default=1.0, help="cover: 1 = follow the source closely, low = free interpretation.")
    p.add_argument("--repaint-start", type=float, help="repaint: first second to regenerate.")
    p.add_argument("--repaint-end", type=float, help="repaint: last second to regenerate (-1 = to the end).")
    args = p.parse_args(argv)

    if not (args.model_dir / "checkpoints").is_dir():
        p.error(f"{args.model_dir} must hold a checkpoints folder.")
    if not args.caption.strip():
        p.error("--caption must not be empty.")
    if args.duration is not None and not MIN_DURATION_SECONDS <= args.duration <= MAX_DURATION_SECONDS:
        p.error(f"--duration must be between {MIN_DURATION_SECONDS:g} and {MAX_DURATION_SECONDS:g} seconds.")
    if args.bpm is not None and not 30 <= args.bpm <= 300:
        p.error("--bpm must be between 30 and 300.")
    if not 1 <= args.steps <= 200:
        p.error("--steps must be between 1 and 200.")
    if not 0 <= args.guidance_scale <= 30:
        p.error("--guidance-scale must be between 0 and 30.")
    if args.seed < 0:
        p.error("--seed must be >= 0.")
    if not 0.0 <= args.cover_strength <= 1.0:
        p.error("--cover-strength must be between 0 and 1.")
    if args.mode in ("cover", "repaint") and (args.src_audio is None or not args.src_audio.is_file()):
        p.error(f"--mode {args.mode} needs an existing --src-audio file.")
    if args.mode == "repaint":
        if args.repaint_start is None or args.repaint_end is None:
            p.error("--mode repaint needs --repaint-start and --repaint-end.")
        if args.repaint_start < 0 or (args.repaint_end != -1 and args.repaint_end <= args.repaint_start):
            p.error("--repaint-start must be >= 0 and less than --repaint-end (or --repaint-end -1).")
    return args


def configure_environment(model_dir: Path) -> None:
    """Must run before torch / acestep are imported."""
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(model_dir / "checkpoints")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def load_pipeline(model_dir: Path):
    """Load the diffusion transformer and the language model onto the GPU. Returns (dit, llm)."""
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    started = time.time()
    print(f"Loading ACE-Step 1.5 ({DIT_MODEL} + {LM_MODEL}) from {model_dir / 'checkpoints'}", flush=True)
    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=str(model_dir),
        config_path=DIT_MODEL,
        device="cuda",
        use_flash_attention=True,
        offload_to_cpu=False,
    )
    if not ok:
        raise RuntimeError(f"The diffusion model failed to load: {status}")
    llm = LLMHandler()
    status, ok = llm.initialize(
        checkpoint_dir=str(model_dir / "checkpoints"),
        lm_model_path=LM_MODEL,
        backend="vllm",
        device="cuda",
        offload_to_cpu=False,
    )
    if not ok:
        raise RuntimeError(f"The language model failed to load: {status}")
    print(f"Loaded in {time.time() - started:.0f}s", flush=True)
    return dit, llm


def build_params(args: argparse.Namespace):
    from acestep.inference import GenerationParams

    lyrics = INSTRUMENTAL_LYRICS if args.instrumental and not args.lyrics.strip() else args.lyrics
    params = dict(
        caption=args.caption,
        lyrics=lyrics,
        instrumental=args.instrumental,
        vocal_language=args.vocal_language,
        duration=args.duration if args.duration is not None else -1.0,
        bpm=args.bpm,
        keyscale=args.keyscale,
        timesignature=args.time_signature,
        inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    if args.mode == "generate":
        params.update(task_type="text2music", thinking=args.thinking)
    elif args.mode == "cover":
        # Cover and repaint are conditioned on the source audio; upstream skips the language model for them.
        params.update(task_type="cover", src_audio=str(args.src_audio), audio_cover_strength=args.cover_strength, thinking=False)
    else:
        params.update(
            task_type="repaint",
            src_audio=str(args.src_audio),
            repainting_start=args.repaint_start,
            repainting_end=args.repaint_end,
            chunk_mask_mode="explicit",  # pass the exact range as the mask (upstream's documented repaint setting)
            thinking=False,
        )
    return GenerationParams(**params)


def generate(pipe, args: argparse.Namespace, started: float) -> int:
    """Run one request on loaded models and write args.output_path (FLAC)."""
    import torch
    from acestep.inference import GenerationConfig, generate_music

    dit, llm = pipe
    torch.cuda.reset_peak_memory_stats()

    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[args.seed], audio_format="flac")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    # Upstream names its own output files; collect them in a scratch folder next to the output.
    scratch = Path(tempfile.mkdtemp(prefix=".ace_", dir=args.output_path.parent))
    try:
        generate_started = time.time()
        result = generate_music(dit, llm, build_params(args), config, save_dir=str(scratch))
        if not result.success or not result.audios:
            print(f"ERROR: ACE-Step failed: {result.error or result.status_message}", file=sys.stderr)
            return 1
        print(f"Generated in {time.time() - generate_started:.1f}s", flush=True)
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        shutil.move(result.audios[0]["path"], args.output_path)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

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
