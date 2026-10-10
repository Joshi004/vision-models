#!/usr/bin/env python3
"""Runs Breeze TTS 2's own `infer.py` for one text-to-speech request -- the
single unit of work model-api's Slurm job submits for one `/synthesize`
request. See `model-api/services/breeze_tts/dispatch.py`, which builds the
argv below, and `model-api/common/run_pipeline_job.py`, which invokes this
on the compute node (that wrapper owns the partial-path-rename-on-success /
`.failed`-marker-on-failure convention; this script only needs to write
`--output-path` and exit non-zero on any real failure).

Must be run with this project's own venv interpreter
(`ModelService_BreezeTTS2/.venv/bin/python`) -- torch, qwen-tts and
imageio-ffmpeg only exist there.

Three modes, selected purely by which flags are present (same as upstream):
  * Voice Design    : --text [--instruction]
  * Voice Clone     : --text --ref-audio-path --ref-text
  * Voice Direction : --text --ref-audio-path --ref-text --instruction

Usage:
    run_synthesize.py --model-dir DIR --work-dir DIR --output-path FILE \\
        --text=TEXT [--instruction=TEXT] \\
        [--ref-audio-path FILE --ref-text=TEXT] \\
        [--seed N] [--cfg-scale X]

Text-like flags are passed as `--flag=value` (not `--flag value`) so a value
starting with "-" is never mistaken for an option. Nothing here goes through
a shell, so user text needs no escaping.

--ref-audio-path  Reference recording (any format ffmpeg can decode). It is
                  first decoded to `{work-dir}/reference.wav`, because
                  upstream reads it with soundfile, which can't open M4A/AAC.

Eager mode only (no --fast-all): its CUDA-graph warmup costs more than it
saves on a one-shot job.

Exit code 0 only if --output-path was actually written.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from normalize_audio import AudioDecodeError, to_wav

REPO_ROOT = Path(__file__).resolve().parents[3]  # <repo>/models/breeze_tts/runner/<this file>
UPSTREAM_DIR = REPO_ROOT / "third_party" / "breeze-tts"
INFER_SCRIPT = UPSTREAM_DIR / "infer.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="Local Breeze TTS 2 checkpoint directory.")
    parser.add_argument("--work-dir", required=True, help="Directory for the decoded reference.wav.")
    parser.add_argument("--output-path", required=True, help="Where to write the generated WAV.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction")
    parser.add_argument("--ref-audio-path")
    parser.add_argument("--ref-text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model_dir = Path(args.model_dir)
    work_dir = Path(args.work_dir)
    output_path = Path(args.output_path)

    if not model_dir.is_dir():
        print(f"ERROR: model directory not found: {model_dir}", file=sys.stderr)
        return 1
    if not args.text.strip():
        print("ERROR: --text is empty.", file=sys.stderr)
        return 1

    has_ref_audio = bool(args.ref_audio_path)
    has_ref_text = bool(args.ref_text and args.ref_text.strip())
    if has_ref_audio != has_ref_text:
        print("ERROR: --ref-audio-path and --ref-text must be provided together.", file=sys.stderr)
        return 1

    infer_argv = [
        sys.executable, str(INFER_SCRIPT), str(model_dir),
        f"--text={args.text}",
        f"--seed={args.seed}",
        f"--cfg-scale={args.cfg_scale}",
        f"--output={output_path}",
    ]
    if args.instruction and args.instruction.strip():
        infer_argv.append(f"--instruction={args.instruction}")

    if has_ref_audio:
        ref_source = Path(args.ref_audio_path)
        if not ref_source.is_file():
            print(f"ERROR: reference audio not found: {ref_source}", file=sys.stderr)
            return 1
        reference_wav = work_dir / "reference.wav"
        print(f"Normalizing '{ref_source}' -> '{reference_wav}'...", flush=True)
        try:
            to_wav(ref_source, reference_wav)
        except AudioDecodeError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        infer_argv += [f"--ref-audio={reference_wav}", f"--ref-text={args.ref_text.strip()}"]

    print("argv:", infer_argv, flush=True)
    start = time.monotonic()
    # cwd=UPSTREAM_DIR: upstream resolves its own imports/configs relative to
    # its repo root; run from there to match its documented usage.
    proc = subprocess.run(infer_argv, cwd=str(UPSTREAM_DIR))
    elapsed = time.monotonic() - start
    print(f"=== infer: finished in {elapsed:.1f}s (exit {proc.returncode}) ===", flush=True)
    if proc.returncode != 0:
        print(f"ERROR: infer.py exited with code {proc.returncode}", file=sys.stderr)
        return proc.returncode

    if not output_path.is_file():
        print(f"ERROR: infer.py exited successfully but wrote no file at {output_path}", file=sys.stderr)
        return 1

    print(f"Synthesis completed: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
