#!/usr/bin/env python3
"""Runs Applio's own `core.py infer` on a single recording, after first
decoding it to a WAV -- the single unit of work model-api's Slurm job
submits for one `/convert` request -- see
`model-api/services/rvc/dispatch.py`, which builds the argv below, and
`model-api/common/run_pipeline_job.py`, which actually invokes this on the
compute node (that wrapper owns the partial-path-rename-on-success /
`.failed`-marker-on-failure convention; this script only needs to write
`--output-path` and exit non-zero on any real failure).

Supersedes `run_infer.sh` (a plain bash `cd` + exec, kept on disk but no
longer referenced by model-api) for two reasons:

1. **cwd-relative internal lookup.** Same root cause `run_infer.sh` was
   originally written for: core.py's own module-level code resolves an
   internal file (`rvc/lib/tools/tts_voices.json`) relative to the
   process's cwd, not its own `__file__`. Fixed the same way -- invoke
   `core.py` with this project's own root as cwd, not whatever directory
   the Slurm job happens to start in.
2. **No real support for the input formats callers actually send.**
   Applio's own `load_audio_infer()` (`rvc/lib/utils.py`) only ever calls
   `soundfile.read()`, and this deployment's libsndfile (1.2.2) can't
   decode AAC/M4A -- a real iPhone-recorded `.m4a` voice note uploaded
   through this API failed outright on exactly this gap. Fixed by decoding
   the input to WAV first (`normalize_audio.py`'s `to_wav()`, using the
   ffmpeg binary already bundled in this project's own venv via
   imageio-ffmpeg -- no system ffmpeg exists on this cluster) before
   `core.py` ever sees it.

Must be run with *Applio's own* venv interpreter (`RVC_PYTHON` in
`model-api/services/rvc/config.py`, i.e.
`ModelService_RVC-Applio/.venv/bin/python`) -- both `core.py` and this
script's own `normalize_audio` import (which needs `imageio-ffmpeg`) only
exist there.

Usage:
    run_convert.py --input-path FILE --work-dir DIR --output-path FILE \\
        [any other `core.py infer` flag ...]

--input-path   The original recording to convert (any format ffmpeg can
               decode -- see model-api/services/rvc/config.py's
               INPUT_EXTENSIONS for what's actually offered to callers).
--work-dir     Directory to write the normalized "source.wav" into --
               already exists by the time this runs (it's the job's own
               job_dir) and is not cleaned up afterwards, same as
               run_batch.py's own --output-folder convention.
--output-path  Forwarded to `core.py infer` unchanged. Also intercepted
               here (rather than left to fall through in `extra_args`)
               purely so this script can explicitly verify it was written
               before declaring success -- defense in depth, not strictly
               required, since model-api/common/run_pipeline_job.py
               already checks this same path's existence on its own; see
               run_batch.py's own per-file verification for the same
               reasoning applied to a batch.

Every other flag (--pth-path, --index-path, --pitch, --f0-method,
--export-format, ...) is forwarded to `core.py infer` verbatim, in its
original order.

Exit code 0 only if --output-path was actually written; non-zero on any
failure (an undecodable input, or a non-zero/missing-output exit from
`infer` itself).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from normalize_audio import AudioDecodeError, to_wav

APPLIO_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "Applio"  # <repo>/models/rvc/runner/<this file>
CORE_SCRIPT = APPLIO_ROOT / "core.py"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", required=True, help="The original recording to convert.")
    parser.add_argument("--work-dir", required=True, help="Directory to write the normalized source.wav into.")
    parser.add_argument("--output-path", required=True, help="Forwarded to core.py infer unchanged.")
    # parse_known_args, not parse_args: everything this parser doesn't
    # recognize (--pth-path, --pitch, --f0-method, --export-format, ...)
    # comes back as a flat list in `extra`, in its original order --
    # exactly what gets forwarded to `core.py infer` unchanged below. Same
    # convention run_batch.py already uses for the same reason.
    return parser.parse_known_args()


def main() -> int:
    args, extra_args = _parse_args()
    input_path = Path(args.input_path)
    work_dir = Path(args.work_dir)
    output_path = Path(args.output_path)

    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    normalized_path = work_dir / "source.wav"
    print(f"Normalizing '{input_path}' -> '{normalized_path}'...", flush=True)
    try:
        to_wav(input_path, normalized_path)
    except AudioDecodeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    # cwd=APPLIO_ROOT: same cwd fix run_infer.sh used to provide (see
    # module docstring's gap #1).
    core_argv = [
        sys.executable, str(CORE_SCRIPT), "infer",
        "--input-path", str(normalized_path),
        "--output-path", str(output_path),
        *extra_args,
    ]
    print("argv:", core_argv, flush=True)
    start = time.monotonic()
    proc = subprocess.run(core_argv, cwd=str(APPLIO_ROOT))
    elapsed = time.monotonic() - start
    print(f"=== infer: finished in {elapsed:.1f}s (exit {proc.returncode}) ===", flush=True)
    if proc.returncode != 0:
        print(f"ERROR: core.py infer exited with code {proc.returncode}", file=sys.stderr)
        return proc.returncode

    if not output_path.is_file():
        export_ext = output_path.suffix.lstrip(".").upper() or "the requested format"
        print(
            f"ERROR: infer exited successfully, but the expected output file is missing: "
            f"{output_path}. This usually means Applio's own WAV-to-{export_ext} conversion step "
            "silently failed (it prints a warning but does not raise) -- check the 'An error "
            "occurred converting the audio format' lines above for the real cause.",
            file=sys.stderr,
        )
        return 1

    print(f"Conversion completed: {normalized_path} -> {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
