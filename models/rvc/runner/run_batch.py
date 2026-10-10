#!/usr/bin/env python3
"""Runs Applio's own `core.py batch-infer` on a folder of input recordings,
then packages every successfully converted file into a single zip -- the
single unit of work model-api's Slurm job submits for a batch conversion --
see model-api/services/rvc/dispatch.py, which builds the argv below, and
model-api/common/run_pipeline_job.py, which actually invokes it on the
compute node (that wrapper owns the partial-path-rename-on-success /
.failed-marker-on-failure convention; this script only needs to write
--output-path and exit non-zero on any real failure).

Four real gaps in `core.py batch-infer` this wrapper exists to paper over
(found by reading rvc/infer/infer.py's VoiceConverter.convert_audio /
convert_audio_batch and rvc/lib/utils.py's load_audio_infer directly, not
assumed) -- plus one real naming trap already hit once while building
this: the click command is registered as `batch-infer` (hyphen), not
`batch_infer` (the Python function's own name, `def batch_infer(...)` in
core.py) -- click rewrites underscores to hyphens in a command's default
name. Confirmed directly against a real `core.py --help` on this cluster
after guessing wrong the first time, not assumed from the function name
alone -- unlike `infer`/`tts` (no underscore to rewrite), this one
actually differs from its Python name.

1. **cwd-relative internal lookup.** Same root cause as run_infer.sh:
   core.py's own module-level code resolves an internal file
   (rvc/lib/tools/tts_voices.json) relative to the process's cwd, not its
   own __file__. Fixed the same way -- invoke core.py with this project's
   own root as cwd, not whatever directory the Slurm job happens to start
   in.
2. **The output folder is never created.** convert_audio_batch() does
   os.listdir(audio_input_paths) (must already exist -- fine, dispatch.py
   creates it) but never touches audio_output_path itself; sf.write()
   into a missing directory raises. Fixed by mkdir'ing it here first.
3. **A failed format conversion doesn't fail the job.** Per file,
   convert_audio() always writes a plain .wav first, then -- only if
   export_format != WAV -- calls convert_audio_format() to also produce
   the requested format. That second step's own exception handling prints
   and swallows any error instead of raising, and its caller
   (convert_audio_batch) never checks convert_audio()'s return value at
   all. Net effect: core.py can exit 0 having silently produced only the
   intermediate .wav for some file, not the format the request actually
   asked for. Fixed here by explicitly verifying one "<stem>_output.<ext>"
   file per input after core.py returns, before declaring success.
4. **No real support for the input formats convert_audio_batch() claims to
   accept.** Its own os.listdir() filter matches .m4a/.mp4/.aac/.wma/
   .webm/.ac3/... by extension, but every actual read goes through
   soundfile.read() (load_audio_infer(), rvc/lib/utils.py) -- and this
   deployment's libsndfile (1.2.2) can only decode WAV/MP3/FLAC/OGG/Opus/
   AIFF, not AAC/M4A or the rest. A real iPhone-recorded .m4a voice note
   uploaded through this API failed outright on exactly this gap. Fixed by
   decoding every input file to WAV first (normalize_audio.py's to_wav(),
   using the ffmpeg binary already bundled in this project's own venv via
   imageio-ffmpeg -- no system ffmpeg exists on this cluster) into a
   separate --work-folder, then pointing batch-infer at that folder
   instead of the caller's original one.

Must be run with *Applio's own* venv interpreter (RVC_PYTHON in
model-api/services/rvc/config.py, i.e. ModelService_RVC-Applio/.venv/bin/
python) -- both core.py and this script's own normalize_audio import
(which needs imageio-ffmpeg) only exist there.

Usage:
    run_batch.py --input-folder DIR --work-folder DIR --output-folder DIR \\
        --output-path ZIP --export-format FMT \\
        [any other `core.py batch-infer` flag ...]

--input-folder   Folder of input recordings to convert (already populated
                  by the caller -- this script only reads it; may be a
                  folder of symlinks to real files elsewhere, see
                  model-api/services/rvc/dispatch.py's _symlink_inputs).
--work-folder    Folder this script decodes a WAV copy of every input file
                  into (created here if missing) -- what's actually passed
                  to `core.py batch-infer` as --input-folder. Kept separate
                  from --input-folder itself since that folder may only
                  contain symlinks this script shouldn't write into.
--output-folder  Folder Applio itself writes its per-file outputs into
                  (created here if missing; not cleaned up afterwards --
                  the caller owns this directory's lifecycle, same as
                  run_replace.py's --work-dir).
--output-path    Final zip path to write. Written directly (this script
                  does not do its own "partial" renaming -- see
                  run_pipeline_job.py). Not a `core.py batch-infer` flag --
                  read here, not forwarded.
--export-format  Forwarded to `core.py batch-infer` unchanged, and also
                  used here to know which output extension to verify/zip.

Every other flag (--pth-path, --index-path, --pitch, --f0-method, ...) is
forwarded to `core.py batch-infer` verbatim, in its original order --
this script only needs to intercept the five flags above.

Exit code 0 only if --output-path was actually written with one converted
file per input inside it; non-zero on any failure (an undecodable input, a
non-zero exit from `batch-infer` itself, a missing expected output, or a
failed zip write).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from normalize_audio import AudioDecodeError, to_wav

APPLIO_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "Applio"  # <repo>/models/rvc/runner/<this file>
CORE_SCRIPT = APPLIO_ROOT / "core.py"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-folder", required=True, help="Folder of input recordings to convert.")
    parser.add_argument("--work-folder", required=True, help="Folder to write normalized (WAV) copies of each input into.")
    parser.add_argument("--output-folder", required=True, help="Folder for core.py's own per-file outputs.")
    parser.add_argument("--output-path", required=True, help="Final zip path to write.")
    parser.add_argument("--export-format", required=True, help="Output audio format (WAV/MP3/FLAC/OGG).")
    # parse_known_args, not parse_args: everything this parser doesn't
    # recognize (--pth-path, --pitch, --f0-method, ...) comes back as a
    # flat list in `extra`, in its original order -- exactly what gets
    # forwarded to `core.py batch-infer` unchanged below. Confirmed
    # directly (not assumed) that argparse preserves flag/value pairing
    # and order for unrecognized tokens.
    return parser.parse_known_args()


def _run_step(name: str, argv: list[str], cwd: Path) -> int:
    print(f"=== {name}: starting ===", flush=True)
    print("argv:", argv, flush=True)
    start = time.monotonic()
    proc = subprocess.run(argv, cwd=str(cwd))
    elapsed = time.monotonic() - start
    print(f"=== {name}: finished in {elapsed:.1f}s (exit {proc.returncode}) ===", flush=True)
    return proc.returncode


def main() -> int:
    args, extra_args = _parse_args()
    input_folder = Path(args.input_folder).resolve()
    work_folder = Path(args.work_folder).resolve()
    output_folder = Path(args.output_folder).resolve()
    output_path = Path(args.output_path).resolve()
    export_ext = args.export_format.lower()

    if not input_folder.is_dir():
        print(f"ERROR: input folder not found: {input_folder}", file=sys.stderr)
        return 1

    # Same os.path.splitext(...)[0] convention convert_audio_batch itself
    # uses to name each output ("<stem>_output.<ext>") -- computed here so
    # the verification step below can check for exactly the files Applio
    # should have produced, regardless of what it actually did. Computed
    # from the original (pre-normalization) filenames -- normalizing only
    # ever changes a file's extension, never its stem, so this set is the
    # same either way.
    input_stems = sorted(p.stem for p in input_folder.iterdir() if p.is_file())
    if not input_stems:
        print(f"ERROR: no input files found in {input_folder}", file=sys.stderr)
        return 1

    # Gap #4 (see module docstring): decode every input to WAV first --
    # Applio's own convert_audio_batch() only ever calls soundfile.read(),
    # which can't decode AAC/M4A/... at all. Written into a separate
    # --work-folder, not input_folder itself, since the latter may only
    # contain symlinks to real uploads elsewhere (see model-api/services/
    # rvc/dispatch.py's _symlink_inputs) that this script shouldn't write
    # into.
    work_folder.mkdir(parents=True, exist_ok=True)
    decode_failures: list[str] = []
    for source in sorted(input_folder.iterdir()):
        if not source.is_file():
            continue
        try:
            to_wav(source, work_folder / f"{source.stem}.wav")
        except AudioDecodeError as error:
            decode_failures.append(f"{source.name}: {error}")
    if decode_failures:
        print("ERROR: could not decode " + "; ".join(decode_failures), file=sys.stderr)
        return 1

    # Gap #2 (see module docstring): convert_audio_batch never creates its
    # own output folder.
    output_folder.mkdir(parents=True, exist_ok=True)

    # Gap #1 (see module docstring): cwd=APPLIO_ROOT, not whatever
    # directory the Slurm job actually started in.
    # "batch-infer" (hyphen), not "batch_infer" -- click's default command
    # naming rewrites the Python function's own underscored name
    # (`def batch_infer(...)`) to a hyphenated CLI command name. See the
    # module docstring's own note on this -- confirmed directly against a
    # real `core.py --help` after this exact line first shipped wrong.
    core_argv = [
        sys.executable, str(CORE_SCRIPT), "batch-infer",
        "--input-folder", str(work_folder),
        "--output-folder", str(output_folder),
        "--export-format", args.export_format,
        *extra_args,
    ]
    returncode = _run_step("batch-infer", core_argv, cwd=APPLIO_ROOT)
    if returncode != 0:
        print(f"ERROR: core.py batch-infer exited with code {returncode}", file=sys.stderr)
        return returncode

    # Gap #3 (see module docstring): core.py can exit 0 without actually
    # having produced every requested-format file. Verify explicitly
    # rather than trusting the exit code alone.
    missing: list[str] = []
    output_files: list[Path] = []
    for stem in input_stems:
        expected = output_folder / f"{stem}_output.{export_ext}"
        if expected.is_file():
            output_files.append(expected)
        else:
            missing.append(expected.name)
    if missing:
        print(
            "ERROR: batch-infer exited successfully, but these expected output files are "
            f"missing: {', '.join(missing)}. This usually means Applio's own "
            f"WAV-to-{args.export_format} conversion step silently failed for those files "
            "(it prints a warning but does not raise) -- check the 'An error occurred "
            "converting the audio format' lines above for the real cause.",
            file=sys.stderr,
        )
        return 1

    print(f"Zipping {len(output_files)} converted file(s) into {output_path}...", flush=True)
    try:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in output_files:
                zf.write(f, arcname=f.name)
    except OSError as error:
        print(f"ERROR: could not write {output_path}: {error}", file=sys.stderr)
        return 1

    if not output_path.is_file():
        print(f"ERROR: zip step finished but {output_path} does not exist.", file=sys.stderr)
        return 1

    print(f"Batch conversion completed: {len(output_files)} file(s) -> {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
