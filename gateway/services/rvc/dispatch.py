"""Builds RVC's (Applio's `core.py`) invocations and submits them as Slurm
jobs -- one file via `infer` (see build_convert_args/dispatch_convert), or
many files in one job via `batch-infer` (see
build_batch_convert_args/dispatch_batch_convert). Neither goes straight to
`core.py`: both go through a small wrapper (run_convert.py / run_batch.py)
that fixes two real gaps shared by both endpoints, plus two more specific
to batch (see run_batch.py's own header comment) and a real naming trap
(`batch-infer`, hyphenated, differs from its own Python function's name,
`batch_infer` -- click rewrites underscores to hyphens in a command's
default name; confirmed directly against a real `core.py --help` on this
cluster, not assumed).

The two gaps shared by both endpoints:
1. **cwd-relative internal lookup.** core.py's own module-level code
   resolves an internal file (rvc/lib/tools/tts_voices.json) relative to
   the process's cwd, not its own __file__, so it fails outright unless
   invoked with Applio's project root as cwd -- a Slurm job doesn't start
   there by default. See config.RUN_CONVERT_SCRIPT/RUN_BATCH_SCRIPT and
   each wrapper's own header comment.
2. **No real support for the input formats callers actually send.**
   Applio's own load_audio_infer()/convert_audio_batch() only ever call
   soundfile.read(), and this deployment's libsndfile (1.2.2) can't decode
   AAC/M4A and other real-world recording formats -- real uploads
   (iPhone-recorded .m4a voice notes) hit this directly. Both wrappers
   decode every input to WAV first (normalize_audio.py's to_wav(), via the
   ffmpeg binary already bundled in Applio's own venv through
   imageio-ffmpeg) before core.py ever sees it -- see config.
   INPUT_EXTENSIONS for what's actually offered to callers, checked here
   (_check_extension/_validate_batch_sources) before a job is ever
   submitted.

Applio's CLI itself already takes exactly the flags each endpoint needs --
verified directly against a real `core.py infer --help`/`core.py --help`
on this cluster (flag names are hyphenated, e.g. --input-path, --pth-path,
--f0-method -- easy to get wrong guessing); both wrappers forward every
flag they don't themselves intercept straight through, unchanged.

See services/ltx/dispatch.py for the fuller rationale behind the
submission plumbing mirrored below (capacity check, partial-path rename
convention).
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from common import config as common_config
from common import job_store, storage
from common.slurm import resolve_partition
from services.rvc import config
from services.rvc.schemas import BatchConvertRequest, BatchSource, ConvertRequest


class DispatchError(RuntimeError):
    """Could not submit the job (bad asset reference, sbatch failure, ...)."""


class CapacityError(RuntimeError):
    """Too many jobs already in flight (see common.config.MAX_INFLIGHT_JOBS)."""


class InputError(RuntimeError):
    """A source recording can't actually be used -- either an unsupported
    file extension (see config.INPUT_EXTENSIONS; checked by both
    build_convert_args and _validate_batch_sources), or, for a batch
    request specifically, an unknown/missing asset_id. Asset lookup
    failures on the *single*-file /convert endpoint still surface as the
    plain KeyError/FileNotFoundError storage.resolve_asset_path itself
    raises, not this -- only its extension check goes through InputError
    there. For a batch request, this collects *every* problem found across
    every source into one message (see _validate_batch_sources), not just
    the first, so a caller can fix a whole batch request in one pass
    instead of one rejection at a time."""


def _check_capacity() -> None:
    if (
        common_config.MAX_INFLIGHT_JOBS is not None
        and job_store.count_inflight_jobs() >= common_config.MAX_INFLIGHT_JOBS
    ):
        raise CapacityError(
            f"At capacity: {common_config.MAX_INFLIGHT_JOBS} job(s) already in flight. Try again shortly."
        )


def _check_extension(path: Path) -> None:
    """Raises InputError if `path`'s extension isn't one this deployment's
    Slurm-side wrappers (run_convert.py / run_batch.py) can actually decode
    -- see config.INPUT_EXTENSIONS. Shared by build_convert_args (single
    file) and _validate_batch_sources (many files) so the two endpoints
    can't drift apart on what's accepted."""
    if path.suffix.lower() not in config.INPUT_EXTENSIONS:
        accepted = ", ".join(sorted(config.INPUT_EXTENSIONS))
        raise InputError(f"Unsupported file extension {path.suffix!r} (accepted: {accepted}): {path.name}")


def build_convert_args(req: ConvertRequest, job_dir: Path, output_path: Path) -> list[str]:
    """Args *after* the interpreter+script path, matching run_convert.py's
    own CLI (--input-path/--work-dir/--output-path, then every `core.py
    infer` flag unchanged -- see that script's own header comment).
    --output-path is already set to `output_path`; _submit_slurm_job
    rewrites it to a partial path before actually submitting, same
    convention every other backend's dispatch module uses."""
    source_path = storage.resolve_asset_path(req.source_audio_asset_id)  # raises KeyError/FileNotFoundError
    _check_extension(source_path)  # raises InputError
    voice = config.VOICE_REGISTRY[req.voice]  # schema validation already guarantees this key exists
    return [
        "--input-path", str(source_path),
        "--work-dir", str(job_dir),
        "--output-path", str(output_path),
        "--pth-path", str(voice["pth_path"]),
        "--index-path", str(voice["index_path"]),
        "--pitch", str(req.pitch),
        "--f0-method", req.f0_method,
        "--index-rate", str(req.index_rate),
        "--protect", str(req.protect),
        "--export-format", req.export_format,
    ]


def _validate_batch_sources(sources: list[BatchSource]) -> list[Path]:
    """Resolves every source's asset_id to a real file and checks its
    extension against config.INPUT_EXTENSIONS, collecting *every*
    problem found (not just the first) into one InputError -- see that
    class's own docstring for why. Returns the resolved paths, in the same
    order as `sources`, but only if every one of them is valid; raises
    before returning anything otherwise, so a partially-valid batch is
    never partially acted on."""
    resolved: list[Path] = []
    problems: list[str] = []
    accepted = ", ".join(sorted(config.INPUT_EXTENSIONS))
    for i, source in enumerate(sources, start=1):
        try:
            path = storage.resolve_asset_path(source.asset_id)
        except KeyError:
            problems.append(f"source #{i} ({source.asset_id}): unknown asset_id")
            continue
        except FileNotFoundError:
            problems.append(f"source #{i} ({source.asset_id}): upload file missing on disk")
            continue
        if path.suffix.lower() not in config.INPUT_EXTENSIONS:
            problems.append(
                f"source #{i} ({path.name}): unsupported extension {path.suffix!r} (accepted: {accepted})"
            )
            continue
        resolved.append(path)
    if problems:
        raise InputError(f"{len(problems)} of {len(sources)} source(s) rejected: " + "; ".join(problems))
    return resolved


def _symlink_inputs(resolved_paths: list[Path], input_dir: Path) -> None:
    """Populates input_dir with one symlink per resolved source path,
    named "<NNN>_<original filename>" (1-based, zero-padded to 3 digits --
    plenty for BATCH_MAX_FILES's default of 50). The numeric prefix means:
    (a) two sources that happen to share an original filename can never
    collide, and (b) the batch's own output filenames (run_batch.py names
    each "<stem>_output.<ext>", stem being this symlink's own name minus
    its extension) stay traceable back to each source's position in the
    request."""
    input_dir.mkdir(parents=True)
    for i, path in enumerate(resolved_paths, start=1):
        (input_dir / f"{i:03d}_{path.name}").symlink_to(path)


def build_batch_convert_args(
    req: BatchConvertRequest, input_dir: Path, work_dir: Path, output_dir: Path, output_path: Path
) -> list[str]:
    """Args *after* the interpreter+script path, matching run_batch.py's
    own CLI exactly (--output-path is read by run_batch.py itself, not
    forwarded to `core.py batch-infer` -- see that script's own header
    comment). --output-path is already set to `output_path`;
    _submit_slurm_job rewrites it to a partial path before actually
    submitting, same convention every other backend's dispatch module
    uses."""
    voice = config.VOICE_REGISTRY[req.voice]  # schema validation already guarantees this key exists
    return [
        "--input-folder", str(input_dir),
        "--work-folder", str(work_dir),
        "--output-folder", str(output_dir),
        "--output-path", str(output_path),
        "--pth-path", str(voice["pth_path"]),
        "--index-path", str(voice["index_path"]),
        "--pitch", str(req.pitch),
        "--f0-method", req.f0_method,
        "--index-rate", str(req.index_rate),
        "--protect", str(req.protect),
        "--export-format", req.export_format,
    ]


def _submit_slurm_job(
    job_id: str,
    job_dir: Path,
    script_argv_prefix: list[str],
    module_args: list[str],
    output_path: Path,
    partition: str,
    *,
    job_name: str,
    time_limit: str,
    extra_sbatch_args: list[str] | None = None,
) -> str:
    """Write args.json + submit via sbatch (non-blocking). Returns the
    Slurm job id.

    `script_argv_prefix` is everything before the flags themselves --
    [RVC_PYTHON, RUN_CONVERT_SCRIPT] for /convert or [RVC_PYTHON,
    RUN_BATCH_SCRIPT] for /batch-convert (both wrapper scripts must be run
    with Applio's own venv interpreter explicitly, not whatever `env
    python3` resolves to on the compute node -- see run_batch.py's own
    header comment for why that distinction matters; run_convert.py's own
    `import normalize_audio` needs the same venv for the same reason).
    `job_name`/`time_limit` let each caller keep its own existing values;
    `extra_sbatch_args` is only non-empty for batch jobs today (see
    dispatch_batch_convert's own --dependency=singleton)."""
    partial_path = job_dir / f"output.partial{output_path.suffix}"
    args = list(module_args)
    idx = args.index("--output-path")
    args[idx + 1] = str(partial_path)

    full_argv = [*script_argv_prefix, *args]
    (job_dir / "args.json").write_text(json.dumps(full_argv))
    (job_dir / "output_path.txt").write_text(str(output_path))

    wrap_argv = ["python3", str(config.RUN_PIPELINE_JOB_SCRIPT), str(job_dir)]
    wrap_str = "HF_HUB_OFFLINE=1 " + shlex.join(wrap_argv)
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        "--gres=gpu:1",
        f"--cpus-per-task={config.CPUS_PER_GPU}",
        f"--mem-per-gpu={config.MEM_PER_GPU}",
        f"--time={time_limit}",
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}{job_name}",
        f"--output={job_dir / 'slurm.log'}",
        *(extra_sbatch_args or []),
        f"--wrap={wrap_str}",
    ]
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DispatchError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip().split(";")[0]  # --parsable prints "<id>" or "<id>;<cluster>"


def dispatch_convert(req: ConvertRequest) -> str:
    """Shared submit path: validate capacity + assets, then create the job
    directory/DB row/Slurm job, in that order, so an invalid request never
    leaves debris behind."""
    _check_capacity()
    partition = resolve_partition(req.partition)
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    output_path = job_dir / f"output.{req.export_format.lower()}"

    # Raises KeyError/FileNotFoundError (unknown/missing asset) or
    # InputError (unsupported extension) here (before anything is created
    # on disk or in the DB) if source_audio_asset_id is invalid.
    module_args = build_convert_args(req, job_dir, output_path)

    job_dir.mkdir(parents=True)
    job_store.create_job(job_id, "rvc:convert", job_dir, req.model_dump(mode="json"), partition)
    try:
        slurm_job_id = _submit_slurm_job(
            job_id,
            job_dir,
            [str(config.RVC_PYTHON), str(config.RUN_CONVERT_SCRIPT)],
            module_args,
            output_path,
            partition,
            job_name=f"rvc-api-{job_id[:8]}",
            time_limit=config.JOB_TIME_LIMIT,
        )
    except DispatchError:
        job_store.mark_failed(job_id, "Could not submit the Slurm job.")
        raise
    job_store.set_slurm_job_id(job_id, slurm_job_id)
    return job_id


def dispatch_batch_convert(req: BatchConvertRequest) -> str:
    """Shared submit path: validate capacity + every source, then create
    the job directory (with every source symlinked in)/DB row/Slurm job,
    in that order, so an invalid request never leaves debris behind.

    Submitted under a fixed job name with --dependency=singleton (see
    config.BATCH_SLURM_JOB_NAME) so at most one batch runs at a time --
    Applio's own convert_audio_batch() writes and then deletes one shared,
    hardcoded pid file in its own project root for the whole batch's
    duration; two batches overlapping would race on that file. Single-file
    /convert jobs never touch that file and aren't part of this
    singleton group."""
    _check_capacity()
    partition = resolve_partition(req.partition)

    # Raises InputError here (before anything is created on disk or in the
    # DB) if any source's asset_id/extension is invalid.
    resolved_paths = _validate_batch_sources(req.sources)

    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    input_dir = job_dir / "inputs"
    work_dir = job_dir / "normalized"
    output_dir = job_dir / "outputs"
    output_path = job_dir / "output.zip"

    job_dir.mkdir(parents=True)
    _symlink_inputs(resolved_paths, input_dir)  # creates input_dir itself

    module_args = build_batch_convert_args(req, input_dir, work_dir, output_dir, output_path)

    job_store.create_job(job_id, "rvc:batch-convert", job_dir, req.model_dump(mode="json"), partition)
    try:
        slurm_job_id = _submit_slurm_job(
            job_id,
            job_dir,
            [str(config.RVC_PYTHON), str(config.RUN_BATCH_SCRIPT)],
            module_args,
            output_path,
            partition,
            job_name=config.BATCH_SLURM_JOB_NAME,
            time_limit=config.BATCH_JOB_TIME_LIMIT,
            extra_sbatch_args=["--dependency=singleton"],
        )
    except DispatchError:
        job_store.mark_failed(job_id, "Could not submit the Slurm job.")
        raise
    job_store.set_slurm_job_id(job_id, slurm_job_id)
    return job_id
