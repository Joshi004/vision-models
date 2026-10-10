"""Builds Parakeet's transcription invocation and submits it as a Slurm
job. Unlike services/rvc/dispatch.py (which has two recipes sharing a
parameterized _submit_slurm_job), Parakeet has exactly one recipe, so the
submit helper below is not parameterized by job_name/time_limit -- there is
only ever one call site.

The actual work happens in run_transcribe.sh (ModelService_Parakeet/), a
thin wrapper that sets a CUDA-library-path env fix before exec'ing
run_transcribe.py -- see that script's own header comment for the full
story (ffmpeg decode, local .nemo checkpoint load, chunking past 10
minutes of audio).

See services/ltx/dispatch.py / services/rvc/dispatch.py for the fuller
rationale behind the submission plumbing mirrored below (capacity check,
partial-path rename convention).
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from common import config as common_config
from common import job_store, storage
from common.slurm import resolve_partition
from services.parakeet import config
from services.parakeet.schemas import TranscribeRequest


class DispatchError(RuntimeError):
    """Could not submit the job (sbatch failure)."""


class CapacityError(RuntimeError):
    """Too many jobs already in flight (see common.config.MAX_INFLIGHT_JOBS)."""


class InputError(RuntimeError):
    """The referenced asset isn't a file this deployment can actually
    transcribe -- an unsupported file extension (see
    config.INPUT_EXTENSIONS)."""


def _check_capacity() -> None:
    if (
        common_config.MAX_INFLIGHT_JOBS is not None
        and job_store.count_inflight_jobs() >= common_config.MAX_INFLIGHT_JOBS
    ):
        raise CapacityError(
            f"At capacity: {common_config.MAX_INFLIGHT_JOBS} job(s) already in flight. Try again shortly."
        )


def _check_extension(path: Path) -> None:
    """Raises InputError if `path`'s extension isn't one run_transcribe.py's
    own ffmpeg decode step can actually read -- see config.INPUT_EXTENSIONS."""
    if path.suffix.lower() not in config.INPUT_EXTENSIONS:
        accepted = ", ".join(sorted(config.INPUT_EXTENSIONS))
        raise InputError(f"Unsupported file extension {path.suffix!r} (accepted: {accepted}): {path.name}")


def build_transcribe_args(req: TranscribeRequest, job_dir: Path, output_path: Path) -> list[str]:
    """Args *after* run_transcribe.sh's own path, matching its CLI exactly
    (--input-path/--work-dir/--model-path/--output-path -- see that
    script's own header comment). --output-path is already set to
    `output_path`; _submit_slurm_job rewrites it to a partial path before
    actually submitting, same convention every other backend's dispatch
    module uses."""
    source_path = storage.resolve_asset_path(req.audio_asset_id)  # raises KeyError/FileNotFoundError
    _check_extension(source_path)  # raises InputError
    return [
        "--input-path", str(source_path),
        "--work-dir", str(job_dir),
        "--model-path", str(config.MODEL_PATH),
        "--output-path", str(output_path),
    ]


def _submit_slurm_job(job_id: str, job_dir: Path, module_args: list[str], output_path: Path, partition: str) -> str:
    """Write args.json + submit via sbatch (non-blocking). Returns the
    Slurm job id. See services/rvc/dispatch.py's own _submit_slurm_job for
    the fuller rationale behind this shape (partial-path rewrite,
    HF_HUB_OFFLINE, output redirection)."""
    partial_path = job_dir / f"output.partial{output_path.suffix}"
    args = list(module_args)
    idx = args.index("--output-path")
    args[idx + 1] = str(partial_path)

    # argv[0] is run_transcribe.sh itself (executable, with its own
    # shebang) -- not a [python, script] pair like RVC/LTX use, since this
    # wrapper needs to set an env var before python even starts. See that
    # script's own header comment.
    full_argv = [str(config.RUN_TRANSCRIBE_SCRIPT), *args]
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
        f"--time={config.JOB_TIME_LIMIT}",
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}parakeet-api-{job_id[:8]}",
        f"--output={job_dir / 'slurm.log'}",
        f"--wrap={wrap_str}",
    ]
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DispatchError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip().split(";")[0]  # --parsable prints "<id>" or "<id>;<cluster>"


def dispatch_transcribe(req: TranscribeRequest) -> str:
    """Shared submit path: validate capacity + asset, then create the job
    directory/DB row/Slurm job, in that order, so an invalid request never
    leaves debris behind."""
    _check_capacity()
    partition = resolve_partition(req.partition)
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    output_path = job_dir / "output.json"

    # Raises KeyError/FileNotFoundError (unknown/missing asset) or
    # InputError (unsupported extension) here (before anything is created
    # on disk or in the DB) if audio_asset_id is invalid.
    module_args = build_transcribe_args(req, job_dir, output_path)

    job_dir.mkdir(parents=True)
    job_store.create_job(job_id, "parakeet:transcribe", job_dir, req.model_dump(mode="json"), partition)
    try:
        slurm_job_id = _submit_slurm_job(job_id, job_dir, module_args, output_path, partition)
    except DispatchError:
        job_store.mark_failed(job_id, "Could not submit the Slurm job.")
        raise
    job_store.set_slurm_job_id(job_id, slurm_job_id)
    return job_id
