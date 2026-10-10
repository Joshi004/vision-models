"""Builds Wan-Animate v1's (Wan2.2-Animate-14B, replace mode) invocation and
submits it as a Slurm job.

Unlike LTX-2.3 (a Python module run directly by its own venv), Wan-Animate's
"pipeline" is this project's own wrapper script -- see
ModelService_Wan-Animate-2/v1/run_replace.py, which itself shells out to the
official Wan2.2 repo's preprocess_data.py and generate.py in sequence, then
re-muxes the source video's audio. See that script's own docstring for why
it exists as a wrapper rather than calling generate.py directly, and
services/ltx/dispatch.py for the fuller rationale behind the submission
plumbing mirrored below (capacity check, partial-path rename convention).
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from common import config as common_config
from common import job_store, storage
from common.slurm import resolve_partition
from services.wan_animate import config
from services.wan_animate.schemas import ReplaceRequest


class DispatchError(RuntimeError):
    """Could not submit the job (bad asset reference, sbatch failure, ...)."""


class CapacityError(RuntimeError):
    """Too many jobs already in flight (see common.config.MAX_INFLIGHT_JOBS)."""


def _check_capacity() -> None:
    if (
        common_config.MAX_INFLIGHT_JOBS is not None
        and job_store.count_inflight_jobs() >= common_config.MAX_INFLIGHT_JOBS
    ):
        raise CapacityError(
            f"At capacity: {common_config.MAX_INFLIGHT_JOBS} job(s) already in flight. Try again shortly."
        )


def build_replace_args(req: ReplaceRequest, output_path: Path) -> list[str]:
    """Args *after* the interpreter+script path (i.e. starting with
    "--video"), matching run_replace.py's own CLI exactly. --output-path is
    already set to `output_path`; _submit_slurm_job rewrites it to a
    partial path before actually submitting, same convention every other
    backend's dispatch module uses."""
    video_path = storage.resolve_asset_path(req.video_asset_id)  # raises KeyError/FileNotFoundError
    image_path = storage.resolve_asset_path(req.image_asset_id)
    args = [
        "--video", str(video_path),
        "--image", str(image_path),
        "--work-dir", str(output_path.parent),
        "--output-path", str(output_path),
        "--seed", str(req.seed),
    ]
    if not req.use_relighting_lora:
        args.append("--no-relighting-lora")
    if not req.keep_audio:
        args.append("--no-audio")
    return args


def _submit_slurm_job(job_id: str, job_dir: Path, script_args: list[str], output_path: Path, partition: str) -> str:
    """Write args.json + submit via sbatch (non-blocking). Returns the Slurm job id."""
    partial_path = job_dir / f"output.partial{output_path.suffix}"
    args = list(script_args)
    idx = args.index("--output-path")
    args[idx + 1] = str(partial_path)

    full_argv = [str(config.WAN_ANIMATE_PYTHON), str(config.RUN_REPLACE_SCRIPT), *args]
    (job_dir / "args.json").write_text(json.dumps(full_argv))
    (job_dir / "output_path.txt").write_text(str(output_path))

    job_name = f"wananimate-api-{job_id[:8]}"
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
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}{job_name}",
        f"--output={job_dir / 'slurm.log'}",
        f"--wrap={wrap_str}",
    ]
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DispatchError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip().split(";")[0]  # --parsable prints "<id>" or "<id>;<cluster>"


def dispatch_replace(req: ReplaceRequest) -> str:
    """Shared submit path: validate capacity + assets, then create the job
    directory/DB row/Slurm job, in that order, so an invalid request never
    leaves debris behind."""
    _check_capacity()
    partition = resolve_partition(req.partition)
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    output_path = job_dir / "output.mp4"

    # Raises KeyError/FileNotFoundError here (before anything is created on
    # disk or in the DB) if an asset_id is invalid.
    script_args = build_replace_args(req, output_path)

    job_dir.mkdir(parents=True)
    job_store.create_job(job_id, "wan-animate:replace", job_dir, req.model_dump(mode="json"), partition)
    try:
        slurm_job_id = _submit_slurm_job(job_id, job_dir, script_args, output_path, partition)
    except DispatchError:
        job_store.mark_failed(job_id, "Could not submit the Slurm job.")
        raise
    job_store.set_slurm_job_id(job_id, slurm_job_id)
    return job_id
