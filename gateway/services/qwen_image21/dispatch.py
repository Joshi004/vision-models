"""Builds Qwen-Image-2.1's invocation and submits it as a Slurm job.

The job runs models/qwen_image21/runner/run_generate.py (with that model's own
environment). See services/breeze_tts/dispatch.py for the fuller rationale
behind the submission plumbing mirrored here (capacity check, partial-path
rename convention enforced by common/run_pipeline_job.py).

Text-like values are passed as `--flag=value` single argv entries (and the
job is launched from args.json, never through a shell string), so user text
needs no escaping and can safely start with "-" or contain quotes/newlines.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from common import config as common_config
from common import job_store, retention, storage
from common.slurm import resolve_partition
from services.qwen_image21 import config
from services.qwen_image21.schemas import QwenImageEditRequest, QwenImageGenerateRequest


class DispatchError(RuntimeError):
    """Could not submit the job (sbatch failure, ...)."""


class CapacityError(RuntimeError):
    """Too many jobs already in flight (see common.config.MAX_INFLIGHT_JOBS)."""


class InputError(RuntimeError):
    """A reference image can't be used (unsupported file extension)."""


def _check_capacity() -> None:
    if (
        common_config.MAX_INFLIGHT_JOBS is not None
        and job_store.count_inflight_jobs() >= common_config.MAX_INFLIGHT_JOBS
    ):
        raise CapacityError(
            f"At capacity: {common_config.MAX_INFLIGHT_JOBS} job(s) already in flight. Try again shortly."
        )


def _check_image_extension(path: Path) -> None:
    if path.suffix.lower() not in config.REFERENCE_IMAGE_EXTENSIONS:
        accepted = ", ".join(sorted(config.REFERENCE_IMAGE_EXTENSIONS))
        raise InputError(f"Unsupported file extension {path.suffix!r} (accepted: {accepted}): {path.name}")


def build_args(req: QwenImageGenerateRequest | QwenImageEditRequest, output_path: Path) -> list[str]:
    """Args *after* the interpreter+script path, matching run_generate.py's
    own CLI. --output-path is already set to `output_path`; _submit_slurm_job
    rewrites it to a partial path before actually submitting."""
    args = [
        "--model-dir", str(config.MODEL_DIR_BY_MODE[req.mode]),
        "--mode", req.mode,
        "--output-path", str(output_path),
        f"--prompt={req.prompt}",
        "--seed", str(req.seed),
    ]
    if req.mode == "quality":
        args += ["--steps", str(req.num_inference_steps or config.DEFAULT_QUALITY_STEPS)]
    if req.aspect_ratio is not None:
        width, height = config.ASPECT_RATIOS[req.aspect_ratio]
        args += ["--width", str(width), "--height", str(height)]
    if isinstance(req, QwenImageEditRequest):
        for asset_id in req.reference_image_asset_ids:
            image_path = storage.resolve_asset_path(asset_id)  # raises KeyError/FileNotFoundError
            _check_image_extension(image_path)  # raises InputError
            args += ["--image", str(image_path)]
    return args


def _submit_slurm_job(job_id: str, job_dir: Path, module_args: list[str], output_path: Path, partition: str) -> str:
    """Write args.json + submit via sbatch (non-blocking). Returns the Slurm
    job id."""
    partial_path = job_dir / f"output.partial{output_path.suffix}"
    args = list(module_args)
    idx = args.index("--output-path")
    args[idx + 1] = str(partial_path)

    full_argv = [str(config.QWEN_IMAGE21_PYTHON), str(config.RUN_GENERATE_SCRIPT), *args]
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
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}qwen-image21-api-{job_id[:8]}",
        f"--output={retention.slurm_log_path(job_dir)}",
        f"--wrap={wrap_str}",
    ]
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DispatchError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip().split(";")[0]  # --parsable prints "<id>" or "<id>;<cluster>"


def _dispatch(req: QwenImageGenerateRequest | QwenImageEditRequest, pipeline: str) -> str:
    """Validate capacity + reference assets, then create the job directory/DB
    row/Slurm job, in that order, so an invalid request never leaves debris
    behind."""
    _check_capacity()
    partition = resolve_partition(req.partition)
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    output_path = job_dir / "output.png"

    # Raises KeyError/FileNotFoundError (unknown/missing asset) or InputError
    # (unsupported extension) here, before anything is created on disk or in
    # the DB.
    module_args = build_args(req, output_path)

    job_dir.mkdir(parents=True)
    job_store.create_job(job_id, pipeline, job_dir, req.model_dump(mode="json"), partition)
    try:
        slurm_job_id = _submit_slurm_job(job_id, job_dir, module_args, output_path, partition)
    except DispatchError:
        job_store.mark_failed(job_id, "Could not submit the Slurm job.")
        raise
    job_store.set_slurm_job_id(job_id, slurm_job_id)
    return job_id


def dispatch_generate(req: QwenImageGenerateRequest) -> str:
    return _dispatch(req, "qwen-image21:generate")


def dispatch_edit(req: QwenImageEditRequest) -> str:
    return _dispatch(req, "qwen-image21:edit")
