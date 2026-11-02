"""Builds ACE-Step 1.5's invocation and queues it for a warm worker.

A request has no Slurm job of its own: a few long-lived (as many as the
GPU capacity page allows) worker jobs (models/ace_step15/runner/worker.py, with that model's own
environment) each load the models once and serve queued requests one at a
time. common/worker_pool.py starts, feeds and stops them; this module only
writes the job's files (args.json, output_path.txt: the same
partial-path-rename convention common/run_pipeline_job.py enforces for cold
jobs, which models/common/pool_worker.py follows) and puts the request in the
queue.

Text-like values are passed as `--flag=value` single argv entries (and the
request is read from args.json, never through a shell string), so user text
needs no escaping and can safely start with "-" or contain quotes/newlines.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from common import config as common_config
from common import dedup, job_store
from common.audio_inputs import AudioInputError, resolve_audio_asset
from common.slurm import resolve_partition
from common.worker_pool import PoolError, PoolRequest, WorkerPool
from services.ace_step15 import config
from services.ace_step15.schemas import AceStepCoverRequest, AceStepGenerateRequest, AceStepRepaintRequest

# One kind of worker: every endpoint runs on the same loaded models.
POOL_KEY = "default"


class DispatchError(RuntimeError):
    """Could not queue the request (no worker could be started, ...)."""


class CapacityError(RuntimeError):
    """Too many jobs already queued or running for this model (see common.config.MAX_QUEUED_PER_BACKEND)."""


class InputError(RuntimeError):
    """A source recording or its time range can't be used."""


def _check_capacity() -> None:
    if job_store.count_inflight_for(pool.pipelines) >= common_config.MAX_QUEUED_PER_BACKEND:
        raise CapacityError(
            f"This model already has {common_config.MAX_QUEUED_PER_BACKEND} job(s) queued or running. "
            "Try again shortly."
        )


def _source_audio(asset_id: str) -> tuple[Path, float]:
    try:
        return resolve_audio_asset(asset_id, config.MAX_SOURCE_SECONDS)
    except AudioInputError as e:
        raise InputError(str(e)) from e


def build_args(req: AceStepGenerateRequest | AceStepCoverRequest | AceStepRepaintRequest, mode: str, output_path: Path) -> list[str]:
    """Args *after* the interpreter+script path, matching run_generate.py's own CLI.
    --output-path is already set to `output_path`; _write_job_files rewrites it to a partial path."""
    args = [
        "--model-dir", str(config.MODEL_DIR),
        "--mode", mode,
        "--output-path", str(output_path),
        f"--caption={req.caption}",
        f"--lyrics={req.lyrics}",
        f"--vocal-language={req.vocal_language}",
        "--steps", str(req.num_inference_steps),
        "--guidance-scale", str(req.guidance_scale),
        "--seed", str(req.seed),
    ]
    if req.instrumental:
        args.append("--instrumental")
    if isinstance(req, AceStepGenerateRequest):
        if req.duration is not None:
            args += ["--duration", str(req.duration)]
        if req.bpm is not None:
            args += ["--bpm", str(req.bpm)]
        if req.keyscale:
            args.append(f"--keyscale={req.keyscale}")
        if req.time_signature is not None:
            args.append(f"--time-signature={req.time_signature}")
        args.append("--thinking" if req.thinking else "--no-thinking")
    else:
        source_path, source_seconds = _source_audio(req.source_audio_asset_id)
        args += ["--src-audio", str(source_path)]
        if isinstance(req, AceStepCoverRequest):
            args += ["--cover-strength", str(req.cover_strength)]
        else:
            if req.repaint_start >= source_seconds:
                raise InputError(
                    f"repaint_start ({req.repaint_start:g}s) is past the end of the source recording ({source_seconds:.1f}s)."
                )
            end = -1.0 if req.repaint_end is None else min(req.repaint_end, source_seconds)
            args += [f"--repaint-start={req.repaint_start}", f"--repaint-end={end}"]
    return args


def _write_job_files(job_dir: Path, module_args: list[str], output_path: Path) -> None:
    """args.json (the full argv of a cold run of run_generate.py, with --output-path rewritten to the
    partial file) and output_path.txt (the final path), read by the worker."""
    partial_path = job_dir / f"output.partial{output_path.suffix}"
    args = list(module_args)
    idx = args.index("--output-path")
    args[idx + 1] = str(partial_path)

    full_argv = [str(config.ACE_STEP15_PYTHON), str(config.RUN_GENERATE_SCRIPT), *args]
    (job_dir / "args.json").write_text(json.dumps(full_argv))
    (job_dir / "output_path.txt").write_text(str(output_path))


def _worker_sbatch_command(worker_id: str, worker_dir: Path, key: str, partition: str, gpus: int) -> list[str]:
    """The sbatch command for one warm worker."""
    worker_argv = [
        str(config.ACE_STEP15_PYTHON),
        str(config.WORKER_SCRIPT),
        "--worker-dir", str(worker_dir),
        "--model-dir", str(config.MODEL_DIR),
        # A safety net only: the gateway stops an idle worker after IDLE_MINUTES.
        "--idle-exit-seconds", str(int(config.IDLE_MINUTES * 60 * 2)),
    ]
    return [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        f"--gres=gpu:{gpus}",
        f"--cpus-per-task={config.CPUS_PER_GPU}",
        f"--mem-per-gpu={config.MEM_PER_GPU}",
        f"--time={config.WORKER_TIME_LIMIT}",
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}ace-step15-worker-{worker_id[:8]}",
        f"--output={worker_dir / 'worker.log'}",
        f"--wrap=HF_HUB_OFFLINE=1 {shlex.join(worker_argv)}",
    ]


pool = WorkerPool(
    "ace_step15",
    label="ACE-Step 1.5",
    pipelines=("ace-step15:generate", "ace-step15:cover", "ace-step15:repaint"),
    default_max_jobs=4,
    idle_seconds=config.IDLE_MINUTES * 60,
    worker_time_seconds=config.worker_time_limit_seconds(),
    build_sbatch_command=_worker_sbatch_command,
)


def _dispatch(req: AceStepGenerateRequest | AceStepCoverRequest | AceStepRepaintRequest, mode: str) -> dedup.Submission:
    """Validate the source audio, reuse an identical earlier request if there is one, else create the job directory/DB row and queue the request,
    in that order, so an invalid request never leaves debris behind."""
    partition = resolve_partition(req.partition)
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    output_path = job_dir / "output.flac"

    # Raises KeyError/FileNotFoundError (unknown/missing asset) or InputError (unusable audio or
    # range) here, before anything is created on disk or in the DB.
    module_args = build_args(req, mode, output_path)

    pipeline = f"ace-step15:{mode}"
    dedup_key = None if req.incognito else dedup.content_key(pipeline, req.model_dump(mode="json"), config.RESULT_VERSION)
    with dedup.lock:
        earlier = dedup.find_reusable(dedup_key)
        if earlier is not None:
            return dedup.reuse(earlier, req.priority)
        _check_capacity()
        job_dir.mkdir(parents=True)
        job_store.create_job(job_id, pipeline, job_dir, req.model_dump(mode="json"), partition, incognito=req.incognito, dedup_key=dedup_key)
    try:
        _write_job_files(job_dir, module_args, output_path)
        pool.enqueue(
            PoolRequest(
                job_id=job_id,
                key=POOL_KEY,
                partition=partition,
                gpus=1,
                limit_seconds=config.job_time_limit_seconds(),
            ),
            job_dir,
        )
    except (OSError, PoolError) as e:
        job_store.mark_failed(job_id, "Could not queue the request.")
        raise DispatchError(str(e)) from e
    return dedup.Submission(job_id, "queued")


def dispatch_generate(req: AceStepGenerateRequest) -> dedup.Submission:
    return _dispatch(req, "generate")


def dispatch_cover(req: AceStepCoverRequest) -> dedup.Submission:
    return _dispatch(req, "cover")


def dispatch_repaint(req: AceStepRepaintRequest) -> dedup.Submission:
    return _dispatch(req, "repaint")
