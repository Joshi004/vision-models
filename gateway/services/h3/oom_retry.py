"""Runs an H3 job again on 8 GPUs when its 4-GPU run ran out of GPU memory.

A 4-GPU job gives the text/vision encoder two GPUs; an 8-GPU job gives it four (and the denoising
sequence is split 8 ways instead of 4). So a request that ran out of memory in one of those two
stages may fit on 8 GPUs. The poller calls `retry_on_more_gpus` when it finds a job's `.failed`
marker, before it marks the job failed (common/poller.py).

Only these failures are retried (read from the runner's metrics files, see models/h3/runner/run_h3.py):
  * the encoder: metrics.condition.json, failed_stage "encode", with out-of-memory numbers;
  * the denoising: failed_stage "denoise" with out-of-memory numbers, in a denoise worker's file
    (metrics.worker<r>.json) or in rank 0's (metrics.run.json, when the failure was its own).
Reference encoding, the decode and the weight copy run on one GPU next to the whole transformer, so
more GPUs do not help them: those fail as before.

A job is retried at most once: it needs 4 GPUs in jobs.db, and the retry sets 8.

What a retry does, in this order:
  1. jobs.db: the row goes back to "queued" with 8 GPUs and the 8-GPU estimate (job_store.requeue_on_more_gpus;
     it fails, and nothing more happens, if the job is no longer running, e.g. it was cancelled).
  2. The usage log gets a line for the 4-GPU attempt (outcome "oom_retry"), so the numbers of both attempts are kept.
  3. run.log, the metrics files, error.txt and sizing.json of the first attempt move into attempt-4gpu/ in the job
     folder (the second attempt would overwrite them); args.json gets --gpus 8, sizing.json the 8-GPU sizing with a
     "retried_after_oom" note, and the .failed marker is removed.
  4. The pool queues the request again, first in line, for an 8-GPU worker.
If anything after step 1 goes wrong the job is failed with its original error, as it would have been without a retry.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from common import job_store, retention
from common.worker_pool import PoolRequest
from services.h3 import config, sizing, usage_log

logger = logging.getLogger("model-api.h3-oom-retry")

FROM_GPUS = 4
TO_GPUS = 8
ATTEMPT_DIR = f"attempt-{FROM_GPUS}gpu"
_MAX_ERROR_CHARS = 4000  # the same length the poller keeps of error.txt


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def oom_stage(job_dir: Path, gpus: int) -> dict | None:
    """Which stage ran out of memory, if it is one that more GPUs help: {"role": the metrics file, "stage": the
    runner's stage name}. None for any other failure (or if the metrics files cannot be read)."""
    condition = _read_json(job_dir / "metrics.condition.json")
    if condition and condition.get("oom") and condition.get("failed_stage") == "encode":
        return {"role": "condition", "stage": "encode"}
    for rank in range(1, gpus):
        worker = _read_json(job_dir / f"metrics.worker{rank}.json")
        if worker and worker.get("oom") and worker.get("failed_stage") == "denoise":
            return {"role": f"worker{rank}", "stage": "denoise"}
    run = _read_json(job_dir / "metrics.run.json")
    if run and run.get("oom") and run.get("failed_stage") == "denoise" and run.get("failed_in") != "worker":
        return {"role": "run", "stage": "denoise"}
    return None


def _read_error(job_dir: Path) -> str:
    try:
        return (job_dir / "error.txt").read_text(errors="replace")[:_MAX_ERROR_CHARS]
    except OSError:
        return "Job failed."


def _log_first_attempt(row, failure: dict) -> None:
    """The usage-log line of the 4-GPU attempt. `row` is the job row as it was before the retry (4 GPUs, and the
    4-GPU metrics files are still in place). Best effort."""
    try:
        record = usage_log.build_record(row)
        finished_at = datetime.now(timezone.utc).isoformat()
        record["outcome"] = "oom_retry"
        record["finished_at"] = finished_at
        record["run_s"] = usage_log._seconds_between(row["started_at"], finished_at)
        record["oom_retry"] = {**failure, "from_gpus": FROM_GPUS, "to_gpus": TO_GPUS}
        usage_log.append(record)
    except Exception:
        logger.exception("oom retry: could not write the usage-log line of the first attempt of job %s", row["job_id"])


def _move_first_attempt_files(job_dir: Path, gpus: int) -> None:
    """Moves what the first run left (the second run overwrites it) into attempt-4gpu/. Files are named, not listed:
    the folder is on a shared filesystem where a listing can lag."""
    attempt_dir = job_dir / ATTEMPT_DIR
    attempt_dir.mkdir(exist_ok=True)
    names = ["run.log", "error.txt", "metrics.run.json", "metrics.condition.json"]
    names += [f"metrics.worker{rank}.json" for rank in range(1, gpus)]
    for name in names:
        source = job_dir / name
        if source.exists():
            shutil.move(str(source), str(attempt_dir / name))
    shutil.copy2(job_dir / "sizing.json", attempt_dir / "sizing.json")


def _set_gpus_in_args(job_dir: Path) -> None:
    args_path = job_dir / "args.json"
    argv = json.loads(args_path.read_text())
    argv[argv.index("--gpus") + 1] = str(TO_GPUS)
    args_path.write_text(json.dumps(argv))


def _fail_as_before(job_id: str, error: str, why: str) -> None:
    logger.warning("oom retry: job %s could not be queued again on %d GPUs (%s); failing it.", job_id, TO_GPUS, why)
    job_store.mark_failed(job_id, error)
    retention.on_job_failed(job_id)


def retry_on_more_gpus(row) -> bool:
    """True if this failed H3 job was taken over for a retry on 8 GPUs (the poller then leaves it alone); False if
    it is not eligible and must be failed as usual. Never raises."""
    try:
        return _retry(row)
    except Exception:
        # Only reachable before the job was handed over (everything after that is handled in _retry), so the
        # poller can fail the job as usual.
        logger.exception("oom retry: unexpected error for job %s", row["job_id"])
        return False


def _retry(row) -> bool:
    if not config.OOM_RETRY or not row["pipeline"].startswith("h3:") or row["gpus"] != FROM_GPUS:
        return False
    job_id = row["job_id"]
    job_dir = Path(row["job_dir"])
    failure = oom_stage(job_dir, FROM_GPUS)
    if failure is None:
        return False
    old_sizing = _read_json(job_dir / "sizing.json")
    old_request = _read_json(job_dir / "pool_request.json")
    if old_sizing is None or old_request is None:
        return False

    from services.h3 import dispatch  # here, not at the top: dispatch builds the whole worker pool

    error = _read_error(job_dir)
    new_sizing = sizing.for_gpus(old_sizing, TO_GPUS)
    new_sizing["retried_after_oom"] = {"from_gpus": FROM_GPUS, **failure}
    request = PoolRequest(
        job_id=job_id,
        key=dispatch.worker_key_with_gpus(old_request["key"], TO_GPUS),
        partition=old_request["partition"],
        gpus=TO_GPUS,
        limit_seconds=int(new_sizing["time_limit_seconds"]),
    )

    if not job_store.requeue_on_more_gpus(job_id, TO_GPUS, new_sizing["estimated_run_seconds"]):
        return False
    logger.info("oom retry: job %s ran out of memory in %s (%s); queued again on %d GPUs.", job_id, failure["stage"], failure["role"], TO_GPUS)
    _log_first_attempt(row, failure)
    try:
        _move_first_attempt_files(job_dir, FROM_GPUS)
        _set_gpus_in_args(job_dir)
        (job_dir / "sizing.json").write_text(json.dumps(new_sizing, indent=1))
        (job_dir / ".failed").unlink(missing_ok=True)
        dispatch.pool.requeue_on_more_gpus(request, job_dir)
    except Exception as problem:  # the job is already queued in jobs.db: it must be failed here or it would hang
        _fail_as_before(job_id, f"{error}\n\n(A retry on {TO_GPUS} GPUs could not be started: {type(problem).__name__}.)", str(problem))
    return True
