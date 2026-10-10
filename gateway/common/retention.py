"""What this server keeps on disk, and when it deletes it.

Everything here is a no-op when config.PRESERVE_JOB_DATA is true (the old
behavior: keep everything until config.JOB_RETENTION_DAYS pass or the job is
purged). When it is false (the default):

  * Slurm's own log is not written at all (slurm_log_path -> /dev/null).
    run.log still exists while a job is queued/running -- the live-progress
    reader and the failure reason need it -- and is deleted when the job ends.
  * When a job succeeds, everything in its folder except the output file is
    deleted, and so are its uploaded inputs (unless another queued/running
    job still references them).
  * When a job fails or is cancelled, its folder and inputs are deleted; only
    a one-line error is kept.
  * When a result has been downloaded completely (server.py's job_result), the
    output and the folder are deleted.
  * Whenever a job's data is removed, its database row is scrubbed to a
    minimal record (job_store.scrub_job): status, pipeline and timings stay;
    the prompt/settings, asset ids and output path are gone.
  * Uploads nothing references are deleted after
    config.UNUSED_UPLOAD_TTL_MINUTES.

sweep() re-applies these rules to everything finished, so a crash between a
job ending and its cleanup, a job whose submission failed, and data that
already existed when the flag was switched off are all cleaned up too.

Public functions never raise: a cleanup problem is logged and retried by the
next sweep, and must not break the poller or a download.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common import config, job_store, purge, storage

logger = logging.getLogger("model-api.retention")

_LOG_TAIL_MARKER = "--- last lines of run.log ---"
_MAX_ERROR_CHARS = 500


def slurm_log_path(job_dir: Path) -> str:
    """Where `sbatch --output=` should write: the job's slurm.log when
    preserving, otherwise /dev/null so no Slurm log is ever stored."""
    return str(job_dir / "slurm.log") if config.PRESERVE_JOB_DATA else "/dev/null"


def short_error(error: str | None) -> str | None:
    """Reduce a failure message (reason + the last 40 lines of run.log) to its
    reason line plus the last non-empty log line, e.g. 'Exit code: 1 --
    torch.OutOfMemoryError: CUDA out of memory.'. Already-short messages are
    returned unchanged."""
    if not error:
        return error
    head, _, tail = error.partition(_LOG_TAIL_MARKER)
    reason = head.strip().splitlines()[0] if head.strip() else "Job failed."
    log_lines = [line.strip() for line in tail.splitlines() if line.strip()]
    text = f"{reason} -- {log_lines[-1]}" if log_lines else reason
    return text[:_MAX_ERROR_CHARS]


def _job_dir(row) -> Path | None:
    """The job's folder, or None if it is missing or not where it must be
    (same safety check as purge.py: never delete outside JOBS_DIR/<job_id>)."""
    job_dir = Path(row["job_dir"])
    if job_dir.resolve() != (config.JOBS_DIR / row["job_id"]).resolve():
        logger.error("Refusing to clean unexpected job directory: %s", job_dir)
        return None
    return job_dir if job_dir.exists() else None


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _delete_unreferenced_inputs(row) -> None:
    """Delete this job's uploaded inputs, except any that another queued or
    running job still needs."""
    for asset_id in sorted(purge.asset_ids_in(json.loads(row["request_json"]))):
        if not job_store.asset_in_flight(asset_id):
            storage.delete_upload(asset_id)


def on_job_succeeded(job_id: str) -> None:
    """Keep only the output file; delete the inputs."""
    if config.PRESERVE_JOB_DATA:
        return
    try:
        row = job_store.get_job(job_id)
        if row is None or job_store.is_scrubbed(row):
            return
        job_dir = _job_dir(row)
        if job_dir is not None:
            keep = Path(row["output_path"]).resolve() if row["output_path"] else None
            for entry in job_dir.iterdir():
                if keep is None or entry.resolve() != keep:
                    _remove(entry)
        _delete_unreferenced_inputs(row)
    except Exception:
        logger.exception("retention: cleanup after success failed for job %s", job_id)


def on_job_failed(job_id: str) -> None:
    """Delete the whole folder and the inputs; keep a one-line error."""
    if config.PRESERVE_JOB_DATA:
        return
    try:
        row = job_store.get_job(job_id)
        if row is None or job_store.is_scrubbed(row):
            return
        job_dir = _job_dir(row)
        if job_dir is not None:
            shutil.rmtree(job_dir)
        _delete_unreferenced_inputs(row)
        job_store.scrub_job(job_id, error=short_error(row["error"]))
    except Exception:
        logger.exception("retention: cleanup after failure failed for job %s", job_id)


def on_result_downloaded(job_id: str) -> None:
    """The result was sent in full: delete the output and the folder, and scrub
    the record (job_store.scrub_job also stamps downloaded_at)."""
    if config.PRESERVE_JOB_DATA:
        return
    try:
        row = job_store.get_job(job_id)
        if row is None:
            return
        # Record the download first: if removing the folder fails below (e.g.
        # another download of the same file is still open on the shared
        # filesystem), the job must still read as downloaded. Any leftover is
        # removed with the job's row by the retention-days sweep.
        job_store.scrub_job(job_id, downloaded=True)
        job_dir = _job_dir(row)
        if job_dir is not None:
            shutil.rmtree(job_dir)
        _delete_unreferenced_inputs(row)  # normally already gone; covers a retry after a partial cleanup
    except Exception:
        logger.exception("retention: cleanup after download failed for job %s", job_id)


def sweep_unused_uploads() -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=config.UNUSED_UPLOAD_TTL_MINUTES)).isoformat()
    for row in job_store.list_uploads_before(cutoff):
        try:
            if not job_store.asset_in_flight(row["asset_id"]):
                storage.delete_upload(row["asset_id"])
                logger.info("Deleted unused upload %s", row["asset_id"])
        except Exception:
            logger.exception("retention: could not delete upload %s", row["asset_id"])


def sweep() -> None:
    """Apply every rule above to everything that is already finished."""
    if config.PRESERVE_JOB_DATA:
        return
    for row in job_store.list_finished_unscrubbed():
        if row["status"] == "succeeded":
            on_job_succeeded(row["job_id"])
        else:
            on_job_failed(row["job_id"])
    sweep_unused_uploads()
