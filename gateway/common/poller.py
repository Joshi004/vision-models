"""Background task: polls in-flight jobs and updates their status.

Shared by every backend -- nothing here is LTX-specific (or specific to any
other backend); it only knows about the job_dir/output_path.txt/.failed
conventions that any backend's dispatch module follows.

Runs as an asyncio task inside the FastAPI process (started in server.py's
lifespan handler). For each in-flight job, checked in this order:

  1. Does the final output file exist? -> succeeded.
  2. Did the job's wrapper script leave a .failed marker? -> failed (with
     the reason from error.txt).
  3. Otherwise, ask Slurm directly via `sacct` -- catches states the
     wrapper script never got to record (the node itself failing, admin
     cancellation, or simply still queued/running).

Also runs a periodic sweep that deletes old finished jobs' directories,
per config.JOB_RETENTION_DAYS.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common import config, job_store

logger = logging.getLogger("model-api.poller")

# sacct States (see `man sacct`) that mean "still going" -- anything else
# observed is terminal (COMPLETED, FAILED, CANCELLED, TIMEOUT, PREEMPTED,
# NODE_FAIL, OUT_OF_MEMORY, ...).
_ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING"}

_CLEANUP_EVERY_N_POLLS = 120  # e.g. every ~10 minutes at the default 5s poll interval

# The project root (and every job dir under it) lives on a shared NFS mount
# (confirmed: `df /` on this cluster points at an NFS server). A file
# written/renamed by the *compute* node that ran the job is not always
# immediately visible over NFS to *this* process running on the login node
# -- observed live: sacct reported the job COMPLETED and the output file's
# own mtime showed it already existed, several seconds before this process
# could see it with a plain Path.exists() check. So: once Slurm considers a
# job terminal but we can't yet see an output file or a .failed marker,
# don't declare failure immediately -- keep re-checking for a grace period
# first, since the file may simply not have propagated to this node's view
# of the filesystem yet.
_OUTPUT_VISIBILITY_GRACE_SECONDS = 90.0
_pending_since: dict[str, float] = {}  # job_id -> monotonic time first seen "terminal but unresolved"


def _sacct_state(slurm_job_id: str) -> str | None:
    """The Slurm State for exactly this job id (not its .batch/.extern steps),
    or None if sacct couldn't answer right now (try again next poll)."""
    try:
        result = subprocess.run(
            ["sacct", "-j", slurm_job_id, "-n", "-P", "-o", "JobID,State"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("sacct check failed for job %s: %s", slurm_job_id, e)
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) == 2 and parts[0] == slurm_job_id:
            return parts[1].split()[0]  # e.g. "CANCELLED by 1234" -> "CANCELLED"
    return None


def _expected_output_path(job_dir: Path) -> Path | None:
    marker = job_dir / "output_path.txt"
    if not marker.exists():
        return None
    return Path(marker.read_text().strip())


def _read_error(job_dir: Path) -> str:
    error_file = job_dir / "error.txt"
    if error_file.exists():
        return error_file.read_text()[:4000]
    return "Job failed (no further detail was recorded)."


def _check_one(row: sqlite3.Row) -> None:
    job_id = row["job_id"]
    job_dir = Path(row["job_dir"])

    expected_output = _expected_output_path(job_dir)
    if expected_output is not None and expected_output.exists():
        job_store.mark_succeeded(job_id, expected_output)
        _pending_since.pop(job_id, None)
        return

    if (job_dir / ".failed").exists():
        job_store.mark_failed(job_id, _read_error(job_dir))
        _pending_since.pop(job_id, None)
        return

    slurm_job_id = row["slurm_job_id"]
    if not slurm_job_id:
        return  # sbatch hasn't returned yet -- shouldn't normally happen

    state = _sacct_state(slurm_job_id)
    if state is None:
        return  # sacct not ready / transient hiccup -- retry next poll
    if state in _ACTIVE_STATES:
        if state == "RUNNING":
            job_store.mark_running(job_id)
        _pending_since.pop(job_id, None)  # no longer terminal, if it briefly looked like it was
        return

    # Slurm considers the job terminal, but neither an output file nor a
    # .failed marker is visible yet -- give the shared filesystem a grace
    # period to catch up before concluding real failure (see module docstring).
    now = time.monotonic()
    first_seen = _pending_since.setdefault(job_id, now)
    if now - first_seen < _OUTPUT_VISIBILITY_GRACE_SECONDS:
        return  # keep waiting, re-check next poll

    _pending_since.pop(job_id, None)
    if state == "COMPLETED":
        job_store.mark_failed(job_id, "Slurm job completed but no output file ever appeared.")
        return
    job_store.mark_failed(job_id, f"Slurm job ended with state: {state}")


def _cleanup_old_jobs() -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=config.JOB_RETENTION_DAYS)).isoformat()
    for row in job_store.list_finished_before(cutoff):
        job_dir = Path(row["job_dir"])
        shutil.rmtree(job_dir, ignore_errors=True)
        job_store.delete_job_row(row["job_id"])
        logger.info("Cleaned up expired job %s (%s)", row["job_id"], job_dir)


async def poll_forever() -> None:
    iteration = 0
    while True:
        try:
            for row in job_store.list_inflight_jobs():
                _check_one(row)
            iteration += 1
            if iteration % _CLEANUP_EVERY_N_POLLS == 0:
                _cleanup_old_jobs()
        except Exception:
            logger.exception("poller iteration failed")
        await asyncio.sleep(config.POLL_INTERVAL_SECONDS)
