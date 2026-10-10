"""Permanently deletes a finished job: its own directory, its database row,
and any uploaded input it referenced that no other remaining job still
uses.

Distinct from the routine cleanup common/poller.py already does (which only
ever removes a job's directory + row once it's older than
config.JOB_RETENTION_DAYS, and never touches uploads at all) -- this is the
user-initiated "delete this specific job right now" path behind
DELETE /v1/jobs/{job_id}/purge in server.py. That endpoint is responsible for
checking the job has actually finished (not queued/running) before calling
into this module at all; everything here assumes that's already true.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from common import config, job_store, storage


class PurgeError(RuntimeError):
    """Base class for anything that stops a job from being purged. Always
    raised before or during deletion, never after -- server.py maps each
    subclass below to its own HTTP status."""


class PurgeSafetyError(PurgeError):
    """The job's own directory didn't resolve to where it should -- refuses
    to delete anything at all rather than risk removing the wrong path.
    Should never actually happen: job_dir is always JOBS_DIR / job_id,
    written by this same codebase's own dispatch modules, never by a
    caller of this API."""


class PurgeRetryableError(PurgeError):
    """The job's directory couldn't be fully removed just now (e.g. an NFS
    hiccup, or a file still briefly open on some node). The job's database
    row is deliberately left in place when this happens, so the job stays
    visible in job history and the delete can simply be retried, rather
    than silently losing track of a half-deleted job."""


def _asset_ids(value: Any) -> set[str]:
    """Every `asset_id`/`*_asset_id` string value nested anywhere in a job's
    stored request body (e.g. ReplaceRequest.image_asset_id, or one entry of
    TextToVideoRequest.images[].asset_id). Walks dicts/lists generically so
    this needs no per-backend knowledge of any one request schema's shape --
    every backend's own asset-reference fields already follow this same
    `asset_id`/`..._asset_id` naming (see services/*/schemas.py)."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if (key == "asset_id" or key.endswith("_asset_id")) and isinstance(item, str):
                found.add(item)
            else:
                found |= _asset_ids(item)
    elif isinstance(value, list):
        for item in value:
            found |= _asset_ids(item)
    return found


def purge_job(row: sqlite3.Row) -> tuple[list[str], list[str]]:
    """Delete `row`'s own job directory and database row, then any upload it
    referenced that's now orphaned. Returns (removed_asset_ids,
    kept_asset_ids), each sorted for a stable, testable order.

    The job's row is deleted only after its directory is confirmed fully
    gone -- so a failure partway through (see PurgeRetryableError) always
    leaves the job still listed, never silently vanished with debris left
    behind on disk.

    Upload cleanup happens *after* the job row is gone, so
    job_store.asset_in_use below is checking every *other* remaining job,
    not this one (which no longer has a row to match against).
    """
    job_dir = Path(row["job_dir"])
    expected_dir = config.JOBS_DIR / row["job_id"]
    if job_dir.resolve() != expected_dir.resolve():
        raise PurgeSafetyError(f"Refusing to delete unexpected job directory: {job_dir}")

    if job_dir.exists():
        try:
            shutil.rmtree(job_dir)
        except OSError as e:
            raise PurgeRetryableError(
                f"Could not fully remove the job directory yet ({e}). The job record was kept -- try again shortly."
            ) from e
    if job_dir.exists():
        raise PurgeRetryableError(
            "The job directory still exists after attempting to remove it. The job record was kept -- try again shortly."
        )

    job_store.delete_job_row(row["job_id"])

    request = json.loads(row["request_json"])
    removed: list[str] = []
    kept: list[str] = []
    for asset_id in sorted(_asset_ids(request)):
        if job_store.asset_in_use(asset_id):
            kept.append(asset_id)
        else:
            storage.delete_upload(asset_id)
            removed.append(asset_id)
    return removed, kept
