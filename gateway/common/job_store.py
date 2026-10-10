"""SQLite-backed job (and upload) tracking, shared by every backend.

Deliberately simple: a single-file SQLite database, one row per job/upload,
a short-lived connection per call rather than a long-held connection or
pool. That's appropriate for the low request volume this service is
designed for (see the discussion that led here) -- if volume ever grows
enough for this to matter, swap in a real queue/DB then, not before.

One shared `jobs` table across every backend (LTX-2.3 today, more later):
the `pipeline` column identifies which backend/recipe created a given row
(e.g. "ltx:text-to-video"), so nothing here needs to know about any
specific backend.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    pipeline TEXT NOT NULL,
    status TEXT NOT NULL,
    slurm_job_id TEXT,
    job_dir TEXT NOT NULL,
    partition TEXT,
    output_path TEXT,
    request_json TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS uploads (
    asset_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


@contextlib.contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    config.ensure_directories()
    with _connect() as conn:
        conn.executescript(_SCHEMA)


# --- jobs --------------------------------------------------------------


def create_job(
    job_id: str,
    pipeline: str,
    job_dir: Path,
    request: dict[str, Any],
    partition: str | None = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, pipeline, status, job_dir, partition, request_json, created_at) "
            "VALUES (?, ?, 'queued', ?, ?, ?, ?)",
            (job_id, pipeline, str(job_dir), partition, json.dumps(request), _now()),
        )


def set_slurm_job_id(job_id: str, slurm_job_id: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE jobs SET slurm_job_id = ? WHERE job_id = ?", (slurm_job_id, job_id))


def mark_running(job_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'running', started_at = COALESCE(started_at, ?) "
            "WHERE job_id = ? AND status = 'queued'",
            (_now(), job_id),
        )


def mark_succeeded(job_id: str, output_path: Path) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'succeeded', output_path = ?, error = NULL, finished_at = ? "
            "WHERE job_id = ? AND status != 'succeeded'",
            (str(output_path), _now(), job_id),
        )


def mark_failed(job_id: str, error: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'failed', error = ?, finished_at = ? "
            "WHERE job_id = ? AND status NOT IN ('succeeded', 'failed')",
            (error, _now(), job_id),
        )


def get_job(job_id: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()


def list_inflight_jobs() -> list[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute("SELECT * FROM jobs WHERE status IN ('queued', 'running')").fetchall()


def list_recent_jobs(limit: int) -> list[sqlite3.Row]:
    """Most recently created jobs first, across every backend -- backs GET
    /v1/jobs (the web UI's job history), capped at `limit`."""
    with _connect() as conn:
        return conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()


def recent_run_seconds(pipeline: str, limit: int = 20) -> list[float]:
    """Wall-clock run time (started_at -> finished_at) of the most recent
    succeeded jobs for this exact pipeline string (e.g. "ltx:text-to-video"),
    most recent first -- backs the "typically takes about..." estimate the
    API reports alongside a queued/running job of the same recipe (see
    server.py's _typical_run_seconds). Empty if there's no history yet,
    e.g. right after this database was created or for a pipeline that has
    never actually succeeded."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT started_at, finished_at FROM jobs "
            "WHERE pipeline = ? AND status = 'succeeded' "
            "AND started_at IS NOT NULL AND finished_at IS NOT NULL "
            "ORDER BY finished_at DESC LIMIT ?",
            (pipeline, limit),
        ).fetchall()
    durations = []
    for row in rows:
        started = datetime.fromisoformat(row["started_at"])
        finished = datetime.fromisoformat(row["finished_at"])
        durations.append((finished - started).total_seconds())
    return durations


def list_finished_before(cutoff_iso: str) -> list[sqlite3.Row]:
    """Jobs finished (succeeded or failed) before cutoff_iso -- used for cleanup."""
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM jobs WHERE finished_at IS NOT NULL AND finished_at < ?", (cutoff_iso,)
        ).fetchall()


def delete_job_row(job_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))


def count_inflight_jobs() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued', 'running')").fetchone()
        return int(row["n"])


# --- uploads -------------------------------------------------------------


def create_upload(asset_id: str, filename: str, path: Path, size_bytes: int) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO uploads (asset_id, filename, path, size_bytes, created_at) VALUES (?, ?, ?, ?, ?)",
            (asset_id, filename, str(path), size_bytes, _now()),
        )


def get_upload(asset_id: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute("SELECT * FROM uploads WHERE asset_id = ?", (asset_id,)).fetchone()


def asset_in_use(asset_id: str) -> bool:
    """True if any job row's stored request body still contains this
    asset_id -- a cheap, backend-agnostic way to check "is any job still
    using this upload" without needing to know each request schema's exact
    shape (see common/purge.py's own asset_id walker, which extracts the
    asset_ids to check in the first place). asset_id is always a
    32-character uuid4 hex string (new_id()), so a false-positive substring
    match inside some unrelated field is not realistically possible."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE instr(request_json, ?) > 0 LIMIT 1", (asset_id,)
        ).fetchone()
        return row is not None


def delete_upload_row(asset_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM uploads WHERE asset_id = ?", (asset_id,))
