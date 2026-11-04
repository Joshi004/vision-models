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

from common import config, job_source

# Priority runs from 1 (most urgent) to 10 (least urgent); a request that names none gets 10.
MIN_PRIORITY = 1
MAX_PRIORITY = 10
DEFAULT_PRIORITY = MAX_PRIORITY

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
    finished_at TEXT,
    downloaded_at TEXT,
    incognito INTEGER NOT NULL DEFAULT 0,
    lip_sync_json TEXT,
    gpus INTEGER,
    estimated_run_seconds REAL
);

CREATE TABLE IF NOT EXISTS uploads (
    asset_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


@contextlib.contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    # A burst of submissions writes from several threads at once: wait for the lock
    # (up to 30 s) instead of failing at once with "database is locked".
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
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
        # Databases created before `downloaded_at` existed: CREATE TABLE IF NOT
        # EXISTS above leaves them as they were, so add the column here.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        if "downloaded_at" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN downloaded_at TEXT")
        # Same for `incognito`; jobs created before it existed were kept in
        # full, so 0 (not incognito) is the right value for them.
        if "incognito" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN incognito INTEGER NOT NULL DEFAULT 0")
        # Same for `lip_sync_json` (the outcome of the optional lip-sync step,
        # see run_pipeline_job.py); NULL for every job that did not ask for it.
        if "lip_sync_json" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN lip_sync_json TEXT")
        # `gpus` and `estimated_run_seconds`: what the backend decided when the job was
        # submitted (H3 sizes every request, see services/h3/sizing.py); NULL for every
        # job of a backend that does not, and for jobs created before they existed.
        if "gpus" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN gpus INTEGER")
        if "estimated_run_seconds" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN estimated_run_seconds REAL")
        # `dedup_key`: what makes two requests "the same" (common/dedup.py); NULL when the
        # job is never matched (incognito, Qwen3.8 without an Idempotency-Key, old jobs).
        if "dedup_key" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN dedup_key TEXT")
        # `attempts`: how many times the job has been given to a worker (1 = first try).
        if "attempts" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 1")
        # `priority`: 1 (most urgent) to 10 (the default); jobs created before it existed get 10.
        if "priority" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 10")
        # `sha256` of an upload's content; NULL for uploads saved before it existed.
        upload_columns = {row["name"] for row in conn.execute("PRAGMA table_info(uploads)")}
        if "sha256" not in upload_columns:
            conn.execute("ALTER TABLE uploads ADD COLUMN sha256 TEXT")
        # Looked up by POST /v1/uploads/by-hash.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_uploads_sha256 ON uploads(sha256)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_dedup_key ON jobs(dedup_key)")
        # `source`: "ui" or "api" -- who submitted the job (common/job_source.py); NULL for jobs
        # created before it existed (shown as "unknown").
        if "source" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN source TEXT")
        # Backs the jobs list's ORDER BY created_at DESC (GET /v1/jobs, POST /v1/jobs/search).
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)")


# --- jobs --------------------------------------------------------------


def create_job(
    job_id: str,
    pipeline: str,
    job_dir: Path,
    request: dict[str, Any],
    partition: str | None = None,
    incognito: bool = False,
    gpus: int | None = None,
    estimated_run_seconds: float | None = None,
    dedup_key: str | None = None,
) -> None:
    """`gpus` and `estimated_run_seconds` are what the backend decided at submit time, for
    the backends that size their jobs (H3); None otherwise. `dedup_key` is what a later
    identical request is matched on (common/dedup.py); None for a job that is never matched.
    The job's priority is taken from the request body (`priority`, see PriorityOptionMixin);
    DEFAULT_PRIORITY if the request has none."""
    priority = request.get("priority", DEFAULT_PRIORITY)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, pipeline, status, job_dir, partition, request_json, created_at, incognito, "
            "gpus, estimated_run_seconds, dedup_key, priority, source) "
            "VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                pipeline,
                str(job_dir),
                partition,
                json.dumps(request),
                _now(),
                1 if incognito else 0,
                gpus,
                estimated_run_seconds,
                dedup_key,
                priority,
                job_source.current_source(),
            ),
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


def mark_succeeded(job_id: str, output_path: Path, lip_sync: dict[str, Any] | None = None) -> None:
    """`lip_sync` is the outcome of the optional lip-sync step ({"requested",
    "applied", "error"}) for a job that asked for it, else None."""
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'succeeded', output_path = ?, error = NULL, finished_at = ?, "
            "lip_sync_json = ? WHERE job_id = ? AND status != 'succeeded'",
            (str(output_path), _now(), json.dumps(lip_sync) if lip_sync is not None else None, job_id),
        )


def mark_failed(job_id: str, error: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'failed', error = ?, finished_at = ? "
            "WHERE job_id = ? AND status NOT IN ('succeeded', 'failed')",
            (error, _now(), job_id),
        )


def requeue(job_id: str) -> bool:
    """Put a job that was lost for reasons that are not its own fault (its worker or the model
    server vanished) back to 'queued' for another try. False, and nothing changed, if the job is
    already finished or has used up config.MAX_ATTEMPTS."""
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE jobs SET status = 'queued', slurm_job_id = NULL, started_at = NULL, attempts = attempts + 1 "
            "WHERE job_id = ? AND status IN ('queued', 'running') AND attempts < ?",
            (job_id, config.MAX_ATTEMPTS),
        )
        return cursor.rowcount == 1


def requeue_on_more_gpus(job_id: str, gpus: int, estimated_run_seconds: float) -> bool:
    """Put a job that ran out of memory back to 'queued' to run again on `gpus` GPUs, which is more than it had
    (so this works once per job). Clears the earlier attempt's error. False, and nothing changed, if the job is
    not queued or running (finished, cancelled) or already has `gpus` GPUs or more."""
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE jobs SET status = 'queued', slurm_job_id = NULL, started_at = NULL, error = NULL, "
            "attempts = attempts + 1, gpus = ?, estimated_run_seconds = ? "
            "WHERE job_id = ? AND status IN ('queued', 'running') AND gpus < ?",
            (gpus, estimated_run_seconds, job_id, gpus),
        )
        return cursor.rowcount == 1


def statuses(job_ids: list[str]) -> dict[str, str]:
    """job_id -> status for these jobs in as few queries as possible (a job id that is not in the
    table is left out). Chunked so a long queue stays under SQLite's limit on query parameters."""
    found: dict[str, str] = {}
    with _connect() as conn:
        for start in range(0, len(job_ids), 500):
            chunk = job_ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            for row in conn.execute(f"SELECT job_id, status FROM jobs WHERE job_id IN ({marks})", chunk):
                found[row["job_id"]] = row["status"]
    return found


def queued_priorities(job_ids: list[str]) -> dict[str, int]:
    """job_id -> priority for those of these jobs that are still 'queued' (a job that is running,
    finished or unknown is left out, which is how a caller finds the ones that stopped waiting).
    Chunked like `statuses`."""
    found: dict[str, int] = {}
    with _connect() as conn:
        for start in range(0, len(job_ids), 500):
            chunk = job_ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            for row in conn.execute(
                f"SELECT job_id, priority FROM jobs WHERE status = 'queued' AND job_id IN ({marks})", chunk
            ):
                found[row["job_id"]] = int(row["priority"])
    return found


def set_priority(job_id: str, priority: int) -> bool:
    """Change the priority of a job that is still queued. False, and nothing changed, if the job
    does not exist or is no longer queued."""
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE jobs SET priority = ? WHERE job_id = ? AND status = 'queued'", (priority, job_id)
        )
        return cursor.rowcount == 1


def raise_priority(job_id: str, priority: int) -> None:
    """Make a queued job at least as urgent as `priority` (a lower number); never lowers its urgency."""
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET priority = ? WHERE job_id = ? AND status = 'queued' AND priority > ?",
            (priority, job_id, priority),
        )


def count_inflight_for(pipelines: tuple[str, ...]) -> int:
    """Queued or running jobs of these pipelines -- backs each backend's own queue limit."""
    marks = ",".join("?" * len(pipelines))
    with _connect() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued', 'running') AND pipeline IN ({marks})",
            pipelines,
        ).fetchone()
        return int(row["n"])


def find_by_dedup_key(dedup_key: str) -> list[sqlite3.Row]:
    """Every job created with this dedup key, newest first."""
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM jobs WHERE dedup_key = ? ORDER BY created_at DESC", (dedup_key,)
        ).fetchall()


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


def _escape_like(text: str) -> str:
    """Make `text` match literally inside a LIKE pattern that uses ESCAPE '\\'."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_jobs(
    *,
    offset: int,
    limit: int,
    statuses: list[str] | None = None,
    pipelines: list[str] | None = None,
    source: str | None = None,
    job_ids: list[str] | None = None,
    exclude_job_ids: list[str] | None = None,
    q: str | None = None,
) -> tuple[list[sqlite3.Row], int]:
    """One page of jobs, newest first, plus the total number of jobs matching the filters --
    backs POST /v1/jobs/search. Every filter is optional; the ones given are combined with AND.
    `source` is "ui", "api" or "unknown" (jobs created before `source` existed). `q` matches
    anywhere in the job id or the request's prompt. All values are passed as query parameters,
    never spliced into the SQL."""
    clauses: list[str] = []
    params: list[Any] = []

    if statuses:
        clauses.append(f"status IN ({','.join('?' * len(statuses))})")
        params.extend(statuses)
    if pipelines:
        clauses.append(f"pipeline IN ({','.join('?' * len(pipelines))})")
        params.extend(pipelines)
    if source == "unknown":
        clauses.append("source IS NULL")
    elif source:
        clauses.append("source = ?")
        params.append(source)
    if job_ids is not None:
        # An empty list is a real filter ("only these jobs", and there are none): matches nothing.
        clauses.append(f"job_id IN ({','.join('?' * len(job_ids))})")
        params.extend(job_ids)
    if exclude_job_ids:
        clauses.append(f"job_id NOT IN ({','.join('?' * len(exclude_job_ids))})")
        params.extend(exclude_job_ids)
    if q:
        pattern = f"%{_escape_like(q.lower())}%"
        clauses.append(
            "(lower(job_id) LIKE ? ESCAPE '\\' OR lower(COALESCE(json_extract(request_json, '$.prompt'), '')) LIKE ? ESCAPE '\\')"
        )
        params.extend([pattern, pattern])

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect() as conn:
        total = int(conn.execute(f"SELECT COUNT(*) AS n FROM jobs {where}", params).fetchone()["n"])
        rows = conn.execute(
            f"SELECT * FROM jobs {where} ORDER BY created_at DESC, job_id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
    return rows, total


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


def list_finished_jobs(pipeline_prefix: str) -> list[sqlite3.Row]:
    """Finished (succeeded or failed) jobs whose pipeline starts with this prefix (e.g. "h3:"), oldest
    first -- backs the H3 usage log's backfill (scripts/h3_usage_log.py)."""
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM jobs WHERE finished_at IS NOT NULL AND substr(pipeline, 1, ?) = ? ORDER BY created_at",
            (len(pipeline_prefix), pipeline_prefix),
        ).fetchall()


def list_finished_before(cutoff_iso: str) -> list[sqlite3.Row]:
    """Jobs finished (succeeded or failed) before cutoff_iso -- used for cleanup."""
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM jobs WHERE finished_at IS NOT NULL AND finished_at < ?", (cutoff_iso,)
        ).fetchall()


def delete_job_row(job_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))


# Request body stored on a job whose data has been scrubbed (see scrub_job).
SCRUBBED_REQUEST_JSON = "{}"


def is_scrubbed(row: sqlite3.Row) -> bool:
    """True once scrub_job has removed this job's request body and output."""
    return row["request_json"] == SCRUBBED_REQUEST_JSON


def scrub_job(job_id: str, *, error: str | None = None, downloaded: bool = False) -> None:
    """Reduce a job to a minimal record: pipeline, status, partition, Slurm id
    and timestamps stay (so the jobs list and the typical-run-time estimates
    keep working); the request body (prompt, settings, asset ids) and the
    output path are removed. `error` replaces the stored error text (used to
    shorten a failure's log tail); `downloaded` stamps downloaded_at."""
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET request_json = ?, output_path = NULL, "
            "error = COALESCE(?, error), "
            "downloaded_at = CASE WHEN ? THEN COALESCE(downloaded_at, ?) ELSE downloaded_at END "
            "WHERE job_id = ?",
            (SCRUBBED_REQUEST_JSON, error, 1 if downloaded else 0, _now(), job_id),
        )


def list_finished_unscrubbed_incognito() -> list[sqlite3.Row]:
    """Finished (succeeded or failed) incognito jobs that still hold their
    request body and possibly files -- what retention.sweep re-checks. Jobs
    that are not incognito are never scrubbed, so they are not listed."""
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM jobs WHERE incognito = 1 AND finished_at IS NOT NULL AND request_json != ?",
            (SCRUBBED_REQUEST_JSON,),
        ).fetchall()


def count_inflight_jobs() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued', 'running')").fetchone()
        return int(row["n"])


def inflight_counts_by_pipeline() -> dict[tuple[str, str], int]:
    """(pipeline, status) -> number of queued or running jobs, in one query -- backs the GPU
    capacity accounting of the backends that run one Slurm job per request (common/capacity.py)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT pipeline, status, COUNT(*) AS n FROM jobs WHERE status IN ('queued', 'running') "
            "GROUP BY pipeline, status"
        ).fetchall()
    return {(row["pipeline"], row["status"]): int(row["n"]) for row in rows}


# --- settings ------------------------------------------------------------


def get_setting(key: str) -> str | None:
    with _connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None


def set_setting(key: str, value: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


# --- uploads -------------------------------------------------------------


def create_upload(asset_id: str, filename: str, path: Path, size_bytes: int, sha256: str | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO uploads (asset_id, filename, path, size_bytes, created_at, sha256) VALUES (?, ?, ?, ?, ?, ?)",
            (asset_id, filename, str(path), size_bytes, _now(), sha256),
        )


def upload_sha256s(asset_ids: list[str]) -> dict[str, str | None]:
    """asset_id -> content sha256 (None for an upload saved before hashes were kept); an unknown
    asset_id is left out."""
    found: dict[str, str | None] = {}
    with _connect() as conn:
        for start in range(0, len(asset_ids), 500):
            chunk = asset_ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            for row in conn.execute(f"SELECT asset_id, sha256 FROM uploads WHERE asset_id IN ({marks})", chunk):
                found[row["asset_id"]] = row["sha256"]
    return found


def get_upload(asset_id: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute("SELECT * FROM uploads WHERE asset_id = ?", (asset_id,)).fetchone()


def find_uploads_by_sha256(sha256: str) -> list[sqlite3.Row]:
    """Every upload with this content hash, newest first."""
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM uploads WHERE sha256 = ? ORDER BY created_at DESC", (sha256,)
        ).fetchall()


def touch_upload(asset_id: str) -> None:
    """Make the upload count as uploaded now, so the unused-upload sweep
    (retention.sweep_unused_uploads) waits its full time again."""
    with _connect() as conn:
        conn.execute("UPDATE uploads SET created_at = ? WHERE asset_id = ?", (_now(), asset_id))


def asset_used_by_incognito(asset_id: str) -> bool:
    """True if any incognito job's stored request still references this asset_id (such an
    upload is deleted when that job ends, so it must not be offered to anyone else)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE incognito = 1 AND instr(request_json, ?) > 0 LIMIT 1", (asset_id,)
        ).fetchone()
        return row is not None


def asset_needed_by_other_job(asset_id: str) -> bool:
    """True if a job still needs this upload: one that is queued or running (its GPU job may
    still read the file), or one that is not incognito (kept in full, so Retry needs it).
    Called while cleaning up a finished incognito job, which is neither."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE instr(request_json, ?) > 0 "
            "AND (status IN ('queued', 'running') OR incognito = 0) LIMIT 1",
            (asset_id,),
        ).fetchone()
        return row is not None


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


def asset_in_flight(asset_id: str) -> bool:
    """Like asset_in_use, but only counts jobs that are still queued or running
    -- i.e. whose GPU job may still need to read this upload."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE status IN ('queued', 'running') AND instr(request_json, ?) > 0 LIMIT 1",
            (asset_id,),
        ).fetchone()
        return row is not None


def list_uploads_before(cutoff_iso: str) -> list[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute("SELECT * FROM uploads WHERE created_at < ?", (cutoff_iso,)).fetchall()


def delete_upload_row(asset_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM uploads WHERE asset_id = ?", (asset_id,))
