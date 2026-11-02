"""Recognising a request that was already sent, so it is not run twice.

A job row carries a `dedup_key`. When a request arrives with a key that an earlier job
already has, and that job is still queued, still running, or finished and its result is
still on disk, the earlier job is returned instead of a new one.

Two kinds of key:

  * content_key: a hash of what the request asks for (its settings, with every uploaded
    file replaced by the sha256 of its content), used by the warm-pool backends. Their
    requests carry a seed with a fixed default, so the same request gives the same result.
  * idempotency_key: a hash of a client-chosen key (the Idempotency-Key header), used by
    Qwen3.8, whose answers are not repeatable and so are never matched by content.

A request with `incognito: true` has no key and is never matched.

A failed job is never reused; sending the request again creates a new job.

`lock` must be held from the lookup until the new job's row exists (job_store.create_job),
so two identical requests arriving together cannot both create a job. One lock in one
process is enough because the gateway runs as a single process.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import job_store, purge

lock = threading.Lock()

# Request fields that do not change what is produced.
_IGNORED_FIELDS = ("partition", "incognito")


@dataclass(frozen=True)
class Submission:
    """What a submit endpoint answers with: the job, its current status, and whether it was
    an earlier job found again instead of a new one."""

    job_id: str
    status: str
    reused: bool = False


class _NotMatchable(Exception):
    """The request refers to an upload whose content hash is unknown."""


def _replace_assets(value: Any, hashes: dict[str, str | None]) -> Any:
    """A copy of the request in which every asset id is replaced by the sha256 of its file.
    The fields that hold asset ids are found by the same naming rule as purge.asset_ids_in."""

    def hashed(asset_id: Any) -> str:
        digest = hashes.get(asset_id) if isinstance(asset_id, str) else None
        if digest is None:
            raise _NotMatchable(str(asset_id))
        return f"sha256:{digest}"

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if (key == "asset_id" or key.endswith("_asset_id")) and isinstance(item, str):
                result[key] = hashed(item)
            elif key.endswith("_asset_ids") and isinstance(item, list):
                result[key] = [hashed(entry) if isinstance(entry, str) else entry for entry in item]
            else:
                result[key] = _replace_assets(item, hashes)
        return result
    if isinstance(value, list):
        return [_replace_assets(item, hashes) for item in value]
    return value


def content_key(pipeline: str, request: dict[str, Any], version: str) -> str | None:
    """The key of a request, from its validated body (defaults filled in), or None if it
    cannot be matched (it uses an upload saved before content hashes were kept).

    `version` is the backend's RESULT_VERSION: change it when the weights or the generation
    code change what a request produces, so old results are not handed out for it."""
    body = {key: item for key, item in request.items() if key not in _IGNORED_FIELDS}
    hashes = job_store.upload_sha256s(sorted(purge.asset_ids_in(body)))
    try:
        body = _replace_assets(body, hashes)
    except _NotMatchable:
        return None
    text = json.dumps(
        {"pipeline": pipeline, "version": version, "request": body},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "content:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def idempotency_key(pipeline: str, client_key: str) -> str:
    return "idem:" + hashlib.sha256(f"{pipeline}\n{client_key}".encode("utf-8")).hexdigest()


def pick_reusable(rows: list[sqlite3.Row]) -> sqlite3.Row | None:
    """The newest of these jobs (newest first) that can answer a repeated request: one still
    queued or running, or one that succeeded and whose result file is still there."""
    for row in rows:
        if row["status"] in ("queued", "running"):
            return row
        if (
            row["status"] == "succeeded"
            and not row["downloaded_at"]
            and row["output_path"]
            and Path(row["output_path"]).exists()
        ):
            return row
    return None


def find_reusable(dedup_key: str | None) -> sqlite3.Row | None:
    if dedup_key is None:
        return None
    return pick_reusable(job_store.find_by_dedup_key(dedup_key))


def reuse(row: sqlite3.Row) -> Submission:
    return Submission(job_id=row["job_id"], status=row["status"], reused=True)
