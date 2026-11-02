"""Upload storage: save incoming files, resolve asset_id -> path on disk.

Shared by every backend -- POST /v1/uploads is a common endpoint, and any
backend's dispatch module can resolve an asset_id from it.
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from pathlib import Path
from typing import BinaryIO

from common import config, job_store

_CHUNK_BYTES = 1024 * 1024


class UploadTooLarge(Exception):
    """The upload is bigger than the allowed maximum."""


def save_upload_stream(filename: str, fileobj: BinaryIO, max_bytes: int) -> tuple[str, Path, int, str]:
    """Save an uploaded file under a fresh asset_id and record it.

    The file is copied to disk in 1 MiB chunks, so memory use does not depend on its size, and
    its size and sha256 are worked out on the way. If it grows past `max_bytes` the partial file
    is deleted and UploadTooLarge is raised; any other failure also leaves no partial file behind.

    Returns (asset_id, path, size_bytes, sha256).
    """
    asset_id = uuid.uuid4().hex
    safe_name = Path(filename or "upload").name  # strip any directory components
    dest = config.UPLOADS_DIR / f"{asset_id}__{safe_name}"
    # Created at startup too, but the folder can be removed while the server
    # runs (e.g. by hand while clearing data); recreate it rather than fail.
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with dest.open("wb") as out:
            while chunk := fileobj.read(_CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLarge()
                digest.update(chunk)
                out.write(chunk)
        # The content hash lets two uploads of the same file be recognised as the same input
        # (common/dedup.py), and lets a client skip uploading a file the server already has
        # (find_reusable_upload below).
        job_store.create_upload(asset_id, safe_name, dest, size, digest.hexdigest())
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return asset_id, dest, size, digest.hexdigest()


def find_reusable_upload(sha256: str) -> sqlite3.Row | None:
    """A stored upload with this content hash that can be handed to another client, or None
    (never uploaded, deleted, purged, expired, or saved before hashes were kept).

    An upload is skipped if its file is missing or has the wrong size on disk, or if an
    incognito job references it (that job deletes it when it ends). The one returned is
    marked as uploaded now, so the unused-upload sweep gives its new user the full wait."""
    for row in job_store.find_uploads_by_sha256(sha256):
        path = Path(row["path"])
        try:
            if path.stat().st_size != row["size_bytes"]:
                continue
        except OSError:
            continue
        if job_store.asset_used_by_incognito(row["asset_id"]):
            continue
        job_store.touch_upload(row["asset_id"])
        return row
    return None


def resolve_asset_path(asset_id: str) -> Path:
    """Look up a previously uploaded file by asset_id.

    Raises KeyError if the asset_id is unknown, FileNotFoundError if it was
    recorded but the underlying file is missing (e.g. cleaned up) -- both are
    mapped to a 400 by the API layer (each backend's router), never to a 500.
    """
    row = job_store.get_upload(asset_id)
    if row is None:
        raise KeyError(f"Unknown asset_id: {asset_id}")
    path = Path(row["path"])
    if not path.exists():
        raise FileNotFoundError(f"Upload file missing on disk for asset_id: {asset_id}")
    return path


def delete_upload(asset_id: str) -> None:
    """Delete an uploaded file and its DB row. A no-op if the asset_id is
    unknown (already deleted, or never existed) -- common/purge.py is the
    only caller today, and only ever after confirming (via
    job_store.asset_in_use) that no remaining job still references this
    asset_id."""
    row = job_store.get_upload(asset_id)
    if row is None:
        return
    path = Path(row["path"])
    if path.parent.resolve() != config.UPLOADS_DIR.resolve():
        # Should never happen -- path always comes from this same module's
        # own save_upload_bytes -- but refusing beats deleting whatever the
        # row happens to point at if it somehow did.
        raise RuntimeError(f"Refusing to delete upload outside the uploads directory: {path}")
    path.unlink(missing_ok=True)
    job_store.delete_upload_row(asset_id)
