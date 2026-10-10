"""Upload storage: save incoming files, resolve asset_id -> path on disk.

Shared by every backend -- POST /v1/uploads is a common endpoint, and any
backend's dispatch module can resolve an asset_id from it.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from common import config, job_store


def save_upload_bytes(filename: str, data: bytes) -> tuple[str, Path]:
    """Save uploaded bytes under a fresh asset_id, record it, return (asset_id, path)."""
    asset_id = uuid.uuid4().hex
    safe_name = Path(filename or "upload").name  # strip any directory components
    dest = config.UPLOADS_DIR / f"{asset_id}__{safe_name}"
    dest.write_bytes(data)
    job_store.create_upload(asset_id, safe_name, dest, len(data))
    return asset_id, dest


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
