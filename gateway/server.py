"""FastAPI app: the shared async job-submission API for GPU-cluster
model-serving backends on this node.

Backend-agnostic pieces (job tracking, uploads, Slurm polling,
partition selection) live in common/. Each backend (LTX-2.3, LTX-2.5,
Wan-Animate v1, RVC, Parakeet and Breeze TTS 2 today; more may be added later) owns its
own request schemas and Slurm dispatch logic under services/<name>/,
mounted here under its own path prefix.

The long-form documentation shown in /docs and /openapi.json (app
description, tags, and the common endpoints' descriptions/examples) lives
in common/openapi_docs.py; each backend keeps its own endpoint docs next to
its router (e.g. services/ltx/openapi_docs.py). GET /v1/guide (below)
assembles all of that, plus a few narrative files, into the one document a
third-party integration should actually read -- see common/guide.py.

Run with (see run.sh for the tmux-wrapped version):
    cd model-api
    venv/bin/uvicorn server:app --host 0.0.0.0 --port 8012
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi import Path as PathParam
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from common import config, guide, job_store, openapi_docs, partition_status, poller, purge, storage
from common.schemas import (
    CancelResponse,
    GuideResponse,
    HealthResponse,
    JobListItem,
    JobListResponse,
    JobStatusResponse,
    PartitionStatus,
    PartitionStatusResponse,
    PurgeResponse,
    UploadResponse,
)
from common.slurm import cancel_slurm_job
from services.breeze_tts import config as breeze_tts_config
from services.breeze_tts.router import router as breeze_tts_router
from services.ltx import config as ltx_config
from services.ltx.router import router as ltx_router
from services.ltx25 import config as ltx25_config
from services.ltx25.router import router as ltx25_router
from services.parakeet import config as parakeet_config
from services.parakeet.router import router as parakeet_router
from services.rvc import config as rvc_config
from services.rvc.router import router as rvc_router
from services.wan_animate import config as wan_animate_config
from services.wan_animate import progress as wan_animate_progress
from services.wan_animate.router import router as wan_animate_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-api")

_poller_task: asyncio.Task | None = None


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    config.ensure_directories()
    job_store.init_db()
    global _poller_task
    _poller_task = asyncio.create_task(poller.poll_forever())
    for warning in guide.lint(app):
        logger.warning("docs lint: %s", warning)
    logger.info("model-api ready.")
    yield
    if _poller_task is not None:
        _poller_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _poller_task


app = FastAPI(
    title="GPU Cluster Model-Serving API",
    version="0.7.0",
    description=openapi_docs.APP_DESCRIPTION,
    openapi_tags=openapi_docs.TAGS_METADATA,
    lifespan=lifespan,
)

# Backend routers, each namespaced under its own prefix. Add future backends
# (e.g. MAGI-2.2, Wan-Animate v2) the same way once their own
# services/<name>/ module exists.
app.include_router(ltx_router, prefix="/v1/ltx")
app.include_router(ltx25_router, prefix="/v1/ltx25")
app.include_router(wan_animate_router, prefix="/v1/wan-animate")
app.include_router(rvc_router, prefix="/v1/rvc")
app.include_router(parakeet_router, prefix="/v1/parakeet")
app.include_router(breeze_tts_router, prefix="/v1/breeze-tts")

# Registered backends, for GET /v1/guide (common/guide.py) -- each entry is
# a small, enumerable set of facts: display name, path prefix, and where
# that backend's own narrative guide file lives (relative to services/).
# Add a line here when a new backend's router joins the ones above;
# forgetting to does NOT hide that backend's endpoints from the guide --
# they still show up under "Other endpoints" there -- it only means they're
# missing their own curated section until this list catches up.
_BACKENDS: list[guide.BackendInfo] = [
    guide.BackendInfo(id="ltx", name="LTX-2.3", prefix="/v1/ltx", guide_file="ltx/GUIDE.md"),
    guide.BackendInfo(id="ltx25", name="LTX-2.5", prefix="/v1/ltx25", guide_file="ltx25/GUIDE.md"),
    guide.BackendInfo(id="wan-animate", name="Wan-Animate v1", prefix="/v1/wan-animate", guide_file="wan_animate/GUIDE.md"),
    guide.BackendInfo(id="rvc", name="RVC voice conversion", prefix="/v1/rvc", guide_file="rvc/GUIDE.md"),
    guide.BackendInfo(id="parakeet", name="Parakeet transcription", prefix="/v1/parakeet", guide_file="parakeet/GUIDE.md"),
    guide.BackendInfo(id="breeze-tts", name="Breeze TTS 2 text-to-speech", prefix="/v1/breeze-tts", guide_file="breeze_tts/GUIDE.md"),
]


@app.get(
    "/v1/guide",
    tags=["Integration guide"],
    summary="Full integration guide for building a third-party integration against this API",
    description=openapi_docs.GUIDE,
    responses=openapi_docs.GUIDE_RESPONSES,
    response_model=None,
)
def get_guide(
    request: Request,
    fmt: Annotated[
        Literal["markdown", "json"],
        Query(
            alias="format",
            description=(
                "'markdown' (default): plain-text prose, meant to be read directly. 'json': the same "
                "content plus structured facts (limits, registered backends, ...) for code "
                "to consume."
            ),
        ),
    ] = "markdown",
) -> Response:
    base_url = str(request.base_url).rstrip("/")
    markdown, facts, content_hash, generated_at = guide.render(app, _BACKENDS, base_url)
    etag = f'"{content_hash}-{fmt}"'
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    if fmt == "json":
        payload = GuideResponse(
            api_version=app.version,
            generated_at=generated_at,
            content_hash=content_hash,
            facts=facts,
            markdown=markdown,
        )
        return JSONResponse(payload.model_dump(mode="json"), headers=headers)
    return Response(content=markdown, media_type="text/markdown; charset=utf-8", headers=headers)


@app.get(
    "/v1/health",
    tags=["System"],
    summary="Health check",
    description=openapi_docs.HEALTH,
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        default_partition=config.FALLBACK_PARTITION,
        job_retention_days=config.JOB_RETENTION_DAYS,
    )


@app.get(
    "/v1/partitions",
    tags=["System"],
    summary="Live Slurm partition availability",
    description=openapi_docs.PARTITIONS,
    response_model=PartitionStatusResponse,
    responses=openapi_docs.PARTITIONS_RESPONSES,
)
def get_partitions() -> PartitionStatusResponse:
    try:
        snapshot, generated_at = partition_status.get_partition_status()
    except partition_status.PartitionStatusError as e:
        raise HTTPException(503, f"Partition status is temporarily unavailable: {e}") from e
    return PartitionStatusResponse(
        partitions=[PartitionStatus(**entry) for entry in snapshot],
        generated_at=generated_at,
    )


@app.post(
    "/v1/uploads",
    tags=["Uploads"],
    summary="Upload a file, get back an asset_id",
    description=openapi_docs.UPLOAD,
    response_model=UploadResponse,
    responses=openapi_docs.UPLOAD_RESPONSES,
)
async def upload(
    file: Annotated[UploadFile, File(description="The image, audio, or video file to upload.")],
) -> UploadResponse:
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds the {config.MAX_UPLOAD_BYTES} byte limit.")
    asset_id, _path = storage.save_upload_bytes(file.filename or "upload", data)
    return UploadResponse(asset_id=asset_id, filename=file.filename or "upload", size_bytes=len(data))


_JOB_ID_PARAM = Annotated[
    str, PathParam(description="The job_id returned by a generation endpoint.", examples=["6b54a0bd6cef4b04abe50a8f81f55a80"])
]

# Maps a pipeline's own "<backend>:<recipe>" prefix (i.e. everything before
# the first ":") to that backend's fallback typical-run estimate, used only
# when recent_run_seconds has no history yet for the *exact* pipeline
# string. Extend this table, not _typical_run_seconds itself, when a new
# backend joins.
_TYPICAL_FALLBACK_SECONDS: dict[str, float] = {
    "ltx": ltx_config.TYPICAL_RUN_SECONDS,
    "ltx25": ltx25_config.TYPICAL_RUN_SECONDS,
    "wan-animate": wan_animate_config.TYPICAL_RUN_SECONDS,
    "rvc": rvc_config.TYPICAL_RUN_SECONDS,
    "parakeet": parakeet_config.TYPICAL_RUN_SECONDS,
    "breeze-tts": breeze_tts_config.TYPICAL_RUN_SECONDS,
}

# Same idea, but for live progress readers -- most backends have none
# (JobProgress stays null for them), so this table only needs an entry for
# ones that do. See services/wan_animate/progress.py.
_PROGRESS_READERS = {
    "wan-animate": wan_animate_progress.read_progress,
}


def _typical_run_seconds(pipeline: str, cache: dict[str, tuple[float | None, str | None]]) -> tuple[float | None, str | None]:
    """(typical_run_seconds, typical_basis) for this exact pipeline string --
    the median of recent succeeded runs of it if there are any, else that
    backend's own documented fallback (see _TYPICAL_FALLBACK_SECONDS).
    `cache` is scoped to one request (list_jobs/job_status each create a
    fresh dict): several rows in a jobs list commonly share the same
    pipeline, and job_store.recent_run_seconds is a real query, not worth
    repeating once per row."""
    if pipeline not in cache:
        durations = job_store.recent_run_seconds(pipeline)
        if durations:
            cache[pipeline] = (statistics.median(durations), f"median of {len(durations)} recent run(s)")
        else:
            fallback = _TYPICAL_FALLBACK_SECONDS.get(pipeline.split(":", 1)[0])
            cache[pipeline] = (fallback, "documented estimate" if fallback is not None else None)
    return cache[pipeline]


def _row_to_status(row: sqlite3.Row, typical_cache: dict[str, tuple[float | None, str | None]]) -> JobStatusResponse:
    """Shared by job_status and list_jobs so both report the same fields the
    same way."""
    typical_run_seconds, typical_basis = _typical_run_seconds(row["pipeline"], typical_cache)

    progress = None
    if row["status"] in ("queued", "running"):
        reader = _PROGRESS_READERS.get(row["pipeline"].split(":", 1)[0])
        if reader is not None:
            progress = reader(Path(row["job_dir"]), json.loads(row["request_json"]))

    return JobStatusResponse(
        job_id=row["job_id"],
        pipeline=row["pipeline"],
        status=row["status"],
        partition=row["partition"],
        slurm_job_id=row["slurm_job_id"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        error=row["error"],
        result_ready=row["status"] == "succeeded" and bool(row["output_path"]),
        typical_run_seconds=typical_run_seconds,
        typical_basis=typical_basis,
        progress=progress,
    )


def _row_to_list_item(row: sqlite3.Row, typical_cache: dict[str, tuple[float | None, str | None]]) -> JobListItem:
    return JobListItem(**_row_to_status(row, typical_cache).model_dump(), request=json.loads(row["request_json"]))


@app.get(
    "/v1/jobs",
    tags=["Jobs"],
    summary="List recent jobs",
    description=openapi_docs.JOB_LIST,
    response_model=JobListResponse,
    responses=openapi_docs.JOB_LIST_RESPONSES,
)
def list_jobs(
    limit: Annotated[int, Query(ge=1, le=500, description="Max number of jobs to return, most recent first.")] = 50,
) -> JobListResponse:
    typical_cache: dict[str, tuple[float | None, str | None]] = {}
    jobs = [_row_to_list_item(row, typical_cache) for row in job_store.list_recent_jobs(limit)]
    return JobListResponse(jobs=jobs, server_time=datetime.now(timezone.utc).isoformat())


@app.get(
    "/v1/jobs/{job_id}",
    tags=["Jobs"],
    summary="Get job status",
    description=openapi_docs.JOB_STATUS,
    response_model=JobStatusResponse,
    responses=openapi_docs.JOB_STATUS_RESPONSES,
)
def job_status(job_id: _JOB_ID_PARAM) -> JobStatusResponse:
    row = job_store.get_job(job_id)
    if row is None:
        raise HTTPException(404, "Unknown job_id.")
    return _row_to_status(row, {})


@app.get(
    "/v1/jobs/{job_id}/result",
    tags=["Jobs"],
    summary="Download the finished file",
    description=openapi_docs.JOB_RESULT,
    responses=openapi_docs.JOB_RESULT_RESPONSES,
    # Not a JSON response -- without this, FastAPI's OpenAPI generator still
    # assumes its own default response_class (JSONResponse) for schema
    # purposes only, which injects a spurious, empty "application/json"
    # content entry into the 200 response alongside the real ones above.
    response_class=Response,
)
def job_result(job_id: _JOB_ID_PARAM) -> FileResponse:
    row = job_store.get_job(job_id)
    if row is None:
        raise HTTPException(404, "Unknown job_id.")
    if row["status"] != "succeeded" or not row["output_path"]:
        raise HTTPException(409, f"Job is not finished successfully yet (status: {row['status']}).")
    output_path = Path(row["output_path"])
    if not output_path.exists():
        raise HTTPException(410, "Result file is no longer available (it may have expired).")
    return FileResponse(
        output_path,
        filename=output_path.name,
        # The output file is provably immutable once this can return 200 --
        # job_store.mark_succeeded only ever writes output_path the first
        # time a job reaches "succeeded" -- so a long "immutable" cache
        # lifetime is safe: repeat requests for the same job_id (a second
        # "Load preview" click after the browser's local cache is cleared,
        # or the plain "Download" link) can be served straight from the
        # browser's own HTTP cache instead of re-reading this file over NFS.
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@app.delete(
    "/v1/jobs/{job_id}",
    tags=["Jobs"],
    summary="Cancel a job",
    description=openapi_docs.CANCEL_JOB,
    response_model=CancelResponse,
    response_model_exclude_none=True,
    responses=openapi_docs.CANCEL_RESPONSES,
)
def cancel_job(job_id: _JOB_ID_PARAM) -> CancelResponse:
    row = job_store.get_job(job_id)
    if row is None:
        raise HTTPException(404, "Unknown job_id.")
    if row["status"] not in ("queued", "running"):
        return CancelResponse(cancelled=False, reason=f"job already {row['status']}")
    if row["slurm_job_id"]:
        cancel_slurm_job(row["slurm_job_id"])
    job_store.mark_failed(job_id, "Cancelled by client request.")
    return CancelResponse(cancelled=True)


@app.delete(
    "/v1/jobs/{job_id}/purge",
    tags=["Jobs"],
    summary="Permanently delete a finished job",
    description=openapi_docs.PURGE_JOB,
    response_model=PurgeResponse,
    responses=openapi_docs.PURGE_RESPONSES,
)
def purge_job(job_id: _JOB_ID_PARAM) -> PurgeResponse:
    row = job_store.get_job(job_id)
    if row is None:
        raise HTTPException(404, "Unknown job_id.")
    if row["status"] in ("queued", "running"):
        raise HTTPException(409, "Job is still queued/running -- cancel it first (DELETE /v1/jobs/{job_id}).")
    try:
        removed, kept = purge.purge_job(row)
    except purge.PurgeSafetyError as e:
        raise HTTPException(500, str(e)) from e
    except purge.PurgeRetryableError as e:
        raise HTTPException(409, str(e)) from e
    return PurgeResponse(job_id=job_id, removed_asset_ids=removed, kept_asset_ids=kept)


# ---------------------------------------------------------------------------
# Web UI: a small React/Vite app (model-api/ui/) that calls this same API --
# see ui/README.md for how to build it. Mounted last, under its own /ui
# prefix rather than "/", and only if it's actually been built -- see
# StaticFiles/Mount matching semantics before moving this: a Mount matches
# any path under its prefix, so a route added *below* a "/" mount would
# never be reached; every route above stays reachable either way since
# Starlette tries routes in registration order and returns on the first
# match.
# ---------------------------------------------------------------------------
if config.UI_DIST_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=config.UI_DIST_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def _ui_redirect() -> RedirectResponse:
        return RedirectResponse("/ui/")
else:
    logger.warning("Web UI not built (missing %s) -- /ui disabled. See ui/README.md.", config.UI_DIST_DIR)
