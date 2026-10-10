"""Long-form OpenAPI/Swagger documentation for the parts of this API that
are shared by every backend (app-level description, tags, uploads, job
tracking, and GET /v1/guide itself). Each backend's own endpoint docs (e.g.
services/ltx/openapi_docs.py) live next to that backend's code instead.

Every number quoted here is pulled from common/config.py at import time,
not hardcoded, so the docs can't silently drift from the server's actual
running configuration --
APP_DESCRIPTION in particular is rendered by common/guide.py from the same
narrative file GET /v1/guide itself opens with, so the two can't disagree.
"""

from __future__ import annotations

from common import config, guide
from common.schemas import GuideResponse

# ---------------------------------------------------------------------------
# App-level description (rendered at the top of /docs and /redoc) -- the
# opening section of GET /v1/guide, verbatim. See common/guide.py's
# render_summary() and common/guide_sections/00_summary.md.
# ---------------------------------------------------------------------------

APP_DESCRIPTION = guide.render_summary()

TAGS_METADATA = [
    {
        "name": "Integration guide",
        "description": "Start here. GET /v1/guide is the single, always-current reference for building a third-party integration against this API.",
    },
    {
        "name": "System",
        "description": "Health check and live cluster-partition availability.",
    },
    {
        "name": "Uploads",
        "description": "Upload a file once, then reference its returned asset_id from any generation "
        "request (on any backend) that needs it.",
    },
    {
        "name": "Video generation",
        "description": "Endpoints that produce a video file. Grouped under each backend's own path prefix.",
    },
    {
        "name": "Audio generation",
        "description": "Endpoints that produce audio only, no video. Grouped under each backend's own path prefix.",
    },
    {
        "name": "Voice conversion",
        "description": "Convert an existing recording to sound like one of a curated set of pre-installed voices, keeping its original delivery intact.",
    },
    {
        "name": "Text to speech",
        "description": "Generate speech from text (English and Chinese) -- design a voice from a description, or clone/direct a reference voice.",
    },
    {
        "name": "Transcription",
        "description": "Speech-to-text for an existing recording (or a video's audio track), with word- and segment-level timestamps.",
    },
    {
        "name": "Jobs",
        "description": "Poll status, download the result, cancel, or permanently delete a job submitted by any backend's generation endpoint.",
    },
]

# ---------------------------------------------------------------------------
# GET /v1/guide itself -- see common/guide.py for how it's actually built,
# and server.py for the route (format=markdown|json, ETag/304).
# ---------------------------------------------------------------------------

GUIDE = """
The single, always-current reference for building a third-party integration against this API --
read it once, and re-fetch it (or watch its `ETag`/`content_hash`) when you upgrade your
integration, rather than keeping your own saved copy. Generated fresh from this server's own live
configuration and `/openapi.json` on first request after each restart, so it cannot drift from what
`/docs` shows the way a hand-written document could.

Two forms of the same content:
- **`?format=markdown`** (default): plain prose, meant to be read directly by a developer or an
  LLM coding agent.
- **`?format=json`**: the same prose (in a `markdown` field) plus a `facts` object -- the same
  information, structured for code to consume without parsing text (upload/retention limits,
  registered backends, recommended polling interval, and more).

Cache using the `ETag` response header (or the JSON body's own `content_hash` field, which is the
same value): send it back as `If-None-Match` on a later request and get a `304` with no body if
nothing has changed, rather than re-fetching and re-parsing the whole guide every time.
"""

GUIDE_RESPONSES = {
    200: {
        "description": "The guide. `text/markdown` by default; `application/json` when `?format=json` is given.",
        "content": {
            "text/markdown": {"schema": {"type": "string"}},
            "application/json": {"schema": GuideResponse.model_json_schema()},
        },
    },
    304: {
        "description": "Not Modified -- the `If-None-Match` request header matched this guide's current ETag. No body.",
    },
}

# ---------------------------------------------------------------------------
# GET /v1/health -- moved here from server.py for consistency with every
# other endpoint's docs, which all live in a *_docs module rather than
# inline. No HEALTH_RESPONSES constant: this handler has no error path at
# all, so there is nothing beyond the default 200 to document.
# ---------------------------------------------------------------------------

HEALTH = """
Always returns `ok` if the API process itself is reachable. Does not indicate anything about
GPU/cluster availability -- a healthy API can still take minutes to actually start a job if the
cluster is busy, or report a job as `queued` for a long time. See `GET /v1/partitions` for live
GPU/queue numbers, and the main guide's "Jobs and timing" section for what to expect while a job is
queued or running.
"""

# ---------------------------------------------------------------------------
# Docs for the common endpoints (uploads, jobs) -- shared by every backend.
# ---------------------------------------------------------------------------

UPLOAD = f"""
Upload a single file (image, audio, or video) and get back an `asset_id` to
reference from any generation request (on any backend) that needs it. One
file per call.

Limit: {guide.STATIC_FACTS['upload_mb']} MB. Uploaded files are **not** automatically deleted
(unlike finished job outputs, which expire after {config.JOB_RETENTION_DAYS:g} days) --
re-uploading the same file again produces a new, additional `asset_id` rather
than deduplicating.

Image format note: images are decoded and forced to RGB (the first 3 channels of
whatever `PIL.Image.open()` returns) -- plain grayscale or palette-mode images
may not convert as expected. Standard RGB JPEG/PNG images work reliably.
"""

PARTITIONS = f"""
Live Slurm partition availability, for deciding which `partition` to pass
(or simply accept the default) on a generation request -- see that field's
own description on every generation request schema.

Every number is a snapshot: cached for up to 20 seconds server-side (so
several callers polling at once don't hammer the Slurm controller directly),
and subject to change the instant after you read it. Free GPUs don't
guarantee a job starts immediately -- a higher-priority job may already be
queued for the same nodes -- and the pre-emptible default partition
(`"{config.FALLBACK_PARTITION}"` right now) can lose a running job to
pre-emption at any time; that job keeps reporting `status: "running"` and
restarts automatically rather than failing.

`reclaimable_gpus` is null for the pre-emptible partition itself, and for
every other partition counts GPUs the pre-emptible partition's own jobs are
currently using there -- GPUs a job submitted to that other, higher-priority
partition could take over immediately by pre-empting them.
"""

JOB_LIST = """
The most recently submitted jobs first, across every backend, capped at
`limit` (default 50, max 500). Each entry has the same fields
`GET /v1/jobs/{job_id}` returns, plus `request` -- the exact body the job was
submitted with (e.g. its `prompt`), so you don't need to have kept your own
record of what a given `job_id` was for.

Only ever shows jobs this server still has a row for -- finished jobs (and
their output files) are deleted automatically per the retention window
described in this API's top-level description, so old jobs eventually drop
off this list on their own, the same as `GET /v1/jobs/{job_id}` would then
404 for them.
"""

JOB_STATUS = """
Current status of a previously submitted job, from any backend's generation
endpoint.

`status` values: `"queued"` (submitted, not yet confirmed running), `"running"`,
`"succeeded"`, `"failed"`. Poll at a relaxed interval regardless -- 10-15
seconds is plenty, since every job takes minutes at minimum, and neither
`progress` nor `typical_run_seconds` below update any faster than this
server's own 5-second background poll loop anyway.

`typical_run_seconds`/`typical_basis` give a rough "usually takes about..."
figure for this exact pipeline -- the median of recent successful runs once
enough history exists, otherwise a documented per-backend estimate. Context
for the elapsed time you can already compute from `created_at`/`started_at`,
not a guarantee.

`progress` gives live, best-effort detail while `status` is `"running"`, for the
subset of backends that report it (see `GET /v1/guide` for which) -- e.g. which
stage it's on, which clip/step, and a rough time-remaining estimate. Every other
pipeline always reports `progress: null`; that's expected, not a sign anything
is stuck.

`started_at` is approximate: it's set the first time the server's background
poller observes the Slurm job as running, not the instant it actually started.

`error` is populated only when `status == "failed"`, and holds a human-readable
reason plus (when available) the tail of the pipeline's own log output --
useful for telling a real failure (bad input, out of memory) apart from a
transient one (pre-emption, a shared-filesystem hiccup) that's simply worth
resubmitting unchanged.
"""

JOB_RESULT = """
Downloads the finished file for a job with `status == "succeeded"`. Returns the raw output file
directly, with a `Content-Disposition` header naming it -- never wrapped in a JSON envelope, even
on the one endpoint below whose own output file happens to be JSON. The exact content type depends
on which endpoint created the job: `video/mp4` for every video-generation endpoint, `audio/x-wav`/
`audio/mpeg`/`audio/flac`/`audio/ogg` for an audio-generation or voice-conversion job (following
that request's own `export_format`/output format), `application/json` for a transcription job (the
transcript itself -- see `POST /v1/parakeet/transcribe`'s own description for its exact shape), or
`application/zip` for a batch job that produces more than one output file.

Returns 409 if the job hasn't succeeded yet (check `GET /v1/jobs/{job_id}` first) and 410 if the
result previously existed but has since expired per the retention policy described in the API's
top-level description.
"""

CANCEL_JOB = """
Best-effort cancellation of a job that hasn't finished yet. If the job already
succeeded or failed, this is a no-op (`cancelled: false` in the response) rather
than an error -- safe to call even if you're unsure of the current status. A
cancelled job subsequently reports `status: "failed"`, not a distinct
"cancelled" status.

This only stops the job -- it doesn't delete anything. See
`DELETE /v1/jobs/{job_id}/purge` to permanently remove a finished job's files
and database row.
"""

PURGE_JOB = f"""
Permanently deletes a job that has already finished (`succeeded` or `failed`)
-- its own directory (output file, logs, and any intermediate files), its
database row, and any uploaded input it referenced that no other remaining
job still uses. This cannot be undone.

A queued or running job cannot be purged directly -- cancel it first with
`DELETE /v1/jobs/{{job_id}}`, then purge it once it reports `status: "failed"`.

`removed_asset_ids` in the response lists which of this job's uploaded inputs
were deleted along with it; `kept_asset_ids` lists which ones were left alone
because at least one other job still references them.

Distinct from the automatic {config.JOB_RETENTION_DAYS:g}-day cleanup described
in this API's top-level description -- this deletes on request, immediately,
regardless of how recently the job finished.
"""

# ---------------------------------------------------------------------------
# Shared `responses=` documentation (example error bodies per failure mode),
# reused by every backend's generation endpoints.
# ---------------------------------------------------------------------------

_VALIDATION_ERROR_RESPONSE = {
    422: {
        "description": "Request failed schema validation (e.g. conflicting fields, out-of-range values).",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "type": "value_error",
                            "loc": ["body"],
                            "msg": "Value error, Use either 'orientation' or explicit 'height'/'width', not both.",
                            "input": {},
                        }
                    ]
                }
            }
        },
    },
}

_ASSET_ERROR_RESPONSE = {
    400: {
        "description": "A referenced asset_id doesn't exist, or its uploaded file is missing on disk.",
        "content": {
            "application/json": {"example": {"detail": "Invalid asset reference: 'Unknown asset_id: abc123'"}}
        },
    },
}

_CAPACITY_RESPONSE = {
    429: {
        "description": "The optional soft concurrency cap (MODEL_API_MAX_INFLIGHT) was reached; unset by default.",
        "content": {
            "application/json": {"example": {"detail": "At capacity: 4 job(s) already in flight. Try again shortly."}}
        },
    },
}

_DISPATCH_ERROR_RESPONSE = {
    502: {
        "description": (
            "The cluster's own job scheduler (Slurm) rejected the job after it passed this API's own "
            "checks -- a scheduling-side issue, not a problem with your request. Safe to retry; worth "
            "reporting if it persists."
        ),
        "content": {
            "application/json": {"example": {"detail": "Could not submit the job to Slurm: <scheduler error detail>"}}
        },
    },
}

_JOB_NOT_FOUND_RESPONSE = {
    404: {
        "description": "No job with this job_id exists.",
        "content": {"application/json": {"example": {"detail": "Unknown job_id."}}},
    },
}

SUBMIT_RESPONSES = {
    **_VALIDATION_ERROR_RESPONSE,
    **_ASSET_ERROR_RESPONSE,
    **_CAPACITY_RESPONSE,
    **_DISPATCH_ERROR_RESPONSE,
}

UPLOAD_RESPONSES = {
    413: {
        "description": f"File exceeds the {config.MAX_UPLOAD_BYTES} byte limit.",
        "content": {
            "application/json": {
                "example": {"detail": f"File exceeds the {config.MAX_UPLOAD_BYTES} byte limit."}
            }
        },
    },
}

JOB_LIST_RESPONSES: dict[int, dict] = {}

PARTITIONS_RESPONSES = {
    503: {
        "description": "The live cluster-scheduler status couldn't be read just now, and there's no previous snapshot to fall back to yet. Transient -- retry shortly.",
        "content": {
            "application/json": {
                "example": {"detail": "Partition status is temporarily unavailable: <scheduler query error>"}
            }
        },
    },
}

JOB_STATUS_RESPONSES = {
    **_JOB_NOT_FOUND_RESPONSE,
}

JOB_RESULT_RESPONSES = {
    200: {
        "description": "The finished file, streamed directly (never JSON-wrapped). The exact content type depends on which endpoint created the job -- see this endpoint's own description above.",
        "content": {
            "video/mp4": {},
            "audio/x-wav": {},
            "audio/mpeg": {},
            "audio/flac": {},
            "audio/ogg": {},
            "application/json": {},
            "application/zip": {},
        },
    },
    **_JOB_NOT_FOUND_RESPONSE,
    409: {
        "description": "The job exists but hasn't finished successfully yet.",
        "content": {
            "application/json": {
                "example": {"detail": "Job is not finished successfully yet (status: running)."}
            }
        },
    },
    410: {
        "description": "The result file existed but has since expired / been cleaned up.",
        "content": {
            "application/json": {"example": {"detail": "Result file is no longer available (it may have expired)."}}
        },
    },
}

CANCEL_RESPONSES = {
    **_JOB_NOT_FOUND_RESPONSE,
}

PURGE_RESPONSES = {
    **_JOB_NOT_FOUND_RESPONSE,
    409: {
        "description": "The job is still queued/running (cancel it first), or its directory could not be fully removed yet (safe to retry).",
        "content": {
            "application/json": {
                "example": {"detail": "Job is still queued/running -- cancel it first (DELETE /v1/jobs/{job_id})."}
            }
        },
    },
    500: {
        "description": "A safety check refused to delete the job's directory (should not normally happen); nothing was deleted.",
        "content": {
            "application/json": {"example": {"detail": "Refusing to delete unexpected job directory: ..."}}
        },
    },
}
