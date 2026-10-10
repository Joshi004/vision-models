"""Pydantic pieces shared across every backend: the optional `partition`
field every generation request gets, and the response models backing this
API's common endpoints (uploads, jobs, health) -- none of these are
specific to any one backend, unlike services/<name>/schemas.py's request
bodies.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PartitionOptionMixin(BaseModel):
    """Adds an optional `partition` field to a generation request. Shared by
    every backend so the override behaves identically everywhere -- see
    common/slurm.py::resolve_partition for the exact fallback rule."""

    partition: str | None = Field(
        None,
        description=(
            "Slurm partition to submit this job to (e.g. 'main', 'health'). Optional -- if "
            "omitted, or if the named partition doesn't currently exist on this cluster, the "
            "job silently falls back to the API's default partition instead of erroring (see "
            "GET /v1/health for which one is currently configured). See GET /v1/partitions for "
            "each partition's live GPU availability and queue length before choosing one."
        ),
        examples=[None],
    )


class PartitionStatus(BaseModel):
    """One partition's live availability -- one entry in the response from
    GET /v1/partitions. Every number here is a snapshot, not a promise --
    see that endpoint's own description for exactly how fresh it is and
    why it can change the instant after you read it."""

    name: str = Field(
        ..., description="Slurm partition name, exactly as you'd pass it in a generation request's `partition` field.", examples=["background"]
    )
    is_default: bool = Field(
        ...,
        description="True for the one partition a generation request falls back to when it omits `partition` (see GET /v1/health's own default_partition).",
    )
    preemptible: bool = Field(
        ...,
        description=(
            "True if jobs submitted here can be pre-empted (killed and requeued from scratch) by a job "
            "submitted to a higher-priority partition -- true only for the API's default partition "
            "today. A pre-empted job is not a failure: this API's own poller keeps reporting it as "
            "'running' while Slurm automatically restarts it."
        ),
        examples=[True],
    )
    state: str = Field(..., description="This partition's own Slurm state (normally 'UP').", examples=["UP"])
    nodes: int = Field(..., description="Number of nodes that belong to this partition.", examples=[150])
    total_gpus: int = Field(..., description="Total GPUs across every node in this partition, regardless of current use.", examples=[1200])
    free_gpus: int = Field(
        ...,
        description=(
            "GPUs on this partition's nodes that are idle right now. A job can only start immediately "
            "if enough of these sit on a single node -- every backend here requests its GPU(s) from one "
            "node, never split across several."
        ),
        examples=[15],
    )
    reclaimable_gpus: int | None = Field(
        None,
        description=(
            "GPUs on this partition's nodes currently held by the pre-emptible partition's own jobs, "
            "which a job submitted here could reclaim by pre-empting them. Null for the pre-emptible "
            "partition itself, where this concept doesn't apply."
        ),
        examples=[393],
    )
    waiting_jobs: int = Field(
        ...,
        description=(
            "Jobs (from any user or tool on this shared cluster, not just this API) currently queued "
            "for this partition's GPUs, excluding ones held by their owner/an admin or blocked on a "
            "dependency -- neither of those is actually waiting on GPU availability."
        ),
        examples=[12],
    )


class PartitionStatusResponse(BaseModel):
    """Response from GET /v1/partitions."""

    partitions: list[PartitionStatus] = Field(
        ..., description="Every partition this API offers through its `partition` field, most relevant (the current default) first."
    )
    generated_at: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of when this snapshot was taken. Cached briefly server-side (see the endpoint description), so it may be a few seconds old.",
        examples=["2026-09-30T17:24:23.314768+00:00"],
    )


class UploadResponse(BaseModel):
    """Response from POST /v1/uploads."""

    asset_id: str = Field(..., description="Opaque id to reference this file from a generation request.", examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"])
    filename: str = Field(..., description="Original filename as sent by the client (for your own reference only).", examples=["reference_photo.jpg"])
    size_bytes: int = Field(..., description="Size of the uploaded file in bytes.", examples=[184320])


class JobSubmitResponse(BaseModel):
    """Response from every generation endpoint, on every backend."""

    job_id: str = Field(..., description="Id to poll GET /v1/jobs/{job_id} and later download GET /v1/jobs/{job_id}/result.", examples=["6b54a0bd6cef4b04abe50a8f81f55a80"])
    status: str = Field(..., description="Always 'queued' immediately after submission.", examples=["queued"])


class JobStage(BaseModel):
    """One step of a multi-step pipeline (e.g. Wan-Animate's preprocess ->
    generate -> mux_audio) -- part of JobProgress.stages."""

    name: str = Field(..., description="Stage name, e.g. 'preprocess', 'generate', 'mux_audio'.", examples=["generate"])
    state: Literal["done", "running", "pending"] = Field(
        ..., description="Whether this stage has finished, is currently running, or hasn't started yet."
    )
    seconds: float | None = Field(
        None,
        description="Wall-clock seconds this stage took, once it's done. Null while running or pending.",
        examples=[146.4],
    )


class JobProgress(BaseModel):
    """Live progress for a running job, read directly from its own log file
    -- only some backends populate this (see GET /v1/guide for which);
    every pipeline that doesn't always reports `progress: null` on
    JobStatusResponse instead, which is expected, not a sign anything is
    stuck.

    Best-effort by design: this is a reading of that backend's own log
    output, not a documented interface it promises to keep stable. If that
    format ever changes, individual fields quietly go back to null instead
    of this request failing."""

    stages: list[JobStage] = Field(
        ..., description="Every stage this job's recipe runs through, in order, each with its own done/running/pending state."
    )
    current_stage: str | None = Field(
        None,
        description="Name of the stage currently running, or null if none is (queued, or every stage already finished).",
        examples=["generate"],
    )
    clip: int | None = Field(
        None,
        description="Which clip (1-indexed) is currently being denoised, for recipes that process a driving video in clips. Null outside that phase (e.g. still preprocessing or loading the model).",
        examples=[3],
    )
    clip_count: int | None = Field(
        None, description="Total number of clips this job will process. Null until that's known.", examples=[6]
    )
    step: int | None = Field(None, description="Denoising step reached within the current clip.", examples=[12])
    step_count: int | None = Field(None, description="Total denoising steps per clip.", examples=[20])
    eta_seconds: float | None = Field(
        None,
        description=(
            "Estimated seconds remaining, from the current clip/step and this job's own measured "
            "seconds-per-step -- covers sampling time only, not per-clip save/offload overhead, so "
            "treat it as a rough 'about' figure, not a countdown to the second. Null when there isn't "
            "enough information yet."
        ),
        examples=[840.0],
    )


class JobStatusResponse(BaseModel):
    """Response from GET /v1/jobs/{job_id}."""

    job_id: str = Field(..., description="Echoes the job_id you polled for.", examples=["6b54a0bd6cef4b04abe50a8f81f55a80"])
    pipeline: str = Field(
        ...,
        description="Which backend/recipe created this job, as '<backend>:<recipe>'.",
        examples=[
            "ltx:text-to-video",
            "ltx:keyframe-interpolation",
            "ltx:audio-to-video",
            "ltx:retake",
            "ltx:text-to-audio",
            "ltx25:text-to-video",
            "ltx25:interpolate",
            "ltx25:retake",
            "wan-animate:replace",
            "rvc:convert",
            "rvc:batch-convert",
            "parakeet:transcribe",
            "breeze-tts:synthesize",
        ],
    )
    status: Literal["queued", "running", "succeeded", "failed"] = Field(
        ...,
        description=(
            "queued -> running -> succeeded|failed. A cancelled job reports 'failed', not a separate "
            "status. See `progress` for finer-grained detail within 'running' -- only some backends "
            "populate it (see GET /v1/guide for which); every other pipeline always reports it as "
            "null, which is expected, not a sign of a stuck job."
        ),
    )
    partition: str | None = Field(
        None,
        description="The Slurm partition this job actually submitted to, after the partition fallback rule was applied.",
        examples=["background"],
    )
    slurm_job_id: str | None = Field(
        None,
        description="The underlying Slurm job id. Mainly useful if you (or whoever operates this deployment) need to cross-reference this job directly against the cluster's own scheduler.",
        examples=["418249"],
    )
    created_at: str = Field(..., description="ISO 8601 UTC timestamp when the job was submitted.", examples=["2026-09-24T19:23:17.095881+00:00"])
    started_at: str | None = Field(
        None,
        description="ISO 8601 UTC timestamp of the first poll that observed the job running. Approximate, not the exact Slurm start time.",
        examples=[None],
    )
    finished_at: str | None = Field(None, description="ISO 8601 UTC timestamp when the job reached succeeded/failed.", examples=[None])
    error: str | None = Field(
        None,
        description="Populated only when status='failed': a human-readable reason, plus (when available) the tail of the pipeline's own log output.",
        examples=[None],
    )
    result_ready: bool = Field(
        False, description="True iff status='succeeded' and the output file is available for download right now."
    )
    typical_run_seconds: float | None = Field(
        None,
        description=(
            "How long this exact pipeline typically takes end to end (started_at to finished_at), for "
            "context alongside the elapsed time you can already compute from created_at/started_at. "
            "See typical_basis for where this number came from. Null only if a pipeline has neither "
            "history nor a documented fallback, which shouldn't normally happen."
        ),
        examples=[360.0],
    )
    typical_basis: str | None = Field(
        None,
        description=(
            "Where typical_run_seconds came from: 'median of N recent run(s)' once enough succeeded "
            "jobs of this exact pipeline exist, otherwise 'documented estimate' (a fixed per-backend "
            "fallback)."
        ),
        examples=["median of 5 recent run(s)"],
    )
    progress: JobProgress | None = Field(
        None,
        description=(
            "Live progress while status='running', for the subset of backends that report it (see "
            "GET /v1/guide for which). Null for every other backend's pipelines, and for "
            "queued/finished jobs either way -- rely on typical_run_seconds and the elapsed time "
            "instead when this is null."
        ),
    )


class JobListItem(JobStatusResponse):
    """One entry in the response from GET /v1/jobs -- everything
    JobStatusResponse has, plus the request body the job was submitted
    with."""

    request: dict = Field(
        ...,
        description="The exact request body this job was submitted with, for context alongside its status (e.g. its prompt).",
        examples=[{"prompt": "A golden retriever puppy runs across a sunlit lawn.", "mode": "fast", "seed": 10}],
    )


class JobListResponse(BaseModel):
    """Response from GET /v1/jobs."""

    jobs: list[JobListItem] = Field(
        ..., description="Most recently submitted jobs first, across every backend, capped at the requested `limit`."
    )
    server_time: str = Field(
        ...,
        description=(
            "This server's own current UTC time (ISO 8601), captured when this response was built. "
            "Compute elapsed/remaining time against this instead of the caller's own clock, which may "
            "be skewed."
        ),
        examples=["2026-09-27T19:23:17.095881+00:00"],
    )


class HealthResponse(BaseModel):
    """Response from GET /v1/health."""

    status: Literal["ok"] = Field("ok", description="Always 'ok' if the API process is reachable at all.", examples=["ok"])
    default_partition: str = Field(
        ...,
        description="The Slurm partition a generation request falls back to when it omits `partition`, or names one that doesn't currently exist.",
        examples=["background"],
    )
    job_retention_days: float = Field(
        ...,
        description=(
            "Finished jobs (and their output files) are deleted automatically this many days after "
            "they finish. A finished job can also be deleted immediately on request -- see "
            "DELETE /v1/jobs/{job_id}/purge."
        ),
        examples=[7.0],
    )


class CancelResponse(BaseModel):
    """Response from DELETE /v1/jobs/{job_id}."""

    cancelled: bool = Field(..., description="True if a queued/running job was cancelled; false if it had already finished (not an error).")
    reason: str | None = Field(None, description="Present only when cancelled=false, explaining why.", examples=["job already succeeded"])


class PurgeResponse(BaseModel):
    """Response from DELETE /v1/jobs/{job_id}/purge."""

    job_id: str = Field(..., description="Echoes the job_id that was purged.", examples=["6b54a0bd6cef4b04abe50a8f81f55a80"])
    removed_asset_ids: list[str] = Field(
        ...,
        description=(
            "Uploaded asset_ids this job referenced that were deleted along with it, because no other "
            "remaining job still uses them."
        ),
        examples=[["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"]],
    )
    kept_asset_ids: list[str] = Field(
        ...,
        description=(
            "Uploaded asset_ids this job referenced that were left alone, because at least one other "
            "remaining job still uses them."
        ),
        examples=[[]],
    )


# ---------------------------------------------------------------------------
# GET /v1/guide -- see common/guide.py for how these are actually built.
# Split into small, focused models (rather than one flat dict) so each
# fact has its own description here, visible to anyone inspecting the
# schema directly instead of only in prose.
# ---------------------------------------------------------------------------


class LimitFacts(BaseModel):
    """Numeric limits for GET /v1/guide, read live from this server's own
    configuration -- never hardcoded, so they can't drift from reality."""

    max_upload_mb: int = Field(..., description="Maximum size, in MB, of a single file sent to POST /v1/uploads.", examples=[2048])
    job_retention_days: float = Field(
        ..., description="Finished jobs (and their output files) are deleted automatically this many days after they finish.", examples=[7.0]
    )
    max_inflight_jobs: int | None = Field(
        None,
        description="Optional soft cap on jobs in flight across every backend at once. Null if this deployment has no fixed cap.",
        examples=[None],
    )


class PollingFacts(BaseModel):
    """Recommended polling behavior for GET /v1/guide."""

    recommended_interval_seconds: list[int] = Field(
        ...,
        description="Suggested [min, max] seconds between GET /v1/jobs/{job_id} polls while a job is queued/running.",
        examples=[[10, 15]],
    )
    webhooks: bool = Field(
        ..., description="True if this API can push job-completion notifications instead of being polled. Always false today.", examples=[False]
    )


class PartitionFacts(BaseModel):
    """Default-partition facts for GET /v1/guide."""

    default: str = Field(
        ...,
        description="The Slurm partition a generation request falls back to when it omits partition, or names one that doesn't currently exist.",
        examples=["background"],
    )
    live_status_endpoint: str = Field(
        "/v1/partitions", description="Call this endpoint for live free-GPU/queue-length numbers before choosing a partition."
    )


class BackendFacts(BaseModel):
    """One registered backend, as listed in GET /v1/guide."""

    id: str = Field(..., description="Short id used internally and in a job's own 'pipeline' field (as '<id>:<recipe>').", examples=["ltx"])
    name: str = Field(..., description="Human-readable backend name.", examples=["LTX-2.3"])
    prefix: str = Field(..., description="Path prefix every one of this backend's own endpoints is mounted under.", examples=["/v1/ltx"])


class GuideFacts(BaseModel):
    """Machine-readable facts bundled with GET /v1/guide?format=json -- the
    same facts the prose repeats, structured for code instead of a human to
    read."""

    base_url: str = Field(
        ..., description="This API's own base URL, exactly as the request that fetched this guide reached it.", examples=["http://localhost:8012"]
    )
    limits: LimitFacts
    polling: PollingFacts
    partitions: PartitionFacts
    backends: list[BackendFacts] = Field(..., description="Every backend currently registered behind this API.")


class GuideResponse(BaseModel):
    """Response from GET /v1/guide?format=json -- the full integration
    guide, plus the same facts in machine-readable form, in one call."""

    api_version: str = Field(
        ...,
        description="This API's own version string -- bump on any integrator-visible change; see the guide's own changelog section.",
        examples=["0.2.0"],
    )
    generated_at: str = Field(
        ..., description="ISO 8601 UTC timestamp of when this response was generated.", examples=["2026-10-04T10:15:00+00:00"]
    )
    content_hash: str = Field(
        ...,
        description=(
            "Hash of this guide's own content, independent of base_url/generated_at -- unchanged "
            "means nothing about the guide's substance has changed since you last read it. Same "
            "value as this response's own ETag header."
        ),
        examples=["sha256:3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d"],
    )
    facts: GuideFacts
    markdown: str = Field(..., description="The exact same guide GET /v1/guide (without ?format=json) returns as plain text/markdown.")
