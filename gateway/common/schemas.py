"""Pydantic pieces shared across every backend: the optional `partition`
field every generation request gets, and the response models backing this
API's common endpoints (uploads, jobs, health) -- none of these are
specific to any one backend, unlike services/<name>/schemas.py's request
bodies.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from common import job_store


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


class IncognitoOptionMixin(BaseModel):
    """Adds an optional `incognito` field to a generation request. Shared by
    every backend so the opt-out behaves identically everywhere -- see
    common/retention.py for exactly what is deleted, and when."""

    incognito: bool = Field(
        False,
        description=(
            "If true, this job is not kept any longer than needed: no log is stored, its uploaded "
            "inputs are deleted as soon as the job ends, its result can be downloaded only once (the "
            "file is deleted right after it has been sent in full -- download it before it is gone), "
            "and once it is done only a minimal record remains (status, pipeline, timings; `request` "
            "becomes `{}`). If false (the default), everything about this job -- prompt/settings, "
            "inputs, logs and output -- is kept until it is deleted by hand "
            "(DELETE /v1/jobs/{job_id}/purge) or `job_retention_days` pass (see GET /v1/health)."
        ),
        examples=[False],
    )


class PriorityOptionMixin(BaseModel):
    """Adds an optional `priority` field to a generation request. Shared by every backend so
    the field is accepted everywhere; see common/worker_pool.py and common/capacity.py for how
    it is used."""

    priority: int = Field(
        job_store.DEFAULT_PRIORITY,
        ge=job_store.MIN_PRIORITY,
        le=job_store.MAX_PRIORITY,
        description=(
            "How urgent this job is, from 1 (most urgent) to 10 (least urgent). Optional -- if omitted, "
            "the job gets 10. When jobs wait for a GPU, the more urgent one is served first, "
            "across models too; jobs with the same priority are served in the order they were submitted. "
            "A queued job's priority can be changed later with PUT /v1/jobs/{job_id}/priority. "
            "Not used for ordering by LTX-2.3, LTX-2.5, Wan-Animate, RVC and TBDub: those jobs are "
            "accepted with this field but always count as 10 when they compete for GPUs."
        ),
        examples=[10],
    )


class JobPriorityUpdate(BaseModel):
    """Body of PUT /v1/jobs/{job_id}/priority."""

    priority: int = Field(
        ...,
        ge=job_store.MIN_PRIORITY,
        le=job_store.MAX_PRIORITY,
        description="The new priority, from 1 (most urgent) to 10 (least urgent).",
        examples=[1],
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


QueueState = Literal["scheduling", "model_limit", "gpu_limit", "priority", "slurm_pending", "model_loading"]
_QUEUE_STATE_DOC = (
    "'scheduling': the gateway has not looked at it yet. "
    "'model_limit': its model is at its own max jobs (for Qwen3.8: every request slot of the server is in use). "
    "'gpu_limit': the global GPU limit is reached. "
    "'priority': a more urgent job of another model is waiting for the free GPUs first. "
    "'slurm_pending': waiting for the cluster (Slurm) to allocate the GPUs. "
    "'model_loading': GPUs are allocated and the worker or server is loading the model."
)


class CapacityModel(BaseModel):
    """One model's limit and usage -- one entry in the response from GET /v1/capacity."""

    key: str = Field(..., description="Id to use in a PUT /v1/capacity `max_jobs` entry.", examples=["h3"])
    label: str = Field(..., description="Display name.", examples=["MiniMax H3"])
    kind: Literal["pool", "oneshot", "fixed"] = Field(
        ...,
        description=(
            "'pool': served by warm workers that load the model once. 'oneshot': one Slurm job per request. "
            "'fixed': a single shared server with no limit of its own (Qwen3.8); its GPU still counts toward the global limit."
        ),
    )
    max_jobs: int | None = Field(
        None,
        description="The most jobs of this model that run at once (for 'pool', also the most warm workers). Null for 'fixed'.",
        examples=[8],
    )
    planning_gpus: int = Field(..., description="GPUs one job is counted as when the limits are validated.", examples=[4])
    gpus_held: int = Field(
        ...,
        description="GPUs this model holds right now, allocated or still pending in Slurm (what counts toward the global limit).",
        examples=[8],
    )
    gpus_allocated: int = Field(..., description="Of `gpus_held`, the GPUs Slurm has allocated (jobs and workers running).", examples=[4])
    gpus_pending: int = Field(..., description="Of `gpus_held`, the GPUs still waiting in Slurm for an allocation.", examples=[4])
    running_jobs: int = Field(..., description="Jobs of this model running right now.", examples=[1])
    workers_pending: int = Field(..., description="Workers whose Slurm job is still waiting for a GPU ('pool' only).", examples=[1])
    workers_loading: int = Field(..., description="Workers that have a GPU and are loading the model ('pool' only).", examples=[0])
    queued_by_reason: dict[str, int] = Field(
        ...,
        description=f"Queued jobs of this model, counted by why they wait (a `queue_state` value). {_QUEUE_STATE_DOC}",
        examples=[{"slurm_pending": 1, "model_limit": 2}],
    )
    busy: int = Field(..., description="Workers running a request ('pool'), or running jobs ('oneshot').", examples=[1])
    idle: int = Field(..., description="Warm workers waiting for a request.", examples=[0])
    starting: int = Field(..., description="Workers still starting or loading the model.", examples=[1])
    queued: int = Field(..., description="Requests waiting for a worker ('pool'), or jobs waiting in Slurm ('oneshot').", examples=[3])


class CapacityResponse(BaseModel):
    """Response from GET /v1/capacity and PUT /v1/capacity."""

    global_gpus: int = Field(..., description="The most GPUs this service may hold at once, across every model.", examples=[30])
    gpus_in_use: int = Field(..., description="GPUs held right now by all models together.", examples=[20])
    planned_gpus: int = Field(
        ...,
        description="What the per-model limits add up to (each job counted at its model's `planning_gpus`, plus the Qwen3.8 server). Never more than `global_gpus`.",
        examples=[28],
    )
    gpus_allocated: int = Field(..., description="Of `gpus_in_use`, the GPUs Slurm has allocated.", examples=[16])
    gpus_pending: int = Field(..., description="Of `gpus_in_use`, the GPUs still waiting in Slurm for an allocation.", examples=[4])
    running_jobs: int = Field(..., description="Jobs running right now, all models together.", examples=[4])
    queued_jobs: int = Field(..., description="Jobs queued right now, all models together.", examples=[7])
    queued_by_reason: dict[str, int] = Field(
        ...,
        description=f"Queued jobs of all models, counted by why they wait (a `queue_state` value). {_QUEUE_STATE_DOC}",
        examples=[{"slurm_pending": 3, "model_limit": 4}],
    )
    models: list[CapacityModel] = Field(..., description="Every model with its limit and current usage.")


class CapacityUpdateRequest(BaseModel):
    """Body of PUT /v1/capacity."""

    global_gpus: int = Field(..., description="The new global GPU limit (1 or more).", examples=[30])
    max_jobs: dict[str, int] = Field(
        default_factory=dict,
        description="New `max_jobs` (1 or more) for the models named here, by `key`; models not named keep their limit.",
        examples=[{"h3": 4, "parakeet": 2}],
    )


class UploadResponse(BaseModel):
    """Response from POST /v1/uploads."""

    asset_id: str = Field(..., description="Opaque id to reference this file from a generation request.", examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"])
    filename: str = Field(..., description="Original filename as sent by the client (for your own reference only).", examples=["reference_photo.jpg"])
    size_bytes: int = Field(..., description="Size of the uploaded file in bytes.", examples=[184320])
    sha256: str = Field(
        ...,
        description=(
            "Lowercase hex SHA-256 of the file as the server stored it. Compare it (and `size_bytes`) with your own "
            "to confirm the whole file arrived intact; also usable with POST /v1/uploads/by-hash next time."
        ),
        examples=["9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"],
    )
    reused: bool = Field(
        False,
        description=(
            "True only in a POST /v1/uploads/by-hash answer: the file was already stored, this is its existing "
            "asset_id, and nothing was uploaded. Always false from POST /v1/uploads."
        ),
        examples=[False],
    )


class UploadByHashRequest(BaseModel):
    """Request body of POST /v1/uploads/by-hash."""

    sha256: str = Field(
        ...,
        description="Lowercase or uppercase hex SHA-256 (64 characters) of the file you would otherwise upload.",
        examples=["9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"],
    )

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        value = value.strip().lower()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("sha256 must be 64 hexadecimal characters.")
        return value


class JobSubmitResponse(BaseModel):
    """Response from every generation endpoint, on every backend."""

    job_id: str = Field(..., description="Id to poll GET /v1/jobs/{job_id} and later download GET /v1/jobs/{job_id}/result.", examples=["6b54a0bd6cef4b04abe50a8f81f55a80"])
    status: str = Field(
        ...,
        description=(
            "'queued' for a new job. When `reused` is true, the current status of the earlier job that was "
            "found again: 'queued', 'running' or 'succeeded'."
        ),
        examples=["queued"],
    )
    reused: bool = Field(
        False,
        description=(
            "True if no new job was created because an identical request (same settings, same uploaded file "
            "content) from before is still queued or running, or has finished and its result is still "
            "available; `job_id` is then that earlier job. Only the warm-pool backends do this, plus Qwen3.8 "
            "for a request that carries the same Idempotency-Key header. Always false for a request with "
            "`incognito: true`."
        ),
        examples=[False],
    )
    gpus: int | None = Field(
        None,
        description=(
            "How many GPUs this job was given, chosen from the size of the request. Only backends that size "
            "their jobs report it (MiniMax H3 today); null for every other backend."
        ),
        examples=[4],
    )
    estimated_run_seconds: float | None = Field(
        None,
        description=(
            "Estimated seconds this job will run once it starts (loading the model through writing the video; "
            "the time waiting in the queue is not included), worked out from the size of the request. A rough "
            "'about' figure from formulas that are still being calibrated, not a promise. Only backends that "
            "size their jobs report it (MiniMax H3 today); null for every other backend."
        ),
        examples=[805.0],
    )


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


class LipSyncResult(BaseModel):
    """What happened to the optional lip-sync step of an LTX audio-to-video
    job submitted with `lip_sync: true`."""

    requested: bool = Field(True, description="Always true when this object is present.")
    applied: bool | None = Field(
        None,
        description=(
            "true: the result is lip-synced. false: lip sync did not happen and the result is the video "
            "WITHOUT lip sync (see `error`; if the job itself failed there is no result at all). "
            "null: not known yet, the job is still queued or running."
        ),
        examples=[True],
    )
    error: str | None = Field(
        None,
        description=(
            "Why lip sync was not applied (only when `applied` is false), with the end of the lip-sync log. "
            "The job still reports status 'succeeded' in that case. To retry only the sync without "
            "regenerating the video, call POST /v1/tbdub/videos/lip-sync with this job's id as `source_job_id`."
        ),
        examples=[None],
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
            "ltx25:audio-to-video",
            "ltx25:retake",
            "wan-animate:replace",
            "rvc:convert",
            "rvc:batch-convert",
            "parakeet:transcribe",
            "breeze-tts:synthesize",
            "qwen-image21:generate",
            "qwen-image21:edit",
            "tbdub:lip-sync",
            "h3:generate",
            "h3:audio-to-video",
            "ace-step15:generate",
            "ace-step15:cover",
            "ace-step15:repaint",
            "stable-audio-open:generate",
            "stable-audio-open:transform",
            "stable-audio3:generate",
            "stable-audio3:transform",
            "stable-audio3:inpaint",
            "yue2:generate",
            "yue2:cover",
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
    downloaded_at: str | None = Field(
        None,
        description=(
            "ISO 8601 UTC timestamp when the result was downloaded in full and deleted from the server. "
            "Only ever set for a job submitted with `incognito: true`. "
            "Once set, GET /v1/jobs/{job_id}/result returns 410."
        ),
        examples=[None],
    )
    error: str | None = Field(
        None,
        description=(
            "Populated only when status='failed': a human-readable reason, plus (when available) the tail "
            "of the pipeline's own log output -- or, for a job submitted with `incognito: true`, "
            "just the reason and the last log line."
        ),
        examples=[None],
    )
    incognito: bool = Field(
        False,
        description=(
            "True if this job was submitted with `incognito: true`: its data is deleted as soon as "
            "it is no longer needed, and only a minimal record of it (this one) remains. False: "
            "everything about the job is kept until it is purged or `job_retention_days` pass."
        ),
        examples=[False],
    )
    priority: int = Field(
        10,
        description=(
            "The job's priority, from 1 (most urgent) to 10 (least urgent, the default). See the "
            "`priority` field of a generation request."
        ),
        examples=[10],
    )
    priority_editable: bool = Field(
        False,
        description=(
            "True if PUT /v1/jobs/{job_id}/priority would change this job right now: it is still queued "
            "and belongs to a backend that orders its queue by priority (every backend except LTX-2.3, "
            "LTX-2.5, Wan-Animate, RVC and TBDub)."
        ),
        examples=[False],
    )
    queue_state: QueueState | None = Field(
        None,
        description=f"Only while status='queued': why the job is not running yet, as of the last scheduling pass. {_QUEUE_STATE_DOC} Null for any other status.",
        examples=["slurm_pending"],
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
    gpus: int | None = Field(
        None,
        description=(
            "How many GPUs this job was given (decided when it was submitted, from the size of the request). "
            "Only backends that size their jobs report it (MiniMax H3 today); null for every other backend "
            "and for jobs submitted before this field existed."
        ),
        examples=[4],
    )
    estimated_run_seconds: float | None = Field(
        None,
        description=(
            "The estimate, made when the job was submitted, of how long it runs once it starts (the queue wait "
            "is not included). Prefer it over typical_run_seconds when present: it accounts for this request's "
            "own size, where typical_run_seconds is the median of whatever ran before. A rough 'about' figure from "
            "formulas that are still being calibrated. Only backends that size their jobs report it (MiniMax H3 "
            "today); null for every other backend and for older jobs."
        ),
        examples=[805.0],
    )
    attempts: int = Field(
        1,
        description=(
            "How many times this job has been started. It is above 1 when the model worker (or model "
            "server) running it was lost, for example pre-empted, and the job was put in the queue again "
            "under the same job_id. A job that fails because of its own request (a model error) is not "
            "run again, except that a MiniMax H3 job that ran out of GPU memory on 4 GPUs is run once "
            "more on 8 (then 'gpus' is 8 and this number is one higher)."
        ),
        examples=[1],
    )
    progress: JobProgress | None = Field(
        None,
        description=(
            "Live progress while status='running', for the subset of backends that report it (see "
            "GET /v1/guide for which). Null for every other backend's pipelines, and for "
            "queued/finished jobs either way -- rely on typical_run_seconds and the elapsed "
            "time instead when this is null."
        ),
    )
    lip_sync: LipSyncResult | None = Field(
        None,
        description=(
            "Only for an audio-to-video job submitted with `lip_sync: true` (null for every other job, "
            "including the standalone lip-sync endpoint, whose failure is an ordinary failed job). "
            "Check `lip_sync.applied` on a succeeded job: if it is false, the downloaded video has no "
            "lip sync and `lip_sync.error` says why."
        ),
    )
    source: Literal["ui", "api"] | None = Field(
        None,
        description=(
            "Who submitted the job: 'ui' if the request carried the header `X-Job-Source: ui` (the web UI "
            "sends it), 'api' for any other client. Null for jobs submitted before this field existed. "
            "A label only, not authentication: any client can send that header."
        ),
        examples=["api"],
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


class JobPageResponse(JobListResponse):
    """Response from POST /v1/jobs/search: one page of the jobs that match the filters."""

    total: int = Field(..., description="How many jobs match the filters in all, across every page.", examples=[137])
    offset: int = Field(..., description="Echoes the requested `offset`.", examples=[0])
    limit: int = Field(..., description="Echoes the requested `limit`.", examples=[50])


_JobIdList = list[Annotated[str, Field(max_length=64)]]


class JobSearchRequest(BaseModel):
    """Body of POST /v1/jobs/search. Every filter is optional; the ones given are combined with AND."""

    offset: int = Field(0, ge=0, description="How many matching jobs to skip (newest first).", examples=[0])
    limit: int = Field(50, ge=1, le=500, description="Page size.", examples=[50])
    statuses: list[Literal["queued", "running", "succeeded", "failed"]] | None = Field(
        None, max_length=4, description="Only jobs in one of these states."
    )
    pipelines: list[Annotated[str, Field(max_length=100)]] | None = Field(
        None,
        max_length=200,
        description="Only jobs of one of these pipelines, each exactly as in `pipeline` (e.g. 'ltx:text-to-video').",
    )
    source: Literal["ui", "api", "unknown"] | None = Field(
        None,
        description="Only jobs submitted from the web UI ('ui'), by another client ('api'), or before `source` was recorded ('unknown').",
    )
    job_ids: _JobIdList | None = Field(
        None,
        max_length=10000,
        description="Only these jobs (an empty list matches nothing). The web UI sends the jobs it has cached for its 'cached' filter.",
    )
    exclude_job_ids: _JobIdList | None = Field(
        None, max_length=10000, description="Leave these jobs out (the web UI's 'not cached' filter)."
    )
    q: str | None = Field(
        None,
        max_length=200,
        description="Text to find anywhere in the job id or the request's `prompt` (case-insensitive).",
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
            "DELETE /v1/jobs/{job_id}/purge. A job submitted with `incognito: true` does not wait "
            "this long: its data is deleted as soon as it is no longer needed."
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
