## Jobs and timing

**Status values:** `queued` -> `running` -> `succeeded` or `failed`. A cancelled job reports
`failed`, not a separate status. Poll `GET /v1/jobs/{job_id}` at a relaxed interval -- **10-15
seconds is plenty**; every job takes minutes at an absolute minimum, so polling faster wastes calls
for no benefit.

**Why "queued" can last a while:** every request dynamically submits a new job to this cluster's
job scheduler. Except for the Qwen3.8 text model (below), no backend here keeps a resident, "warm"
model in GPU memory across requests -- each job cold-starts by loading model weights from storage
before it does any actual work, then releases its allocation afterward. The Qwen3.8 text model keeps
one shared server running while it is in use and stops it after a period without requests: its
requests are answered in seconds when the server is up, and a request that arrives while it is
stopped stays `queued` for the several minutes the server takes to start. On top of that cold start, a job can also sit genuinely queued for a while
if the cluster is busy; that queue wait has no fixed upper bound. Budget at least several minutes
per request before any backend-specific timing on top of that -- see each backend's own section
below for real measured numbers.

**Choosing a partition:** every generation request, on every backend, may optionally set a
top-level `partition` field to choose which pool of GPUs its job runs on. If omitted, or if the
named partition doesn't currently exist, the job silently falls back to this deployment's own
default (`{{default_partition}}` right now -- `GET /v1/health` echoes this live) instead of erroring
either way. `GET /v1/partitions` reports each partition's live free-GPU count and queue length so
you can choose intentionally instead of guessing; see that endpoint's own description for one
important caveat -- this deployment's default partition can have a running job pre-empted (killed
and automatically restarted) by higher-priority work, which surfaces as the job continuing to report
`status: "running"` through the restart, not as a failure.

**If a job's `status` becomes `"failed"` due to pre-emption** (the `error` field will say so), the
correct response is to resubmit the identical request unchanged, not to treat it as a bad input.

**GPUs and an estimate for this request:** a backend that sizes each request before it queues the job
(MiniMax H3 today) also reports two numbers: `gpus`, how many GPUs the job was given, and
`estimated_run_seconds`, about how long it will run once it starts (the queue wait is not included). They are
in the response that submits the job and in `GET /v1/jobs/{job_id}` and `GET /v1/jobs`. Prefer
`estimated_run_seconds` over `typical_run_seconds` when it is present: it accounts for this request's own
size, where the typical figure is the median of whatever ran before. Both are rough "about" figures. Every
other backend reports `gpus` and `estimated_run_seconds` as `null`. Such a backend can also refuse a request
that is valid but too big for its GPUs, with HTTP 422 and a sentence saying why, before anything is queued.

**Live progress:** some backends report fine-grained progress (current stage, step, a rough ETA) in
the `progress` field while `status` is `"running"`; every other backend simply reports
`progress: null` for the whole run -- that's expected, not a sign anything is stuck. Check each
backend's own section below for whether it's one of the ones that does.

**Cancel vs. delete:** `DELETE /v1/jobs/{job_id}` stops a job that hasn't finished yet -- a safe,
best-effort no-op (not an error) if it already finished either way. `DELETE /v1/jobs/{job_id}/purge`
permanently deletes a job that **has** already finished, along with any uploaded input no other job
still references; this cannot be undone, and a queued/running job must be cancelled first.

{{retention_text}} Re-uploading the same file again produces a new, additional `asset_id` rather than
deduplicating.

**Concurrency:** {{max_inflight_line}}