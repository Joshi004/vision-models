## Jobs and timing

**Status values:** `queued` -> `running` -> `succeeded` or `failed`. A cancelled job reports
`failed`, not a separate status. Poll `GET /v1/jobs/{job_id}` at a relaxed interval -- **10-15
seconds is plenty**; every job takes minutes at an absolute minimum, so polling faster wastes calls
for no benefit.

**Why "queued" can last a while:** every request dynamically submits a new job to this cluster's
job scheduler. No backend here keeps a resident, "warm" model in GPU memory across requests -- each
job cold-starts by loading model weights from storage before it does any actual work, then releases
its allocation afterward. On top of that cold start, a job can also sit genuinely queued for a while
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

**Live progress:** some backends report fine-grained progress (current stage, step, a rough ETA) in
the `progress` field while `status` is `"running"`; every other backend simply reports
`progress: null` for the whole run -- that's expected, not a sign anything is stuck. Check each
backend's own section below for whether it's one of the ones that does.

**Cancel vs. delete:** `DELETE /v1/jobs/{job_id}` stops a job that hasn't finished yet -- a safe,
best-effort no-op (not an error) if it already finished either way. `DELETE /v1/jobs/{job_id}/purge`
permanently deletes a job that **has** already finished, along with any uploaded input no other job
still references; this cannot be undone, and a queued/running job must be cancelled first.

**Retention:** finished jobs (and their output files) are deleted automatically
{{job_retention_days}} day(s) after they finish. Uploaded assets are **not** auto-deleted --
re-uploading the same file again produces a new, additional `asset_id` rather than deduplicating.

**Concurrency:** {{max_inflight_line}}