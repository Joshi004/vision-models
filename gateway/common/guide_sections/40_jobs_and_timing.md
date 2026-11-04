## Jobs and timing

**Status values:** `queued` -> `running` -> `succeeded` or `failed`. A cancelled job reports
`failed`, not a separate status. Poll `GET /v1/jobs/{job_id}` at a relaxed interval -- **10-15
seconds is plenty**; every job takes minutes at an absolute minimum, so polling faster wastes calls
for no benefit.

**Why "queued" can last a while:** most backends submit a new job to this cluster's job scheduler for
every request. That job cold-starts by loading model weights from storage before it does any actual
work, then releases its allocation afterward. Several backends keep GPU work running between requests
instead. The Qwen3.8 text model keeps one shared server running while it is in use and stops it after a
period without requests: its requests are answered in seconds when the server is up, and a request that
arrives while it is stopped stays `queued` for the several minutes the server takes to start.
Qwen-Image-2.1, MiniMax H3, the four music and audio backends, Parakeet and Breeze TTS keep a limited number of
long-lived GPU jobs (by default at most 8 each for Qwen-Image-2.1 and H3 and 4 each for the others, stopped after 10
idle minutes): a request that finds one free starts at once, and loads much faster than a cold start when its
GPU job has run a request before; otherwise it stays `queued` until a GPU job has started or become free. On top of that cold start, a job can also sit genuinely queued for a while
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

**Automatic retries and repeated requests:** the backends that keep GPU jobs running between requests
(Qwen-Image-2.1, MiniMax H3, the four music and audio backends, Parakeet and Breeze TTS) and the Qwen3.8 text model put a request back in
the queue, under the same `job_id`, when the GPU job or model server running it is lost (for example
pre-empted), up to 3 starts in total; `attempts` on a job's status says how many. A request that fails because of
its own content is not run again, with one exception: a MiniMax H3 job given 4 GPUs that runs out of GPU memory in
the text encoder or the denoising is run once more on 8 GPUs (same `job_id`, `gpus` becomes 8, `attempts` goes up by
one). Those backends also recognise a request that is identical to an earlier one
(same settings and seed, same uploaded file content) and answer with the earlier job's `job_id` and
`reused: true` instead of running it again -- unless the earlier job failed, or the request has
`incognito: true`. Qwen3.8 does this only for a request that carries the same `Idempotency-Key` header. A shared
job is cancelled for everyone by `DELETE /v1/jobs/{job_id}`. Each backend's section has the details.

**Live progress:** some backends report fine-grained progress (current stage, step, a rough ETA) in
the `progress` field while `status` is `"running"`; every other backend simply reports
`progress: null` for the whole run -- that's expected, not a sign anything is stuck. Check each
backend's own section below for whether it's one of the ones that does.

**Cancel vs. delete:** `DELETE /v1/jobs/{job_id}` stops a job that hasn't finished yet -- a safe,
best-effort no-op (not an error) if it already finished either way. `DELETE /v1/jobs/{job_id}/purge`
permanently deletes a job that **has** already finished, along with any uploaded input no other job
still references; this cannot be undone, and a queued/running job must be cancelled first.

{{retention_text}} Re-uploading the same file again produces a new, additional `asset_id` rather than
deduplicating; to avoid sending a file the server may already hold, ask `POST /v1/uploads/by-hash` first (see
"Skip re-uploading a file the server already has" in the Quickstart).

**Priority:** every generation request may carry an optional top-level `priority`, an integer from 1 (most
urgent) to 10 (least urgent). If it is omitted the job gets 10; a value outside 1-10 is refused with 422. The job's
status reports it as `priority`. While jobs wait, the more urgent one goes first, and jobs of the same priority go
in the order they were submitted. This holds for every backend except LTX-2.3, LTX-2.5, Wan-Animate, RVC and
TBDub, and it also holds between models: when the global GPU limit is reached, the most urgent waiting job of any
model gets the next free GPUs, and a less urgent job of another model does not take them (an idle GPU job of
another model may be stopped early so its GPUs can be used). The five excluded backends accept the field but
always count as 10 against other models: while a more urgent job of another model is waiting for GPUs, they answer
**429** to a new request. Priority is strict, without ageing: a steady stream of urgent jobs can keep a priority-10
job waiting. A repeated request that is recognised as an earlier, still queued job makes that job at least as
urgent as the new request asked. To change a queued job's priority (whether or not it was given one), call
`PUT /v1/jobs/{job_id}/priority` with `{"priority": 1}`: it answers 409 for a job that is no longer queued or whose
backend does not use priority, and `priority_editable` on a job says whether the call would change it now. A change
takes effect within a few seconds.

**Finding jobs (filters and pages):** `GET /v1/jobs` returns only the newest `limit` jobs. To page through the
rest, or to find jobs by their properties, `POST /v1/jobs/search` takes a JSON body with `offset` and `limit` (at
most 500) and any of these optional filters, combined with AND: `statuses`, `pipelines` (exact
`<backend>:<recipe>` strings), `source`, `job_ids` / `exclude_job_ids` (at most 10000 each) and `q` (text to find in
the job id or the request's `prompt`). The answer has the same `jobs` entries as `GET /v1/jobs` plus `total`, the
number of matching jobs across all pages. A page is a snapshot: while jobs are being submitted, a job can show up
on two pages or be missed when you move from one page to the next.

**Who submitted a job:** every job reports `source`: `"ui"` if the request carried the header `X-Job-Source: ui`
(the web UI sends it), `"api"` for any other client, null for jobs submitted before the field existed. Send the
header yourself only if you want your jobs listed as UI jobs; it is a label, not authentication. A repeated
request that is answered with an earlier job keeps that job's `source`.

**Concurrency:** the operator limits how many GPUs this service uses in total and how many jobs of each
model run at once; both can change at any time, and `GET /v1/capacity` shows them with current usage. A
model served by warm workers (every backend except LTX-2.3, LTX-2.5, RVC, TBDub and Wan-Animate) accepts
your request and keeps it `queued` until a slot is free. The five that run one Slurm job per request
answer **429** when their own limit or the global GPU limit is reached: nothing was queued, retry shortly.
Lowering a limit never cancels a job that is already running.

**Why a job is still queued:** while a job is `queued`, `queue_state` says why, as of the last scheduling pass (a few
seconds old at most); it is null for any other status. `scheduling`: not looked at yet. `model_limit`: its model is at
its own limit of jobs (for Qwen3.8, every request slot of the server is in use). `gpu_limit`: the service's global GPU
limit is reached. `priority`: a more urgent job of another model gets the free GPUs first. `slurm_pending`: waiting for
the cluster to allocate GPUs. `model_loading`: GPUs are allocated and the worker or server is loading the model. A job
of LTX-2.3, LTX-2.5, Wan-Animate, RVC or TBDub only ever shows `slurm_pending`. `GET /v1/capacity` counts the queued
jobs by these reasons (`queued_by_reason`), per model and in total.