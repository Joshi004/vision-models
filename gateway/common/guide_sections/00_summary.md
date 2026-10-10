# GPU Cluster Model-Serving API -- Integration Guide

A single asynchronous HTTP API for GPU-cluster model-serving backends on this node. Each backend
(video generation, character replacement in existing video, voice conversion, speech
transcription, text-to-speech, and more over time) is namespaced under its own path prefix, but every one of them
shares the same upload, job-submission, and job-tracking endpoints, described once below rather
than once per backend.

**Start here if you're integrating a third-party system against this API:** `GET /v1/guide` (add
`?format=json` for the same content plus machine-readable facts) is this exact document, generated
fresh from this server's own live configuration and `/openapi.json` -- it cannot drift from what
`/docs` shows, and it updates automatically as endpoints change. Re-fetch it (or watch its `ETag`)
when you upgrade your integration, rather than keeping a saved copy of this text.

### Core workflow, in short

1. **Upload any file inputs first**, one at a time: `POST /v1/uploads`. Each call returns an
   `asset_id`; reference that `asset_id` from the generation request that needs it.
2. **Submit a generation request** under the backend's own prefix. Every generation endpoint on
   every backend returns immediately with a `job_id` and `status: "queued"` -- it does **not** wait
   for the result to finish.
3. **Poll `GET /v1/jobs/{job_id}`** every 10-15 seconds until `status` is `"succeeded"` or
   `"failed"`.
4. **Download `GET /v1/jobs/{job_id}/result`** once succeeded.

See "Quickstart" below for a runnable version of this exact flow, and "Jobs and timing" for what to
expect while a job is queued or running.