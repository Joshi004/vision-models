## Overview

This API fronts a shared GPU cluster: a request submits a job to the cluster's own job scheduler,
which runs it on a free GPU whenever one becomes available, and the API tracks that job's status
until you poll for the finished result. There is no synchronous "generate and return the file in
this same request" endpoint anywhere in this API -- every generation call is submit-then-poll, by
design, because even the fastest recipe here takes minutes, not seconds.

**Backends currently registered on this server:** see "Endpoint reference" below for the full,
current list with their own path prefixes -- this guide lists whichever backends are actually
mounted on this running server, not a fixed set that might drift from reality.

**What's shared vs. backend-specific:**
- Shared, used the same way regardless of which backend created a job: uploading a file
  (`POST /v1/uploads`), a generation request's optional `partition` field, and every job endpoint
  (`GET /v1/jobs`, `GET /v1/jobs/{job_id}`, `GET /v1/jobs/{job_id}/result`,
  `DELETE /v1/jobs/{job_id}`, `DELETE /v1/jobs/{job_id}/purge`).
- Backend-specific: everything under that backend's own path prefix -- its request fields, its
  output format, and its own timing and limitations.