## Errors

| HTTP status | Meaning | What to do |
|---|---|---|
| `400` | Something referenced in the request doesn't exist or isn't accepted -- an unknown `asset_id`, or (on endpoints that validate file type) an unsupported input format | Fix the request; the `detail` field names what was wrong |
| `422` | The request body failed schema validation (missing/conflicting/out-of-range fields) | Fix the request; the `detail` array names the exact field and rule that failed |
| `429` | An optional concurrency cap was reached (not configured on every deployment) | Wait and retry -- this isn't specific to your request |
| `502` | The job was rejected by the cluster's own scheduler after passing this API's checks | Safe to retry once; if it persists, treat it as worth reporting |
| `404` | No job (or resource) exists with the id you gave | Double-check the id; it may also have already expired (see retention, above) |
| `409` | The job exists, but isn't in the right state yet for what you asked (e.g. downloading a result before it's finished) | Check `GET /v1/jobs/{job_id}` first and retry once it's in the expected state |
| `410` | The thing you asked for existed once but has since expired or been cleaned up | Nothing to recover; resubmit if you still need the content |
| `413` | The uploaded file is larger than this deployment's own limit | Reduce the file size (limit: {{upload_mb}} MB) |

A job reaching `status: "failed"` is not itself an HTTP error -- the submit call already succeeded
with a `200`. Always check the job's own `error` field for *why* it failed (bad input vs. a
transient cluster issue worth simply resubmitting) rather than treating every failed job the same
way.