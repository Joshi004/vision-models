## Quickstart

A complete, minimal example: upload an image, generate a video from it, poll until done, download
the result. Every backend follows this same submit-then-poll shape; swap the endpoint and request
body for a different backend's own recipe (see "Endpoint reference" below).

### curl

```bash
BASE="{{base_url}}"

# 1. Upload an image (skip this step for pure text-to-video)
curl -s -X POST "$BASE/v1/uploads" \
  -F "file=@photo.jpg"
# -> {"asset_id": "...", "filename": "photo.jpg", "size_bytes": ..., "sha256": "...", "reused": false}
#    (compare size_bytes and sha256 with `stat` / `sha256sum photo.jpg` to confirm it arrived intact)

# 2. Submit a generation request (this example: LTX-2.3 text/image-to-video)
curl -s -X POST "$BASE/v1/ltx/videos/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A golden retriever puppy runs across a sunlit lawn, tongue out, tail wagging, chasing a bright yellow tennis ball. Bright daylight, shallow depth of field, handheld camera following the puppy.",
    "mode": "fast",
    "orientation": "landscape",
    "duration_seconds": 5
  }'
# -> {"job_id": "...", "status": "queued"}

# 3. Poll every 10-15 seconds
curl -s "$BASE/v1/jobs/<job_id>"
# -> {"status": "queued" | "running" | "succeeded" | "failed", ...}

# 4. Download once status is "succeeded"
curl -s "$BASE/v1/jobs/<job_id>/result" -o output.mp4
```

### Python

```python
import time
import requests

BASE = "{{base_url}}"

resp = requests.post(
    f"{BASE}/v1/ltx/videos/generate",
    json={
        "prompt": "A golden retriever puppy runs across a sunlit lawn, tongue out, tail "
                  "wagging, chasing a bright yellow tennis ball. Bright daylight, shallow "
                  "depth of field, handheld camera following the puppy.",
        "mode": "fast",
        "orientation": "landscape",
        "duration_seconds": 5,
    },
)
resp.raise_for_status()
job_id = resp.json()["job_id"]

while True:
    status = requests.get(f"{BASE}/v1/jobs/{job_id}").json()
    if status["status"] in ("succeeded", "failed"):
        break
    time.sleep(15)

if status["status"] == "succeeded":
    video = requests.get(f"{BASE}/v1/jobs/{job_id}/result")
    open("output.mp4", "wb").write(video.content)
else:
    print("failed:", status["error"])
```

### Skip re-uploading a file the server already has

Optional. Before uploading, send the file's SHA-256 to `POST /v1/uploads/by-hash`. **200** means the
server holds a copy: use the `asset_id` it returns (`reused: true`) and send nothing. **404** means it has
none -- never uploaded, or since deleted, purged or expired -- so upload the file normally and use the
`asset_id` that returns. If a later generation request answers 400 that an `asset_id` is unknown (the copy
was removed in between), upload the file again the same way.

```bash
HASH=$(sha256sum photo.jpg | cut -d' ' -f1)
curl -s -X POST "$BASE/v1/uploads/by-hash" \
  -H "Content-Type: application/json" -d "{\"sha256\": \"$HASH\"}"
# 200 -> {"asset_id": "...", "filename": "photo.jpg", "size_bytes": ..., "sha256": "...", "reused": true}
# 404 -> {"detail": "No stored upload has this sha256 ... Upload the file with POST /v1/uploads."}
#        -> then: curl -s -X POST "$BASE/v1/uploads" -F "file=@photo.jpg"
```

```python
import hashlib
import requests

BASE = "{{base_url}}"

def upload_asset(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    found = requests.post(f"{BASE}/v1/uploads/by-hash", json={"sha256": digest.hexdigest()})
    if found.status_code == 200:
        return found.json()["asset_id"]  # already on the server, nothing was sent
    if found.status_code != 404:
        found.raise_for_status()

    with open(path, "rb") as f:
        uploaded = requests.post(f"{BASE}/v1/uploads", files={"file": f})
    uploaded.raise_for_status()
    result = uploaded.json()
    if result["sha256"] != digest.hexdigest():
        raise RuntimeError("the server stored different bytes than were sent; upload again")
    return result["asset_id"]
```