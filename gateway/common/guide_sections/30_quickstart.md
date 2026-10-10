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
# -> {"asset_id": "...", "filename": "photo.jpg", "size_bytes": ...}

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