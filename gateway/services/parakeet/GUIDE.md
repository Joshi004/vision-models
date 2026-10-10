## Parakeet speech transcription

**What this is for:** turning an existing recording into text, with both word-level and
segment-level timestamps. Speech-to-text only -- there's nothing to configure about *how* the
output sounds, only what was said and when.

**Use this when** you have a real recording (audio, or a video you only care about the audio
track of) and want a transcript with timing, not a voice conversion or a generated video/audio
file. The model auto-detects among 25 supported European languages, so there's no `language` field
to set.

**Inputs:** one file (`audio_asset_id`), uploaded first via `POST /v1/uploads`. A wide range of
real-world recording *and* video container formats is accepted -- see the endpoint's own
description for the exact list -- and if you pass a video, only its first audio stream is used.

**Determinism:** the same input and model always produce the same output -- there is no `seed`
field.

### Workflow

1. Upload the recording.
2. Submit a transcription job.
3. Poll until it finishes.
4. Download and parse the transcript JSON.

**curl:**
```bash
BASE="http://localhost:8012"

# 1. Upload
ASSET_ID=$(curl -s -X POST "$BASE/v1/uploads" \
  -F "file=@recording.m4a" | python3 -c "import sys,json; print(json.load(sys.stdin)['asset_id'])")

# 2. Submit
JOB_ID=$(curl -s -X POST "$BASE/v1/parakeet/transcribe" \
  -H "Content-Type: application/json" \
  -d "{\"audio_asset_id\": \"$ASSET_ID\"}" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# 3. Poll (every 10-15s)
curl -s "$BASE/v1/jobs/$JOB_ID"

# 4. Download the transcript once status == "succeeded"
curl -s "$BASE/v1/jobs/$JOB_ID/result" -o transcript.json
```

**Python:**
```python
import time
import requests

BASE = "http://localhost:8012"

# 1. Upload
with open("recording.m4a", "rb") as f:
    asset_id = requests.post(f"{BASE}/v1/uploads", files={"file": f}).json()["asset_id"]

# 2. Submit
job_id = requests.post(
    f"{BASE}/v1/parakeet/transcribe", json={"audio_asset_id": asset_id}
).json()["job_id"]

# 3. Poll
while True:
    status = requests.get(f"{BASE}/v1/jobs/{job_id}").json()
    if status["status"] in ("succeeded", "failed"):
        break
    time.sleep(12)

if status["status"] == "failed":
    raise RuntimeError(status["error"])

# 4. Download and parse
transcript = requests.get(f"{BASE}/v1/jobs/{job_id}/result").json()
print(transcript["transcription"])
for word in transcript["word_timestamps"]:
    print(f"{word['start']:.2f}-{word['end']:.2f}  {word['word']}")
```

See `POST /v1/parakeet/transcribe`'s own description for the full transcript JSON shape
(`transcription`, `processing_time`, `word_timestamps[]`, `segment_timestamps[]`, `metadata`),
units, and what `metadata.duration` means.

### Migrating from the old always-on service

This replaces the previous standalone Parakeet service (`POST :8006/transcribe`, body
`{"audio_url": "..."}`, synchronous response), which needed a GPU held for its entire lifetime
regardless of whether a request was in flight. The new flow above holds a GPU only for the
length of one Slurm job:

- Upload the file instead of pointing at a URL the old service had to download itself.
- The response is now asynchronous (`job_id` + poll), not an immediate HTTP response body.
- **The transcript JSON itself is unchanged** -- `transcription`, `processing_time`,
  `word_timestamps`, `segment_timestamps`, and `metadata` all keep the exact same field names and
  meanings, so existing code that parses that JSON doesn't need to change, only the code that
  submits the request and retrieves the result.

### Long recordings

Recordings longer than 10 minutes (decoded duration, not compressed file size) are chunked
automatically into overlapping 10-minute pieces and stitched back together -- no action needed on
your part, and nothing about the request or the result's shape changes either way. Expect
proportionally longer processing time for proportionally longer recordings.
