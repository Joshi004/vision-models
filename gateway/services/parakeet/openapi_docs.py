"""Long-form OpenAPI/Swagger documentation text for Parakeet's own
endpoint, kept out of router.py so the route itself stays readable. Docs
shared by every backend (uploads, jobs, app-level description) live in
common/openapi_docs.py instead.
"""

from __future__ import annotations

from common import openapi_docs as common_docs
from services.parakeet import config

_ACCEPTED_FORMATS_LIST = ", ".join(sorted(ext.lstrip(".") for ext in config.INPUT_EXTENSIONS))

TRANSCRIBE = f"""
Transcribes an existing audio or video recording to text, with both
**word-level** and **segment-level** timestamps. Speech-to-text, not voice
conversion or generation: there's nothing to configure about *how* it
sounds, only what was said and when.

**Language:** auto-detected among 25 supported European languages --
there's no `language` field to set. **Determinism:** the same input and
model always produce the same output, so there's no `seed` field either,
unlike LTX/Wan-Animate's diffusion-based sampling.

**Accepted file types:** {_ACCEPTED_FORMATS_LIST}. If you pass a video
file, only its first audio stream is used -- everything else about the
request is identical to passing a plain audio file. An unsupported file is
rejected immediately (400) before any job is submitted.

**Long audio is chunked automatically.** Recordings longer than 10 minutes
are split into overlapping 10-minute chunks (10-second overlap), transcribed
one at a time, then stitched back together with duplicate words in the
overlap removed and segment boundaries re-joined across the cut -- no
action needed on your part, and the result looks the same as a single-pass
transcription either way.

**The result is a JSON file, not audio/video** -- download it from
`GET /v1/jobs/{{job_id}}/result` once `GET /v1/jobs/{{job_id}}` reports
`status: "succeeded"`. Its shape (unchanged from this service's previous
always-on API, so existing downstream parsing code keeps working):
```json
{{
  "transcription": "Thanks for calling. How can I help you today?",
  "processing_time": 4.12,
  "word_timestamps": [
    {{"word": "Thanks", "start": 0.08, "end": 0.34}},
    {{"word": "for", "start": 0.34, "end": 0.5}},
    {{"word": "calling.", "start": 0.5, "end": 0.95}}
  ],
  "segment_timestamps": [
    {{"text": "Thanks for calling.", "start": 0.08, "end": 0.95, "word_count": 3}},
    {{"text": "How can I help you today?", "start": 1.2, "end": 2.6, "word_count": 7}}
  ],
  "metadata": {{"total_segments": 2, "total_words": 10, "duration": 2.6}}
}}
```
- All times (`start`, `end`, `processing_time`, `metadata.duration`) are in **seconds**, as floats.
- `metadata.duration` is the **end timestamp of the last word**, not the recording's own full
  length -- trailing silence after the last spoken word isn't counted. Same behavior as this
  service's previous always-on API.
- `word_timestamps`/`segment_timestamps` can be empty lists for a recording with no detected speech
  (silence, music-only, or an unsupported language); `metadata.duration` is then `0.0`.

**Timing:** requests are run by a small number of long-lived GPU jobs that
load the model once and transcribe one recording after another; one with
no request for 10 minutes is stopped. A request that finds one free starts
at once and skips loading the model; the first request after an idle stop
waits for a GPU job to start and load. When every GPU job was started
fresh for its request, measured times were 63-117 seconds end to end
(queueing + node allocation + loading + the run itself), from a 10-second
clip up to a 35-minute chunked recording, with loading dominating for
anything under a few chunks; a request on an already loaded GPU job takes
correspondingly less (not yet measured). See `typical_run_seconds` on
`GET /v1/jobs/{{job_id}}` for a live, continuously-updated estimate. Very
long recordings take proportionally longer once chunk count dominates.

**Repeated requests:** an identical request (same uploaded file content,
same settings) while the earlier job is queued, running or succeeded with
its result still available returns that job's `job_id` with `reused: true`
instead of running again; a request whose GPU job is lost is queued again
under the same `job_id`. See the Parakeet section of `GET /v1/guide`.
"""

TRANSCRIBE_EXAMPLES = {
    "basic_transcription": {
        "summary": "Transcribe an uploaded recording",
        "value": {
            "audio_asset_id": "<asset_id of the recording to transcribe, from POST /v1/uploads>",
        },
    },
    "on_a_specific_partition": {
        "summary": "Same request, submitted to a specific partition",
        "value": {
            "audio_asset_id": "<asset_id of the recording to transcribe, from POST /v1/uploads>",
            "partition": "main",
        },
    },
}

# ---------------------------------------------------------------------------
# 400 responses -- richer than common_docs's generic _ASSET_ERROR_RESPONSE,
# because an unsupported input file is a real, distinct, common 400 cause
# here, same reasoning as services/rvc/openapi_docs.py's own
# CONVERT_RESPONSES. Two named examples under the same 400 status: a
# human/LLM reading /docs sees both causes at a glance.
# ---------------------------------------------------------------------------
TRANSCRIBE_RESPONSES = {
    **common_docs.SUBMIT_RESPONSES,
    400: {
        "description": "The referenced asset doesn't exist, or its file isn't one of the accepted formats.",
        "content": {
            "application/json": {
                "examples": {
                    "unknown_asset": {
                        "summary": "Unknown or missing asset_id",
                        "value": {"detail": "Invalid asset reference: 'Unknown asset_id: abc123'"},
                    },
                    "unsupported_format": {
                        "summary": "Unsupported recording file type",
                        "value": {
                            "detail": (
                                f"Unsupported file extension '.xyz' (accepted: {_ACCEPTED_FORMATS_LIST}): "
                                "recording.xyz"
                            )
                        },
                    },
                }
            }
        },
    },
}
