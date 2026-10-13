"""Long-form OpenAPI/Swagger documentation text for TBDub's own endpoint,
kept out of router.py so the route itself stays readable. Docs shared
by every backend (uploads, jobs, app-level description) live in
common/openapi_docs.py instead.
"""

from __future__ import annotations

from common import openapi_docs as common_docs
from services.tbdub import config

LIP_SYNC = f"""
Redraws the mouth of the speaker in an existing video so it matches a given audio track exactly,
using **TBDub** (30-step Teacher model). The audio is never regenerated: your exact audio is muxed
into the result at its original quality.

**Two ways to say what to sync**
- **`video_asset_id` + `audio_asset_id`** -- a video and an audio file (or a video whose audio track
  is used), both uploaded first via `POST /v1/uploads`. `audio_start_time` is where in the audio file
  the video starts (default 0).
- **`source_job_id`** -- a *succeeded* LTX audio-to-video job (`/v1/ltx/videos/audio-to-video` or
  `/v1/ltx25/videos/audio-to-video`). Its un-synced video and its audio are reused, so nothing is
  uploaded again. This is the retry path for a job that came back with `lip_sync.applied: false`.
  It is not available if that job was deleted, was an `incognito` job, or lost its audio upload
  (a 404/409 says which); upload the files instead.

**This is always its own job**, with its own `job_id` and its own entry in the jobs list, even when
started from `source_job_id` (the source's video and audio are copied into the new job). Deleting
either job (`DELETE /v1/jobs/{{job_id}}/purge`) never affects the other.

**What it changes, and what it does not.** TBDub edits only what speech moves: lips, teeth, jaw and the
skin around them. Eyes, eyebrows, head motion and the overall expression come from the input video, so a
laugh or a raised eyebrow has to be in the video already (for LTX, describe it in the prompt). A laugh
in the *audio* is likely to come out as ordinary talking mouth shapes.

**What the input must look like**
- **One speaker, face visible.** Only one face is tracked. If no face is found for more than ~0.4 s
  (a cutaway, someone turning fully away) the job fails.
- Any frame rate is accepted: the video is converted to 25 fps first (same duration), and the result is
  always **25 fps**.
- The face is processed as a 512x512 crop and pasted back, so on a very tight close-up the mouth can
  look slightly softer than the rest of the frame.
- The result has the same resolution as the input video.

**Settings.** `seed`, and two optional dials that only matter for a retry: `audio_cfg_scale` ("follow
the voice": raise it if the mouth under-articulates) and `ref_cfg_scale` ("stay like the original": raise
it if the face drifts). Omit both for TBDub's own defaults.

**Time and size.** Defaults to a single GPU with a job limit of {config.JOB_TIME_LIMIT} (HH:MM:SS); the cost
grows with video length. Each job loads ~15 GB of model weights first.

**Result.** An MP4, same resolution as the input, 25 fps, with the audio from `audio_asset_id`
(starting at `audio_start_time`) cut to the video's length. Download it via
`GET /v1/jobs/{{job_id}}/result`.

Only sync recordings and faces you have the right to use. Generated media should be disclosed as such.
"""

LIP_SYNC_EXAMPLES = {
    "retry_a_job": {
        "summary": "Re-sync an LTX audio-to-video job whose lip sync failed",
        "value": {"source_job_id": "<job_id of the audio-to-video job>"},
    },
    "retry_with_stronger_voice_following": {
        "summary": "Same, with the mouth following the audio more strongly",
        "value": {"source_job_id": "<job_id of the audio-to-video job>", "audio_cfg_scale": 8, "seed": 11},
    },
    "uploaded_files": {
        "summary": "Sync an uploaded video to an uploaded audio file",
        "value": {
            "video_asset_id": "<asset_id of the video, from POST /v1/uploads>",
            "audio_asset_id": "<asset_id of the audio, from POST /v1/uploads>",
        },
    },
}

LIP_SYNC_RESPONSES = {
    **common_docs.SUBMIT_RESPONSES,
    400: {
        "description": (
            "A referenced asset_id doesn't exist or its file is missing, or `source_job_id` is not an "
            "audio-to-video job."
        ),
        "content": {
            "application/json": {
                "examples": {
                    "unknown_asset": {
                        "summary": "Unknown or missing asset_id",
                        "value": {"detail": "Invalid asset reference: 'Unknown asset_id: abc123'"},
                    },
                    "wrong_job_kind": {
                        "summary": "source_job_id is not an audio-to-video job",
                        "value": {
                            "detail": (
                                "source_job_id must be an audio-to-video job (ltx25:audio-to-video, "
                                "ltx:audio-to-video); this one is rvc:convert."
                            )
                        },
                    },
                }
            }
        },
    },
    404: {
        "description": "`source_job_id` does not exist (it never did, or the job was deleted).",
        "content": {
            "application/json": {"example": {"detail": "Unknown source_job_id (it may have been deleted)."}}
        },
    },
    409: {
        "description": (
            "`source_job_id` exists but cannot be re-synced: it has not succeeded, its data was removed "
            "(an incognito job), or its video or audio is no longer on disk. Upload the files instead."
        ),
        "content": {
            "application/json": {
                "example": {"detail": "The source job has not succeeded (status: running)."}
            }
        },
    },
}
