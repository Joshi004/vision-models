"""Long-form OpenAPI/Swagger documentation text for ACE-Step 1.5's endpoints,
kept out of router.py so the routes themselves stay readable. Docs shared by
every backend (uploads, jobs, app-level description) live in
common/openapi_docs.py instead.
"""

from __future__ import annotations

from common import openapi_docs as common_docs
from common.audio_inputs import ACCEPTED_AUDIO_FORMATS
from services.ace_step15 import config

_COMMON_NOTES = f"""
**Result:** `output.flac` (48 kHz stereo, 24-bit FLAC), downloaded with
`GET /v1/jobs/{{job_id}}/result`. The same inputs and `seed` give the same audio
on the same hardware.

**Model:** ACE-Step 1.5 XL (the SFT diffusion transformer, {config.DEFAULT_STEPS} steps) with its 4B
language-model planner, both loaded in bf16 without quantization.

**Timing:** requests are run by long-lived GPU jobs that load the models once.
A request that has to start one takes minutes; one that finds a loaded, free
worker starts at once. A worker idle for {config.IDLE_MINUTES:g} minutes is stopped. No live
progress: `progress` is always null.

**License:** the ACE-Step code is MIT and its authors state that generated music
may be used commercially. Generated music can resemble existing works: check
originality and disclose AI involvement where required.
"""

GENERATE = f"""
Generates a full song (with vocals) or an instrumental from a **caption** and
**lyrics** with **ACE-Step 1.5**. Strong choices: songs in many languages (50+),
exact control of tempo, key and length, and fast iteration.

How to write the inputs:

- `caption` -- genre, mood, instruments, timbre, era, production and vocal
  character in one description. Don't put tempo or key in it.
- `bpm`, `keyscale`, `time_signature`, `duration` -- structured fields. Leave them
  out to let the model decide (`thinking: true` plans them).
- `lyrics` -- section tags on their own lines (`[Verse 1]`, `[Chorus]`, `[Bridge]`,
  `[Instrumental]`, ...), at most one modifier per tag (`[Chorus - anthemic]`),
  about 6-10 syllables per line. For no vocals set `instrumental: true`.
- Keep caption and lyrics consistent, and **fix `seed` while tuning**.
{_COMMON_NOTES}
"""

COVER = f"""
Restyles an existing recording with **ACE-Step 1.5**, keeping its structure.
Upload the source with `POST /v1/uploads` (at most {config.MAX_SOURCE_SECONDS:g} seconds; {ACCEPTED_AUDIO_FORMATS}) and pass
its `asset_id` as `source_audio_asset_id`.

`caption` describes the new style (`jazz piano version`); `lyrics` is optional new
lyrics. `cover_strength` sets how closely the result follows the source: `1.0`
very close, `0.5` balanced, about `0.2` a loose style transfer. The output is as
long as the source. The language model is not used for covers.
{_COMMON_NOTES}
"""

REPAINT = f"""
Regenerates one time window of an existing recording with **ACE-Step 1.5** and
keeps the rest. Upload the source with `POST /v1/uploads` (at most
{config.MAX_SOURCE_SECONDS:g} seconds; {ACCEPTED_AUDIO_FORMATS}) and pass its `asset_id` as
`source_audio_asset_id`.

`repaint_start` and `repaint_end` (seconds; leave `repaint_end` out to go to the
end) mark the window, which must start inside the recording. `caption` describes
what the new section should contain (`smooth transition with piano solo`). Use it
to fix a section, add variation or build a transition. The language model is not
used for repaints.
{_COMMON_NOTES}
"""

GENERATE_EXAMPLES = {
    "song": {
        "summary": "A song with lyrics",
        "value": {
            "caption": "melancholic indie folk, fingerpicked acoustic guitar, soft female vocal, warm intimate room sound, light brushed drums",
            "lyrics": "[Verse 1]\nWalking home through the quiet rain\nStreetlights humming my name\n\n[Chorus - soft]\nHold on to the little light\nIt will carry us tonight",
            "bpm": 78,
            "keyscale": "A minor",
            "vocal_language": "en",
        },
    },
    "instrumental": {
        "summary": "An instrumental of a chosen length",
        "value": {
            "caption": "calm lo-fi hip hop, dusty drums, mellow electric piano, vinyl crackle, late-night study mood",
            "instrumental": True,
            "duration": 90,
            "bpm": 80,
        },
    },
}

COVER_EXAMPLES = {
    "restyle": {
        "summary": "Jazz piano version of a recording",
        "value": {
            "source_audio_asset_id": "<asset_id from POST /v1/uploads>",
            "caption": "jazz piano trio version, brushed drums, upright bass",
            "cover_strength": 0.8,
        },
    },
}

REPAINT_EXAMPLES = {
    "section": {
        "summary": "Replace seconds 10-20 with a piano solo",
        "value": {
            "source_audio_asset_id": "<asset_id from POST /v1/uploads>",
            "caption": "smooth transition into an expressive piano solo",
            "repaint_start": 10.0,
            "repaint_end": 20.0,
        },
    },
}

GENERATE_RESPONSES = {**common_docs.SUBMIT_RESPONSES}

AUDIO_INPUT_RESPONSES = {
    **common_docs.SUBMIT_RESPONSES,
    400: {
        "description": "The source recording's asset_id doesn't exist, the file isn't usable audio, or it is too long.",
        "content": {
            "application/json": {
                "examples": {
                    "unknown_asset": {
                        "summary": "Unknown or missing asset_id",
                        "value": {"detail": "Invalid asset reference: 'Unknown asset_id: abc123'"},
                    },
                    "unsupported_format": {
                        "summary": "Unsupported file type",
                        "value": {
                            "detail": f"Unsupported file extension '.xyz' (accepted: {ACCEPTED_AUDIO_FORMATS}): source.xyz"
                        },
                    },
                    "bad_range": {
                        "summary": "Repaint window outside the recording",
                        "value": {"detail": "repaint_start (90s) is past the end of the source recording (60.0s)."},
                    },
                }
            }
        },
    },
}
