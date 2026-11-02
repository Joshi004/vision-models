"""Long-form OpenAPI/Swagger documentation text for Stable Audio 3 Medium's
endpoints, kept out of router.py so the routes themselves stay readable. Docs
shared by every backend (uploads, jobs, app-level description) live in
common/openapi_docs.py instead.
"""

from __future__ import annotations

from common import openapi_docs as common_docs
from common.audio_inputs import ACCEPTED_AUDIO_FORMATS
from services.stable_audio3 import config

_COMMON_NOTES = f"""
**Result:** `output.flac` (44.1 kHz stereo, 24-bit FLAC), downloaded with
`GET /v1/jobs/{{job_id}}/result`. The same inputs and `seed` give the same audio
on the same hardware.

**Model:** Stable Audio 3 Medium (1.4B), the post-trained checkpoint: {config.DEFAULT_STEPS} steps,
no `cfg_scale` and no `negative_prompt` (they have no effect on it). Clips of up to
{config.MAX_DURATION_SECONDS:g} seconds. **No intelligible vocals**: for songs with sung lyrics use
ACE-Step (`/v1/ace-step15`) or YuE2 (`/v1/yue2`).

**Timing:** requests are run by long-lived GPU jobs that load the model once;
generation itself takes seconds. A request that has to start a GPU job takes
minutes; one that finds a loaded, free worker starts at once. A worker idle for
{config.IDLE_MINUTES:g} minutes is stopped. No live progress: `progress` is always null.

**License:** Stable Audio 3 is under the **Stability AI Community License** (free
for individuals and organizations below USD 1M annual revenue; above that a
separate licence is needed). Whoever operates this deployment is responsible for
how it is used; this API does not enforce that restriction itself.
"""

GENERATE = f"""
Generates **instrumental music, stems, solo instruments or sound effects** from a
text prompt with **Stable Audio 3 Medium**.

For music, describe genre, each instrument and how it sounds, mood and energy, BPM
and production style, in long descriptive prose if you like. Tags help:
`TrackType: Music, VocalType: Instrumental` (more coherent), `Genre: Funk`,
`Instruments: Guitar, Saxophone, Bass`; `TrackType: Instrument` for one isolated
instrument; `TrackType: SFX` for sound effects (say what makes the sound, how it is
triggered and for how long, and use a short `duration`).

`duration` is up to {config.MAX_DURATION_SECONDS:g} seconds; pick one that fits the description.
{_COMMON_NOTES}
"""

TRANSFORM = f"""
Transforms an existing recording from a prompt (**audio-to-audio**) with **Stable
Audio 3 Medium**: change its timbre, style or genre while keeping its musical
shape. Upload the source with `POST /v1/uploads` (at most {config.MAX_SOURCE_SECONDS:g} seconds;
{ACCEPTED_AUDIO_FORMATS}) and pass its `asset_id` as `source_audio_asset_id`.

`init_noise_level` (0 to 1) sets how much of the source is kept: `0` keeps it, `1`
ignores it. About `0.4` suits timbre transfer (violin to metal guitar), about `0.5`
beatbox to lo-fi beat, about `0.6` a style shift; `0.1` is a close variation. It
needs tinkering per input, so keep `seed` fixed while you try values. Leave
`duration` out to use the source's length.
{_COMMON_NOTES}
"""

INPAINT = f"""
Regenerates **part of a recording, or extends it**, with **Stable Audio 3 Medium**.
Upload the source with `POST /v1/uploads` (at most {config.MAX_SOURCE_SECONDS:g} seconds;
{ACCEPTED_AUDIO_FORMATS}) and pass its `asset_id` as `source_audio_asset_id`.

`mask_start` and `mask_end` (seconds) mark the section to regenerate; it must start
inside the recording. The audio around the section matters more than the prompt, so
write a prompt that is plausible for that context ("dubstep drop" in a house track,
not "cat meow"). Masking only a few seconds gives output very close to the original:
start with a larger section and shrink it.

**To extend a clip:** set `mask_start` to the clip's length (or a little earlier),
`mask_end` to where the new audio should end, and leave `duration` out (it becomes
`mask_end`) or set it at least that long.
{_COMMON_NOTES}
"""

GENERATE_EXAMPLES = {
    "music": {
        "summary": "An instrumental",
        "value": {
            "prompt": (
                "TrackType: Music, VocalType: Instrumental, Genre: Funk. A funky hip hop instrumental with live recorded "
                "instrumentation that has the vibe of a 70's TV show theme. Warm, textured analogue sound."
            ),
            "duration": 60,
        },
    },
    "sfx": {
        "summary": "A sound effect",
        "value": {
            "prompt": "TrackType: SFX. A heavy wooden door creaks open slowly in a large stone hall, long reverb tail.",
            "duration": 8,
        },
    },
}

TRANSFORM_EXAMPLES = {
    "timbre": {
        "summary": "Violin to metal guitar",
        "value": {
            "source_audio_asset_id": "<asset_id from POST /v1/uploads>",
            "prompt": "TrackType: Instrument. A distorted metal electric guitar playing a melodic line.",
            "init_noise_level": 0.4,
        },
    },
}

INPAINT_EXAMPLES = {
    "section": {
        "summary": "Replace seconds 10-20",
        "value": {
            "source_audio_asset_id": "<asset_id from POST /v1/uploads>",
            "prompt": "A dubstep drop with heavy wobbling bass.",
            "mask_start": 10.0,
            "mask_end": 20.0,
        },
    },
    "extend": {
        "summary": "Extend a 30-second clip to 60 seconds",
        "value": {
            "source_audio_asset_id": "<asset_id from POST /v1/uploads>",
            "prompt": "The same groove continues and builds in energy.",
            "mask_start": 30.0,
            "mask_end": 60.0,
        },
    },
}

GENERATE_RESPONSES = {**common_docs.SUBMIT_RESPONSES}

AUDIO_INPUT_RESPONSES = {
    **common_docs.SUBMIT_RESPONSES,
    400: {
        "description": "The source recording's asset_id doesn't exist, the file isn't usable audio, it is too long, or the mask starts outside it.",
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
                    "bad_mask": {
                        "summary": "Mask outside the recording",
                        "value": {"detail": "mask_start (90s) is past the end of the source recording (60.0s)."},
                    },
                }
            }
        },
    },
}
