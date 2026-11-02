"""Long-form OpenAPI/Swagger documentation text for Stable Audio Open 1.0's
endpoints, kept out of router.py so the routes themselves stay readable. Docs
shared by every backend (uploads, jobs, app-level description) live in
common/openapi_docs.py instead.
"""

from __future__ import annotations

from common import openapi_docs as common_docs
from common.audio_inputs import ACCEPTED_AUDIO_FORMATS
from services.stable_audio_open import config

_COMMON_NOTES = f"""
**Result:** `output.flac` (44.1 kHz stereo, 24-bit FLAC), downloaded with
`GET /v1/jobs/{{job_id}}/result`. The same inputs and `seed` give the same audio
on the same hardware.

**Limits:** at most {config.MAX_DURATION_SECONDS:g} seconds (the model's fixed window), English prompts,
no intelligible vocals. For longer or more structured music use Stable Audio 3
(`/v1/stable-audio3`); for songs with vocals use ACE-Step or YuE2.

**Timing:** requests are run by long-lived GPU jobs that load the model once.
A request that has to start one takes minutes; one that finds a loaded, free
worker starts at once. A worker idle for {config.IDLE_MINUTES:g} minutes is stopped. No live
progress: `progress` is always null.

**License:** Stable Audio Open 1.0 is under the **Stability AI Community
License** (free for individuals and organizations below USD 1M annual revenue;
above that a separate licence is needed). Whoever operates this deployment is
responsible for how it is used; this API does not enforce that restriction itself.
"""

GENERATE = f"""
Generates a **sound effect, foley hit, drum loop, ambience or short clip** from a
text prompt with **Stable Audio Open 1.0**.

Write a short, concrete description: what makes the sound, what it is made of,
the room, and the tempo for loops -- `128 BPM tech house drum loop`, `The sound of a
hammer hitting a wooden surface.` `negative_prompt` (for example `Low quality.`)
steers away from unwanted qualities. `duration` is up to {config.MAX_DURATION_SECONDS:g} seconds.
The model is weaker at long musical structure than Stable Audio 3.
{_COMMON_NOTES}
"""

TRANSFORM = f"""
Re-imagines an existing recording from a prompt (**audio-to-audio**) with
**Stable Audio Open 1.0**. Upload the source with `POST /v1/uploads` (at most
{config.MAX_SOURCE_SECONDS:g} seconds; {ACCEPTED_AUDIO_FORMATS}) and pass its `asset_id` as
`source_audio_asset_id`. Only the first {config.MAX_DURATION_SECONDS:g} seconds are used.

`init_noise_level` sets how far the result departs from the source: low (about
`0.1`) stays very close, higher follows the prompt more. It is in the sampler's own
units ({config.MIN_INIT_NOISE_LEVEL:g} to {config.MAX_INIT_NOISE_LEVEL:g}), so try a few values and keep `seed` fixed
while you do. Leave `duration` out to use the source's length.
{_COMMON_NOTES}
"""

GENERATE_EXAMPLES = {
    "loop": {
        "summary": "A drum loop",
        "value": {"prompt": "128 BPM tech house drum loop", "duration": 20, "negative_prompt": "Low quality."},
    },
    "sfx": {
        "summary": "A sound effect",
        "value": {"prompt": "The sound of a hammer hitting a wooden surface.", "duration": 6},
    },
}

TRANSFORM_EXAMPLES = {
    "variation": {
        "summary": "A variation of a recording",
        "value": {
            "source_audio_asset_id": "<asset_id from POST /v1/uploads>",
            "prompt": "lo-fi hip hop drum loop, dusty vinyl",
            "init_noise_level": 0.5,
        },
    },
}

GENERATE_RESPONSES = {**common_docs.SUBMIT_RESPONSES}

TRANSFORM_RESPONSES = {
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
                }
            }
        },
    },
}
