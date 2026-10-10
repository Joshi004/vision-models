"""Long-form OpenAPI/Swagger documentation text for Breeze TTS 2's endpoint,
kept out of router.py so the route itself stays readable. Docs shared by
every backend (uploads, jobs, app-level description) live in
common/openapi_docs.py instead.
"""

from __future__ import annotations

from common import openapi_docs as common_docs
from services.breeze_tts import config

_ACCEPTED_FORMATS_LIST = ", ".join(sorted(ext.lstrip(".") for ext in config.REFERENCE_EXTENSIONS))

SYNTHESIZE = f"""
Generates speech from text with **Breeze TTS 2** (BreezeBlue; English and
Chinese). One endpoint covers three modes, selected by which optional fields
you send:

| Mode | Send | Result |
|------|------|--------|
| **Voice Design** | `text` (+ `instruction`) | A new voice created from a natural-language description. No reference audio. |
| **Voice Clone** | `text` + `reference_audio_asset_id` + `reference_text` | The reference speaker's timbre, rhythm, emotion and style. |
| **Voice Direction** | Voice Clone fields + `instruction` | The reference speaker's identity, with tone/emotion/pace steered by the instruction. |

`reference_audio_asset_id` and `reference_text` must be sent **together** (the
text must be the exact transcript of the recording); sending only one is
rejected with 422. Upload the reference first with `POST /v1/uploads`.
Accepted reference formats: {_ACCEPTED_FORMATS_LIST} (decoded to WAV before
synthesis, so phone voice notes work). Use a clean recording with minimal
background noise.

**Vocal events** can be written inline in `text`: `(laugh)`, `(cough)`,
`(clears throat)`, `(sigh)` in English; `[笑]`, `[咳嗽]`, `[清嗓子]`, `[叹气]`
in Chinese. Write `instruction` in the same language as `text`.

**`cfg_scale`:** 1 (default) is fine for plain synthesis and cloning. Upstream
recommends **4** whenever `instruction` is set, to strengthen
instruction-following.

**Limits:** `text` is at most {config.MAX_TEXT_CHARS} characters;
`instruction` at most {config.MAX_INSTRUCTION_CHARS}. A single request can
generate about two minutes of audio at most; anything beyond that is cut off
rather than rejected. {config.MAX_TEXT_CHARS} English characters measured at
~92 s of audio. Chinese is denser per character, so keep Chinese text to
roughly half that (an estimate, not measured) -- or split long scripts across
several requests.

**Result:** `output.wav` -- mono, 24 kHz, 16-bit PCM. Download with
`GET /v1/jobs/{{job_id}}/result`. The same inputs and `seed` give the same
audio on the same hardware.

**Timing:** each request is a one-shot GPU job. Loading the model dominates
for short text (measured: ~20 s for a short sentence on a warm node, ~70 s on
a cold node; ~170 s for a 1000-character passage on a cold node), so expect
roughly one to three minutes end to end including queueing. No live progress: `progress` is always null.

**License and responsible use:** the Breeze TTS 2 *weights and self-hosted
outputs* are licensed for **research and non-commercial use only**. Only clone
voices you have the rights and consent to use; impersonation and fraud are
prohibited by the model's license.
"""

SYNTHESIZE_EXAMPLES = {
    "voice_design": {
        "summary": "Voice Design: create a voice from a description",
        "value": {
            "text": "(sigh) Welcome aboard. Your journey begins now.",
            "instruction": "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery.",
            "cfg_scale": 4,
        },
    },
    "voice_clone": {
        "summary": "Voice Clone: speak new text in a reference voice",
        "value": {
            "text": "It is good to hear your voice again after all this time.",
            "reference_audio_asset_id": "<asset_id of the reference recording, from POST /v1/uploads>",
            "reference_text": "This is the exact transcript of the reference audio.",
        },
    },
    "voice_direction": {
        "summary": "Voice Direction: reference voice, steered delivery",
        "value": {
            "text": "(clears throat) We need to discuss what happened last night.",
            "reference_audio_asset_id": "<asset_id of the reference recording, from POST /v1/uploads>",
            "reference_text": "This is the exact transcript of the reference audio.",
            "instruction": "Speak slowly with a restrained, serious tone.",
            "cfg_scale": 4,
        },
    },
}

SYNTHESIZE_RESPONSES = {
    **common_docs.SUBMIT_RESPONSES,
    400: {
        "description": "The reference recording's asset_id doesn't exist, or its file isn't an accepted format.",
        "content": {
            "application/json": {
                "examples": {
                    "unknown_asset": {
                        "summary": "Unknown or missing asset_id",
                        "value": {"detail": "Invalid asset reference: 'Unknown asset_id: abc123'"},
                    },
                    "unsupported_format": {
                        "summary": "Unsupported reference file type",
                        "value": {
                            "detail": (
                                f"Unsupported file extension '.xyz' (accepted: {_ACCEPTED_FORMATS_LIST}): "
                                "reference.xyz"
                            )
                        },
                    },
                }
            }
        },
    },
}
