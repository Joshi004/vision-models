"""Pydantic request model for Breeze TTS 2's text-to-speech endpoint.

A reference recording is always referenced by an asset_id obtained from
POST /v1/uploads, same convention as every other backend's file inputs --
see services/ltx/schemas.py for the fuller rationale.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from common.schemas import PartitionOptionMixin
from services.breeze_tts import config

_ACCEPTED_REFERENCE_FORMATS = ", ".join(sorted(ext.lstrip(".") for ext in config.REFERENCE_EXTENSIONS))


class SynthesizeRequest(PartitionOptionMixin):
    """Request body for POST /v1/breeze-tts/synthesize.

    The mode is chosen by which optional fields are present:
    - **Voice Design**: `text` (+ `instruction`) -- no reference audio.
    - **Voice Clone**: `text` + `reference_audio_asset_id` + `reference_text`.
    - **Voice Direction**: Voice Clone fields + `instruction`.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=config.MAX_TEXT_CHARS,
        description=(
            "The text to speak (English or Chinese). Inline vocal events are supported: "
            "parentheses in English, e.g. `(laugh)`, `(cough)`, `(clears throat)`, `(sigh)`; "
            "square brackets in Chinese, e.g. `[笑]`, `[咳嗽]`, `[清嗓子]`, `[叹气]`. "
            f"At most {config.MAX_TEXT_CHARS} characters."
        ),
        examples=["(sigh) Welcome aboard. Your journey begins now."],
    )
    instruction: str | None = Field(
        None,
        max_length=config.MAX_INSTRUCTION_CHARS,
        description=(
            "Natural-language voice description (Voice Design, with no reference audio) or "
            "delivery direction (Voice Direction, with reference audio) -- e.g. tone, emotion, "
            "pace. Write it in the same language as `text`. Use `cfg_scale` 4 to strengthen "
            "how closely it is followed."
        ),
        examples=["A warm, thoughtful young woman with a clear voice and a calm, reflective delivery."],
    )
    reference_audio_asset_id: str | None = Field(
        None,
        description=(
            "asset_id (from POST /v1/uploads) of a clean recording of the voice to clone, with "
            "minimal background noise. Must be sent together with `reference_text`. "
            f"Accepted file types: {_ACCEPTED_REFERENCE_FORMATS}."
        ),
        examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"],
    )
    reference_text: str | None = Field(
        None,
        max_length=config.MAX_REFERENCE_TEXT_CHARS,
        description=(
            "The exact transcript of the reference recording. Must be sent together with "
            "`reference_audio_asset_id`."
        ),
        examples=["This is the exact transcript of the reference audio."],
    )
    seed: int = Field(
        42,
        ge=0,
        le=2**31 - 1,
        description="Random seed. The same inputs and seed give the same audio on the same hardware.",
        examples=[42],
    )
    cfg_scale: float = Field(
        1.0,
        gt=0,
        le=10,
        description=(
            "Classifier-free guidance scale. Upstream recommends 4 when `instruction` is given "
            "(stronger instruction-following); 1 (default) otherwise."
        ),
        examples=[1.0],
    )

    @model_validator(mode="after")
    def _reference_fields_go_together(self) -> "SynthesizeRequest":
        has_audio = self.reference_audio_asset_id is not None
        has_text = bool(self.reference_text and self.reference_text.strip())
        if has_audio != has_text:
            raise ValueError("reference_audio_asset_id and reference_text must be provided together.")
        if not self.text.strip():
            raise ValueError("text must not be blank.")
        return self
