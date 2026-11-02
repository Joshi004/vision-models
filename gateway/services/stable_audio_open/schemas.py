"""Pydantic request models for Stable Audio Open 1.0's audio endpoints.

Source recordings are always referenced by an asset_id obtained from
POST /v1/uploads, same convention as every other backend's file inputs --
see services/ltx/schemas.py for the fuller rationale.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from common.audio_inputs import ACCEPTED_AUDIO_FORMATS
from common.schemas import IncognitoOptionMixin, PartitionOptionMixin
from services.stable_audio_open import config


class _AudioRequestBase(PartitionOptionMixin, IncognitoOptionMixin):
    """Fields shared by generate and transform."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=config.MAX_PROMPT_CHARS,
        description=(
            "A short, concrete description of the sound or loop, in English: `128 BPM tech house drum loop`, "
            "`The sound of a hammer hitting a wooden surface.` Name the source, the material, the room and "
            f"the tempo if there is one. At most {config.MAX_PROMPT_CHARS} characters."
        ),
        examples=["128 BPM tech house drum loop"],
    )
    negative_prompt: str | None = Field(
        None,
        max_length=config.MAX_PROMPT_CHARS,
        description="What to steer away from, e.g. `Low quality.` Optional.",
        examples=[None],
    )
    num_inference_steps: int = Field(
        config.DEFAULT_STEPS,
        ge=1,
        le=config.MAX_STEPS,
        description=f"Diffusion steps (default {config.DEFAULT_STEPS}). The model card's own example uses 200; more steps are slower.",
        examples=[config.DEFAULT_STEPS],
    )
    cfg_scale: float = Field(
        config.DEFAULT_CFG_SCALE,
        ge=0,
        le=30,
        description="How strictly to follow the prompt (classifier-free guidance). The default is the model's usual setting.",
        examples=[config.DEFAULT_CFG_SCALE],
    )
    seed: int = Field(
        42,
        ge=0,
        le=2**31 - 1,
        description="Random seed. The same inputs and seed give the same audio on the same hardware.",
        examples=[42],
    )

    @model_validator(mode="after")
    def _check_common(self) -> "_AudioRequestBase":
        if not self.prompt.strip():
            raise ValueError("prompt must not be blank.")
        return self


class StableAudioOpenGenerateRequest(_AudioRequestBase):
    """Request body for POST /v1/stable-audio-open/audio/generate (text-to-audio)."""

    duration: float = Field(
        config.DEFAULT_DURATION_SECONDS,
        gt=0,
        le=config.MAX_DURATION_SECONDS,
        description=f"Length in seconds, up to {config.MAX_DURATION_SECONDS:g} (the model's fixed window).",
        examples=[config.DEFAULT_DURATION_SECONDS],
    )


class StableAudioOpenTransformRequest(_AudioRequestBase):
    """Request body for POST /v1/stable-audio-open/audio/transform (audio-to-audio)."""

    source_audio_asset_id: str = Field(
        ...,
        description=(
            f"asset_id (from POST /v1/uploads) of the recording to start from, at most {config.MAX_SOURCE_SECONDS:g} "
            f"seconds long; only its first {config.MAX_DURATION_SECONDS:g} seconds are used. Accepted file types: "
            f"{ACCEPTED_AUDIO_FORMATS}."
        ),
        examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"],
    )
    init_noise_level: float = Field(
        config.DEFAULT_INIT_NOISE_LEVEL,
        ge=config.MIN_INIT_NOISE_LEVEL,
        le=config.MAX_INIT_NOISE_LEVEL,
        description=(
            "How much noise is added to the source before regenerating it, in the sampler's own units. Low values "
            "(about 0.1) stay very close to the source, around 1 changes the sound but keeps the character, 5 or more ignores it almost entirely. "
            "There is no fixed 'right' value: try a few."
        ),
        examples=[config.DEFAULT_INIT_NOISE_LEVEL],
    )
    duration: float | None = Field(
        None,
        gt=0,
        le=config.MAX_DURATION_SECONDS,
        description=f"Length in seconds, up to {config.MAX_DURATION_SECONDS:g}. Leave out to use the source's length (capped).",
        examples=[None],
    )
