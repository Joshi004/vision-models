"""Pydantic request models for Stable Audio 3's audio endpoints.

Source recordings are always referenced by an asset_id obtained from
POST /v1/uploads, same convention as every other backend's file inputs --
see services/ltx/schemas.py for the fuller rationale.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from common.audio_inputs import ACCEPTED_AUDIO_FORMATS
from common.schemas import IncognitoOptionMixin, PartitionOptionMixin
from services.stable_audio3 import config

_EXAMPLE_PROMPT = (
    "TrackType: Music, VocalType: Instrumental, Genre: Funk. A funky hip hop instrumental with live recorded "
    "instrumentation that has the vibe of a 70's TV show theme. Warm, textured analogue sound."
)


class _AudioRequestBase(PartitionOptionMixin, IncognitoOptionMixin):
    """Fields shared by generate, transform and inpaint."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=config.MAX_PROMPT_CHARS,
        description=(
            "What to generate, in English. Long descriptive prose works: genre, each instrument and how it sounds, "
            "mood and energy, BPM, production style. Metadata-style tags help: `TrackType: Music`, "
            "`VocalType: Instrumental`, `Genre: Funk, Genre: Jazz`, `Instruments: Guitar, Saxophone, Bass`; "
            "`TrackType: Instrument` for one isolated instrument; `TrackType: SFX` for sound effects. "
            "**No intelligible vocals.** For a sound effect say what makes the sound, how it is triggered and for how "
            f"long. At most {config.MAX_PROMPT_CHARS} characters."
        ),
        examples=[_EXAMPLE_PROMPT],
    )
    num_inference_steps: int = Field(
        config.DEFAULT_STEPS,
        ge=1,
        le=config.MAX_STEPS,
        description=(
            f"Denoising steps (default {config.DEFAULT_STEPS}, the checkpoint's own setting). There is no `cfg_scale` and no "
            "`negative_prompt`: they have no effect on this checkpoint."
        ),
        examples=[config.DEFAULT_STEPS],
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


_DURATION_NOTE = (
    f"Seconds, up to {config.MAX_DURATION_SECONDS:g}. Pick a length that fits the description: longer is not automatically "
    "better, and sound effects want short durations."
)
_SOURCE_NOTE = (
    f"asset_id (from POST /v1/uploads) of the recording, at most {config.MAX_SOURCE_SECONDS:g} seconds long. "
    f"Accepted file types: {ACCEPTED_AUDIO_FORMATS}."
)


class StableAudio3GenerateRequest(_AudioRequestBase):
    """Request body for POST /v1/stable-audio3/audio/generate (text-to-audio)."""

    duration: float = Field(
        config.DEFAULT_DURATION_SECONDS,
        gt=0,
        le=config.MAX_DURATION_SECONDS,
        description=_DURATION_NOTE,
        examples=[config.DEFAULT_DURATION_SECONDS],
    )


class StableAudio3TransformRequest(_AudioRequestBase):
    """Request body for POST /v1/stable-audio3/audio/transform (audio-to-audio)."""

    source_audio_asset_id: str = Field(..., description=_SOURCE_NOTE, examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"])
    init_noise_level: float = Field(
        config.DEFAULT_INIT_NOISE_LEVEL,
        ge=0,
        le=1,
        description=(
            "How much of the source is kept. `0` keeps it, `1` ignores it. Starting points: about `0.4` for timbre "
            "transfer (violin to metal guitar), about `0.5` for beatbox to lo-fi beat, about `0.6` for a style shift; "
            "`0.1` is a close variation. Needs tinkering per input."
        ),
        examples=[config.DEFAULT_INIT_NOISE_LEVEL],
    )
    duration: float | None = Field(
        None,
        gt=0,
        le=config.MAX_DURATION_SECONDS,
        description=f"{_DURATION_NOTE} Leave out to use the source's length.",
        examples=[None],
    )


class StableAudio3InpaintRequest(_AudioRequestBase):
    """Request body for POST /v1/stable-audio3/audio/inpaint (regenerate part of a recording, or extend it)."""

    source_audio_asset_id: str = Field(..., description=_SOURCE_NOTE, examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"])
    mask_start: float = Field(
        ...,
        ge=0,
        description=(
            "Where the regenerated section starts, in seconds. It must be inside the recording. To **extend** a clip, set "
            "it to the clip's length (or a little earlier) and make `duration` longer."
        ),
        examples=[10.0],
    )
    mask_end: float = Field(..., gt=0, description="Where the regenerated section ends, in seconds. May be past the end of the recording when extending.", examples=[20.0])
    duration: float | None = Field(
        None,
        gt=0,
        le=config.MAX_DURATION_SECONDS,
        description=(
            f"{_DURATION_NOTE} Leave out to use the source's length, or `mask_end` if that is longer. Must be at least `mask_end`."
        ),
        examples=[None],
    )

    @model_validator(mode="after")
    def _check_mask(self) -> "StableAudio3InpaintRequest":
        if self.mask_end <= self.mask_start:
            raise ValueError("mask_end must be greater than mask_start.")
        if self.mask_end > config.MAX_DURATION_SECONDS:
            raise ValueError(f"mask_end can be at most {config.MAX_DURATION_SECONDS:g} seconds.")
        if self.duration is not None and self.duration < self.mask_end:
            raise ValueError("duration must be at least mask_end.")
        return self
