"""Pydantic request models for ACE-Step 1.5's music endpoints.

Source recordings are always referenced by an asset_id obtained from
POST /v1/uploads, same convention as every other backend's file inputs --
see services/ltx/schemas.py for the fuller rationale.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from common.audio_inputs import ACCEPTED_AUDIO_FORMATS
from common.schemas import IncognitoOptionMixin, PartitionOptionMixin
from services.ace_step15 import config

# "unknown" lets the model decide; otherwise an ISO-style language code such as en, zh, ja, pt-BR.
_LANGUAGE_PATTERN = r"^(unknown|[a-z]{2,3}(-[A-Za-z0-9]{2,8})?)$"
# Free text such as "C Major", "A minor", "F# minor", "Bb Major", "Am".
_KEYSCALE_PATTERN = r"^[A-Za-z0-9#♭ ]{0,20}$"


class _MusicRequestBase(PartitionOptionMixin, IncognitoOptionMixin):
    """Fields shared by generate, cover and repaint."""

    caption: str = Field(
        ...,
        min_length=1,
        max_length=config.MAX_CAPTION_CHARS,
        description=(
            "The overall description of the music: genre, mood, instruments, timbre, era, production style "
            "and vocal character, e.g. `melancholic indie folk, fingerpicked acoustic guitar, soft female "
            "vocal, warm intimate room sound`. Do **not** put tempo or key here -- use `bpm` and `keyscale`. "
            "For cover and repaint, describe the style to move to or the content of the repainted section. "
            f"At most {config.MAX_CAPTION_CHARS} characters."
        ),
        examples=["melancholic indie folk, fingerpicked acoustic guitar, soft female vocal, warm intimate room sound"],
    )
    lyrics: str = Field(
        "",
        max_length=config.MAX_LYRICS_CHARS,
        description=(
            "The words, with structure tags on their own lines: `[Intro]`, `[Verse 1]`, `[Pre-Chorus]`, "
            "`[Chorus]`, `[Bridge]`, `[Outro]`, `[Instrumental]`, `[Guitar Solo]`. At most one modifier per tag "
            "(`[Chorus - anthemic]`). About 6-10 syllables per line works best. Leave empty with "
            f"`instrumental: true` for no vocals. At most {config.MAX_LYRICS_CHARS} characters."
        ),
        examples=["[Verse 1]\nWalking home through the quiet rain\nStreetlights humming my name\n\n[Chorus - soft]\nHold on to the little light"],
    )
    instrumental: bool = Field(
        False,
        description="No vocals, whatever `lyrics` says. If `lyrics` is empty it is sent as `[Instrumental]`.",
        examples=[False],
    )
    vocal_language: str = Field(
        "unknown",
        pattern=_LANGUAGE_PATTERN,
        description=(
            "Language of the sung lyrics as a short code (`en`, `zh`, `ja`, `ko`, `es`, `fr`, `de`, `pt`, ...; "
            "50+ languages) or `unknown` to let the model detect it."
        ),
        examples=["en"],
    )
    num_inference_steps: int = Field(
        config.DEFAULT_STEPS,
        ge=1,
        le=config.MAX_STEPS,
        description=f"Diffusion steps. The XL SFT model is a 50-step model (default {config.DEFAULT_STEPS}); fewer is faster and rougher.",
        examples=[config.DEFAULT_STEPS],
    )
    guidance_scale: float = Field(
        config.DEFAULT_GUIDANCE_SCALE,
        ge=0,
        le=30,
        description="How strictly to follow the caption and lyrics. 5-9 is the useful range; the default is upstream's.",
        examples=[config.DEFAULT_GUIDANCE_SCALE],
    )
    seed: int = Field(
        42,
        ge=0,
        le=2**31 - 1,
        description="Random seed. Fix it while tuning other settings, otherwise changes are hidden by randomness.",
        examples=[42],
    )

    @model_validator(mode="after")
    def _check_common(self) -> "_MusicRequestBase":
        if not self.caption.strip():
            raise ValueError("caption must not be blank.")
        return self


class AceStepGenerateRequest(_MusicRequestBase):
    """Request body for POST /v1/ace-step15/music/generate (text-to-music)."""

    duration: float | None = Field(
        None,
        ge=config.MIN_DURATION_SECONDS,
        le=config.MAX_DURATION_SECONDS,
        description=(
            f"Length in seconds, {config.MIN_DURATION_SECONDS:g} to {config.MAX_DURATION_SECONDS:g}. Leave it out "
            "to let the model choose (recommended when lyrics are given)."
        ),
        examples=[None],
    )
    bpm: int | None = Field(None, ge=30, le=300, description="Tempo in beats per minute. Leave out to let the model choose.", examples=[None])
    keyscale: str = Field(
        "",
        pattern=_KEYSCALE_PATTERN,
        description="Musical key, e.g. `C Major`, `A minor`, `F# minor`. Empty lets the model choose.",
        examples=[""],
    )
    time_signature: Literal["2", "3", "4", "6"] | None = Field(
        None,
        description="Beats per bar: `2` (2/4), `3` (3/4), `4` (4/4) or `6` (6/8). Leave out to let the model choose.",
        examples=[None],
    )
    thinking: bool = Field(
        True,
        description=(
            "Let the 4B language model plan first: fill in missing BPM, key and duration, refine the caption and "
            "draft the song's semantic structure. Turn it off when you give precise metadata yourself."
        ),
        examples=[True],
    )


class _SourceAudioRequest(_MusicRequestBase):
    source_audio_asset_id: str = Field(
        ...,
        description=(
            f"asset_id (from POST /v1/uploads) of the source recording, at most {config.MAX_SOURCE_SECONDS:g} seconds "
            f"long. Accepted file types: {ACCEPTED_AUDIO_FORMATS}."
        ),
        examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"],
    )


class AceStepCoverRequest(_SourceAudioRequest):
    """Request body for POST /v1/ace-step15/music/cover (restyle a recording)."""

    cover_strength: float = Field(
        1.0,
        ge=0,
        le=1,
        description=(
            "How closely the result follows the source's structure. `1.0` stays very close, `0.5` is a balanced "
            "transformation, around `0.2` is a loose style transfer."
        ),
        examples=[0.8],
    )


class AceStepRepaintRequest(_SourceAudioRequest):
    """Request body for POST /v1/ace-step15/music/repaint (regenerate part of a recording)."""

    repaint_start: float = Field(..., ge=0, description="Where the regenerated section starts, in seconds.", examples=[10.0])
    repaint_end: float | None = Field(
        None,
        gt=0,
        description="Where it ends, in seconds. Leave out to regenerate to the end of the recording.",
        examples=[20.0],
    )

    @model_validator(mode="after")
    def _check_range(self) -> "AceStepRepaintRequest":
        if self.repaint_end is not None and self.repaint_end <= self.repaint_start:
            raise ValueError("repaint_end must be greater than repaint_start.")
        return self
