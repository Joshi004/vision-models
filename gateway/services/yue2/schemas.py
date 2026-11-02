"""Pydantic request models for YuE2's song endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from common.schemas import IncognitoOptionMixin, PartitionOptionMixin
from services.yue2 import config

_EXAMPLE_STYLE = (
    "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, "
    "lyrical memorable melody, unhurried phrasing, 88 BPM"
)
_EXAMPLE_ABC = (
    "X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=88\n"
    'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
    'V: Ins clef=treble name="Ins Melody" snm="Inst."\n'
    "K:G\n% verse\nV: Vocal\nB8d8c8A8|F16G16|\nV: Ins\nZ2|"
)
_EXAMPLE_LYRICS = "[Verse]\nStreetlights hum along the avenue\nI keep your number in my shoe\n\n[Chorus]\nStay with me until the morning light\nHold the quiet of the night"


class _SongRequestBase(PartitionOptionMixin, IncognitoOptionMixin):
    """Fields shared by generate and cover."""

    style: str = Field(
        ...,
        min_length=1,
        max_length=config.MAX_STYLE_CHARS,
        description=(
            "One line describing the song: **language first**, then genre, instruments, vocal character and "
            "tempo, e.g. `English, warm piano pop, expressive female voice, acoustic piano, light drums, 88 BPM`. "
            "There is no separate BPM field -- tempo goes here. "
            f"At most {config.MAX_STYLE_CHARS} characters."
        ),
        examples=[_EXAMPLE_STYLE],
    )
    lyrics: str = Field(
        ...,
        min_length=1,
        max_length=config.MAX_LYRICS_CHARS,
        description=(
            "The words to sing, with a section tag on its own line before each part (`[Verse]`, `[Chorus]`, "
            "`[Bridge]`, ...) and a blank line between sections. "
            f"At most {config.MAX_LYRICS_CHARS} characters."
        ),
        examples=[_EXAMPLE_LYRICS],
    )
    cfg_scale: float | None = Field(
        None,
        ge=0,
        le=20,
        description=(
            "Text guidance. Leave it out: the defaults are tuned (1.0 with a plan, 1.01 with `cot: off`). "
            "Upstream suggests trying 1.2 for stronger adherence to the text; changing it may change quality."
        ),
        examples=[None],
    )
    seed: int = Field(
        831001,
        ge=0,
        le=config.MAX_SEED,
        description=(
            "Random seed. One request makes one candidate; to choose among several, send several seeds and "
            "pick by ear."
        ),
        examples=[831001],
    )

    @model_validator(mode="after")
    def _check_common(self) -> "_SongRequestBase":
        if not self.style.strip() or not self.lyrics.strip():
            raise ValueError("style and lyrics must not be blank.")
        return self


class Yue2GenerateRequest(_SongRequestBase):
    """Request body for POST /v1/yue2/songs/generate (a new song)."""

    cot: Literal["full", "melody", "off"] = Field(
        "full",
        description=(
            "How much the model plans before singing. `full`: melody and chords (default, best quality). "
            "`melody`: melody only. `off`: no plan, a little faster and clearly lower quality in upstream's "
            "comparison."
        ),
        examples=["full"],
    )


class Yue2CoverRequest(_SongRequestBase):
    """Request body for POST /v1/yue2/songs/cover (a song that follows a score you supply)."""

    abc: str = Field(
        ...,
        min_length=1,
        max_length=config.MAX_ABC_CHARS,
        description=(
            "The melody as ABC notation text in the model's native format: header lines (`X:`, `M:`, `L:`, `Q:`, "
            "`K:`), two monophonic voices declared as `V: Vocal` and `V: Ins`, and section comments such as "
            "`% verse` / `% chorus` between groups of bars. A melody-only score (recommended for covers) has no "
            "chord symbols. Make `lyrics` follow the score's sections in order. Arbitrary ABC dialects may need "
            "conversion first. (Producing an ABC score from a recording is not part of this API; upstream uses a "
            "separate tool, SheetSage2.) "
            f"At most {config.MAX_ABC_CHARS} characters."
        ),
        examples=[_EXAMPLE_ABC],
    )
    cot: Literal["full", "melody"] = Field(
        "melody",
        description="Planning mode around the supplied score. `melody` is recommended for covers; `off` is not allowed because a score is supplied.",
        examples=["melody"],
    )

    @model_validator(mode="after")
    def _check_abc(self) -> "Yue2CoverRequest":
        if not self.abc.strip():
            raise ValueError("abc must not be blank.")
        return self
