"""Pydantic request/response models for RVC's voice-conversion endpoints.

A recording is always referenced by an asset_id obtained from
POST /v1/uploads, same convention as every other backend's file inputs --
see services/ltx/schemas.py for the fuller rationale. Unlike a fresh
per-request upload, `voice` selects a *pre-installed*, reusable RVC voice
model (a .pth + .index pair someone keeps using across many requests, not
uploaded fresh each time) -- see services/rvc/config.py's VOICE_REGISTRY and
VOICE_CONVERSION_RESEARCH.md section 1 for why this is a deliberately
different shape than Wan-Animate's per-request reference image.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from common.schemas import PartitionOptionMixin
from services.rvc import config

# Every accepted source-recording extension, formatted once for reuse in
# both ConvertRequest.source_audio_asset_id's and BatchSource.asset_id's
# own field descriptions below -- see config.INPUT_EXTENSIONS for the full
# reasoning (both endpoints' own Slurm-side wrappers decode any of these
# via ffmpeg before Applio itself ever sees the file).
_ACCEPTED_INPUT_FORMATS = ", ".join(sorted(ext.lstrip(".") for ext in config.INPUT_EXTENSIONS))


class ConversionSettings(PartitionOptionMixin):
    """Fields shared by every RVC conversion request -- which pre-installed
    voice to convert to, and its own tuning knobs (pitch/f0_method/
    index_rate/protect/export_format) -- factored out so ConvertRequest
    (one file) and BatchConvertRequest (many files) can never drift apart
    on these. Only what differs between them -- which file(s) to convert
    -- lives on the two subclasses below. Not itself a request body for
    any endpoint.
    """

    voice: str = Field(
        ...,
        description=(
            "Which pre-installed voice to convert to. See GET /v1/rvc/voices for the current "
            "list -- this is a fixed, curated set of already-installed voice models, not an "
            "arbitrary name or an uploaded file."
        ),
        examples=["<voice name, from GET /v1/rvc/voices>"],
    )
    pitch: int = Field(
        0,
        ge=-24,
        le=24,
        description="Pitch shift in semitones applied on top of the conversion. 0 = no shift.",
        examples=[0],
    )
    f0_method: Literal["rmvpe", "crepe", "crepe-tiny", "fcpe"] = Field(
        "rmvpe",
        description=(
            "Pitch-extraction algorithm. 'rmvpe' (default) is fast and high-accuracy for general "
            "speech. 'crepe' is slower but the most accurate -- worth trying when quality matters "
            "more than turnaround time. 'crepe-tiny' trades accuracy for speed; 'fcpe' is a fast, "
            "modern alternative."
        ),
        examples=["rmvpe"],
    )
    index_rate: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description=(
            "How strongly the target voice's own training-data retrieval index influences the "
            "output. Higher = more faithful to the target voice's exact timbre."
        ),
        examples=[0.75],
    )
    protect: float = Field(
        0.33,
        ge=0.0,
        le=0.5,
        description=(
            "Protects consonants and breath sounds from conversion artifacts. Higher = more "
            "protection, at some cost to how strongly the target voice comes through on those sounds."
        ),
        examples=[0.33],
    )
    export_format: Literal["WAV", "MP3", "FLAC", "OGG"] = Field(
        "WAV",
        description=(
            "Output audio file format. M4A isn't offered: the WAV-to-M4A encoding step isn't "
            "reliable on this deployment (no M4A encoder in the installed audio library), and "
            "can silently produce a file named .m4a that actually contains plain WAV bytes."
        ),
        examples=["WAV"],
    )

    @field_validator("voice")
    @classmethod
    def _voice_must_be_registered(cls, value: str) -> str:
        if value not in config.VOICE_REGISTRY:
            available = ", ".join(sorted(config.VOICE_REGISTRY)) or "(none installed)"
            raise ValueError(f"Unknown voice {value!r}. Available voices: {available}")
        return value


class ConvertRequest(ConversionSettings):
    """Request body for POST /v1/rvc/convert.

    Converts an existing recording to sound like one of the pre-installed
    voices (see GET /v1/rvc/voices for the current list), while keeping the
    recording's own timing, pauses, and delivery -- this is voice
    *conversion*, not text-to-speech voice cloning: there is no
    `text`/`prompt` field, because nothing here is generated. There is also
    no `seed` field -- RVC inference is deterministic given the same
    inputs, unlike LTX/Wan-Animate's diffusion sampling.
    """

    source_audio_asset_id: str = Field(
        ...,
        description=(
            "asset_id (from POST /v1/uploads) of the recording to convert. Its own delivery "
            "(timing, pauses, emphasis, emotion) is preserved -- only the voice's timbre changes. "
            f"Accepted file types: {_ACCEPTED_INPUT_FORMATS}."
        ),
        examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"],
    )


class BatchSource(BaseModel):
    """One entry of BatchConvertRequest.sources."""

    asset_id: str = Field(
        ...,
        description=(
            "asset_id (from POST /v1/uploads) of one recording to convert. "
            f"Accepted file types: {_ACCEPTED_INPUT_FORMATS}."
        ),
        examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"],
    )


class BatchConvertRequest(ConversionSettings):
    """Request body for POST /v1/rvc/batch-convert.

    Converts every recording in `sources` to the same pre-installed voice,
    with the same settings, as a single job -- one GPU processes the files
    one at a time internally, not in parallel, but submitting them together
    means one queue wait instead of `len(sources)` separate ones. The
    result is a single output.zip containing one converted file per source,
    each name prefixed with that source's 1-based position in this request
    (e.g. `001_<original filename stem>_output.<ext>`) so the mapping back
    to `sources` is unambiguous.

    All-or-nothing: if any one file fails to convert, the whole job fails
    and no zip is produced -- there is no partial-success result. At most
    one batch runs at a time on this deployment; a second batch submitted
    while one is already running/queued waits its turn rather than running
    concurrently.
    """

    sources: list[BatchSource] = Field(
        ...,
        min_length=1,
        max_length=config.BATCH_MAX_FILES,
        description=(
            f"Recordings to convert, 1 to {config.BATCH_MAX_FILES} of them. Every one is "
            "converted to the same `voice` with the same settings; the output zip's files are in "
            "this same order."
        ),
        examples=[
            [
                {"asset_id": "3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"},
                {"asset_id": "8a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d"},
            ]
        ],
    )


class VoiceListResponse(BaseModel):
    """Response from GET /v1/rvc/voices."""

    voices: list[str] = Field(
        ...,
        description="Names of every currently-installed voice model, alphabetically -- exactly the values `voice` accepts on POST /v1/rvc/convert.",
        examples=[["<voice_a>", "<voice_b>", "<voice_c>"]],
    )
