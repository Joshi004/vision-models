"""Pydantic request model for Parakeet's transcription endpoint.

A recording is always referenced by an asset_id obtained from
POST /v1/uploads, same convention as every other backend's file inputs --
see services/ltx/schemas.py for the fuller rationale. There is no curated
registry here (unlike services/rvc/schemas.py's `voice`): every request
brings its own file, and there is exactly one model (no selection needed).
"""

from __future__ import annotations

from pydantic import Field

from common.schemas import PartitionOptionMixin
from services.parakeet import config

# Every accepted input extension, formatted once for reuse in
# TranscribeRequest.audio_asset_id's own field description -- see
# config.INPUT_EXTENSIONS for the full reasoning (run_transcribe.py's own
# ffmpeg decode step accepts any of these, video included).
_ACCEPTED_INPUT_FORMATS = ", ".join(sorted(ext.lstrip(".") for ext in config.INPUT_EXTENSIONS))


class TranscribeRequest(PartitionOptionMixin):
    """Request body for POST /v1/parakeet/transcribe.

    Transcribes an existing audio or video recording to text, with both
    word-level and segment-level timestamps -- see that endpoint's own
    description for the exact shape of the resulting transcript. There is
    no `language` field: the model auto-detects among 25 supported
    European languages. There is no `seed` field either: transcription is
    deterministic given the same input and model, unlike LTX/Wan-Animate's
    diffusion-based sampling.
    """

    audio_asset_id: str = Field(
        ...,
        description=(
            "asset_id (from POST /v1/uploads) of the audio or video file to transcribe. If it's a "
            "video, only its first audio stream is used -- otherwise it's treated exactly like a "
            f"plain audio file. Accepted file types: {_ACCEPTED_INPUT_FORMATS}."
        ),
        examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"],
    )
