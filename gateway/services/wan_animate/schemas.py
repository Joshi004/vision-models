"""Pydantic request model for Wan-Animate v1's one generation endpoint
(replace mode).

File inputs (the driving video, the reference image) are referenced by an
asset_id obtained from POST /v1/uploads, not embedded directly in this
request body -- same convention as every LTX-2.3 endpoint, see
services/ltx/schemas.py for the fuller rationale.

Field-level `description`/`examples` here are what an LLM agent (or a human
in /docs) sees when inspecting the schema directly -- written to stand
alone, without needing the endpoint description open at the same time,
though both agree with each other on the details that matter (single-person
video only, what `keep_audio` actually does, etc.).
"""

from __future__ import annotations

from pydantic import Field

from common.schemas import PartitionOptionMixin


class ReplaceRequest(PartitionOptionMixin):
    """Request body for POST /v1/wan-animate/videos/replace.

    Swaps the person in an existing video for a reference character while
    keeping that video's background, camera motion, and lighting --
    "replace" mode in the model's own terminology. There is no `prompt`
    field: the underlying pipeline never forwards one to the model, and the
    model's own authors advise against custom prompts for this mode.
    """

    video_asset_id: str = Field(
        ...,
        description=(
            "asset_id (from POST /v1/uploads) of the existing video to edit. Must show "
            "exactly one person on screen throughout -- the official preprocessing step's "
            "mask extraction is documented as single-person-video-only and may produce "
            "incorrect results, or fail outright, on multi-person footage."
        ),
        examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"],
    )
    image_asset_id: str = Field(
        ...,
        description=(
            "asset_id (from POST /v1/uploads) of a single, clear reference photo of the "
            "character to swap into the video. A front-facing, well-lit, unoccluded photo "
            "works best; a large body-proportion mismatch between this character and the "
            "original person can produce visible artifacts (a known limitation, not checked "
            "by this API)."
        ),
        examples=["7a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d"],
    )
    seed: int = Field(
        10,
        ge=0,
        description="Random seed. Same seed + same inputs = same output; change it for a different take.",
        examples=[10],
    )
    use_relighting_lora: bool = Field(
        True,
        description=(
            "Apply the official Relighting LoRA so the swapped-in character's lighting and "
            "color tone match the original scene instead of looking pasted in. Recommended; "
            "turn off only to compare quality with/without it."
        ),
        examples=[True],
    )
    keep_audio: bool = Field(
        True,
        description=(
            "The model itself only ever produces silent video. If true (default), the source "
            "video's own original audio track is muxed onto the output unchanged. If false, "
            "the output has no audio track at all."
        ),
        examples=[True],
    )
