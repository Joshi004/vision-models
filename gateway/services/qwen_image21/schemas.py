"""Pydantic request models for Qwen-Image-2.1's image endpoints.

Reference images are always referenced by an asset_id obtained from
POST /v1/uploads, same convention as every other backend's file inputs --
see services/ltx/schemas.py for the fuller rationale.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from common.schemas import IncognitoOptionMixin, PartitionOptionMixin
from services.qwen_image21 import config

AspectRatio = Literal["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]

_ASPECT_RATIO_SIZES = ", ".join(f"`{name}` = {w}x{h}" for name, (w, h) in config.ASPECT_RATIOS.items())
_ACCEPTED_IMAGE_FORMATS = ", ".join(sorted(ext.lstrip(".") for ext in config.REFERENCE_IMAGE_EXTENSIONS))


class _ImageRequestBase(PartitionOptionMixin, IncognitoOptionMixin):
    """Fields shared by generate and edit."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=config.MAX_PROMPT_CHARS,
        description=(
            "What to generate (or, for an edit, how to change the reference image(s)). Long, specific "
            "prompts work well; text to be drawn in the image should be quoted in the prompt. For a "
            "**transparent (RGBA) image**, use this wording: `This is an RGBA image with "
            "transparency. <your description>. The image has alpha channel and the background is "
            f"transparent.` At most {config.MAX_PROMPT_CHARS} characters."
        ),
        examples=["A ceramic teapot on a wooden table, soft window light, photorealistic."],
    )
    mode: Literal["fast", "quality"] = Field(
        "fast",
        description=(
            "`fast`: the Turbo checkpoint, 8 denoising steps. `quality`: the base checkpoint, "
            f"{config.DEFAULT_QUALITY_STEPS} steps by default (see `num_inference_steps`). The same "
            "seed gives a different image in each mode -- they are different checkpoints."
        ),
        examples=["fast"],
    )
    num_inference_steps: int | None = Field(
        None,
        ge=1,
        le=config.MAX_QUALITY_STEPS,
        description=(
            f"Denoising steps. Only valid with `mode: quality` (default {config.DEFAULT_QUALITY_STEPS}); "
            "`fast` always uses Turbo's own saved 8-step schedule, so sending it with `fast` is "
            "rejected with 422."
        ),
        examples=[None],
    )
    seed: int = Field(
        42,
        ge=0,
        le=2**31 - 1,
        description="Random seed. The same inputs and seed give the same image on the same hardware.",
        examples=[42],
    )

    @model_validator(mode="after")
    def _check_common(self) -> "_ImageRequestBase":
        if not self.prompt.strip():
            raise ValueError("prompt must not be blank.")
        if self.mode == "fast" and self.num_inference_steps is not None:
            raise ValueError("num_inference_steps is only valid with mode 'quality'.")
        return self


class QwenImageGenerateRequest(_ImageRequestBase):
    """Request body for POST /v1/qwen-image21/images/generate (text-to-image)."""

    aspect_ratio: AspectRatio = Field(
        "1:1",
        description=f"Output shape, at the model's native 2K size: {_ASPECT_RATIO_SIZES}.",
        examples=["1:1"],
    )


class QwenImageEditRequest(_ImageRequestBase):
    """Request body for POST /v1/qwen-image21/images/edit (editing and
    multi-reference composition)."""

    reference_image_asset_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=config.MAX_REFERENCE_IMAGES,
        description=(
            f"asset_ids (from POST /v1/uploads) of 1 to {config.MAX_REFERENCE_IMAGES} reference images. "
            "With several, refer to them in the prompt in order (\"the first character\", ...). "
            f"Accepted file types: {_ACCEPTED_IMAGE_FORMATS}."
        ),
        examples=[["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"]],
    )
    aspect_ratio: AspectRatio | None = Field(
        None,
        description=(
            "Output shape. Optional: when omitted, the result follows the **last** reference image's "
            "aspect ratio, at about 4 megapixels for one image and smaller for several. Otherwise one of: "
            f"{_ASPECT_RATIO_SIZES}."
        ),
        examples=[None],
    )
