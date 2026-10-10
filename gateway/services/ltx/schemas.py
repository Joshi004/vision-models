"""Pydantic request models for LTX-2.3's generation endpoints.

File inputs (images, audio, video) are referenced by an asset_id obtained
from POST /v1/uploads, not embedded directly in these request bodies (see
services/ltx/router.py). That keeps every generation endpoint a plain JSON
request, which avoids the awkwardness of mixing repeated file+metadata
groups (e.g. several --image entries, each with its own frame index and
strength) into a single multipart/form-data request.

Every request body here also gets an optional `partition` field via
common.schemas.PartitionOptionMixin -- see that module for the shared
fallback behavior, identical across every backend.

Field-level `description`/`examples` here are what an LLM agent (or a human
in /docs) sees when inspecting the schema directly -- they're written to
stand alone, without needing the endpoint description or services/ltx/GUIDE.md
open at the same time, though all three agree with each other on the details
that matter (frame_idx semantics, the 8k+1 frame-count rule, which fields are
recipe-specific, etc.).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from common.schemas import PartitionOptionMixin
from services.ltx import config


class ImageConditioning(BaseModel):
    """One image-conditioning entry: an uploaded image plus where and how
    strongly it should guide the output.

    The image is resized and center-cropped to the target video's
    height/width before use.
    """

    asset_id: str = Field(
        ...,
        description="asset_id returned by a prior POST /v1/uploads call for this image.",
        examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"],
    )
    frame_idx: int = Field(
        0,
        ge=0,
        description=(
            "Which output frame this image conditions. 0 makes this image the literal first "
            "frame of the video (true image-to-video). A value > 0 uses it as keyframe guidance "
            "partway through instead. Must be < the request's resolved num_frames."
        ),
        examples=[0],
    )
    strength: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Conditioning strength: 1.0 follows the image exactly at frame_idx, lower values blend it more loosely with the prompt.",
        examples=[1.0],
    )
    crf: int | None = Field(
        None,
        ge=0,
        description=(
            "H.264 compression quality applied to the image before conditioning, to match the "
            "statistics of real video frames (0 = lossless/use the image as-is; higher = more "
            "compression). Defaults to the pipeline's own default (33) when omitted."
        ),
        examples=[None],
    )


class VideoShapeMixin(BaseModel):
    """Shared duration/orientation fields + validation: exactly one of
    (orientation) or (height + width), and exactly one of (duration_seconds)
    or (num_frames).

    height/width must each be a multiple of 64 -- this is enforced by the
    underlying pipeline on the GPU node, not by this API's own validation, so
    a bad value here fails only after a job has already been scheduled.
    """

    orientation: Literal["landscape", "portrait"] | None = Field(
        None,
        description="'landscape' (1920x1088) or 'portrait' (1088x1920). Mutually exclusive with height/width.",
        examples=["landscape"],
    )
    height: int | None = Field(
        None,
        gt=0,
        description="Exact output height in pixels; must be a multiple of 64. Requires width to also be set. Mutually exclusive with orientation.",
        examples=[None],
    )
    width: int | None = Field(
        None,
        gt=0,
        description="Exact output width in pixels; must be a multiple of 64. Requires height to also be set. Mutually exclusive with orientation.",
        examples=[None],
    )
    duration_seconds: float | None = Field(
        None,
        gt=0,
        description=(
            "Video length in seconds; internally converted to the nearest valid frame count "
            "(8*k+1 at the given frame_rate -- e.g. 5s @ 24fps = 121 frames). Mutually exclusive "
            "with num_frames. Omit both for the pipeline's own default (~5s)."
        ),
        examples=[5],
    )
    num_frames: int | None = Field(
        None,
        ge=9,
        description="Exact frame count instead of a duration. Must satisfy 8*k+1 (9, 17, ..., 121, ...); other values are silently snapped to the nearest valid count. Mutually exclusive with duration_seconds.",
        examples=[None],
    )
    frame_rate: float = Field(
        config.DEFAULT_FRAME_RATE,
        gt=0,
        description="Output frames per second, and the rate used to convert duration_seconds into a frame count.",
        examples=[24.0],
    )

    @model_validator(mode="after")
    def _check_shape(self):
        if self.orientation is not None and (self.height is not None or self.width is not None):
            raise ValueError("Use either 'orientation' or explicit 'height'/'width', not both.")
        if (self.height is None) != (self.width is None):
            raise ValueError("'height' and 'width' must be given together.")
        if self.duration_seconds is not None and self.num_frames is not None:
            raise ValueError("Use either 'duration_seconds' or 'num_frames', not both.")
        return self


class TextToVideoRequest(VideoShapeMixin, PartitionOptionMixin):
    """Request body for POST /v1/ltx/videos/generate -- text-to-video, or
    image-to-video when `images` is non-empty. See that endpoint's
    description for the fast-vs-quality tradeoff."""

    prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "Cinematographic description of the desired scene: action first, then appearance, "
            "environment, camera, and lighting, as one flowing paragraph (roughly 200 words or "
            "fewer). Put spoken dialogue in double quotes for lip-synced speech. "
            "See `GET /v1/guide` for the full prompting guide."
        ),
        examples=[
            "A golden retriever puppy runs across a sunlit lawn, tongue out, tail wagging, "
            "chasing a bright yellow tennis ball. Bright daylight, shallow depth of field, "
            "handheld camera following the puppy."
        ],
    )
    mode: Literal["fast", "quality"] = Field(
        "fast",
        description=(
            "'fast': distilled checkpoint, 8+3 steps, no guidance, negative_prompt ignored. "
            "'quality': dev checkpoint with CFG+STG guidance and the distilled LoRA fused for "
            "stage 2; slower, honors negative_prompt."
        ),
        examples=["fast"],
    )
    negative_prompt: str | None = Field(
        None,
        description=(
            "What to steer away from. Only applied when mode='quality' (silently ignored in "
            "'fast' mode, which has no guidance mechanism). Replaces the pipeline's own default "
            "negative prompt rather than adding to it; omit to use that default."
        ),
        examples=[None],
    )
    seed: int = Field(
        10,
        description="Random seed. Same seed + same prompt + same other params = same video; change it for a different take.",
        examples=[10],
    )
    enhance_prompt: bool = Field(
        False,
        description=(
            "If true, Gemma rewrites your prompt into a richer one before generation (using an "
            "image-aware rewrite if `images` is set). The rewritten prompt is used internally but "
            "is not returned in the API response."
        ),
        examples=[False],
    )
    images: list[ImageConditioning] = Field(
        default_factory=list,
        description="Optional image conditioning for image-to-video. Empty = pure text-to-video.",
    )


class KeyframeInterpolationRequest(VideoShapeMixin, PartitionOptionMixin):
    """Request body for POST /v1/ltx/videos/keyframe-interpolation.

    Quality recipe only -- no distilled/fast variant is wired up for this
    pipeline in the underlying repo today."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="Describes the motion/action connecting the keyframes, not the keyframes' static appearance (they're already provided as images).",
        examples=[
            "A flower blooms from a tight bud into a fully open blossom in a smooth, continuous "
            "motion, soft natural light, shallow depth of field."
        ],
    )
    negative_prompt: str | None = Field(
        None,
        description="What to steer away from (this endpoint always runs the guided/quality recipe, so this is always honored, unlike on /v1/ltx/videos/generate).",
        examples=[None],
    )
    seed: int = Field(10, description="Random seed; same seed + inputs = same video.", examples=[10])
    keyframes: list[ImageConditioning] = Field(
        ...,
        min_length=2,
        description="At least 2 keyframe images, each with its own frame_idx/strength. Put the first keyframe at frame_idx=0.",
    )


class AudioToVideoRequest(VideoShapeMixin, PartitionOptionMixin):
    """Request body for POST /v1/ltx/videos/audio-to-video.

    Quality recipe only -- same reason as keyframe interpolation."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="Describes the visible speaker/scene that should match the conditioning audio (e.g. who is speaking, their appearance, the setting).",
        examples=[
            "A man in a blue collared shirt sits at a desk, speaking directly to the camera with "
            "natural hand gestures, soft office lighting behind him."
        ],
    )
    negative_prompt: str | None = Field(None, description="What to steer away from (always honored on this endpoint).", examples=[None])
    seed: int = Field(10, description="Random seed; same seed + inputs = same video.", examples=[10])
    audio_asset_id: str = Field(
        ...,
        description="asset_id (from POST /v1/uploads) of the audio/video file whose audio track conditions generation. This audio is muxed into the output unchanged, not regenerated.",
        examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"],
    )
    audio_start_time: float = Field(
        0.0, ge=0, description="Seconds into the source audio file to start reading from.", examples=[0.0]
    )
    audio_max_duration: float | None = Field(
        None,
        gt=0,
        description="Maximum seconds of audio to read. Defaults to the resolved video duration (num_frames / frame_rate) when omitted.",
        examples=[None],
    )
    images: list[ImageConditioning] = Field(
        default_factory=list, description="Optional additional image conditioning (e.g. to pin the speaker's appearance to a reference photo)."
    )


class RetakeRequest(PartitionOptionMixin):
    """Request body for POST /v1/ltx/videos/retake.

    Distilled/fast checkpoint only -- that's the only variant the underlying
    pipeline's CLI ships for this recipe. Height/width/frame count are not
    accepted: they're inherited from the source video, which must already
    satisfy the model's 8k+1 frame-count and multiple-of-32 resolution
    constraints."""

    video_asset_id: str = Field(
        ...,
        description=(
            "asset_id (from POST /v1/uploads) of the source video. Must have 8k+1 frames "
            "(97, 121, 193, ...) and width/height that are multiples of 32 -- not checked by "
            "this API, only by the pipeline once the job is running."
        ),
        examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"],
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="Describes only the content of the [start_time, end_time] window being replaced, not the whole video.",
        examples=["The man raises his coffee cup and takes a slow sip, then sets it back down on the table."],
    )
    start_time: float = Field(..., ge=0, description="Start of the region to regenerate, in seconds.", examples=[2.0])
    end_time: float = Field(..., gt=0, description="End of the region to regenerate, in seconds. Must be greater than start_time.", examples=[4.0])
    seed: int = Field(10, description="Random seed for the regenerated region.", examples=[10])

    @model_validator(mode="after")
    def _check_times(self):
        if self.start_time >= self.end_time:
            raise ValueError("'start_time' must be less than 'end_time'.")
        return self


class TextToAudioRequest(PartitionOptionMixin):
    """Request body for POST /v1/ltx/audio/generate -- audio only, no video.

    Dev checkpoint only -- that's the only variant the underlying
    pipeline's CLI ships for this recipe."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="Describes the desired soundscape/sound: sources, timing, and character of the sound(s), as a flowing description.",
        examples=[
            "A quiet rainforest at dawn: distant bird calls, a light breeze moving through "
            "leaves, and the occasional drip of water from wet foliage."
        ],
    )
    negative_prompt: str | None = Field(None, description="What to steer away from.", examples=[None])
    seed: int = Field(10, description="Random seed; same seed + prompt = same audio.", examples=[10])
    duration_seconds: float | None = Field(
        None, gt=0, description="Audio length in seconds. Mutually exclusive with num_frames.", examples=[5]
    )
    num_frames: int | None = Field(
        None,
        ge=9,
        description="Alternative to duration_seconds: an exact frame count (8*k+1) used with frame_rate to derive the audio duration.",
        examples=[None],
    )
    frame_rate: float = Field(
        config.DEFAULT_FRAME_RATE,
        gt=0,
        description="Used with num_frames/duration_seconds to derive audio duration (num_frames / frame_rate seconds); does not affect audio sample rate.",
        examples=[24.0],
    )

    @model_validator(mode="after")
    def _check_duration(self):
        if self.duration_seconds is not None and self.num_frames is not None:
            raise ValueError("Use either 'duration_seconds' or 'num_frames', not both.")
        return self
