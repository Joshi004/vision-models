"""Pydantic request models for LTX-2.5's generation endpoints.

Deliberately small: this LTX-2.5 backend ships four recipes -- text/
image-to-video (fast or quality), first/last-frame interpolation, retake, and
audio-to-video.
The shape fields (orientation/
height/width/duration/frame rate), image conditioning, and retake window are
identical to LTX-2.3's, so they're reused from services/ltx/schemas.py
instead of redefined here.

What differs from LTX-2.3, and why:
  * No `negative_prompt` on generate/interpolate: both of their recipes
    (distilled and DFR) run with guidance off, so there is nothing for a
    negative prompt to act on. Audio-to-video is guided, so it has one.
  * `mode="quality"` runs the DFR pipeline (distilled transformer plus a
    detailing IC-LoRA), not LTX-2.3's dev-checkpoint-with-CFG recipe.
  * `auto_duration`: LTX-2.5's duration head can pick the clip length from
    the prompt.

Like every backend's schemas, each class below gets a distinct name so its
OpenAPI component never collides with LTX-2.3's same-purpose class, and the
web UI's per-field help (ui/src/schema.js) can look each one up directly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from common.schemas import IncognitoOptionMixin, PartitionOptionMixin
from services.ltx.schemas import ImageConditioning, RetakeRequest, VideoShapeMixin


class Ltx25TextToVideoRequest(VideoShapeMixin, PartitionOptionMixin, IncognitoOptionMixin):
    """Request body for POST /v1/ltx25/videos/generate -- text-to-video, or
    image-to-video when `images` is non-empty. See that endpoint's
    description for the fast-vs-quality tradeoff."""

    prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "Cinematographic description of the desired scene: action first, then appearance, "
            "environment, camera, and lighting, as one flowing paragraph. Put spoken dialogue in "
            "double quotes for lip-synced speech. LTX-2.5 can also cut between 2-4 shots inside "
            "one clip: write the whole scene as one chronological paragraph and name every cut in "
            "prose (e.g. 'A hard cut transitions to a close-up of the woman in the yellow "
            "raincoat; the rain continues.'). There is no separate multi-shot field. "
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
            "'fast': the distilled pipeline -- fixed 8-step schedule, no guidance; the fastest "
            "option. 'quality': the DFR (Diffusion Fidelity Rendering) pipeline -- the same "
            "distilled transformer, plus generated keyframes and a second-stage detailing pass; "
            "sharper detail, slower and more GPU memory."
        ),
        examples=["fast"],
    )
    seed: int = Field(
        10,
        description="Random seed. Same seed + same prompt + same other params = same video; change it for a different take.",
        examples=[10],
    )
    enhance_prompt: bool = Field(
        False,
        description=(
            "If true, a Gemma instruct model rewrites your prompt into a richer one before "
            "generation (using an image-aware rewrite if `images` is set). The rewritten prompt "
            "is used internally but is not returned in the API response."
        ),
        examples=[False],
    )
    auto_duration: bool = Field(
        False,
        description=(
            "If true, the model's duration head picks the clip length from the prompt (clamped "
            "to 1-20 seconds) instead of a fixed length. Mutually exclusive with "
            "`duration_seconds` and `num_frames`. If false (the default) and neither of those is "
            "given, the clip is ~5 seconds (121 frames)."
        ),
        examples=[False],
    )
    images: list[ImageConditioning] = Field(
        default_factory=list,
        description="Optional image conditioning for image-to-video. Empty = pure text-to-video.",
    )

    @model_validator(mode="after")
    def _check_auto_duration(self):
        if self.auto_duration and (self.duration_seconds is not None or self.num_frames is not None):
            raise ValueError("'auto_duration' cannot be combined with 'duration_seconds' or 'num_frames'.")
        return self


class Ltx25InterpolateRequest(VideoShapeMixin, PartitionOptionMixin, IncognitoOptionMixin):
    """Request body for POST /v1/ltx25/videos/interpolate -- generates the
    motion between a given first frame and a given last frame.

    Same two recipes as text-to-video (`mode`), with the two images placed
    at the clip's first and last frame. There is no `auto_duration` here:
    the last frame's position depends on knowing the clip length up front."""

    prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "Describes the motion and action that connects the two frames, not their static "
            "appearance (the images already establish that). Put spoken dialogue in double quotes "
            "for lip-synced speech."
        ),
        examples=[
            "A flower blooms from a tight bud into a fully open blossom in a smooth, continuous "
            "motion, soft natural light, shallow depth of field."
        ],
    )
    first_frame_asset_id: str = Field(
        ...,
        description=(
            "asset_id (from POST /v1/uploads) of the image that becomes the literal first frame "
            "of the video. It is resized and center-cropped to the output size, so it should have "
            "the same aspect ratio as the output (16:9 for `landscape`, 9:16 for `portrait`)."
        ),
        examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"],
    )
    last_frame_asset_id: str = Field(
        ...,
        description=(
            "asset_id (from POST /v1/uploads) of the image the video should end on (it guides the "
            "final frame). Resized and center-cropped like the first frame."
        ),
        examples=["9a8b7c6d5e4f4a3b2c1d0e9f8a7b6c5d"],
    )
    mode: Literal["fast", "quality"] = Field(
        "fast",
        description=(
            "'fast': the distilled pipeline -- fixed 8-step schedule, no guidance. 'quality': the "
            "DFR pipeline -- same distilled transformer plus a detailing pass; sharper detail, "
            "slower and more GPU memory. Same meaning as on `POST /v1/ltx25/videos/generate`."
        ),
        examples=["fast"],
    )
    seed: int = Field(
        10,
        description="Random seed. Same seed + same inputs = same video; change it for a different take.",
        examples=[10],
    )
    enhance_prompt: bool = Field(
        False,
        description=(
            "If true, a Gemma instruct model rewrites your prompt into a richer one before "
            "generation. The rewritten prompt is used internally but is not returned in the API "
            "response."
        ),
        examples=[False],
    )


class Ltx25AudioToVideoRequest(VideoShapeMixin, PartitionOptionMixin, IncognitoOptionMixin):
    """Request body for POST /v1/ltx25/videos/audio-to-video -- generates video
    to match a given audio track, which is muxed into the output unchanged.

    One recipe only: upstream's guided two-stage A2Vid pipeline (dev
    transformer + distilled LoRA). The distilled and DFR pipelines have no
    audio input, so there is no `mode`.

    Clip length has two mutually exclusive sources, because LTX-2.5's A2Vid
    pipeline refuses a frame count and an audio duration together: either a
    fixed length (`duration_seconds`/`num_frames`, default ~5s) with the audio
    cut to fit, or `audio_max_duration`, which makes the clip as long as the
    audio slice actually read."""

    prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "Describes the visible speaker/scene that should match the conditioning audio (who is "
            "speaking, their appearance, the setting, the camera). The audio itself is not "
            "regenerated, so there is no need to quote its dialogue."
        ),
        examples=[
            "A man in a blue collared shirt sits at a desk, speaking directly to the camera with "
            "natural hand gestures, soft office lighting behind him."
        ],
    )
    negative_prompt: str | None = Field(
        None,
        description=(
            "What to steer away from. Honored here (this recipe runs with guidance), unlike on the "
            "other LTX-2.5 endpoints. When omitted, the pipeline's own built-in negative prompt "
            "(blur, artifacts, bad lip sync, ...) is used; a value given here replaces it."
        ),
        examples=[None],
    )
    seed: int = Field(10, description="Random seed; same seed + inputs = same video.", examples=[10])
    audio_asset_id: str = Field(
        ...,
        description=(
            "asset_id (from POST /v1/uploads) of the audio/video file whose audio track conditions "
            "generation. Mono audio is converted to stereo first. This audio is muxed into the "
            "output unchanged, not regenerated."
        ),
        examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"],
    )
    audio_start_time: float = Field(
        0.0, ge=0, description="Seconds into the source audio file to start reading from.", examples=[0.0]
    )
    audio_max_duration: float | None = Field(
        None,
        gt=0,
        le=20,
        description=(
            "Match the clip length to the audio: read at most this many seconds of audio (from "
            "`audio_start_time`) and make the video exactly as long as the audio actually read -- "
            "shorter if the file ends first. Snapped to the nearest valid frame count. Mutually "
            "exclusive with `duration_seconds` and `num_frames`. When omitted, the clip has a "
            "fixed length (`duration_seconds`/`num_frames`, default ~5s) and the audio is cut to "
            "that length."
        ),
        examples=[None],
    )
    images: list[ImageConditioning] = Field(
        default_factory=list,
        description="Optional image conditioning (e.g. to pin the speaker's appearance to a reference photo at frame 0).",
    )

    @model_validator(mode="after")
    def _check_audio_length(self):
        if self.audio_max_duration is not None and (self.duration_seconds is not None or self.num_frames is not None):
            raise ValueError("'audio_max_duration' cannot be combined with 'duration_seconds' or 'num_frames'.")
        return self


class Ltx25RetakeRequest(RetakeRequest):
    """Request body for POST /v1/ltx25/videos/retake.

    Same fields and rules as LTX-2.3's retake: the distilled checkpoint only,
    and height/width/frame count are inherited from the source video, which
    must already have 8k+1 frames and width/height that are multiples of 32."""
