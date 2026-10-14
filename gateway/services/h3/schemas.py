"""Pydantic request models for MiniMax H3's two endpoints.

  * `H3GenerateRequest`      -- POST /v1/h3/videos/generate: text, or a first and/or
    last frame, to video + audio (the model's "FL2VA" checkpoint).
  * `H3AudioToVideoRequest`  -- POST /v1/h3/videos/audio-to-video: an audio reference
    plus image/video references to video + audio (the "Ref2VA" checkpoint).

File inputs are referenced by an asset_id obtained from POST /v1/uploads (see
services/ltx/schemas.py for why). Every asset field is named `asset_id`,
`*_asset_id` or `*_asset_ids`, which is how common/purge.py finds the uploads a
job used.

The shape fields differ from LTX's on purpose (H3 is fixed at 24 fps, frame
counts are 17 * n + 5, sizes are multiples of 32 -- see services/h3/config.py),
so they are defined here instead of reusing services/ltx/schemas.py's mixin.

Like every backend's schemas, each class below gets a distinct name so its
OpenAPI component never collides with another backend's same-purpose class, and
the web UI's per-field help (ui/src/schema.js) can look each one up directly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from common.schemas import IncognitoOptionMixin, PartitionOptionMixin
from services.h3 import config


class H3ShapeMixin(BaseModel):
    """Shared canvas and length fields: at most one of (orientation) or
    (height + width), and at most one of (duration_seconds) or (num_frames)."""

    orientation: Literal["landscape", "portrait"] | None = Field(
        None,
        description=(
            f"'landscape' ({config.LANDSCAPE_WIDTH}x{config.LANDSCAPE_HEIGHT}) or 'portrait' "
            f"({config.PORTRAIT_WIDTH}x{config.PORTRAIT_HEIGHT}). Mutually exclusive with height/width. "
            "Omit orientation, height and width to let H3 choose: its own 16:9 canvas, or, on "
            "`/videos/generate` with a first frame, that frame's aspect ratio."
        ),
        examples=["landscape"],
    )
    height: int | None = Field(
        None,
        ge=config.MIN_SIDE,
        le=config.MAX_SIDE,
        description=(
            f"Exact output height in pixels, a multiple of {config.SIZE_MULTIPLE} between {config.MIN_SIDE} and "
            f"{config.MAX_SIDE}. Requires width. Mutually exclusive with orientation. The model was trained at "
            f"{config.LANDSCAPE_WIDTH}x{config.LANDSCAPE_HEIGHT}; smaller canvases are much faster (960x544 runs "
            "about 2.3x faster per step) and use less GPU memory, larger ones are untested here."
        ),
        examples=[None],
    )
    width: int | None = Field(
        None,
        ge=config.MIN_SIDE,
        le=config.MAX_SIDE,
        description=f"Exact output width in pixels, a multiple of {config.SIZE_MULTIPLE}. Requires height.",
        examples=[None],
    )
    duration_seconds: float | None = Field(
        None,
        ge=config.MIN_DURATION_SECONDS,
        le=config.MAX_DURATION_SECONDS,
        description=(
            f"Video length in seconds ({config.MIN_DURATION_SECONDS:g} to {config.MAX_DURATION_SECONDS:g}). H3 runs at "
            f"a fixed {config.FPS} fps and only decodes frame counts of the form 17*n+5, so the length is rounded up "
            f"to the next such count and kept within {config.MIN_FRAMES} to {config.MAX_FRAMES} frames "
            f"({config.MIN_FRAMES / config.FPS:.2f} to {config.MAX_FRAMES / config.FPS:.1f} s): 15 s becomes "
            f"{config.MAX_FRAMES} frames. Mutually exclusive with num_frames. Omit both for the default, "
            f"{config.DEFAULT_FRAMES} frames ({config.DEFAULT_FRAMES / config.FPS:.2f} s)."
        ),
        examples=[5],
    )
    num_frames: int | None = Field(
        None,
        ge=int(config.MIN_DURATION_SECONDS * config.FPS),
        le=int(config.MAX_DURATION_SECONDS * config.FPS),
        description=(
            "Exact frame count at 24 fps instead of a duration; rounded up to the next 17*n+5 and kept within "
            f"{config.MIN_FRAMES} to {config.MAX_FRAMES} frames, like duration_seconds. Mutually exclusive with "
            "duration_seconds."
        ),
        examples=[None],
    )

    @model_validator(mode="after")
    def _check_shape(self):
        if self.orientation is not None and (self.height is not None or self.width is not None):
            raise ValueError("Use either 'orientation' or explicit 'height'/'width', not both.")
        if (self.height is None) != (self.width is None):
            raise ValueError("'height' and 'width' must be given together.")
        if self.height is not None and (self.height % config.SIZE_MULTIPLE or self.width % config.SIZE_MULTIPLE):
            raise ValueError(f"'height' and 'width' must be multiples of {config.SIZE_MULTIPLE}.")
        if self.duration_seconds is not None and self.num_frames is not None:
            raise ValueError("Use either 'duration_seconds' or 'num_frames', not both.")
        return self


class H3SamplingMixin(BaseModel):
    """Shared sampling fields."""

    seed: int = Field(
        10,
        description="Random seed. Same seed + same prompt + same other params = same video; change it for a different take.",
        examples=[10],
    )
    num_inference_steps: int | None = Field(
        None,
        ge=4,
        le=100,
        description=(
            "Denoising steps. Omit for the pipeline's own default (50). The weights are guidance-distilled, so each step "
            "is one forward pass and there is no guidance scale or negative prompt; fewer steps are proportionally "
            "faster and usually rougher. (An independent benchmark used 20.)"
        ),
        examples=[None],
    )


class H3GenerateRequest(H3ShapeMixin, H3SamplingMixin, PartitionOptionMixin, IncognitoOptionMixin):
    """Request body for POST /v1/h3/videos/generate -- text-to-video, or image-to-video
    with a first and/or last frame."""

    prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "The prompt, exactly as it should reach the model: nothing rewrites it (the hosted MiniMax service's "
            "prompt-expansion model is not part of the open release). H3 was trained on a structured format: an "
            "`integrated_multimodal_description:` of `[Shot N]` blocks, then `overall_soundscape:` and "
            "`non_diegetic_music:`, preceded by one instruction line when a frame is given. A plain sentence is "
            "accepted but gives the model far less to go on. Spoken words go inside `<d>[Language] ...</d>` with a "
            "speaker id such as `(S1)`; refer to the frames as `<Picture 1>` (and `<Picture 2>`). "
            "See `GET /v1/guide` (the MiniMax H3 section) for the format, with examples."
        ),
        examples=[
            "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker "
            "opening the shutters of a small street bakery before sunrise. The camera pushes in with small amplitude "
            "at slow speed as the middle-aged baker with a calm, slightly raspy voice (S1) places a fresh loaf on the "
            "wooden counter and says: <d>[English] First batch of the morning.</d>\n\n"
            "overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink softly inside the "
            "bakery.\n\nnon_diegetic_music: N/A"
        ],
    )
    first_frame_asset_id: str | None = Field(
        None,
        description=(
            "asset_id (from POST /v1/uploads) of the image the video starts from. Optional. It is stretched onto the "
            "canvas, which by default follows its own aspect ratio. Write the prompt's first line as the instruction "
            "that places `<Picture 1>` at 0.00 seconds (see the guide)."
        ),
        examples=[None],
    )
    last_frame_asset_id: str | None = Field(
        None,
        description=(
            "asset_id (from POST /v1/uploads) of the image the video ends on. Optional; may be given without a first "
            "frame (the video then lands on it), or with one (first-to-last-frame interpolation). It is cover-cropped "
            "onto the canvas."
        ),
        examples=[None],
    )


class H3AudioToVideoRequest(H3ShapeMixin, H3SamplingMixin, PartitionOptionMixin, IncognitoOptionMixin):
    """Request body for POST /v1/h3/videos/audio-to-video -- video + audio from an audio
    reference together with image and/or video references (H3's reference mode).

    NOT the same guarantee as LTX's audio-to-video. There the uploaded audio is muxed in
    unchanged. Here the audio is a reference the model reads and then generates its own
    soundtrack from; what it does with it (copy it, or only follow its voice and style) is
    decided by how the prompt describes it."""

    prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "The prompt, exactly as it should reach the model: nothing rewrites it. Reference mode was trained on a "
            "six-part format: `subject_definitions:`, `summary:`, `retention_analysis:`, `detailed_description:`, "
            "`overall_soundscape:` and `non_diegetic_music:`. The references are labelled in the order you give "
            "them, each kind counted on its own: `<Picture 1>` is the first of `image_asset_ids`, `<Video 1>` the "
            "first of `video_asset_ids`, `<Audio 1>` the audio. How the audio is used is stated in the prompt "
            "(`fully_copy`, `partially_copy`, `reference`, `weak_reference`). Spoken words go inside "
            "`<d>[Language] ...</d>` with a speaker id such as `(S1)`. See `GET /v1/guide` for the format, with examples."
        ),
        examples=[
            "subject_definitions:\n<Subject 1> is the young woman in <Picture 1>, with long dark hair and a blue cardigan.\n"
            "<Audio 1> is the voice-timbre reference for <Subject 1> (S1).\n\n"
            "summary:\n[reference generation + audio reference] <Subject 1> speaks to the camera in a sunlit kitchen.\n\n"
            "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - her face, hair and cardigan are retained.\n"
            "<Audio 1>: reference - her voice timbre guides the delivery without copying the original signal.\n\n"
            "detailed_description:\nThe target video is live-action with soft window light.\n"
            "[Shot 1] A medium close-up of <Subject 1> (S1) at a kitchen counter. She smiles and says, using the voice "
            "timbre referenced from <Audio 1>, <d>[English] It is a quiet morning, and the coffee is almost ready.</d>\n\n"
            "overall_soundscape:\nSoft kitchen room tone.\n\nnon_diegetic_music:\nN/A"
        ],
    )
    audio_asset_id: str = Field(
        ...,
        description=(
            "asset_id (from POST /v1/uploads) of the audio, or of a video whose audio track is used, that serves as "
            "the audio reference (`<Audio 1>` in the prompt). 2 to 15 seconds. Mono or stereo at any sample rate "
            "(resampled to 32 kHz). Never used on its own: it needs at least one image or video reference. "
            "This is a reference, not audio that is muxed into the result unchanged -- see the endpoint description."
        ),
        examples=["3f2a9c1e4b7d4a6c9e8f1a2b3c4d5e6f"],
    )
    image_asset_ids: list[str] = Field(
        default_factory=list,
        max_length=config.MAX_REFERENCE_IMAGES,
        description=(
            f"asset_ids (from POST /v1/uploads) of up to {config.MAX_REFERENCE_IMAGES} reference images -- the speaker's "
            "appearance, a scene, a style (`<Picture 1>`, `<Picture 2>`, ... in the prompt, in this order). They do "
            "not set the output's shape (use orientation, or height and width). At least one image or video is required."
        ),
        examples=[["9a8b7c6d5e4f4a3b2c1d0e9f8a7b6c5d"]],
    )
    video_asset_ids: list[str] = Field(
        default_factory=list,
        max_length=config.MAX_REFERENCE_VIDEOS,
        description=(
            f"asset_ids (from POST /v1/uploads) of up to {config.MAX_REFERENCE_VIDEOS} reference videos, 2 to 15 seconds "
            "each, for motion and camera work (`<Video 1>`, ... in the prompt). A video's own soundtrack is read too. "
            "Their frame rate is read from the file. At least one image or video is required."
        ),
        examples=[[]],
    )
    match_audio_duration: bool = Field(
        False,
        description=(
            "If true, the video is as long as the audio reference (rounded onto the 17*n+5 frame grid and kept within "
            f"{config.MIN_FRAMES} to {config.MAX_FRAMES} frames, so an audio shorter than 5.17 s or longer than 14.4 s "
            "is padded or cut). Mutually exclusive with duration_seconds and num_frames."
        ),
        examples=[False],
    )

    @model_validator(mode="after")
    def _check_references(self):
        if not self.image_asset_ids and not self.video_asset_ids:
            raise ValueError("Give at least one image (image_asset_ids) or video (video_asset_ids) reference: audio cannot be the only reference.")
        if self.match_audio_duration and (self.duration_seconds is not None or self.num_frames is not None):
            raise ValueError("'match_audio_duration' cannot be combined with 'duration_seconds' or 'num_frames'.")
        return self
