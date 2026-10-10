"""Long-form OpenAPI/Swagger documentation text for LTX-2.3's own endpoints,
kept out of router.py so the routes themselves stay readable. Docs shared
by every backend (uploads, jobs, app-level description) live in
common/openapi_docs.py instead.

This is the machine-readable half of LTX-2.3's docs, meant to be read via
/openapi.json or /docs. The companion services/ltx/GUIDE.md (same directory,
folded into GET /v1/guide's own LTX-2.3 section at request time) is the
prose half -- the model, prompting technique, and operational quirks that a
schema alone can't capture -- meant to be read once, in full, before
building against this backend.
"""

from __future__ import annotations

GENERATE_VIDEO = """
Text-to-video, or image-to-video if you attach one or more `images`.

**Two recipes, chosen by `mode`:**
- `"fast"` (default) -- the distilled checkpoint, 8-step stage 1 + 3-step stage 2,
  no guidance. Fastest and cheapest; `negative_prompt` is accepted but silently
  ignored (the fast pipeline has no negative-prompt/CFG mechanism at all).
- `"quality"` -- the full dev checkpoint with classifier-free guidance (CFG) and
  spatio-temporal guidance (STG), plus the distilled LoRA fused in for stage-2
  refinement. Slower (~30 stage-1 steps instead of 8) but higher fidelity, and
  honors `negative_prompt`.

Output: MP4, H.264 video + AAC audio, muxed together. Video and audio are
generated *jointly* by the same model -- quoted dialogue in the prompt comes out
lip-synced, not added afterward.

`images[].frame_idx == 0` makes that image the literal first frame (true
image-to-video); `frame_idx > 0` uses it as keyframe guidance partway through
instead. See `GET /v1/guide` for the full prompting guide and worked examples.
"""

KEYFRAME_INTERPOLATION = """
Generates a video that passes through 2 or more given keyframe images at their
specified `frame_idx` positions, with `prompt` describing the motion/action that
connects them.

**Quality recipe only** -- there is no fast/distilled variant of this pipeline in
the underlying repo, so every call here pays the slower, guided cost regardless
of preference (there is no `mode` field on this endpoint for that reason).

`keyframes` reuses the same image-conditioning mechanism as `images` on
`/v1/ltx/videos/generate`: at least 2 entries are required, each with its own
`frame_idx` and `strength`. Put your first keyframe at `frame_idx: 0`.

Output: MP4, H.264 + AAC. Audio is generated fresh by the model, not carried over
from the keyframe images (which have none).
"""

AUDIO_TO_VIDEO = """
Generates video conditioned on a given audio track -- typically speech, for
lip-synced video generation from a pre-recorded voice line.

**Quality recipe only** (same reason as keyframe interpolation: no fast variant
exists for this pipeline in the underlying repo).

The input audio is **not regenerated**: it is decoded from `audio_asset_id` and
muxed into the output unchanged; only the video is generated to match it. Use
`audio_start_time` / `audio_max_duration` to select a slice of a longer audio
file -- by default, audio covering the full requested video duration is read
starting at 0s.

Output: MP4, H.264 video + the original audio muxed in unchanged.
"""

RETAKE = """
Regenerates only the `[start_time, end_time]` window of an existing video,
keeping everything outside that window byte-for-byte identical to the source.
Both video and audio inside the window are regenerated together, driven by
`prompt`.

**Distilled/fast checkpoint only** -- the only variant the underlying pipeline's
CLI ships.

**The source video must already satisfy the model's own shape constraints.**
This is not checked by this API -- a video that doesn't will fail only once the
job is already running on a GPU: frame count must be `8k+1` (97, 121, 193, ...)
and width/height must each be a multiple of 32.

There are no `height`/`width`/`num_frames`/`orientation` fields on this
endpoint: the output always matches the source video's resolution, frame count,
and frame rate exactly.
"""

GENERATE_AUDIO = """
Text-to-audio only -- no video is produced or required. Useful for a standalone
sound effect or ambient soundscape, or for generating a voice line to condition
a later `/v1/ltx/videos/audio-to-video` call with.

**Dev checkpoint only** -- the only variant the underlying pipeline's CLI ships
(despite there being no separate "audio model": this reuses the same 22B
checkpoint's audio-only weights).

Output: WAV, 16-bit PCM, at the model's native sample rate.
"""

# ---------------------------------------------------------------------------
# Named request-body examples (Swagger's "Examples" dropdown per endpoint)
# ---------------------------------------------------------------------------

GENERATE_VIDEO_EXAMPLES = {
    "fast_landscape": {
        "summary": "Fast mode, landscape, 5s -- good default for iterating",
        "value": {
            "prompt": (
                "A golden retriever puppy runs across a sunlit lawn, tongue out, tail wagging, "
                "chasing a bright yellow tennis ball. Bright daylight, shallow depth of field, "
                "handheld camera following the puppy."
            ),
            "mode": "fast",
            "orientation": "landscape",
            "duration_seconds": 5,
            "seed": 42,
        },
    },
    "quality_with_negative_prompt": {
        "summary": "Quality mode with a custom negative prompt",
        "value": {
            "prompt": (
                "A chef in a white uniform plates a dessert in a busy restaurant kitchen, steam "
                "rising from the pan behind her, camera slowly pushing in."
            ),
            "mode": "quality",
            "negative_prompt": "blurry, low quality, extra fingers, static camera",
            "orientation": "landscape",
            "duration_seconds": 5,
        },
    },
    "image_to_video": {
        "summary": "Image-to-video: animate an uploaded photo as the first frame",
        "value": {
            "prompt": (
                'The man in the photo turns his head slowly to look directly at the camera and '
                'says, "I think it\'s ready." Soft natural light, shallow depth of field.'
            ),
            "mode": "fast",
            "orientation": "landscape",
            "duration_seconds": 5,
            "images": [{"asset_id": "<asset_id from POST /v1/uploads>", "frame_idx": 0, "strength": 1.0}],
        },
    },
    "portrait_with_dialogue": {
        "summary": "Portrait orientation with quoted, lip-synced dialogue",
        "value": {
            "prompt": (
                "Style: cinematic-realistic. A woman in a cream turtleneck sits by a window, "
                'sunlight on her face. She smiles and says in a warm, clear voice, "I think we\'re '
                'right on time." Soft ambient room tone.'
            ),
            "mode": "quality",
            "orientation": "portrait",
            "duration_seconds": 5,
        },
    },
}

KEYFRAME_EXAMPLES = {
    "first_and_last_frame": {
        "summary": "Interpolate between a start and end keyframe",
        "value": {
            "prompt": (
                "A flower blooms from a tight bud into a fully open blossom in a smooth, "
                "continuous motion, soft natural light, shallow depth of field."
            ),
            "orientation": "landscape",
            "duration_seconds": 5,
            "keyframes": [
                {"asset_id": "<asset_id of the starting frame>", "frame_idx": 0, "strength": 1.0},
                {"asset_id": "<asset_id of the ending frame>", "frame_idx": 120, "strength": 1.0},
            ],
        },
    },
}

AUDIO_TO_VIDEO_EXAMPLES = {
    "lipsync_to_uploaded_speech": {
        "summary": "Generate a lip-synced speaker for an uploaded voice line",
        "value": {
            "prompt": (
                "A man in a blue collared shirt sits at a desk, speaking directly to the camera "
                "with natural hand gestures, soft office lighting behind him."
            ),
            "orientation": "landscape",
            "duration_seconds": 5,
            "audio_asset_id": "<asset_id from POST /v1/uploads>",
            "audio_start_time": 0,
        },
    },
}

RETAKE_EXAMPLES = {
    "regenerate_a_window": {
        "summary": "Replace the 2.0s-4.0s window of an existing video",
        "value": {
            "video_asset_id": "<asset_id of the source video>",
            "prompt": "The man raises his coffee cup and takes a slow sip, then sets it back down on the table.",
            "start_time": 2.0,
            "end_time": 4.0,
        },
    },
}

GENERATE_AUDIO_EXAMPLES = {
    "ambient_soundscape": {
        "summary": "Standalone ambient soundscape",
        "value": {
            "prompt": (
                "A quiet rainforest at dawn: distant bird calls, a light breeze moving through "
                "leaves, and the occasional drip of water from wet foliage."
            ),
            "duration_seconds": 5,
        },
    },
}
