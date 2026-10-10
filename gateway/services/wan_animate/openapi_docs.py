"""Long-form OpenAPI/Swagger documentation text for Wan-Animate v1's own
endpoint, kept out of router.py so the route itself stays readable. Docs
shared by every backend (uploads, jobs, app-level description) live in
common/openapi_docs.py instead.
"""

from __future__ import annotations

REPLACE = """
Swaps the person in an existing video for a reference character while
preserving that video's background, camera motion, and lighting --
"replace" mode, in the underlying model's (Wan2.2-Animate-14B, "v1") own
terminology. This is character replacement in *existing* footage, not
text/image-to-video generation: nothing here is generated from a prompt,
and there is no `prompt` field on this endpoint at all.

**Pipeline, run as a single job end to end:**
1. Preprocessing extracts pose, face, background, and character-mask videos
   from `video_asset_id` and `image_asset_id`, using the reference
   implementation's own replace-mode preprocessing step.
2. Generation runs Wan2.2-Animate-14B itself on those extracted materials
   (single GPU), using the reference implementation's own replace-mode
   generation step.
3. Since the model only ever produces **silent** video, the source video's
   original audio track is muxed back on afterward (unless `keep_audio` is
   false).

**Known hard limits, from the official docs, not just this API's own
observation:**
- **Single-person video only.** Preprocessing's mask-extraction step is
  documented as designed for single-person footage; a video with more than
  one person on screen may produce an incorrect mask, a garbled swap, or an
  outright failure.
- **Body-proportion mismatches between the reference character and the
  original person can produce visible artifacts** (the pose skeleton driving
  the swap doesn't reshape to fit a very different body shape).
- Cost scales with the driving video's length: the model processes it in
  ~2.5-second segments (77 frames at its own working rate of 30 fps)
  rather than all at once.

**A real measured run, on this deployment (a 6.8s, 1280x720 driving
video):** ~25 minutes of wall-clock time end to end. That's long enough
that this endpoint's job spends meaningfully longer exposed to this
cluster's default `background` partition's pre-emption risk than a typical
LTX-2.3 request does (a job killed mid-run by higher-priority cluster work
surfaces as `status: "failed"`, and the whole ~25 minutes is lost, not just
the remainder). Set `partition: "main"` on the request if that risk isn't
acceptable for a given job -- see `GET /v1/health` / the app-level
description above for how the `partition` field and its fallback behave.

Output: MP4, H.264 video. Resolution/frame rate follow the source video's
own shape (resized so its pixel area matches roughly 1280x720, aspect ratio
preserved) at 30 fps -- there are no `height`/`width`/`orientation` fields
on this endpoint.
"""

REPLACE_EXAMPLES = {
    "dance_video_character_swap": {
        "summary": "Swap the dancer in an existing clip for a reference character",
        "value": {
            "video_asset_id": "<asset_id of the existing video, from POST /v1/uploads>",
            "image_asset_id": "<asset_id of the reference character photo, from POST /v1/uploads>",
            "seed": 10,
        },
    },
    "without_relighting_lora": {
        "summary": "Same swap, comparing quality with the relighting LoRA turned off",
        "value": {
            "video_asset_id": "<asset_id of the existing video, from POST /v1/uploads>",
            "image_asset_id": "<asset_id of the reference character photo, from POST /v1/uploads>",
            "seed": 10,
            "use_relighting_lora": False,
        },
    },
}
