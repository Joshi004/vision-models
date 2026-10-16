"""Long-form OpenAPI/Swagger documentation text for MiniMax H3's own endpoints,
kept out of router.py so the routes themselves stay readable. Docs shared by
every backend (uploads, jobs, app-level description) live in
common/openapi_docs.py instead.

This is the machine-readable half of H3's docs, meant to be read via
/openapi.json or /docs. The companion services/h3/GUIDE.md (same directory,
folded into GET /v1/guide's own MiniMax H3 section at request time) is the prose
half -- the model, the prompt format, and operational quirks that a schema alone
can't capture.
"""

from __future__ import annotations

GENERATE = """
Text-to-video, or first-frame / last-frame / first-to-last-frame video, with generated sound, on
MiniMax H3 (open weights, run on this cluster). Video and audio come out of the same denoising
loop, so quoted dialogue in the prompt is spoken with matching lip movement.

**What decides the recipe.** No frame: text-to-video. `first_frame_asset_id` and/or
`last_frame_asset_id` (upload them first with `POST /v1/uploads`): the video starts from, ends on,
or runs between those frames. All of these use the same model checkpoint.

**Two GPUs per job.** Each job reserves two H100s on one node (the text/vision encoder on one, the
transformer on the other), runs once, and releases them. There is no warm model between requests.

**The prompt is used exactly as sent.** Nothing here rewrites or enlarges it. H3 was trained on a
structured prompt format; write it yourself (see the MiniMax H3 section of `GET /v1/guide`). There is
no `negative_prompt` and no guidance scale: the weights are guidance-distilled.

**Shape and length.** 24 fps, 768 pixels on the short edge by default (1344x768 landscape, 768x1344
portrait). Lengths are rounded up to the next 17*n+5 frame count; the model is documented for 5.17 to
14.4 seconds and anything shorter or longer is **experimental** (quality unknown, long clips may run out
of GPU memory). Only 768p is available from the open weights; the hosted service's 2K upscaling pass is not part of them.

Output: MP4, H.264 video + AAC stereo audio (32 kHz), muxed together.
"""

AUDIO_TO_VIDEO = """
Video with generated sound from references -- an audio reference together with images and/or
videos -- on MiniMax H3's reference mode. Typical use: a speaker from a photo, saying lines in the
voice of a recording. Upload every file first with `POST /v1/uploads`.

**This is not the same guarantee as the LTX audio-to-video endpoints.** There, your audio is muxed
into the output unchanged and only the picture is generated. Here the audio is a *reference*: the
model reads it and then generates its own soundtrack together with the picture. What it does with
the reference is decided by how the prompt describes it -- `fully_copy` (reuse it as the soundtrack),
`partially_copy`, `reference` (follow its voice timbre and delivery without copying it) or
`weak_reference`. Either way the soundtrack passes through the model and is not guaranteed to be a
bit-exact copy. If you need your exact recording with the mouth matched to it, use
`POST /v1/ltx25/videos/audio-to-video` with `lip_sync: true` (or `POST /v1/tbdub/videos/lip-sync`).

**References.** One audio (`audio_asset_id`, 2 to 15 s) plus at least one image
(`image_asset_ids`, up to 9) or video (`video_asset_ids`, up to 3): an audio reference cannot stand
alone. In the prompt they are `<Picture 1>`..., `<Video 1>`... and `<Audio 1>`, in the order given,
each kind counted on its own.

**Length.** `duration_seconds` / `num_frames` (default 124 frames, 5.17 s), or
`match_audio_duration: true` to make the video as long as the audio, rounded up to the next 17*n+5
frames. There is no minimum length; outside the documented 5.17 to 14.4 s is experimental.

**Canvas.** References do not set the output's shape: the default is 16:9 (1344x768). Use
`orientation` or `height` + `width` for anything else.

**Two GPUs per job, prompt used exactly as sent, no negative prompt or guidance scale** -- as on
`POST /v1/h3/videos/generate`. The prompt format for this mode is different (six parts, with
reference labels); see the MiniMax H3 section of `GET /v1/guide`.

Output: MP4, H.264 video + AAC stereo audio (32 kHz), muxed together.
"""

# ---------------------------------------------------------------------------
# Named request-body examples (Swagger's "Examples" dropdown per endpoint)
# ---------------------------------------------------------------------------

_BAKERY_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening "
    "the shutters of a small street bakery before sunrise. The camera pushes in with small amplitude at slow speed as "
    "the middle-aged baker with a calm, slightly raspy voice (S1) places a fresh loaf on the wooden counter and says: "
    "<d>[English] First batch of the morning.</d>\n\n"
    "overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink softly inside the bakery. "
    "The doorbell rings once.\n\n"
    "non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo, with a gentle fade at the end."
)

_FIRST_FRAME_PROMPT = (
    "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, the young woman shown in <Picture 1> "
    "remains beside the rain-covered train window, preserving her appearance, clothing, seat position, and the "
    "carriage layout. The camera trucks right with small amplitude at slow speed as she lifts her gaze from the "
    "folded letter toward the passing city lights. The quiet, breathy young woman (S1) says: "
    "<d>[English] I get off at the next station.</d>\n\n"
    "overall_soundscape: The train wheels produce a steady metallic rhythm beneath a low ventilation hum. Rain ticks "
    "against the window while paper rustles softly in her hands.\n\n"
    "non_diegetic_music: Sustained cello notes at a slow tempo, gradually decreasing in volume."
)

_REFERENCE_PROMPT = (
    "subject_definitions:\n"
    "<Subject 1> is the young woman in <Picture 1>, with long dark hair, a blue cardigan, and a thin silver necklace.\n"
    "<Audio 1> is the voice-timbre reference for <Subject 1> (S1).\n\n"
    "summary:\n"
    "[reference generation + audio reference] <Subject 1> speaks to the camera in a sunlit kitchen, using <Audio 1> "
    "as the voice-timbre reference.\n\n"
    "retention_analysis:\n"
    "<Subject 1> (appears in [Shot 1]): fully_preserved - her face, long dark hair and blue cardigan are retained.\n"
    "<Audio 1>: reference - the target speaker follows <Audio 1>'s voice timbre and measured delivery without "
    "copying the original signal.\n\n"
    "detailed_description:\n"
    "The target video is live-action with soft window light and a shallow depth of field.\n"
    "[Shot 1] A medium close-up of <Subject 1> (S1) at a kitchen counter, a steaming mug beside her. She smiles at "
    "the camera and says, using the voice timbre referenced from <Audio 1>, "
    "<d>[English] It is a quiet morning, and the coffee is almost ready.</d> Her lips close in a relaxed smile as "
    "the line ends.\n\n"
    "overall_soundscape:\n"
    "Soft kitchen room tone and the faint hiss of a coffee machine.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)

GENERATE_EXAMPLES = {
    "text_to_video": {
        "summary": "Text-to-video, landscape, default length (5.17 s)",
        "value": {
            "prompt": _BAKERY_PROMPT,
            "orientation": "landscape",
            "seed": 42,
        },
    },
    "portrait_10s": {
        "summary": "Portrait, about 10 seconds",
        "value": {
            "prompt": _BAKERY_PROMPT,
            "orientation": "portrait",
            "duration_seconds": 10,
        },
    },
    "first_frame": {
        "summary": "First frame to video (image-to-video)",
        "value": {
            "prompt": _FIRST_FRAME_PROMPT,
            "first_frame_asset_id": "<asset_id from POST /v1/uploads>",
            "duration_seconds": 6,
        },
    },
    "first_and_last_frame": {
        "summary": "First and last frame (the prompt describes the path between them)",
        "value": {
            "prompt": (
                "How the reference pictures align with the target video -- Picture 1 (from Shot 1) aligns with the "
                "0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 5.17-second mark of "
                "the target video.\n\n"
                "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a rain-soaked cyclist begins in "
                "the position and framing established by Picture 1, holding a closed black umbrella beside a silver "
                "bicycle. The camera pulls out with small amplitude at slow speed as she raises the umbrella above "
                "her shoulder and presses the runner upward until the canopy opens, settling into the pose and "
                "composition established by Picture 2 at the end of the shot.\n\n"
                "overall_soundscape: Rain falls steadily on the pavement, followed by the metallic click of the "
                "umbrella runner and the soft snap of the canopy opening.\n\n"
                "non_diegetic_music: N/A"
            ),
            "first_frame_asset_id": "<asset_id of the starting frame>",
            "last_frame_asset_id": "<asset_id of the ending frame>",
        },
    },
}

AUDIO_TO_VIDEO_EXAMPLES = {
    "voice_reference": {
        "summary": "A speaker from a photo, voice taken from a recording",
        "value": {
            "prompt": _REFERENCE_PROMPT,
            "audio_asset_id": "<asset_id of the voice recording>",
            "image_asset_ids": ["<asset_id of the speaker's photo>"],
            "orientation": "landscape",
            "duration_seconds": 6,
        },
    },
    "match_audio": {
        "summary": "Video as long as the audio reference",
        "value": {
            "prompt": _REFERENCE_PROMPT,
            "audio_asset_id": "<asset_id of the voice recording>",
            "image_asset_ids": ["<asset_id of the speaker's photo>"],
            "match_audio_duration": True,
        },
    },
}
