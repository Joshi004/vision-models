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
Video with generated sound from references -- images, videos and audio clips, mixed -- on MiniMax
H3's reference mode. Typical uses: a speaker from a photo, saying lines in the voice of a recording;
several characters and places from images alone, brought in by the prompt at the moments you
describe (`[Shot 2] At 00:04.000, <Subject 2> walks in ...`); or editing or continuing a video while
following a voice and keeping its music. The images are references for appearance, not first or
last frames. Upload every file first with `POST /v1/uploads`.

**This is not the same guarantee as the LTX audio-to-video endpoints.** There, your audio is muxed
into the output unchanged and only the picture is generated. Here, audio is a *reference*: the
model reads it and then generates its own soundtrack together with the picture. What it does with
each reference is decided by how the prompt describes it -- `fully_copy` (reuse it as the soundtrack),
`partially_copy`, `reference` (follow its voice timbre and delivery without copying it) or
`weak_reference`. Either way the soundtrack passes through the model and is not guaranteed to be a
bit-exact copy. If you need your exact recording with the mouth matched to it, use
`POST /v1/ltx25/videos/audio-to-video` with `lip_sync: true` (or `POST /v1/tbdub/videos/lip-sync`).

**References and limits.** At least one image or video; audio is optional and cannot stand alone.

| Field | Up to | Notes |
|---|---|---|
| `image_asset_ids` | 9 | appearance of a character, place or style; any shape from 1:4 to 4:1 |
| `video_asset_ids` | 3 | motion, camera, or a video to edit or continue; documented as 2 to 15 s each, 15 s in all |
| `audio_asset_ids` | 3 | a voice, music or sound to follow or reuse; documented as 2 to 15 s each, 15 s in all |
| all together | 12 | |

The clip-length rules are MiniMax's and are **not enforced here**: every reference is simply cut to
the length of the output, so treat anything else as experimental. Within each list, the order you
give is the order the model reads them.

**Labels in the prompt.** Pictures and videos are `<Picture 1>`..., `<Video 1>`..., in the order
given. Audio is `<Audio 1>`... and is counted in this order: a reference video that has a soundtrack
takes the next `<Audio N>` just before its `<Video N>` (`video_audio` switches this off per video),
and then the `audio_asset_ids` follow. So with one video that has sound and one audio file,
`<Audio 1>` is the video's own sound and `<Audio 2>` is the file. A video with no sound takes no
`<Audio N>`. Only mention labels that exist.

**Length.** `duration_seconds` / `num_frames` (default 124 frames, 5.17 s), or
`match_audio_duration: true` (needs at least one `audio_asset_ids` entry; the first one sets the
length) to make the video as long as that audio, rounded up to the next 17*n+5 frames. There is no
minimum length; outside the documented 5.17 to 14.4 s is experimental.

**Changed in 0.14.0.** `audio_asset_id` (one audio) became `audio_asset_ids` (a list); a request that
still sends `audio_asset_id` is rejected with 422.

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

_TWO_CHARACTERS_PROMPT = (
    "subject_definitions:\n"
    "<Subject 1> is the young woman in <Picture 1>, with long dark hair and a blue cardigan.\n"
    "<Subject 2> is the bearded man in <Picture 2>, with short grey hair and a brown leather jacket.\n\n"
    "summary:\n"
    "[reference generation] <Subject 1> waits alone in a quiet cafe; <Subject 2> joins her at her table and the two talk.\n\n"
    "retention_analysis:\n"
    "<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - her face, long dark hair and blue cardigan are retained.\n"
    "<Subject 2> (appears in [Shot 2]): fully_preserved - his face, grey hair, beard and brown leather jacket are retained.\n\n"
    "detailed_description:\n"
    "The target video is live-action with warm window light and a shallow depth of field.\n"
    "[Shot 1] A medium shot of <Subject 1> (S1) seated at a cafe table, stirring a cup of tea. She glances toward the "
    "door and says softly, <d>[English] He said he would be here by now.</d>\n"
    "[Shot 2] At 00:04.000, the camera cuts to a two-shot as <Subject 2> (S2) enters from the left, sits down opposite "
    "<Subject 1> and says in a warm, low voice, <d>[English] Sorry I am late.</d> <Subject 1> (S1) smiles and "
    "replies, <d>[English] You are just in time.</d>\n\n"
    "overall_soundscape:\n"
    "Quiet cafe chatter, a spoon against porcelain and a door chime.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)

_VIDEO_EDIT_PROMPT = (
    "subject_definitions:\n"
    "<Subject 1> is the man in <Video 1>, with short dark hair and a green waxed jacket.\n"
    "<Video 1> is the source video for the target video edit.\n"
    "<Audio 1> is the synchronized audio track of <Video 1>, providing the background music.\n"
    "<Audio 2> is the voice-timbre reference for <Subject 1> (S1).\n\n"
    "summary:\n"
    "[video editing + audio reference + audio reuse] The target video is an edited version of <Video 1>. "
    "<Subject 1> stands at the end of a wooden pier at dusk and now speaks a new line in the voice of <Audio 2>, "
    "while the music of <Audio 1> keeps playing underneath.\n\n"
    "retention_analysis:\n"
    "<Subject 1> (appears in [Shot 1]): fully_preserved - his face, short dark hair and green waxed jacket are "
    "retained, with his mouth newly animated to speak.\n"
    "<Video 1> (source video editing): fully_preserved - the framing, dusk light and pier setting are kept.\n"
    "<Audio 1>: partially_copy - the background music is reused beneath the new dialogue.\n"
    "<Audio 2>: reference - the target speaker follows the voice timbre of <Audio 2> without copying the original "
    "signal.\n\n"
    "detailed_description:\n"
    "The target video is in a realistic photographic style with warm dusk light.\n"
    "[Shot 1] The shot begins from the source <Video 1>: <Subject 1>, a man with short dark hair and a green waxed "
    "jacket, stands at the end of a wooden pier looking out over calm water. The music of <Audio 1> plays "
    "continuously. <Subject 1> (S1) turns toward the camera and says, with the voice timbre referenced from "
    "<Audio 2>, <d>[English] The tide turns at midnight.</d> His lips close, and he looks back at the horizon as the "
    "camera holds.\n\n"
    "overall_soundscape:\n"
    "Small waves lap against the pier posts, beneath the music of <Audio 1>.\n\n"
    "non_diegetic_music:\n"
    "The background music from <Audio 1> is reused as the continuous score, playing quietly beneath the dialogue."
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
            "image_asset_ids": ["<asset_id of the speaker's photo>"],
            "audio_asset_ids": ["<asset_id of the voice recording>"],
            "orientation": "landscape",
            "duration_seconds": 6,
        },
    },
    "video_edit_with_voice": {
        "summary": "Edit a video that has sound: keep its music (<Audio 1>), follow a voice recording (<Audio 2>)",
        "value": {
            "prompt": _VIDEO_EDIT_PROMPT,
            "video_asset_ids": ["<asset_id of the source video, which has a soundtrack>"],
            "audio_asset_ids": ["<asset_id of the voice recording>"],
            "duration_seconds": 5,
        },
    },
    "images_only": {
        "summary": "Several characters from images alone (no audio); the second one enters at 4 s",
        "value": {
            "prompt": _TWO_CHARACTERS_PROMPT,
            "image_asset_ids": ["<asset_id of the first character>", "<asset_id of the second character>"],
            "orientation": "landscape",
            "duration_seconds": 8,
        },
    },
    "match_audio": {
        "summary": "Video as long as the audio reference",
        "value": {
            "prompt": _REFERENCE_PROMPT,
            "image_asset_ids": ["<asset_id of the speaker's photo>"],
            "audio_asset_ids": ["<asset_id of the voice recording>"],
            "match_audio_duration": True,
        },
    },
}
