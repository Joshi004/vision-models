## MiniMax H3 -- text / frame / reference to video, with generated sound

This section is the prose companion to this backend's own endpoints. The endpoint reference generated below
(straight from this server's own `/openapi.json`) is the source of truth for exact field names, types and
validation; this section explains the model, the prompt format it expects, and how it differs from the LTX
backends on the same server.

### 1. What this is

MiniMax H3 is MiniMax's open-weight audio+video model: a 33B transformer (with Qwen3-VL-32B as its text and
vision encoder) that generates the picture and a stereo soundtrack **together, in one denoising loop**, so quoted
dialogue comes out with matching lip movement. Every endpoint of this backend lives under `/v1/h3/...`. Job
submission/tracking and file uploads (`/v1/uploads`, `/v1/jobs/...`) are shared infrastructure with no `/h3`
prefix, used the same way regardless of which backend created the job -- see the main guide's "Jobs and timing"
section. Every request may also set the optional top-level `partition` and `incognito` fields, as on every other
backend.

This server runs the **open release only** (the "H3-Base" checkpoints, bf16, no quantization). MiniMax's hosted
service adds two more models that are not in the open weights and therefore not here:

- **Context-IR**, which rewrites a plain request into the structured prompt below. Here **nothing rewrites your
  prompt** -- it reaches the model exactly as sent, so whatever writes your prompts should write the structured
  format itself (section 4).
- **A 2K upscaler.** The open weights stop at 768p.

### 2. What's different from the LTX backends

| | LTX-2.3 (`/v1/ltx`) / LTX-2.5 (`/v1/ltx25`) | MiniMax H3 (`/v1/h3`) |
|---|---|---|
| GPUs per job | 1 | **2** on one node (the encoder on one, the transformer on the other) |
| Prompt | free prose; optional `enhance_prompt` | a structured format (section 4), never rewritten |
| Guidance | `negative_prompt` on some recipes | none -- the weights are guidance-distilled, one forward pass per step |
| Resolution | up to 1920x1088 | 768 px short edge: 1344x768 landscape, 768x1344 portrait |
| Frame rate | 24 (25 with `lip_sync`) | fixed 24 |
| Length | up to ~20 s | any length, rounded up to 17*n+5 frames; 5.17 to 14.4 s is the documented range, outside it is experimental |
| Cuts inside a clip | LTX-2.5: in prose | `[Shot N]` blocks with cut times |
| Audio-to-video | your audio is muxed in unchanged | the audio is a **reference**; the model generates the soundtrack (section 5) |
| `lip_sync` option | yes (TBDub) | no -- use `POST /v1/tbdub/videos/lip-sync` if you need your exact audio |

### 3. Endpoint chooser

| Endpoint | Produces | Checkpoint | Key inputs |
|---|---|---|---|
| `POST /v1/h3/videos/generate` | MP4 (video + audio) | "FL2VA" (`transformer/`) | `prompt`; optional `first_frame_asset_id` and/or `last_frame_asset_id` |
| `POST /v1/h3/videos/audio-to-video` | MP4 (video + audio) | "Ref2VA" (`transformer_ref/`) | `prompt`, at least one of `image_asset_ids` (up to 9) / `video_asset_ids` (up to 3); optional `audio_asset_ids` (up to 3) and `video_audio`; 12 references in all |

MP4 outputs are H.264 video + AAC stereo audio at 32 kHz, muxed together. Upload every input first with
`POST /v1/uploads`.

### 4. Writing the prompt

H3 was trained on a structured prompt. A plain sentence is accepted, but gives the model far less to go on. The
authoritative references are MiniMax's own two guides; they are included in the downloaded weights, at
`weights/h3/MiniMax-H3/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md` (text and frames) and
`VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` (references) under the data directory -- hand them to whatever writes your
prompts. What follows is the short version.

**Text and frames (`/videos/generate`).** Three fields, separated by blank lines:

```
integrated_multimodal_description: [Shot 1] <style, composition, subjects, action, camera, speech> [Shot 2] At 00:03.500, <cut ...>

overall_soundscape: <ambience, physical sounds, non-verbal human sounds -- no dialogue>

non_diegetic_music: <instrumentation, tempo, dynamics of score the characters cannot hear, or N/A>
```

- **Shots.** `[Shot 1]` has no time; every later shot starts `[Shot N] At MM:SS.mmm, ...` with strictly increasing cut
  times inside the video's length. State the overall style (live-action, cinematic, 2D-animated, ...) and the
  opening composition at the start of Shot 1. One shot is the right choice when interpolating between two frames.
- **Camera.** Write it as natural English with a motion type, and an amplitude and speed when they matter: "The camera
  pushes in with small amplitude at slow speed toward the folded letter." Types: zoom in/out, push in/pull out, pan
  left/right, truck left/right, tilt up/down, pedestal up/down, arc shot, tracking shot, static shot, shake slightly/
  strongly, POV, roll clockwise/counterclockwise.
- **Speech.** Give each voice a stable id, `(S1)`, `(S2)`, and describe it once (age, pitch, pace, accent). Put the
  words inside `<d>[Language] ...</d>`, verbatim and in their own language; everything else in the prompt is English:
  `The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>`. For a
  voice-over use "says in an off-screen voiceover" and state that the on-screen lips stay closed. Keep the amount of
  speech in proportion to the clip's length: a long line crammed into a short clip degrades the result.
- **Frames.** With a first frame, the prompt's first line is
  `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
  With a first and a last frame: `How the reference pictures align with the target video -- Picture 1 (from Shot 1)
  aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the
  target video.` (S.SS is the video's length in seconds, two decimals: 124 frames = 5.17.) With only a last frame:
  `How the reference pictures align with the target video -- <Picture 1> (from [Shot N]) aligns with the S.SS-second mark
  of the target video.` Then describe the *path* between the frames, not two static pictures.
- **Sound fields.** `overall_soundscape` is 1 to 4 sentences; `non_diegetic_music` is 1 to 3 sentences about
  instrumentation, tempo and dynamics, or `N/A`.

**References (`/videos/audio-to-video`).** Six fields, in this order: `subject_definitions:`, `summary:`,
`retention_analysis:`, `detailed_description:`, `overall_soundscape:`, `non_diegetic_music:`.

- **What you can send.** At least one image or video; audio is optional and never stands alone.

  | Field | Up to | Used for |
  |---|---|---|
  | `image_asset_ids` | 9 | how a character, place or style looks (not first or last frames); any shape from 1:4 to 4:1 |
  | `video_asset_ids` | 3 | motion and camera work, or a video to edit or continue; 2 to 15 s each, 15 s in all (documented) |
  | `audio_asset_ids` | 3 | a voice, music or sound to follow or reuse; 2 to 15 s each, 15 s in all (documented) |
  | all together | 12 | |

  The 2 to 15 s clip rules are MiniMax's; this server does not enforce them. It cuts every reference to the length of
  the output, so anything outside them is experimental. The order inside each list is the order the model reads them.
- **Labels.** Pictures and videos are labelled in the order you give them, **each kind counted on its own**:
  `<Picture 1>` is the first of `image_asset_ids`, `<Video 1>` the first of `video_asset_ids`. `<Audio N>` counts all
  the audio, in this order: a reference video that has a soundtrack takes the next `<Audio N>` **just before its
  `<Video N>`**, and then the files in `audio_asset_ids` follow. With one video that has sound and one audio file,
  `<Audio 1>` is the video's own sound and `<Audio 2>` is the file. Set `video_audio` to `false` for a video to use its
  picture only: it then takes no `<Audio N>` (a video with no sound takes none either). Only mention labels that
  exist. `<Subject N>` is not a file: it names a person, object or place you define from them.
- **Images only, several characters.** Audio is optional. Give one image per character or place and define
  each as a `<Subject N>`; the images are references for how they look, not first or last frames. When someone
  should appear later (say at 4 s), start a new shot: `[Shot 2] At 00:04.000, <Subject 2> walks in ...`. Use
  `[reference generation]` in `summary`, omit every `<Audio N>` line, and give each speaker an `(S1)`, `(S2)` id.
  The images guide the model rather than pin frames, so how closely a mid-clip entrance is followed is not guaranteed.
- **A video with sound, plus a voice.** For an edit that keeps the video's music and speaks in another voice, send the
  video in `video_asset_ids` and the recording in `audio_asset_ids`. Then `<Audio 1>` is the video's soundtrack and
  `<Audio 2>` the voice:
  `<Audio 1> is the synchronized audio track of <Video 1>, providing the background music.` and
  `<Audio 2> is the voice-timbre reference for <Subject 1> (S1).` In `retention_analysis`, `<Audio 1>: partially_copy`
  (or `fully_copy`) and `<Audio 2>: reference`. The `[video editing + audio reference + audio reuse]` example in
  the endpoint's Swagger examples shows the whole prompt.
- `subject_definitions` defines each label and what it stands for. `summary` starts with the task type in brackets,
  one or more joined with ` + ` (`reference generation`, `keyframe completion`, `video editing`, `video continuation`,
  `audio reuse`, `audio reference`), for example `[reference generation + audio reference]`. Give a reference the task
  that matches what it does, not just that it exists: a video that only supplies camera movement is
  `reference generation`, a video that is changed is `video editing`.
  `retention_analysis` states, per label, whether it is `fully_preserved`, `partially_preserved`, `attribute_transfer` or
  `weak_reference`; for audio, `fully_copy`, `partially_copy`, `reference` or `weak_reference`.
- `detailed_description` is the body: an English sentence or two on the style, then `[Shot N]` blocks as above with the
  labels inserted where they apply. A speaking subject is written `<Subject 1> (S1) ... says <d>[English] ...</d>`.
  Aim for 350 to 500 words for a generation task.

### 5. What the audio references do (and do not do)

On `/videos/audio-to-video` the audio you upload (`audio_asset_ids`, and the soundtrack of a reference video) is
**not** muxed into the result unchanged, as it is on the LTX audio-to-video endpoints. The model reads it as a
reference and generates its own soundtrack together with the picture. How each audio is used is **written in the
prompt**, per `<Audio N>`, so up to three clips (plus video soundtracks) can each do something different, such as one
voice per speaker, or a voice and a music track:

- `reference` -- follow its voice timbre, delivery or music style, without copying it. The words spoken come from the
  `<d>...</d>` text in your prompt. This is the way to keep one character sounding like the same person across shots:
  reuse the same recording (the same `<Audio N>`) every time.
- `fully_copy` / `partially_copy` -- reuse it as the soundtrack (or part of it). The result is the model's rendition of
  it, which has passed through the audio encoder and the generative model: expect it to be close, not bit-exact.
  (MiniMax documents these markers; how close the copy is in practice has not been measured on this cluster.)

If you need your exact recording with the speaker's mouth matched to it, use `POST /v1/ltx25/videos/audio-to-video` with
`lip_sync: true`, or `POST /v1/tbdub/videos/lip-sync`, instead.

Audio is optional: send up to 3 files in `audio_asset_ids`, or none. Each is documented as 2 to 15 seconds, and all of
them together as at most 15 seconds (a reference video's soundtrack, and the videos themselves, have the same
rules). None of it stands alone: audio needs at least one image or video reference, and the 12-reference limit counts
audio files too. Mono or stereo at any sample rate is accepted (it is resampled to 32 kHz). A reference longer than
the video is cut to the video's length; the server does not check the 2 to 15 s or 15 s rules, so a clip outside
them is experimental.

`match_audio_duration: true` needs at least one audio file, and makes the video as long as the **first** one
(`audio_asset_ids[0]`), rounded up to the next 17*n+5 frames. There is no minimum length; that first audio may not be
longer than the server's ceiling (60 s by default). The model is documented for 5.17 to 14.4 s, so audio outside that
range is experimental: very short clips may look or sound wrong or fail, and long ones may run out of GPU memory.

**Upgrading from 0.13.x.** `audio_asset_id` (a single audio) is now `audio_asset_ids` (a list). A request that still
sends `audio_asset_id` is rejected with 422 instead of being quietly ignored. Retrying an older job in the web UI
fills the list from the old field.

### 6. Job lifecycle, timing and resources

Identical to the other backends: submit, poll `GET /v1/jobs/{job_id}` every 10-15 seconds, then download
`GET /v1/jobs/{job_id}/result`. Every request is an independent cold start -- there is no warm model between
requests -- on **two H100s on one node**: the Qwen3-VL-32B encoder (62 GiB of bf16 weights) on one and the 33B
transformer (61.7 GiB) on the other, both loaded at the same time. When the encoder is done, half of the transformer's
blocks move to its GPU, so the two GPUs share the generation. The weights are read from network storage on every
job, so most of a short job's wall time is loading.

**Timing has not been measured on this cluster yet.** Budget more than 10 minutes per request until real runs have
replaced this paragraph; `typical_run_seconds` on a job starts from a pessimistic guess and switches to the median of
real runs by itself. Larger canvases and longer clips cost far more than linearly (diffusers' documentation: 960x544 is
about 2.3x faster per step than 1344x768).

**GPU memory.** With the whole transformer on one 80 GB card, a 345-frame (14.4 s) clip at the full 1344x768 canvas ran
out of memory, while 124 frames fit. Sharing the transformer between the two GPUs gives each about 45 GiB for the
activations, which is meant to fix that, but **a full-size 14.4 s clip has not been confirmed to complete yet**. If a job
fails with an out-of-memory error, retry with a smaller canvas (for example `height` 544, `width` 960) or a shorter
`duration_seconds`. Every reference adds to what the transformer holds (images are encoded at a 2048 px short edge, and
a video adds frames up to the output's length), so a request near the **12-reference maximum has not been run on this
cluster and may run out of memory** the same way; if it does, send fewer references or use a smaller canvas.

### 7. Licence and responsible use

The weights are under the MiniMax H3 Community License (`LICENSE` in the weights folder). Read it before relying on
this backend. In short: it is **not licensed for use or hosting in the United States, the European Union, the United
Kingdom or South Korea** without MiniMax's separate written authorization, and commercial products above 20 million USD
in yearly revenue need authorization too. This API does not enforce any of that itself.

Only use faces, voices and recordings you have the right to use, and disclose generated media as such.
