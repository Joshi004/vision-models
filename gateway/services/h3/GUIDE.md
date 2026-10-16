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
| `POST /v1/h3/videos/audio-to-video` | MP4 (video + audio) | "Ref2VA" (`transformer_ref/`) | `prompt`, `audio_asset_id`, and at least one of `image_asset_ids` / `video_asset_ids` |

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

- References are labelled in the order you give them, **each kind counted on its own**: `<Picture 1>` is the first of
  `image_asset_ids`, `<Video 1>` the first of `video_asset_ids`, `<Audio 1>` the audio. `<Subject N>` is not a file:
  it names a person, object or place you define from them.
- `subject_definitions` defines each label and what it stands for. `summary` starts with the task type in brackets
  (`[reference generation]`, `[reference generation + audio reference]`, `[keyframe completion]`, ...).
  `retention_analysis` states, per label, whether it is `fully_preserved`, `partially_preserved`, `attribute_transfer` or
  `weak_reference`; for audio, `fully_copy`, `partially_copy`, `reference` or `weak_reference`.
- `detailed_description` is the body: an English sentence or two on the style, then `[Shot N]` blocks as above with the
  labels inserted where they apply. A speaking subject is written `<Subject 1> (S1) ... says <d>[English] ...</d>`.
  Aim for 350 to 500 words for a generation task.

### 5. What the audio reference does (and does not do)

On `/videos/audio-to-video` the audio you upload is **not** muxed into the result unchanged, as it is on the LTX
audio-to-video endpoints. The model reads it as a reference and generates its own soundtrack together with the
picture. How it is used is **written in the prompt**:

- `reference` -- follow its voice timbre, delivery or music style, without copying it. The words spoken come from the
  `<d>...</d>` text in your prompt. This is the way to keep one character sounding like the same person across shots:
  reuse the same recording as `<Audio 1>` every time.
- `fully_copy` / `partially_copy` -- reuse it as the soundtrack (or part of it). The result is the model's rendition of
  it, which has passed through the audio encoder and the generative model: expect it to be close, not bit-exact.
  (MiniMax documents these markers; how close the copy is in practice has not been measured on this cluster.)

If you need your exact recording with the speaker's mouth matched to it, use `POST /v1/ltx25/videos/audio-to-video` with
`lip_sync: true`, or `POST /v1/tbdub/videos/lip-sync`, instead.

An audio reference is 2 to 15 seconds and never stands alone: it needs at least one image or video reference. Mono or
stereo at any sample rate is accepted (it is resampled to 32 kHz). `match_audio_duration: true` makes the video as long
as the audio, rounded up to the next 17*n+5 frames. There is no minimum length, and the audio is not cut; the audio
may not be longer than the server's ceiling (60 s by default). The model is documented for 5.17 to 14.4 s, so audio
outside that range is experimental: very short clips may look or sound wrong or fail, and long ones may run out of GPU
memory.

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
`duration_seconds`.

### 7. Licence and responsible use

The weights are under the MiniMax H3 Community License (`LICENSE` in the weights folder). Read it before relying on
this backend. In short: it is **not licensed for use or hosting in the United States, the European Union, the United
Kingdom or South Korea** without MiniMax's separate written authorization, and commercial products above 20 million USD
in yearly revenue need authorization too. This API does not enforce any of that itself.

Only use faces, voices and recordings you have the right to use, and disclose generated media as such.
