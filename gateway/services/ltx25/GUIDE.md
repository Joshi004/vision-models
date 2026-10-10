## LTX-2.5 -- text/image-to-video, first/last-frame interpolation, and retake, with native multi-shot

This section is the prose companion to this backend's own endpoints. The endpoint reference
generated below (straight from this server's own `/openapi.json`) is the source of truth for exact
field names, types, and validation; this section explains the model, how to prompt it (including
multi-shot), and how it differs from the LTX-2.3 backend on the same server.

### 1. What this is

LTX-2.5 is the newer generation of Lightricks' 22B-parameter audio+video diffusion model, running
alongside the LTX-2.3 backend (`/v1/ltx/...`), which is unchanged. Every endpoint specific to this
backend lives under `/v1/ltx25/...`. Job submission/tracking and file uploads (`/v1/uploads`,
`/v1/jobs/...`) are shared infrastructure with no `/ltx25` prefix, used the same way regardless of
which backend created the job -- see the main guide's "Jobs and timing" section.

Every request may also set an optional top-level `partition` field, exactly as on every other
backend.

### 2. What's different from LTX-2.3

| | LTX-2.3 (`/v1/ltx`) | LTX-2.5 (`/v1/ltx25`) |
|---|---|---|
| Multi-shot (cuts inside one clip) | No -- one continuous shot | Yes, written into the prompt |
| `mode: "quality"` | Dev checkpoint with CFG/STG guidance + distilled LoRA | DFR: distilled transformer + generated keyframes + detailing pass |
| `negative_prompt` | Supported (quality mode) | Not available -- neither recipe uses guidance |
| Clip length | `duration_seconds` / `num_frames` | Same, plus `auto_duration` (model picks from the prompt) |
| Video decoder | Standard VAE | Convolutional VAE on this deployment (LTX-2.5 also ships a heavier diffusion decoder, which is not used here -- it corrupted the last fraction of a second of a 10-second test clip) |
| Text encoder | Gemma 3 12B | Gemma 4 12B, fine-tuned for LTX |
| Recipes on this server | Text/image-to-video, keyframe interpolation, audio-to-video, retake, text-to-audio | Text/image-to-video, first/last-frame interpolation, retake |

Audio-to-video and text-to-audio are not available on this backend yet; use the LTX-2.3 backend
for those. LTX-2.3's keyframe interpolation accepts any number of keyframes; this backend's
`interpolate` takes exactly a first and a last frame (and, unlike 2.3's, has a fast mode).

### 3. Endpoint chooser

| Endpoint | Produces | Recipe | Key inputs |
|---|---|---|---|
| `POST /v1/ltx25/videos/generate` | video (MP4) | fast **or** quality (DFR), your choice | `prompt`, optional `images` |
| `POST /v1/ltx25/videos/interpolate` | video (MP4) | fast **or** quality (DFR), your choice | `first_frame_asset_id`, `last_frame_asset_id`, `prompt` |
| `POST /v1/ltx25/videos/retake` | video (MP4) | distilled only | `video_asset_id`, `prompt`, `start_time`, `end_time` |

MP4 outputs are H.264 video + AAC audio, muxed together. Video and audio are generated jointly:
quoted dialogue in the prompt comes out lip-synced.

### 4. Job lifecycle, timing, and shared-cluster etiquette

Identical to LTX-2.3: submit, poll `GET /v1/jobs/{job_id}` every 10-15 seconds, then download
`GET /v1/jobs/{job_id}/result`. Every request is an independent cold start on one H100 -- there is
no warm model between requests -- so most of a job's wall time is loading weights from network
storage, not denoising.

**Timing, from real measurements on this deployment** (1920x1088, one H100, wall time of the whole
job after it starts running -- not including queue wait). The spread is wide because most of that
time is loading weights from shared network storage, not denoising: three jobs submitted through
this API together took 4.5-4.8 minutes each, while an earlier batch of four to five jobs started at
once on a cold cache took 5.5-10.6 minutes each. **Budget 5-10 minutes per request.**

| Run | Clip | Node wall time | Peak GPU memory |
|---|---|---|---|
| Fast | 4-5s (97-121 frames) | ~4.5-10 min | ~45-50 GB |
| Fast + `enhance_prompt` | 5s | ~7.5 min | ~50 GB |
| Fast, 3-shot prompt | 10s (241 frames) | ~7.5-8 min | ~60-65 GB |
| Quality (DFR) | 5s | ~5-10.5 min | ~56 GB |
| Interpolate (fast or quality) | 5s | ~7 min (14 min once, on a slow node while the cluster was busy) | not measured |
| Retake | 5s source video | ~5-6 min | ~50 GB |

Fast mode with `auto_duration` chose 2.4s for a one-line spoken sentence and 4.0s for a short
scene description. `"quality"` (DFR) costs more than `"fast"` in GPU memory and, on a cold cache,
in time. A job's own status also carries `typical_run_seconds`, which switches from a documented
estimate to the median of this deployment's recent real runs of that recipe once any exist.

Jobs run on a pre-emptible partition by default; a job killed by higher-priority work reports
`status: "failed"` with an `error` saying so, and the correct response is to resubmit the same
request unchanged. Each job consumes a full GPU on a shared cluster, so avoid speculative batches
of requests.

### 5. Video/audio shape rules

- `height`/`width` must each be a **multiple of 64** (enforced on the GPU node, not at submission --
  a bad value only fails once a job is already running). Prefer `orientation` (`landscape` =
  1920x1088, `portrait` = 1088x1920).
- Frame count must be `8k+1`. `duration_seconds` handles this for you (rounds to the nearest valid
  count); an explicit `num_frames` that doesn't satisfy it is silently snapped.

  | duration_seconds | frames (24fps) |
  |---|---|
  | 3 | 73 |
  | 5 | 121 |
  | 8 | 193 |
  | 10 | 241 |

- `auto_duration: true` lets the model choose the length from the prompt, clamped to 1-20 seconds.
  It cannot be combined with `duration_seconds` or `num_frames`. With none of the three set, the
  clip is ~5 seconds.
- `POST /v1/ltx25/videos/interpolate` has no `auto_duration` (the last frame's position needs a known
  length), and both frame images are resized and center-cropped to the output size, so give them
  the output's aspect ratio (16:9 for `landscape`, 9:16 for `portrait`) to avoid unwanted cropping.
- `POST /v1/ltx25/videos/retake`'s source video must **already** have `8k+1` frames and width/height
  that are multiples of 32. A video produced by `POST /v1/ltx25/videos/generate` already satisfies
  this; an arbitrary video you upload from elsewhere might not. Uploads are {{upload_mb}} MB max
  per file, and finished job outputs expire {{job_retention_days}} day(s) after they finish.

### 6. Prompting guide

LTX-2.5 responds to **cinematographic, chronological natural-language description**, not keyword
lists. Write one flowing paragraph in the present tense: the main action first, then movements and
gestures, character and object appearance, background and environment, camera angle and movement,
and lighting and color. Keep to roughly 200 words for a single shot. Prefer plain language over
intensified adjectives ("red dress", not "vibrant crimson dress"). An optional
`Style: <style>, ...` prefix sets an overall visual style.

**Audio**: describe it specifically and in time with the action ("soft footsteps on tile as she
crosses the room"). For speech, put the **exact words in double quotes** with a description of the
voice. The model lip-syncs to whatever dialogue you write, so don't write dialogue you don't want
spoken.

**Image-to-video** (`images`): describe only what *changes* from the attached image. Re-describing
static details already visible in it can pull the result away from the photo.

**Interpolate**: describe the *motion connecting* the two frames, not their static appearance --
the images already establish that. The first frame is the video's literal first frame; the last
frame guides the ending. A bigger change between the two frames needs a longer clip
(`duration_seconds`) to travel believably.

**Choosing the two frames** matters more than the prompt. In testing, two frames from the same
scene with a moderate change (a dog running toward the camera along a beach) produced a smooth,
continuous run. Two completely different compositions (a wide courtyard shot and a close-up of a
woman's face) started and ended exactly on the supplied frames but crossed between them with a
ghosted **cross-dissolve** in the middle instead of a camera move, in both `fast` and `quality`
mode. So: use frames of the same place, subject, and lighting, and make the change between them a
plausible movement. If you want a hard change of shot, use `POST /v1/ltx25/videos/generate` with a
multi-shot prompt instead.

**Retake**: describe only the content of the replaced `[start_time, end_time]` window, as if it
were the whole prompt for a short standalone clip.

#### Multi-shot prompts

Unlike LTX-2.3, LTX-2.5 can cut between several shots inside one clip while holding the same
character, environment, lighting, voice, and visual style. **There is no parameter for it** -- you
write the cuts into the prompt as prose:

- Write the whole scene as **one chronological paragraph**. Numbered shot lists and screenplay
  sluglines are ignored unless you also describe the cut in a sentence.
- At **every cut**, do four things: (1) **name the transition** ("A hard cut transitions to...", "The
  view cuts to a close-up of...", "A match cut connects...", "The image dissolves into..."); (2)
  **re-establish the new shot** -- scale, angle, who is in frame, and lighting if it changed; (3)
  **keep identity consistent** by reusing the same visual tag for anyone who reappears ("the woman
  in the yellow raincoat, earlier under the lamp, now..."); (4) **say what the sound does**
  ("the piano continues across the cut", or "the dialogue drops and only wind remains").
- Use **2-4 shots per clip**, each with a clear job (establish, then detail, then reaction). Each
  shot needs roughly three seconds to register, so match the length to the count -- 8 to 10 seconds
  for three shots (`duration_seconds: 10`).
- Keep action chronological ("initially...", "a moment later...") and avoid unexplained changes of
  place or wardrobe between cuts.

Example (3 shots, `duration_seconds: 10`):
> A wide shot establishes a rainy courtyard at dusk, a woman in a yellow raincoat standing under a
> lamp as soft piano music plays. A hard cut transitions to a close-up of the woman in the yellow
> raincoat, raindrops on her face, as she says quietly, "He said he would be here." The piano
> continues across the cut. A match cut connects to a low-angle shot of her boots in a puddle as the
> music drops to a low drone.

#### Other prompt-related fields

- `enhance_prompt` has a Gemma instruct model rewrite your prompt into a richer one before
  generation. Useful for a short or vague prompt; less useful if you've already written a detailed,
  structured one. **The rewritten prompt is not returned in the API response.** It loads a second
  language model on the GPU node, so it adds to the job's load time.
- `seed`: identical seed plus identical every other field gives an identical output. Vary the seed
  (not the prompt) for a different take.

### 7. Choosing parameters

- Use **fast** while iterating on a prompt or seed; switch to **quality** (DFR) for a final take.
- To compare several seeds, submit several separate requests -- each is a fully independent job.

### 8. Errors and recovery

The same status codes and recovery steps as every other backend -- see the main guide's "Errors"
section. One specific to this backend: a retake job that fails with a message about frame count or
resolution means the uploaded source video doesn't satisfy the shape rules in section 5.

### 9. Known limitations

- Only text/image-to-video, first/last-frame interpolation, and retake are wired up; audio-to-video
  and text-to-audio exist only on the LTX-2.3 backend for now. Interpolation takes exactly two
  frames (first and last), not an arbitrary list of keyframes.
- Height/width and retake-source constraints aren't validated at submission time, only discovered
  after a job is already running.
- The prompt actually used after `enhance_prompt` rewriting is never returned.
- Jobs don't report fine-grained `progress`; `status` is the only signal while a job runs.
- Step counts and guidance are fixed pipeline settings, not request fields.
- Every request cold-starts; there is no warm or resident model option.
- Long-clip windowing, temporal (frame-rate) upscaling, 4K, and HDR output exist in the underlying
  model but are not exposed by this API.
