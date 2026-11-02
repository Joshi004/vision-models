## Changelog

Dates are when a change shipped on this codebase, not necessarily when you're reading this -- the
`api_version` field (`GET /v1/guide?format=json`) is this server's own current version; compare it
against what you last integrated against.

- **0.21.0** (2026-11-01): **GPU capacity limits that the operator changes while the service runs.** A global GPU
  limit and a limit on concurrent jobs per model (for a model served by warm workers, also its most warm workers) are
  now set on a "GPU capacity" page and no longer by environment variables (`MODEL_API_MAX_INFLIGHT` and the
  `*_MAX_WORKERS` settings are gone). New endpoints: `GET /v1/capacity` (the limits, GPUs in use, and per model the
  busy, idle, starting and waiting counts) and `PUT /v1/capacity`. Lowering a limit never cancels a running job:
  busy workers finish, idle ones above the limit are stopped, and nothing new starts until usage is back under it;
  raising a limit starts waiting requests at once. A model served by warm workers keeps requests `queued` until a
  slot is free. LTX-2.3, LTX-2.5, RVC, TBDub and Wan-Animate run one GPU job per request and count 1 GPU each: a
  request that finds their limit or the global limit reached gets **429** (nothing queued), where before they were
  uncapped unless `MODEL_API_MAX_INFLIGHT` was set. `GET /v1/guide?format=json` no longer has
  `limits.max_inflight_jobs`. Otherwise additive: no request or response field changed.
- **0.20.0** (2026-10-31): **Parakeet and Breeze TTS now reuse loaded weights between requests**, like the image
  and music backends. Instead of one GPU job per request, each keeps up to 4 long-lived GPU jobs (operator settings
  `PARAKEET_MAX_WORKERS`, `BREEZE_TTS_MAX_WORKERS`) that run requests one after another and are stopped after 10 idle
  minutes (`PARAKEET_IDLE_MINUTES`, `BREEZE_TTS_IDLE_MINUTES`). A request that finds a loaded GPU job free skips
  loading the model; with more requests than GPU jobs the extra ones wait `queued`. Both backends also get the 0.19.0
  behaviour: at most 1000 queued or running jobs each (429 beyond that), a request whose GPU job is lost queued again
  under the same `job_id` (`attempts`), and an identical request answered with the earlier `job_id` and
  `reused: true`. Requests, results and the transcript JSON are unchanged. Behaviour change: cancelling a request
  that is already running stops its whole GPU job (the next request loads the model again).
- **0.19.0** (2026-10-31): **Safer bursts, automatic retries and repeated requests** for Qwen-Image-2.1, MiniMax H3,
  ACE-Step 1.5, Stable Audio Open, Stable Audio 3, YuE2 and Qwen3.8. (1) Each of these backends accepts at most
  1000 queued or running jobs at a time (operator setting `MODEL_API_MAX_QUEUED_PER_BACKEND`); a request beyond that
  gets 429. (2) A request whose GPU job or model server is lost (pre-empted, crashed, or the gateway restarted) goes
  back in the queue under the same `job_id` instead of failing, up to 3 starts in total (`MODEL_API_MAX_ATTEMPTS`);
  `GET /v1/jobs/{job_id}` has a new field `attempts`. (3) A request identical to an earlier one -- same settings,
  same seed, same uploaded file content -- returns the earlier job's `job_id` with `status` of that job and the
  new response field `reused: true`, instead of running again, while that job is queued, running, or succeeded with
  its result still available; a failed job, or a request with `incognito: true`, is never reused. Qwen3.8 does this
  only for requests that send an `Idempotency-Key` header (the same key with a different request gets 422).
  Cancelling a reused job cancels it for everyone who was given its `job_id`. Additive: no existing field changed
  its meaning, and `reused` is `false` for every other backend.
- **0.18.0** (2026-10-23): **Four music and audio backends, one per model**, each with its own prefix and the same
  submit, poll and download flow as every other backend; the result is always a **FLAC** file. Pipeline values:
  `ace-step15:generate|cover|repaint` (`POST /v1/ace-step15/music/generate|cover|repaint`; songs with vocals, covers and
  local repaint, 48 kHz), `yue2:generate|cover` (`POST /v1/yue2/songs/generate|cover`; whole songs from a style line and
  lyrics, cover from an ABC melody score you supply, 48 kHz, weights are CC BY-NC 4.0), `stable-audio3:generate|transform|inpaint`
  (`POST /v1/stable-audio3/audio/generate|transform|inpaint`; instrumentals, stems and sound effects up to
  380 s, no vocals, 44.1 kHz) and `stable-audio-open:generate|transform` (`POST /v1/stable-audio-open/audio/generate|transform`;
  sound effects and loops up to 47 s, 44.1 kHz). Like Qwen-Image-2.1 and H3, each keeps loaded GPU jobs (operator
  settings `<BACKEND>_MAX_WORKERS`, `<BACKEND>_IDLE_MINUTES`; default 4 and 10 minutes), so a request that finds a loaded
  GPU job free starts at once and the first one after an idle stop waits for the weights to load. Audio inputs are
  uploaded first (`POST /v1/uploads`) and referenced by `source_audio_asset_id`. Each backend's section says how to
  write its prompts; the "Music generation" tag groups the endpoints. Additive: no existing endpoint or field changed.
  **State on release:** every endpoint has produced audio on a real GPU through this API. Once a GPU job is loaded, a
  30 s ACE-Step song takes under 10 s, 37 s of YuE2 audio about 17 s, a Stable Audio 3 clip of any length (up to 380 s)
  a few seconds, and 20 s of Stable Audio Open about 16 s; the first request after an idle stop waits about one to two
  minutes for loading. Output quality has not yet been judged by ear.
- **0.17.0** (2026-10-22): **Qwen-Image-2.1 and MiniMax H3 now reuse loaded weights between requests.** Instead of one
  GPU job per request, each backend keeps up to 8 long-lived GPU jobs (operator settings `QWEN_IMAGE21_MAX_WORKERS`,
  `H3_MAX_WORKERS`) that run requests one after another and are stopped after 10 idle minutes
  (`QWEN_IMAGE21_IDLE_MINUTES`, `H3_IDLE_MINUTES`). A request that finds a loaded GPU job free is much faster than a
  cold start (a tiny H3 request: 441 s cold, 77 s on a loaded one); with more requests than GPU jobs the extra ones wait
  `queued`, and a request `running` is one a GPU job has taken. Behaviour changes: cancelling a request that is
  already running stops its whole GPU job (the next request starts cold), and a request that runs past its time limit
  fails with a message saying so. No endpoint, field or request changed. H3's `estimated_run_seconds` still assumes a
  cold start, so a request on a loaded GPU job finishes sooner than estimated.
- **0.16.1** (2026-10-18): Qwen3.8 (`POST /v1/qwen38/generate`): `max_tokens` now goes up to **49,152** (was
  16,384) and the context window is 65,536 tokens (was 32,768). Long multi-part answers with thinking on (17 scenes of
  frame and video prompts) were cut off at 16,384 with an empty answer because the reasoning used every token; they
  need well over 20,000. A request may now run for up to 30 minutes (was 15). Additive, no other change.
- **0.16.0** (2026-10-18): Added an eighth backend, the **Qwen3.8 text model** (Qwen3.8-27B, FP8; text in, text or
  JSON out, no image/audio/video input): `POST /v1/qwen38/generate` returns a `job_id` like every other endpoint;
  poll `GET /v1/jobs/{job_id}` and download `GET /v1/jobs/{job_id}/result` (JSON with `content`, `json`,
  `reasoning`, `finish_reason`, `usage`, `timings`, `model`). Optional `json_schema` makes the answer always match a
  schema; `thinking` turns reasoning on or off. Pipeline value `qwen38:generate`. Unlike every other backend it keeps
  one shared server on one GPU: answers take seconds when the server is up, a request that arrives while it is
  stopped stays `queued` while it starts (minutes), and it stops itself after 20 idle minutes. New endpoints
  `GET /v1/qwen38/server` (state) and `POST /v1/qwen38/server/warm` (start it ahead of time). Jobs of this backend
  report `queued` until the server picks them up. No change to any existing endpoint; the web UI has a "Text model"
  entry. Declared the `Text model` tag.
- **0.15.0** (2026-10-17): MiniMax H3 (`/v1/h3/videos/generate`, `/v1/h3/videos/audio-to-video`) now sizes every
  request before queueing it and gives each job **4 or 8 GPUs** (it was always two). Additive: the submit response
  (`JobSubmitResponse`), `GET /v1/jobs/{job_id}` and `GET /v1/jobs` gain `gpus` (how many GPUs the job was given) and
  `estimated_run_seconds` (about how long it will run once started, queue wait not included; prefer it over
  `typical_run_seconds` when present); both are `null` for every other backend and for jobs from before this
  version. One behaviour change: a valid H3 request that is too big for the GPUs -- chiefly many or large reference
  images, or reference videos on a long clip -- is now refused with **HTTP 422** and a sentence with the numbers
  (nothing is queued), where before it was queued and failed with an out-of-memory error after loading the weights.
  The size is judged from the prompt, the length and canvas, and the headers of the uploaded files. The limits and
  the estimate are initial guesses and will be refitted; the 4/8-GPU layout has not been run to completion yet.
  The web UI shows the GPU count and the estimate on each H3 job and after submitting.
- **0.14.0** (2026-10-17): **Breaking change** on `POST /v1/h3/videos/audio-to-video` (MiniMax H3's reference mode),
  which now takes the model's full reference set: up to 9 images (`image_asset_ids`), 3 videos (`video_asset_ids`) and
  3 audio clips, 12 in all. (1) `audio_asset_id` (one audio, required) is **replaced by `audio_asset_ids`**, a list of
  up to 3 and optional, so the mode can run from images and/or videos alone (for example several characters from
  images, brought in by the prompt at the moments it describes). A request that still sends `audio_asset_id` is
  rejected with 422, not quietly ignored; to upgrade, send `"audio_asset_ids": ["<the same asset_id>"]`. (2) New
  optional `video_audio`, one true/false per entry of `video_asset_ids`: a reference video that has sound gives its
  soundtrack to the model by default (as before), and `false` uses that video for its picture only. (3) `<Audio N>`
  in the prompt counts every audio: a video with a soundtrack takes the next one just before its `<Video N>`, then
  the `audio_asset_ids` follow (`<Picture N>` and `<Video N>` still count their own kind). The earlier docs said the
  one audio was always `<Audio 1>`, which is only true when no video carries sound. (4) At least one image or video
  is still required; `match_audio_duration: true` needs at least one audio and sizes the clip from the first. The
  per-clip 2 to 15 s and 15 s totals are MiniMax's documented limits and are not enforced here. The web UI form has
  matching lists and a per-video soundtrack switch, and Retry on an older job fills the audio list from the old field.
  The 12-reference maximum and several audio clips have not been run on a GPU yet.
- **0.13.1** (2026-10-16): MiniMax H3 (`/v1/h3/videos/generate`, `/v1/h3/videos/audio-to-video`) no longer limits the
  video to 5 to 15 s. `duration_seconds` (now above 0 up to 60), `num_frames` (now 1 to 1440) and
  `match_audio_duration` accept any length and only round it up to the next 17*n+5 frames, so a 1 s request gives 39
  frames and `match_audio_duration` never cuts the audio. The 124 to 345 frame (5.17 to 14.4 s) range is what the model
  is documented for: **outside it is experimental** -- quality is unknown, very short clips may fail, and long clips may
  run out of GPU memory. With `match_audio_duration`, audio longer than 60 s fails the job (operator setting
  `H3_MAX_SECONDS`). One behaviour change inside the old range: a 15 s request now gives 362 frames (15.08 s) instead of
  being clamped to 345.
- **0.13.0** (2026-10-14): Added a ninth backend, MiniMax H3 (open weights, bf16): `POST /v1/h3/videos/generate`
  (text, or a first and/or last frame, to video with generated stereo sound; `pipeline` `h3:generate`) and
  `POST /v1/h3/videos/audio-to-video` (an audio reference plus image and/or video references to video with
  generated sound; `pipeline` `h3:audio-to-video`). Unlike every other backend a job holds **two** GPUs on one node.
  The prompt is used exactly as sent (no rewriting) and H3 expects a structured prompt format -- see the MiniMax H3
  section. Unlike the LTX audio-to-video endpoints, the uploaded audio is a *reference*, not audio muxed into the
  result unchanged. 768p, fixed 24 fps, 5.17 to 14.4 s. While the video is generated the job's second GPU also runs
  half of the transformer, so the longest clips at the full 1344x768 canvas have the memory they need (a full-size
  14.4 s clip had run out of memory on one GPU; the fix is not yet confirmed by a completed run). No change to any
  existing endpoint.
- **0.12.0** (2026-10-12): Added an eighth backend, TBDub lip sync, in two forms. (1) `POST /v1/ltx/videos/audio-to-video`
  and `POST /v1/ltx25/videos/audio-to-video` accept a new optional boolean `lip_sync` (default `false`): after the
  video is generated, a dedicated lip-sync model redraws the mouth to match the audio exactly, in the same job.
  It forces `frame_rate` to 25 (an explicit `num_frames` is converted to the same duration) and adds several
  minutes. If lip sync fails the job still reports `succeeded` and returns the video **without** lip sync.
  (2) `POST /v1/tbdub/videos/lip-sync` lip-syncs an uploaded video and audio, or re-syncs a succeeded LTX
  audio-to-video job (`source_job_id`) without regenerating it; it is always its own job (`pipeline`
  `tbdub:lip-sync`). New field on every job status: `lip_sync` (`requested`, `applied`, `error`), `null` for jobs that
  did not ask for lip sync -- **check `lip_sync.applied` on a succeeded job that did**. Declared the `Lip sync` tag.
  No existing field or endpoint changed behavior unless `lip_sync: true` is sent.
- **0.11.0** (2026-10-11): **Behavior change: job data is now kept by default, and opting out is per
  request.** Every generation request on every backend accepts a new optional boolean `incognito`
  (default `false`). A normal job keeps everything -- prompt/settings, uploaded inputs, logs and output --
  until it is purged (`DELETE /v1/jobs/{job_id}/purge`) or `job_retention_days` pass, and its result can
  be downloaded any number of times. A job submitted with `incognito: true` gets the no-storage treatment
  that 0.9.0 applied to every job: no logs, inputs deleted as soon as the job ends, a result that can be
  downloaded **once** (then `410`, with `downloaded_at` set), and only a minimal record left in the jobs
  list (`request` is `{}`). Jobs report `incognito` in `GET /v1/jobs` and `GET /v1/jobs/{job_id}`. The
  server-wide `MODEL_API_PRESERVE_JOB_DATA` setting is gone, and `GET /v1/health` no longer reports
  `preserve_job_data`. An uploaded file that no job references is still deleted after 60 minutes.
- **0.10.0** (2026-10-11): Added `POST /v1/ltx25/videos/audio-to-video` to the LTX-2.5 backend --
  generates video to match an uploaded audio track (`audio_asset_id`, `prompt`, optional `images` and
  `negative_prompt`), which is muxed into the output unchanged. One guided recipe, no `mode`. Clip
  length is either fixed (`duration_seconds`/`num_frames`) or matched to the audio with
  `audio_max_duration` (up to 20s); the two can't be combined. Jobs it creates report their
  `pipeline` as `ltx25:audio-to-video`. No change to any existing endpoint.
- **0.9.0** (2026-10-10): **Behavior change: by default this server now keeps no job data.** With
  `MODEL_API_PRESERVE_JOB_DATA` false (the default; `GET /v1/health` reports it as `preserve_job_data`):
  no logs are stored; an uploaded input is deleted as soon as its job ends (or 60 minutes after upload
  if no job used it), so each request needs a fresh upload; and a result can be downloaded **once**
  -- `GET /v1/jobs/{job_id}/result` deletes the file right after sending it in full, then returns 410
  and the job's new `downloaded_at` field is set. An interrupted download keeps the file, and a
  request with a `Range` header does not count. A job whose data was deleted stays in the jobs list as
  a minimal record (`request` is `{}`; a failed job keeps a one-line `error`). Set
  `MODEL_API_PRESERVE_JOB_DATA=true` for the previous behavior (everything kept for
  `job_retention_days`). Added `preserve_job_data` to `GET /v1/health` and `downloaded_at` to job
  status; no endpoint was removed.
- **0.8.0** (2026-10-10): Added a seventh backend, Qwen-Image-2.1 image generation:
  `POST /v1/qwen-image21/images/generate` (text-to-image) and `POST /v1/qwen-image21/images/edit`
  (editing and composition from 1-10 reference images). `mode` is `fast` (Turbo, 8 steps) or `quality`
  (base, 40 steps by default); `aspect_ratio` picks one of seven native 2K sizes; transparent (RGBA) output
  is requested through prompt wording. Returns a PNG. Jobs it creates report their `pipeline` as
  `qwen-image21:generate` or `qwen-image21:edit`. Declared the `Image generation` tag. Model weights are
  under the Qwen Research License (non-commercial use only). No change to any existing endpoint.
- **0.7.0** (2026-10-10): Added a sixth backend, Breeze TTS 2 text-to-speech
  (`POST /v1/breeze-tts/synthesize`). One endpoint covers Voice Design (`text` + `instruction`),
  Voice Clone (`text` + `reference_audio_asset_id` + `reference_text`) and Voice Direction (clone
  plus `instruction`); returns a mono 24 kHz WAV. Jobs it creates report their `pipeline` as
  `breeze-tts:synthesize`. Declared the `Text to speech` tag. Model weights are for research and
  non-commercial use only. No change to any existing endpoint.
- **0.6.0** (2026-10-09): Added `POST /v1/ltx25/videos/interpolate` to the LTX-2.5 backend --
  generates the motion between a given first frame and a given last frame (`first_frame_asset_id`,
  `last_frame_asset_id`, `prompt`), in `fast` or `quality` mode like text-to-video. Jobs it creates
  report their `pipeline` as `ltx25:interpolate`. No change to any existing endpoint.
- **0.5.0** (2026-10-09): Added a fifth backend, LTX-2.5, alongside the unchanged LTX-2.3 backend:
  `POST /v1/ltx25/videos/generate` (text/image-to-video, `fast` or `quality`, with optional
  `auto_duration`) and `POST /v1/ltx25/videos/retake`. LTX-2.5 can cut between several shots inside
  one clip -- written into the prompt, there is no separate field. Jobs it creates report their
  `pipeline` as `ltx25:text-to-video` or `ltx25:retake`.
- **0.4.0** (2026-10-08): Removed authentication entirely. No endpoint requires an `Authorization`
  header any more, so no endpoint returns `401`, and `GET /v1/guide?format=json` no longer has a
  `facts.auth` object. Clients that still send an `Authorization` header are unaffected -- it is
  simply ignored.
- **0.3.0** (2026-10-05): Added a fourth backend, Parakeet speech transcription
  (`POST /v1/parakeet/transcribe`) -- transcribes an existing recording (or a video's audio track)
  to text, with word- and segment-level timestamps; replaces that model's previous dedicated
  always-on service, which held a GPU permanently regardless of whether a request was in flight.
  Declared the `Transcription` tag. Documented `application/json` as a content type
  `GET /v1/jobs/{job_id}/result` can return (the transcript itself, for a transcription job).
- **0.2.0** (2026-10-04): Added this guide (`GET /v1/guide`). Documented the `401` response on every
  endpoint that can return it (previously only shown when a deployment happened to require a token
  at the time its docs were generated). Documented every content type `GET /v1/jobs/{job_id}/result`
  can actually return (previously only `video/mp4`/`audio/wav` were listed; voice-conversion and
  batch jobs also return `audio/mpeg`, `audio/flac`, `audio/ogg`, and `application/zip`). Declared
  the `Voice conversion` tag (the RVC endpoints already used it, but it was missing from this API's
  own tag list). Reworded endpoint descriptions throughout for an external reader -- removed
  references to this deployment's own internal file paths, scripts, and command names that a
  third-party caller has no access to and doesn't need.
- **0.1.0**: Initial tagged version (LTX-2.3, Wan-Animate v1, RVC voice conversion, and the shared
  job-tracking/upload infrastructure all three sit on).