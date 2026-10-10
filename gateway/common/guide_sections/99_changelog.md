## Changelog

Dates are when a change shipped on this codebase, not necessarily when you're reading this -- the
`api_version` field (`GET /v1/guide?format=json`) is this server's own current version; compare it
against what you last integrated against.

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