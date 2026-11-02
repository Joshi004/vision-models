## Breeze TTS 2 text-to-speech

**What this is for:** generating speech from text (English and Chinese) with BreezeBlue's Breeze TTS 2.
This is real **text-to-speech** -- unlike RVC (which converts an existing recording) and Parakeet (which
transcribes one).

**Three modes, one endpoint** (`POST /v1/breeze-tts/synthesize`), chosen by which fields you send:
- **Voice Design** -- `text` (+ `instruction`): invents a voice from a natural-language description. No audio input.
- **Voice Clone** -- `text` + `reference_audio_asset_id` + `reference_text`: speaks `text` in the reference
  speaker's voice. `reference_text` must be the exact transcript of the reference recording; the two fields
  must be sent together.
- **Voice Direction** -- Voice Clone plus `instruction`: keeps the reference speaker's identity while steering
  tone, emotion and pace.

**Inputs:** a reference recording is uploaded first via `POST /v1/uploads` and referenced by `asset_id`. Common
formats are accepted, including phone voice notes (`.m4a`). Use a clean recording with minimal background noise.

**Tips:** write `instruction` in the same language as `text`; use `cfg_scale: 4` whenever you set `instruction`.
Inline vocal events: `(laugh)`, `(sigh)`, `(clears throat)` in English; `[笑]`, `[叹气]` in Chinese.

**Output:** a mono 24 kHz 16-bit WAV via `GET /v1/jobs/{job_id}/result`. Deterministic for the same inputs and `seed`.

**Limits:** text is limited per request (see the endpoint's description for the exact numbers); one request
generates about two minutes of audio at most, so split long scripts across several requests. There is no live
progress and no streaming.

**Timing:** up to 4 long-lived GPU jobs load the model once and run requests one after another; one with no request
for 10 minutes is stopped. A request that finds one free starts at once without loading the model; the first request
after an idle stop waits for a GPU job to start and load the model (about a minute), and with more requests than GPU
jobs the extra ones stay `queued` until one is free.

**Sending the same request again, and retries.** A request identical to an earlier one -- the same text, settings and seed, and the same reference recording if any, and the same uploaded file *content* (uploading the file again is fine) -- does not run a second
time. The response carries the earlier job's `job_id` with `reused: true` and that job's current `status`, as long as
that job is still queued or running, or succeeded and its result is still available. A failed job is never reused:
sending its request again creates a new job. A request with `incognito: true` is never matched. A reused job is shared:
`DELETE /v1/jobs/{job_id}` cancels it for everyone who was given it. If the GPU job running your request is lost (for
example pre-empted), the request goes back in the queue under the same `job_id` -- `attempts` on
`GET /v1/jobs/{job_id}` shows how many times it was started, at most 3 -- while a job that fails because of its own
request (a model error) is not run again. So a client can fire its requests, and send them again after a timeout or a
crash, without checking first whether they are already queued.


**License and responsible use:** the Breeze TTS 2 weights and self-hosted outputs are licensed for **research and
non-commercial use only**. Only clone voices you have the rights and consent to use. Whoever operates this
deployment is responsible for how it is used; this API does not enforce those restrictions itself.
