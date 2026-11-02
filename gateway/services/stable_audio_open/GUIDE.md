## Stable Audio Open 1.0 audio generation

**What this is for:** **short sound effects, foley, drum loops, ambience and textures** (up to 47 seconds) from a text
prompt, plus re-imagining an existing short recording. Trained on Freesound. It does not sing, and it is weaker than
Stable Audio 3 at long musical structure, so for instrumentals and longer cues use `/v1/stable-audio3`; for songs with
vocals use `/v1/ace-step15` or `/v1/yue2`.

**Two endpoints:**
- `POST /v1/stable-audio-open/audio/generate` -- text-to-audio. `prompt` (required), `negative_prompt`, `duration`
  (up to 47 s, default 30), `num_inference_steps` (default 100), `cfg_scale` (default 7), `seed`.
- `POST /v1/stable-audio-open/audio/transform` -- audio-to-audio. Same fields plus `source_audio_asset_id` (an
  `asset_id` from `POST /v1/uploads`, up to 120 s; only the first 47 s are used) and `init_noise_level`.
  `duration` is optional here and defaults to the source's length (capped at 47 s).

**Prompting:** short, concrete, in English. Say what makes the sound, what it is made of, the room, and for loops the
tempo: `128 BPM tech house drum loop`, `The sound of a hammer hitting a wooden surface.` `negative_prompt` can steer
away from unwanted qualities (the model card's example is `Low quality.`). No intelligible vocals.

**Transform:** `init_noise_level` is the sampler's noise level (0.05 to 20, default 0.5), not a 0-1 fraction. Low values
(about 0.1) stay very close to the source; around 1 keeps the rhythm and character but changes the sound; 5 or more
ignores the source almost entirely (measured on a 30 s clip: waveform correlation with the source about 0.94 at 0.1, 0.78
at 1, 0.06 at 5). Try a few with a
fixed `seed`.

**Output:** `output.flac` (44.1 kHz stereo, 24-bit FLAC) via `GET /v1/jobs/{job_id}/result`. Deterministic for the same
inputs and `seed` on the same hardware.

**Timing:** requests are run by long-lived GPU jobs, one per GPU, that load the model once and then run requests one
after another. A request that has to start one takes minutes; a request that finds one already loaded and free starts
at once. As many run at once as the operator's limit for this model allows (`GET /v1/capacity`); a GPU job with nothing to do for 10
minutes is stopped (`STABLE_AUDIO_OPEN_IDLE_MINUTES`). Cancelling a running request stops its GPU job, so the next
request starts cold. No live progress (`progress` is always null).

**License:** the **Stability AI Community License**: free for individuals and organizations below USD 1M annual revenue,
a separate licence above that. Whoever operates this deployment is responsible for how it is used; this API does not
enforce that restriction itself.

**Sending the same request again, and retries.** A request identical to an earlier one -- the same settings,
including the seed, and the same uploaded file *content* (uploading the file again is fine) -- does not run a second
time. The response carries the earlier job's `job_id` with `reused: true` and that job's current `status`, as long as
that job is still queued or running, or succeeded and its result is still available. A failed job is never reused:
sending its request again creates a new job. A request with `incognito: true` is never matched. A reused job is shared:
`DELETE /v1/jobs/{job_id}` cancels it for everyone who was given it. If the GPU job running your request is lost (for
example pre-empted), the request goes back in the queue under the same `job_id` -- `attempts` on
`GET /v1/jobs/{job_id}` shows how many times it was started, at most 3 -- while a job that fails because of its own
request (a model error) is not run again. So a client can fire its requests, and send them again after a timeout or a
crash, without checking first whether they are already queued.
