## Stable Audio 3 audio generation

**What this is for:** **instrumental music, long cues, stems, solo instruments and sound effects**, plus transforming,
repairing and extending an existing recording. Stable Audio 3 Medium (1.4B) generates minutes of audio in seconds. It
**does not produce intelligible vocals**: for songs with sung lyrics use `/v1/ace-step15` or `/v1/yue2`; for short loops
and foley `/v1/stable-audio-open` also works.

**Three endpoints:**
- `POST /v1/stable-audio3/audio/generate` -- text-to-audio. `prompt` (required), `duration` (seconds, default 30, up to
  380), `num_inference_steps` (default 8), `seed`.
- `POST /v1/stable-audio3/audio/transform` -- audio-to-audio. `source_audio_asset_id` (an `asset_id` from
  `POST /v1/uploads`), `prompt`, `init_noise_level` (0-1, default 0.5), optional `duration` (defaults to the source's
  length).
- `POST /v1/stable-audio3/audio/inpaint` -- regenerate part of a recording or extend it. `source_audio_asset_id`,
  `prompt`, `mask_start`, `mask_end` (seconds), optional `duration`.

There is **no `cfg_scale` and no `negative_prompt`**: this is the post-trained checkpoint (8 steps) and they have no effect.

**Prompting:** English, trained on AudioSparx and Freesound with their metadata, so metadata-style prompts work.
- Music: genre, each instrument and how it sounds, mood and energy, BPM, production. Long descriptive prose works well,
  e.g. `A funky hip hop instrumental with live recorded instrumentation that has the vibe of a 70's TV show theme. The
  rhythm guitar strums lightly ... The production has a warm, textured sound associated with analogue gear and tape.`
- Tags that help: `TrackType: Music, VocalType: Instrumental` (higher quality, more coherent), `Genre: Funk, Genre: Jazz`,
  `Instruments: Guitar, Saxophone, Bass, Piano`; `TrackType: Instrument` for one isolated instrument (also `Format: Duo`);
  `TrackType: SFX` for sound effects.
- Sound effects: say what makes the sound, how it is triggered, how long it lasts, and the mic/room/processing. Use a
  short `duration`.
- Pick a `duration` that fits the description. Longer is not automatically better. Send a `seed` for reproducibility;
  unseeded runs would differ every time.

**Transform:** `init_noise_level` is how much of the source is kept: 0 keeps it, 1 ignores it. Starting points: about 0.4
for timbre transfer (violin to metal guitar), about 0.5 for beatbox to lo-fi beat, about 0.6 for a style shift, 0.1 for a
close variation. It needs tinkering per input; keep `seed` fixed while you try values.

**Inpaint:** the audio around the masked section matters more than the prompt. Masking only a few seconds gives output very
close to the original; start with a larger section and shrink it. Use prompts that are plausible in context (a "dubstep
drop" in a house track, not "cat meow"). `mask_start` must be inside the recording. **To extend a clip**, set `mask_start`
to the clip's length (or a little earlier) and `mask_end` to where the new audio should end; `duration` then defaults to
`mask_end`.

**Output:** `output.flac` (44.1 kHz stereo, 24-bit FLAC) via `GET /v1/jobs/{job_id}/result`. Deterministic for the same
inputs and `seed` on the same hardware.

**Timing:** requests are run by long-lived GPU jobs, one per GPU, that load the model once and then run requests one
after another. A request that has to start one takes minutes; a request that finds one already loaded and free starts
at once and generation takes seconds. As many run at once as the operator's limit for this model allows (`GET /v1/capacity`); a GPU job with
nothing to do for 10 minutes is stopped (`STABLE_AUDIO3_IDLE_MINUTES`). Cancelling a running request stops its GPU job,
so the next request starts cold. No live progress (`progress` is always null).

**Limits:** clips of up to 380 seconds (6 min 20 s); a 380 s clip has been generated on this cluster, in under 2 s of GPU
time once loaded.

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
