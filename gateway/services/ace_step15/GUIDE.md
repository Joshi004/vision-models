## ACE-Step 1.5 music generation

**What this is for:** full songs with sung vocals, instrumentals and loops from a text description, plus restyling and
partial regeneration of an existing recording. ACE-Step 1.5 XL (SFT diffusion transformer with a 4B language-model
planner) is the choice when you need **many languages (50+), exact tempo/key/length control, speed, or local edits**.
For the best-sounding English/Chinese full song prefer YuE2 (`/v1/yue2`); for instrumentals and sound design prefer
Stable Audio 3 (`/v1/stable-audio3`).

**Three endpoints:**
- `POST /v1/ace-step15/music/generate` -- text-to-music. `caption` (required), `lyrics`, `instrumental`,
  `vocal_language`, `duration` (10-600 s, leave out to let the model choose), `bpm`, `keyscale`, `time_signature`,
  `thinking`, `num_inference_steps` (default 50), `guidance_scale` (default 7), `seed`.
- `POST /v1/ace-step15/music/cover` -- restyle a recording and keep its structure. `source_audio_asset_id` (an
  `asset_id` from `POST /v1/uploads`, up to 600 s), `caption` for the new style, optional new `lyrics`,
  `cover_strength` (1.0 follows the source closely, about 0.2 is a loose style transfer).
- `POST /v1/ace-step15/music/repaint` -- regenerate one window. `source_audio_asset_id`, `repaint_start`,
  `repaint_end` (omit for "to the end"), `caption` describing the new section. The window must start inside the
  recording.

**Writing the inputs:**
- `caption` (max 512 characters): combine genre, mood, instruments, timbre, era, production style and vocal character,
  e.g. `melancholic indie folk, fingerpicked acoustic guitar, soft female vocal, warm intimate room sound`. Be specific
  ("sad piano ballad with breathy female vocal" beats "a sad song") and avoid contradictions; to get two styles, describe
  how the track evolves ("starts with soft strings, turns to driving rock, ends as hip-hop").
- **Tempo, key and time signature go in `bpm`, `keyscale` and `time_signature`, not in the caption.**
- `lyrics` (max 4096 characters): the script. Put structure tags on their own lines: `[Intro]`, `[Verse 1]`,
  `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`, `[Instrumental]`, `[Guitar Solo]`, `[Build]`, `[Drop]`,
  `[Breakdown]`, `[Fade Out]`. Use at most one modifier per tag (`[Chorus - anthemic]`, `[Bridge - whispered]`);
  stacking several can make the model sing the tag. About 6-10 syllables per line; keep matching lines of a verse
  within one or two syllables of each other.
- For no vocals set `instrumental: true` (an empty `lyrics` is then sent as `[Instrumental]`).
- Keep caption and lyrics consistent: instruments named in the caption should match solo tags, mood should match energy
  tags.
- `duration`: when lyrics are given, leave it out. Instrumentals of 30-180 s work well; 10-20 s is the practical minimum.
- `thinking: true` (default) lets the language model fill in missing BPM, key and duration and refine the caption. Turn
  it off when you supply precise metadata. Cover and repaint never use it.
- **Fix `seed` while tuning**, or randomness hides the effect of your changes. `guidance_scale` 5-9 is the useful range.

**Example (`/music/generate`):**
```json
{
  "caption": "melancholic indie folk, fingerpicked acoustic guitar, soft female vocal, warm intimate room sound, light brushed drums",
  "lyrics": "[Verse 1]\nWalking home through the quiet rain\nStreetlights humming my name\n\n[Chorus - soft]\nHold on to the little light\nIt will carry us tonight",
  "bpm": 78,
  "keyscale": "A minor",
  "vocal_language": "en"
}
```

**Output:** `output.flac` (48 kHz stereo, 24-bit FLAC) via `GET /v1/jobs/{job_id}/result`. Deterministic for the same
inputs and `seed` on the same hardware.

**Timing:** requests are run by long-lived GPU jobs, one per GPU, that load the models (about 28 GB) once and then run
requests one after another. A request that has to start one takes minutes; a request that finds one already loaded and
free starts at once. As many run at once as the operator's limit for this model allows (`GET /v1/capacity`); a GPU job with nothing to do for
10 minutes is stopped (`ACE_STEP15_IDLE_MINUTES`). Cancelling a running request stops its GPU job, so the next request
starts cold. No live progress (`progress` is always null).

**License:** the ACE-Step code is MIT and the authors state generated music may be used commercially. Generated music
can resemble existing works: check originality and disclose AI involvement where required.

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
