## YuE2 song generation

**What this is for:** complete songs with sung vocals. YuE2-3B writes a symbolic score (melody and chords) first and then
renders it, which gives the best-structured full songs among the music backends in the authors' own benchmark (English
and Chinese especially). Pick it for **the best full song with lyrics**, or for **a cover that follows a melody score you
supply**. For a faster song with exact tempo/key/length control, many languages, or local edits, use ACE-Step
(`/v1/ace-step15`); for instrumentals and sound design use Stable Audio 3 (`/v1/stable-audio3`).

**Two endpoints:**
- `POST /v1/yue2/songs/generate` -- a new song. `style` (required), `lyrics` (required), `cot` (`full` default,
  `melody`, `off`), `cfg_scale` (leave out), `seed`.
- `POST /v1/yue2/songs/cover` -- a song that follows your score. Same fields plus `abc` (the melody as ABC text);
  `cot` is `melody` (default) or `full`.

**Writing the inputs:**
- `style` (one line): **language first**, then genre, instruments, vocal character and tempo. There is **no BPM
  field**; write the tempo in `style`. Example:
  `English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM`.
- `lyrics`: the words to sing, each section introduced by a tag on its own line (`[Verse]`, `[Chorus]`, `[Bridge]`) and
  separated by a blank line. The song's length follows the lyrics.
- `cot`: `full` plans melody and chords (best quality, default for new songs); `melody` plans the melody only (use for
  covers, it leaves the accompaniment freer); `off` skips the plan (a little faster, clearly lower quality: experts
  preferred planned songs 49% to 35% in the authors' comparison).
- `cfg_scale`: leave it out. The defaults are tuned (1.0 with a plan, 1.01 with `cot: off`); upstream suggests trying 1.2
  for stronger adherence to the text, and warns that changing it may change quality.
- **One request makes one candidate.** The authors' headline quality comes from generating several and picking the best,
  so to match it send several requests with different `seed`s and choose by ear.

**Covers:** `abc` must be ABC text in the model's native format: header lines, two monophonic voices `V: Vocal` and
`V: Ins`, section comments (`% verse`, `% chorus`) between groups of bars. A melody-only score has **no chord symbols**.
`lyrics` must follow the score's sections in order, and when you translate lyrics, match phrasing and syllable counts to
the melody. **This API does not turn a recording into a score**: upstream uses a separate tool (SheetSage2, Python
3.10/3.11 and FFmpeg 6.1) for that. Run it yourself and review the result, because transcription errors carry into the
cover. Editing a score gives a new full recording; it does not keep the old waveform.

**Example (`/songs/generate`):**
```json
{
  "style": "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM",
  "lyrics": "[Verse]\nStreetlights hum along the avenue\nI keep your number in my shoe\n\n[Chorus]\nStay with me until the morning light\nHold the quiet of the night",
  "cot": "full"
}
```

**Output:** `output.flac` (48 kHz stereo, 24-bit FLAC) via `GET /v1/jobs/{job_id}/result`. Deterministic for the same
inputs and `seed` on the same hardware. The model can reach its token limit and still return playable audio (a song that
ends early); the job's log records when that happened.

**Timing:** requests are run by long-lived GPU jobs, one per GPU, that load the model once and then run requests one
after another. A request that has to start one takes minutes; a request that finds one already loaded and free starts
at once. As many run at once as the operator's limit for this model allows (`GET /v1/capacity`); a GPU job with nothing to do for 10 minutes is
stopped (`YUE2_IDLE_MINUTES`). Cancelling a running request stops its GPU job, so the next request starts cold. No live
progress (`progress` is always null).

**License: non-commercial.** The YuE2 weights are **CC BY-NC 4.0** with creator permission required; companies need a
commercial licence. Whoever operates this deployment is responsible for how it is used; this API does not enforce that
restriction itself. Generated music can resemble existing works: check originality and disclose AI involvement where
required.

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
