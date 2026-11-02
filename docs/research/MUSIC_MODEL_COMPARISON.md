# Music generation models: comparison, use cases and prompting

_Researched 2026-10-22/23 from each model's own README, model card and docs (links at the end). Quality numbers are the
authors' own benchmark runs, not ours; nothing here has been listened to on this cluster yet (see "Not yet verified")._

This file is the basis for the `GUIDE.md` of the four music backends. Four separate backends, one per model:

| Backend | Model | Path prefix | Pipelines |
|---|---|---|---|
| `ace-step15` | ACE-Step 1.5 XL (DiT `acestep-v15-xl-sft` + LM `acestep-5Hz-lm-4B`) | `/v1/ace-step15` | `ace-step15:generate`, `:cover`, `:repaint` |
| `stable-audio-open` | Stable Audio Open 1.0 | `/v1/stable-audio-open` | `stable-audio-open:generate`, `:transform` |
| `stable-audio3` | Stable Audio 3 Medium | `/v1/stable-audio3` | `stable-audio3:generate`, `:transform`, `:inpaint` |
| `yue2` | YuE2-3B (+ YuE2-Vae) | `/v1/yue2` | `yue2:generate`, `:cover` |

## 1. At a glance

| | YuE2-3B | ACE-Step 1.5 (XL) | Stable Audio 3 Medium | Stable Audio Open 1.0 |
|---|---|---|---|---|
| Released | Sep/Oct 2026 | Feb 2026 (XL: 2026-04-02) | May 2026 | Jun 2024 |
| Makes | Full songs with vocals, symbolic score first | Full songs with vocals, instrumentals, loops | Instrumentals, stems, solo instruments, SFX | Short SFX, foley, loops, ambience |
| Vocals | Yes (lyrics) | Yes (lyrics, 50+ languages) | **No** intelligible vocals | No |
| Length | Whole song (about 3-5 min in the demos) | 10 s to 10 min | Up to ~4.75 min (README table) or 6:20 (blog); we measure | **47 s max** |
| Sample rate | 48 kHz stereo | 48 kHz stereo | 44.1 kHz stereo | 44.1 kHz stereo |
| Audio input | Cover from an ABC melody score | Reference audio, cover, repaint | Audio-to-audio, inpaint, continuation | Audio-to-audio (init audio) |
| Speed (authors) | 71 s for 3.6 min on RTX 4090 | under 2 s per song on A100 (turbo); XL sft is 50 steps, slower | under 1 s on H200 | slow: 100-200 steps with CFG |
| Weights licence | **CC BY-NC 4.0** + creator permission; companies need a commercial licence | MIT code; authors state generated music may be used commercially | Stability AI Community Licence (free below USD 1M yearly revenue) | Stability AI Community Licence |
| Gated on HF | No | No | Yes | Yes |

### Benchmark numbers (WildSongBench, 192 prompts, from the YuE2 authors, Sep 2026)

Authors' own benchmark and authors' own model, so read as indicative only. Open models:

| Model | Musicality | SongBench Avg | MuLan (text match) | Phoneme error rate (lower is better) |
|---|---|---|---|---|
| YuE2 | 5.91 | 6.73 | 0.507 | 8.4% |
| ACE-Step 1.5 | 5.16 | 6.01 | 0.437 | 7.5% |
| YuE 1 | 4.08 | 4.92 | 0.262 | 36.4% |

Reference points from the same table: Suno v5 6.87, Suno v4.5 6.70. The ACE-Step authors place their model "between
Suno v4.5 and Suno v5"; the YuE2 table ranks it lower. On lyric clarity (phoneme error rate) ACE-Step is slightly better
than YuE2 in this table. Stable Audio models are not in this benchmark because they do not sing.

## 2. Which one to use

| You want | Use | Why |
|---|---|---|
| The best full song with sung lyrics | **YuE2** | Highest score in the authors' benchmark; plans a score first |
| A cover of a known melody in a new style | **YuE2** `/songs/cover` | Score-conditioned; needs an ABC melody |
| A song fast, with exact tempo/key/length, or in a non-English/Chinese language | **ACE-Step** | 50+ languages, `bpm`, `keyscale`, `duration` fields, much faster |
| Change one section of a song (new solo, new lyrics line) | **ACE-Step** `/music/repaint` | Local regeneration of a time window |
| Restyle an existing recording (keep structure) | **ACE-Step** `/music/cover` | Cover with adjustable strength |
| Commercial use of outputs without a negotiated licence | **ACE-Step** or **Stable Audio 3** | YuE2 weights are non-commercial for companies |
| Instrumental background music, long cues, stems, solo instruments | **Stable Audio 3** | Trained for it; long clips |
| Extend a clip, or regenerate part of it | **Stable Audio 3** `/audio/inpaint` | Inpaint and continuation |
| Timbre or style transfer of an instrumental | **Stable Audio 3** `/audio/transform` | `init_noise_level` control |
| A short sound effect, foley hit, drum loop, ambience | **Stable Audio Open 1.0** (or SA3, which also covers SFX) | Trained on Freesound; 47 s limit |

Rule of thumb for clients: **vocals -> YuE2 or ACE-Step; no vocals -> Stable Audio 3; short SFX -> Stable Audio Open.**
Stable Audio 3 Medium also covers sound effects, so Open 1.0 is mainly here for its different (older) character and
for its short-clip use.

## 3. Prompting, per model

### 3.1 ACE-Step 1.5

Two text inputs plus metadata fields (ACE-Step docs, Tutorial and Inference sections):

- **`caption`** (max 512 chars): the overall portrait. Combine several dimensions: genre, emotion, instruments,
  timbre texture, era, production style, vocal character. "sad piano ballad with female breathy vocal" beats "a sad song".
  Avoid contradictions ("fast slow"); if you want two styles, express them as evolution over time ("starts with soft
  strings, middle turns to driving rock, ends as hip-hop").
- **Do not write tempo, key or time signature in the caption.** Put them in `bpm`, `keyscale`, `timesignature`.
- **`lyrics`** (max 4096 chars): the temporal script. Structure tags: `[Intro]`, `[Verse]`/`[Verse 1]`, `[Pre-Chorus]`,
  `[Chorus]`, `[Bridge]`, `[Outro]`, `[Instrumental]`, `[Guitar Solo]`, `[Build]`, `[Drop]`, `[Breakdown]`, `[Fade Out]`.
  Combine **at most one modifier**: `[Chorus - anthemic]`, `[Bridge - whispered]`. Stacking many modifiers risks the
  model singing them.
- Use `[Instrumental]` as the whole lyrics (or `instrumental: true`) for no vocals.
- Keep caption and lyrics consistent (instruments named in the caption match solo tags; emotion matches energy tags).
- 6-10 syllables per line works best; keep lines at the same position in a verse within +-1-2 syllables.
- `duration`: leave on auto when lyrics are given. Instrumentals 30-180 s work well; 10-20 s is the practical minimum.
- `thinking: true` (default) lets the LM planner fill in missing BPM/key/duration and refine the caption. Turn it off
  when you supply precise metadata. Cover and repaint skip the LM regardless.
- `inference_steps` 50 and `guidance_scale` 7 are the defaults for the sft checkpoint we run (range 5-9 is typical).
- **Fix `seed` while tuning**: otherwise parameter changes are hidden by randomness.
- Cover: `audio_cover_strength` near 1.0 follows the source closely; around 0.2 is a loose style transfer.
- Repaint: describe the new content of `repaint_start`..`repaint_end` in `caption`.

Example:

```json
{
  "caption": "melancholic indie folk, acoustic guitar fingerpicking, soft female vocal, warm intimate room sound, light brushed drums",
  "lyrics": "[Verse 1]\nWalking home through the quiet rain\nStreetlights humming my name\n\n[Chorus - soft]\nHold on to the little light\nIt will carry us tonight",
  "bpm": 78,
  "keyscale": "A minor",
  "vocal_language": "en"
}
```

### 3.2 YuE2

Inputs: `style` (one line), `lyrics` (sectioned), `cot` mode, optional `abc` score (YuE2 generation guide, README).

- **`style`**: language first, then genre, instruments, vocal character, tempo. Upstream example:
  `"English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable
  melody, unhurried phrasing, 88 BPM"`. Tempo and everything else go in this one string (YuE2 has no separate BPM field).
- **`lyrics`**: section tags `[Verse]`, `[Chorus]` (as in the upstream example), the words to sing underneath, a blank
  line between sections.
- **`cot`**: `full` (melody + chords plan, default for new songs), `melody` (melody-only plan, recommended for covers),
  `off` (no plan, slightly faster, lower quality in the authors' comparison: 34.6% vs 49.3% expert preference).
- **`cfg_scale`**: defaults are tuned (1.0 for full/melody, 1.01 for off); upstream suggests trying 1.2 for stronger text
  guidance. Changing it may change quality.
- **One request makes one candidate.** The published best-of-8 number comes from generating 8 and picking; clients that
  want that quality should submit several seeds and choose by ear.
- **Cover**: needs `abc`, a melody score with `Vocal` and `Ins` voices and **no chord symbols**, plus lyrics whose section
  order matches the score, plus the target `style`. Use `cot: melody`. To get the ABC from a recording the upstream
  route is SheetSage2 (`models/SheetSage2/infer.py source.wav --melody-only`) in a separate environment (Python
  3.10/3.11, FFmpeg 6.1). **Not hosted by this API.** Transcription errors carry into the cover, so review the score.
  When translating lyrics, match phrasing and syllable counts to the melody.
- Editing: `plan -> edit score -> render` generates a new full recording; it does not preserve the old waveform. Not
  exposed here (no plan endpoint in v1).
- Truncation: the model can hit a token limit and still produce playable audio; the pipeline reports `truncated`. We
  record that in the job's output sidecar (see the GUIDE).

### 3.3 Stable Audio 3 Medium

(Stable Audio 3 prompting guide.) Trained on AudioSparx and Freesound with their metadata, so metadata-like prompts work.

- **Music**: genre, instruments (and how each sounds), mood and energy, BPM. Long descriptive prose works:
  "A funky hip hop instrumental with live recorded instrumentation that has the vibe of a 70's TV show theme. The
  rhythm guitar strums lightly ... The production has a warm, textured sound associated with analogue gear and tape."
- **Tags that help**: `TrackType: Music, VocalType: Instrumental` (higher quality, more coherent), `Genre: Funk, Genre: Jazz`,
  `Instruments: Guitar, Saxophone, Bass, Piano`; `TrackType: Instrument` for a single isolated instrument (also
  `Format: Duo`); `TrackType: SFX` for sound effects.
- **No intelligible vocals** (at most unintelligible vocal textures). Not for speech. English prompts only.
- **SFX**: say what makes the sound, how it is triggered and how long it lasts, mic placement/room/processing. Use a
  **short duration**.
- **Duration**: pick one that fits the description; very long clips are not automatically better.
- Post-trained checkpoint: 8 steps; `cfg_scale` and `negative_prompt` have **no effect** (they matter only on the base
  checkpoints), so the API does not offer them.
- **Audio-to-audio** (`init_noise_level` 0..1, 1.0 = ignore the input): about 0.4 for timbre transfer (violin to metal
  guitar), about 0.5 for beatbox to lo-fi beat, about 0.6 for a style shift. Lower keeps more of the original (0.1 is a
  close variation). Needs tinkering per input.
- **Inpainting / continuation**: the surrounding audio matters more than the prompt. Masking only a few seconds gives
  output very close to the original; start with a large masked region and shrink. Prompts that are plausible for the
  surrounding context work ("dubstep drop" in a house track, not "cat meow"). To extend, set the mask start to the
  clip's length and `duration` longer.
- Seeds: unseeded runs differ every time; send a `seed` for reproducibility.

### 3.4 Stable Audio Open 1.0

(Model card.) Short, concrete descriptions of a sound or loop: `"128 BPM tech house drum loop"`, `"The sound of a hammer
hitting a wooden surface."` Up to 47 s, English. Supports `negative_prompt` (card example: `"Low quality."`). The card
example uses 200 steps; the `stable-audio-tools` example uses 100 steps with `cfg_scale` 7 and the `dpmpp-3m-sde`
sampler. Output is stereo 44.1 kHz. Good for SFX, foley, drum loops and textures; weaker at musical structure than
Stable Audio 3. Audio-to-audio via init audio and `init_noise_level` is part of `stable-audio-tools`. Unlike Stable Audio 3, its
`init_noise_level` is **not** a 0..1 fraction: it is in the sampler's own noise units and the API accepts 0.05-20
(default 0.5; around 0.1 stays close to the source, larger values follow the prompt more). That range follows
`stable-audio-tools`' own code and examples and has not been listened to on this cluster.

## 4. Operational notes for clients

- All four are **warm-worker backends**: the first request starts a GPU job and loads weights (minutes); later requests
  while a worker is loaded start at once; a worker idle for 10 minutes is stopped. Each backend has its own workers.
- Result is a **FLAC** (`audio/flac`). Same `job_id` / poll / download flow as every other backend.
- Licences differ and the API does not enforce them: YuE2 weights are non-commercial for companies; Stability models
  need the Community Licence terms (free below USD 1M annual revenue); ACE-Step is MIT.
- Generated music can resemble existing works (ACE-Step's own disclaimer): clients should check originality and disclose
  AI involvement where required.

## 5. Not yet verified on this cluster

- Real quality of each model with our settings (nobody has listened yet), load times, generation times, peak VRAM.
- Whether the torch versions each upstream expects run on this cluster's driver (CUDA 12.8 max).
- (Settled 2026-10-23: Stable Audio 3 Medium generated a 380 s clip on this cluster; see its README.)
- The `transform` quality of Stable Audio Open 1.0 via `stable-audio-tools`.

## Sources

- ACE-Step 1.5: https://github.com/ace-step/ACE-Step-1.5 (README, `docs/en/INFERENCE.md`, `docs/en/Tutorial.md`), https://huggingface.co/ACE-Step/Ace-Step1.5
- YuE2: https://huggingface.co/m-a-p/YuE2-3B, https://github.com/multimodal-art-projection/YuE (README, `docs/generation.md`, `docs/covers.md`), paper arXiv:2609.33757
- Stable Audio 3: https://github.com/Stability-AI/stable-audio-3 (README, `docs/guides/prompting.md`, `docs/workflows/inference.md`), https://stability.ai/news-updates/meet-stable-audio-3-the-model-family-built-for-artistic-experimentation-with-open-weight-models
- Stable Audio Open 1.0: https://huggingface.co/stabilityai/stable-audio-open-1.0
