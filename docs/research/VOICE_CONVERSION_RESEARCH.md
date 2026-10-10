# Keep the Performance, Change the Voice: Open-Weight Voice-Conversion Model Options

_Research document for a specific capability: taking a recording as it was actually performed —
its intonation, timing, pauses, emotion — and changing only **who it sounds like**, i.e.
**voice conversion (VC)**, not text-to-speech voice cloning. Not a decision record — nothing
here has been built or run on this cluster yet. Sources are a mix of official repos/HuggingFace/
arXiv (high confidence) and third-party blogs/Reddit threads (lower confidence, flagged where
used). This field moves in weeks, not years — treat exact numbers below as "true as of late
September 2026," not permanent facts._

_**Updated same day** after clarifying real requirements: live/real-time is not needed, GPU and
speed are not constraints, **quality is the only thing that matters**, and **Hindi needs to work
well**. That last point eliminates two of the three models this document originally led with —
see §1, which now supersedes the original priority order everywhere it disagrees with what
follows._

---

## TL;DR

- **Two of the three original top picks (CosyVoice, Amphion Vevo2) do not cover Hindi at
  all** — confirmed directly against their own published training-language lists. Neither
  should be picked for this now. See §1.
- **Given quality-first + Hindi + no real-time requirement, three options actually fit:**
  - **RVC / Applio**, using the **enormous existing library of community-trained Hindi/Bollywood
    voice models** (Neha Kakkar, Jubin Nautiyal, Shreya Ghoshal, Sonu Nigam, Atif Aslam, and
    hundreds more are already trained and downloadable). RVC's biggest previous downside —
    needing a trained model per voice — stops being a downside once speed/effort aren't
    constraints.
  - **`DreamSyncCo/IndicVoiceChanger`** — Seed-VC's architecture (independently shown to have the
    best speaker-similarity score among compared zero-shot VC models) fine-tuned specifically on
    Indic-language data, with real Hindi examples in its own demo. Zero-shot out of the box, and
    also fine-tunable per-voice in ~2 minutes if you want to push quality further for one
    specific target voice.
  - **EZ-VC** (IIT Madras SPRINGLab) — the one *peer-reviewed, academic* option here trained
    explicitly on Hindi (plus 4 other Indian languages and Indian-accented English, 12,840 hours
    total). CC-BY-NC licensed (non-commercial).
- **"Voice cloning" is still ambiguous terminology** — what you want is voice conversion (VC),
  not TTS voice cloning (Chatterbox, IndexTTS2, F5-TTS). This distinction is explained once, in
  §2, and still holds.
- **Since speed/GPU no longer matter, always pick each model's slowest/highest-quality
  configuration** — more diffusion steps, the larger "offline" checkpoint instead of the "tiny
  real-time" one, 48kHz instead of 32kHz, etc. Concrete guidance per model in §1.
- **One caveat carries over and matters more now**: Seed-VC's own maintainer has acknowledged,
  in a real GitHub issue, that it struggles to preserve *strong source emotion* specifically —
  and `IndicVoiceChanger` is built on that same architecture. Worth testing directly, since
  preserving intonation/emotion was your original core ask.
- **Real-time no longer being a requirement actually simplifies the integration story** — see §7.
  Every model recommended here fits this API's existing "submit a job, get a file back" pattern
  with no new infrastructure shape needed.

---

## 1. Your refined constraints, and how they reshape the shortlist

You said: live/real-time conversion isn't needed, GPU isn't a constraint, speed isn't a
constraint, **quality is the only real concern**, and it needs to **work well in Hindi**. That
last point is a hard filter, and checking it directly against each model's actual training data
changes the picture a lot from the original comparison below.

### The disqualification, checked directly against source

| Model (previous rank) | Languages it was actually trained on | Hindi? |
|---|---|---|
| CosyVoice 2/3 *(was top pick #1)* | Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian (+ Chinese dialects) — per its own HuggingFace model card | ❌ Not listed anywhere in its official language coverage |
| Amphion Vevo/Vevo2 *(was top pick #2)* | English, Chinese, German, French, Japanese, Korean — the Emilia dataset it's trained on, confirmed via Amphion's own repo and the dataset card | ❌ Not one of the six languages in its training data |

Both are genuinely excellent for the languages they cover, and their underlying disentanglement
approach is still architecturally the "right idea" for preserving prosody — but neither has ever
seen Hindi during training, and nothing in either project's documentation claims otherwise. For
your stated requirement, they're out.

### What actually covers Hindi, and how well each fits "quality first"

**1. RVC / Applio + an existing Hindi/Bollywood community model**
This project's earlier downside — you need a trained (or downloaded) model per target voice,
not a short reference clip — stops being a real downside once training time and GPU cost don't
matter. And the community coverage for Hindi specifically is enormous and already built: a
direct check of one public model-sharing HuggingFace repo alone turned up ready-to-use RVC
models for **Neha Kakkar, Jubin Nautiyal, Shreya Ghoshal, Sonu Nigam, Atif Aslam, Armaan Malik,
Rahat Fateh Ali Khan, Mohd Rafi, B Praak, Darshan Raval, Sanam Puri, Palak Muchhal, Yo Yo Honey
Singh**, and dozens more — this is a real, currently-active hobbyist ecosystem (AI song covers),
not a hypothetical. RVC is also the option with the single strongest quality reputation of
everything in this document (one Reddit user's blind test: friends couldn't tell trained RVC
output from the real recording). **For quality-first, use the v2 pipeline at 48kHz with a proper
FAISS index (`index_rate` 0.5–0.8), not the fast/low-quality shortcuts** — none of the
speed-vs-quality tradeoffs RVC normally has to make apply to you.

**2. `DreamSyncCo/IndicVoiceChanger`**
A fine-tune of **Seed-VC** — the same architecture that, in an independent third-party benchmark
(SynthVC, arXiv 2510.09245, not affiliated with Seed-VC's own authors), scored the highest
speaker-similarity MOS (4.34) of every zero-shot VC model compared, specifically *because* it's a
diffusion-based model that isn't optimized for real-time — a tradeoff that is exactly in your
favor now. This specific fine-tune re-trained that architecture on Indic-language data, and its
own public demo includes a literal Hindi→Marathi conversion example. It's zero-shot (no training
needed) but also supports fast per-voice fine-tuning (~100 steps, ~2 minutes on a T4) if you want
to push quality further on one specific recurring voice — you can use it either way. **Honest
caveat**: this is a small community project (a HuggingFace Space with a handful of likes, mixed
with an unstated "proprietary dataset"), not a paper with published benchmarks of its own — the
quality claim about the *base* Seed-VC architecture is independently verified; the quality of
*this specific Hindi fine-tune* is not verified by anyone outside that one team. Worth a direct
pilot, not a blind trust.
**Also carries over**: Seed-VC's own maintainer acknowledged in a GitHub issue that preserving
*strong source emotion* specifically (as opposed to general naturalness/speaker-similarity) is
"currently an issue" they're still studying. Since your core ask is preserving intonation, test
this specifically rather than assuming the high MOS score above covers it — MOS measures
naturalness and speaker similarity, not emotion retention, and those are different axes.

**3. EZ-VC** (SPRINGLab, IIT Madras — Joglekar, Singh, Bhatia, Umesh; EMNLP Findings 2025)
The one option here that's an actual **peer-reviewed academic paper**, not a community fine-tune.
Trained from scratch on English (US/European/Indian accents) plus five Indian languages — Bengali,
**Hindi**, Tamil, Telugu, Kannada — totaling 12,840 hours, using a self-supervised Xeus encoder
feeding an F5-TTS-style diffusion-transformer decoder (~300M params), explicitly built and
evaluated for "zero-shot cross-lingual settings even for unseen languages/accents." Code, model
checkpoint, and demo samples are released (`github.com/ez-vc/ez-vc`). **License is CC-BY-NC** —
fine for internal testing/research, a real blocker if this ever needs to ship commercially.

**If it turns out you actually want TTS-style cloning after all** (typed text read aloud in a
cloned voice, rather than converting an existing recording) — none of the Western tools in §2
below cover Hindi either, but **AI4Bharat's IndicF5** (a 0.4B diffusion-transformer TTS model
trained on 1,417 hours of Indic audio, covering Hindi and 10 other Indic languages) is a
well-resourced, purpose-built option worth knowing about for that different task.

---

## 2. "Voice cloning" is ambiguous — here's the split that actually matters

Two genuinely different tasks both get called "voice cloning" in casual use:

| | **Voice Cloning (TTS)** | **Voice Conversion (VC)** — what you described |
|---|---|---|
| Input | New **text** you type, + a reference voice | An **existing recording** of someone speaking (or singing) |
| Where does the delivery/prosody come from? | The TTS model generates it — it's a performance the model invents | **The original recording** — timing, pauses, emphasis, emotion all come from the real performance |
| Output | The reference voice saying something it never actually said | The same words, same performance, same timing — different voice |
| Famous examples | Chatterbox, IndexTTS2, F5-TTS, XTTS, ElevenLabs, AI4Bharat IndicF5 | RVC, Seed-VC (and its `IndicVoiceChanger` fine-tune), EZ-VC, OpenVoice's tone converter, CosyVoice's `inference_vc`, Amphion Vevo |

You asked for something that "preserves the intonations and natural quality of voice but
converts the texture to someone else" — that's the right column, **voice conversion**, not the
left one. This matters because it rules out most of what shows up first when searching "open
source voice cloning":

- **Chatterbox / Chatterbox Turbo** (Resemble AI, MIT) — excellent, genuinely open, and the
  current community favorite for TTS cloning (beat ElevenLabs in blind tests per Podonos
  evaluations). But its core input is **text**, not a recording; the "voice conversion scripts
  included" are a secondary bolt-on, not its validated strength. Also no Hindi in its 23-language
  list.
- **IndexTTS2** (Bilibili) — genuinely impressive emotion/duration control, but still
  fundamentally text-in, and carries a non-standard license (bilibili Model Use License — free
  for most users, but organizations over 100M MAU or ¥1B annual revenue need a separate
  commercial license).
- **F5-TTS, XTTS v2, StyleTTS2** — same shape, same "not a fit" reason, and none cover Hindi.

The rest of this document (beyond §1's Hindi-specific shortlist) only covers the VC column.

---

## 3. How VC models get this to work, mechanically (so the comparison tables make sense)

Every model below follows roughly the same recipe, which is *why* prosody survives the
conversion at all:

1. **Split the source recording into "what was said/how" (content + prosody) vs. "who said it"
   (timbre/speaker identity).** This is usually done with a self-supervised speech encoder
   (HuBERT, WavLM, Whisper, XLSR, or a purpose-built codec) that's been trained to *not* carry
   speaker identity.
2. **Throw away the source's timbre, keep everything else.**
3. **Re-synthesize using a target speaker's timbre** (either a learned embedding from a short
   reference clip, or a small model trained/fine-tuned on that one target voice), through a
   vocoder (HiFiGAN, BigVGAN, Vocos, etc.).

The models differ mainly in *how well step 1 actually keeps timbre out* (poor disentanglement =
"timbre leakage") and *how it gets the target timbre in* (zero-shot embedding vs. train-a-model-
per-voice). That axis is also why some models cross languages better than others: **encoders
trained with a language-agnostic self-supervised objective (XLSR, multilingual HuBERT) tend to
generalize better across languages than ones with heavy text/phoneme conditioning** — which is
part of why Seed-VC's architecture was a reasonable starting point for someone to fine-tune into
`IndicVoiceChanger`, rather than starting from scratch.

---

## 4. Zero-shot candidates — no training, a short reference clip is enough

Fits an on-demand API well: upload a source clip and a target reference clip, get a converted
file back. Rows added for the Hindi-specific findings from §1.

| Model | License | Reference audio needed | Language coverage | Preserves source prosody | Maturity / backing |
|---|---|---|---|---|---|
| **`DreamSyncCo/IndicVoiceChanger`** | Not explicitly stated *(inherits at least Seed-VC's GPL v3.0 lineage — confirm before relying on it beyond a pilot)* | Short reference clip | **Fine-tuned specifically for Indian languages incl. Hindi** — demoed directly | ✅ By architecture; ⚠️ same source-emotion caveat as base Seed-VC (untested for this fine-tune specifically) | Small community project (single HF Space), no independent reviews found |
| **EZ-VC** | **CC-BY-NC** (non-commercial) | Short reference clip | **English (multi-accent) + Hindi, Bengali, Tamil, Telugu, Kannada** — trained from scratch on all of them | ✅ Explicit design goal ("zero-shot cross-lingual... unseen languages") | Peer-reviewed (EMNLP Findings 2025), real code+checkpoint release, from an IIT Madras speech lab |
| Seed-VC (base, no Indic fine-tune) | GPL v3.0 | 1–30s | English/Chinese-centric training; cross-lingual generalization via XLSR/Whisper encoders is architecturally plausible but not validated for Hindi specifically | ⚠️ Same acknowledged source-emotion weakness | Original repo archived/unmaintained; base for `IndicVoiceChanger` above |
| CosyVoice 2/3 | Apache 2.0 | ~3–10s | 9 languages, **no Hindi** — disqualified for this use, see §1 | ✅ (for its covered languages) | Very mature, 22.7k★ |
| Amphion Vevo/Vevo2 | MIT | Short clip | 6 languages, **no Hindi** — disqualified for this use, see §1 | ✅ (for its covered languages) | Research toolkit, continuously updated |
| HierSpeech++ | MIT | Short clip | English + Korean training data; no Hindi | ✅ (own docs note sensitivity to noisy references) | Real lab repo, 1.2k★ |
| OpenVoice V2 | MIT | ~1–10s | English, Spanish, French, Chinese, Japanese, Korean — no Hindi. A real GitHub issue shows a *different* unsupported language (Vietnamese) degrading badly ("mistakes female's tone for male's") — expect the same class of problem with Hindi | ⚠️ By design, but see the real-recording-as-input caveat below | Very popular (37.4k★), but see caveat |
| kNN-VC | "Other" (check directly) | N/A (no training at all — literal k-NN) | WavLM-based (English-centric pretraining); the authors' own follow-up paper reports decent generalization to *unseen* languages in testing, self-reported | ✅ by construction, weakest measured speaker-similarity of the group | Single academic repo, 520★ |

**The OpenVoice catch, unchanged from the original research**: its tone converter is
architecturally capable of taking a real recording as input, but its own GitHub issues show
real users getting robotic/unrecognizable results feeding it genuine recordings rather than its
own TTS output — and this is *before* even accounting for Hindi not being a supported language at
all. Not recommended here despite the star count.

---

## 5. The other shape: train (or download) a small model per target voice

A **different operational pattern**: instead of a short reference clip, you train a small model
on ~10+ minutes of one target voice, or — as established in §1 — download one of the thousands
that already exist, including specifically for Hindi/Bollywood voices.

| Model | License | Setup needed | Language coverage | Preserves source prosody | Maturity / backing |
|---|---|---|---|---|---|
| **RVC / Applio** | **MIT** (some forks are AGPL-3.0 — check which repo) | Train ~10+ min per voice, **or download an existing community model** | Language-agnostic pipeline; **large, active, real Hindi/Bollywood community model library confirmed directly** (Neha Kakkar, Jubin Nautiyal, Shreya Ghoshal, and many more — see §1) | ✅ Widely reported as very natural; a blind test by one user found friends couldn't distinguish it from the real recording | **The most battle-tested option in this entire document** — 36.7k★ on the main WebUI repo, huge model-sharing community |
| so-vits-svc (`-fork` for real-time) | AGPL-3.0 | Train per voice; historically singing-focused | Similarly language-agnostic pipeline, but this research found less direct Hindi-specific community evidence than for RVC specifically | ✅ Good, especially melody/pitch retention | Pre-RVC community favorite, less active today |
| DDSP-SVC | MIT | Train per voice; lighter hardware/time than so-vits-svc | Language-agnostic (DSP-based); no Hindi-specific evidence found either way in this research | ✅ Good — DSP pitch/loudness modeling preserves contour somewhat mechanically | Real, actively-ish maintained, respected but niche |

Given quality is the only real constraint, **RVC/Applio is now the strongest candidate in this
entire document for a fixed, recurring Hindi target voice** — the training-step downside that
mattered in the original comparison simply doesn't apply to you anymore, and its Hindi community
coverage and quality reputation are both real and already validated by a large hobbyist base.

---

## 6. Real-time — no longer a priority, kept here for completeness

You confirmed live conversion isn't something you need, so this section (previously more
central) is now background information only. For the record: RVC remains the best real-time
performer (~90ms with ASIO hardware), and Seed-VC/`IndicVoiceChanger` support a real-time mode
too (~300–430ms) — but since quality, not latency, is your priority, **use each project's
non-real-time / most-diffusion-steps configuration instead** (e.g., Seed-VC's `seed-uvit-whisper-
small-wavenet` offline model rather than its `tiny` real-time model; RVC without the low-latency
shortcuts).

---

## 7. Architecture note — this now fits cleanly, no special infrastructure needed

The original version of this document flagged a real tension: true real-time voice conversion
can't run through this API's existing "one Slurm job per request, cold-start, tear down" pattern
the way LTX-2.3 and Wan-Animate do — it needs an always-on warm process instead.

**That tension goes away entirely now that real-time isn't a requirement.** Every model
recommended in §1 — RVC/Applio, `IndicVoiceChanger`, EZ-VC — works as "submit a source file and a
reference/target, wait, get a converted file back," which is exactly this API's existing job
pattern, with no new infrastructure shape needed. On hardware: all three are small (RVC's
per-voice models are tens of MB; Seed-VC-based models are 25M–200M parameters; EZ-VC is ~300M) —
trivial next to LTX-2.3's 22B or Wan-Animate's 74.6GB VRAM footprint.

---

## 8. Licensing at a glance

| License | Models |
|---|---|
| **MIT** | Amphion Vevo/Vevo2 *(no Hindi — see §1)*, OpenVoice V1/V2, HierSpeech++, Applio (official RVC fork), DDSP-SVC, Chatterbox (TTS, not VC) |
| **Apache 2.0** | CosyVoice v1/v2/v3 *(no Hindi — see §1)* |
| **GPL v3.0** | Seed-VC and its direct forks (base architecture behind `IndicVoiceChanger`) |
| **AGPL-3.0** | so-vits-svc, some RVC WebUI forks (e.g. fumiama's) — **not** the original/Applio MIT lineage |
| **CC-BY-NC (non-commercial)** | **EZ-VC** — fine for internal/research testing, a real blocker for anything shipped commercially |
| **Not explicitly stated** | **`DreamSyncCo/IndicVoiceChanger`** — inherits at least Seed-VC's GPL v3.0 lineage for the architecture/training code; the fine-tuned weights' own license wasn't found stated anywhere in this research. Confirm directly before relying on this beyond a pilot. |
| **Custom, revenue-gated** | IndexTTS2 (TTS, not VC) |

**For your specific case**: if this stays internal/research, all three §1 picks (RVC/Applio's
MIT lineage, `IndicVoiceChanger`, EZ-VC's CC-BY-NC) are usable today. If this ever needs to ship
commercially, RVC/Applio (MIT) is the only one of the three with a clean, unambiguous commercial
license — EZ-VC's CC-BY-NC and `IndicVoiceChanger`'s unstated/GPL-inherited status would both need
a real legal look first.

---

## 9. Research horizon — real papers, not yet something to build on

Turned up while researching, worth knowing about, **not** worth picking today:

- **ProsoCodec** (Jun 2026 arXiv) — a speech codec purpose-built to solve exactly the
  "prosody preservation vs. timbre leakage" tension this whole document is about. Code/weight
  release status wasn't confirmed in this research.
- **DisCo-Speech** (ACL 2026 paper) — similar disentanglement idea; the paper itself states
  "code and weights will be released... upon acceptance," meaning not available yet.
- **X-VC** (Apr 2026) — real released code and checkpoints, unusually for this list, targeting
  streaming VC specifically (less relevant now that real-time isn't a priority for you). A
  single-team release with zero independent reviews so far.
- **OneVoice, MeanVoiceFlow, UniVoice, MeanVC 2, SynthVC** — 2026 papers exploring unified,
  faster, or more robust streaming VC. Directionally interesting, none evaluated on Hindi, and
  code/weight availability was inconsistent across them — not evaluated further here.

---

## 10. Recommendation

1. **Pilot `DreamSyncCo/IndicVoiceChanger` and RVC/Applio (with a downloaded Hindi community
   model) side by side**, on real Hindi source clips that include actual emotional/expressive
   delivery, not flat narration — that's the input type your original ask cares about most, and
   neither option has been independently verified on it by anyone in this workspace yet.
2. **Add EZ-VC to that same pilot if the CC-BY-NC license is acceptable for now** — it's the only
   peer-reviewed, Hindi-trained-from-scratch option here, which makes it a useful quality
   reference point even if licensing rules it out for anything beyond internal testing later.
3. **Specifically listen for source-emotion retention, not just general naturalness**, when
   comparing — Seed-VC's architecture (underlying `IndicVoiceChanger`) has an author-acknowledged
   weak spot there specifically, separate from its otherwise-strong naturalness/similarity scores.
4. **Do not pick CosyVoice or Amphion Vevo2 for this** — both are genuinely strong products, but
   neither has ever seen Hindi in training, confirmed directly against their own published
   language lists.
5. **Use each candidate's highest-quality settings, not its fastest ones** — RVC at 48kHz with a
   proper index, Seed-VC/`IndicVoiceChanger` with more diffusion steps and the larger checkpoint,
   since neither GPU cost nor latency are constraints for you.
6. **If one specific recurring Hindi voice matters more than "any voice, on demand," lean toward
   RVC/Applio** — its Hindi community coverage and quality reputation are the most concretely
   validated of anything in this document, and the one-time training/download cost you'd
   previously want to avoid is no longer a real cost given your priorities.
7. As before: **no integration code yet** — a small hands-on pilot on real Hindi audio, on this
   cluster, comes first.
