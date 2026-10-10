# Should We Wire Up MAGI-2, Or Something Else? A Model Comparison

_Research document, not a decision record. Written before writing any MAGI code,_
_per the request to compare first. Sources are a mix of official repos/HuggingFace/_
_arXiv (high confidence) and third-party comparison sites (lower confidence, flagged_
_where used). This field changes weekly — treat exact numbers below as "true as of_
_late September 2026," not permanent facts._

**Your goal, as stated:** videos of 1 minute or more, with consistent characters and
sound, GPU is not a constraint.

---

## TL;DR

- **MAGI-2 does not clearly get you anything LTX-2.3 doesn't already give you**, for
  your actual goal. It scores slightly better on one leaderboard, at 8x the hardware
  cost, with a much smaller supporting ecosystem, and it has no special ability to
  make long or multi-shot videos — same limitation LTX has.
- **No model — MAGI-2, LTX-2.3, or otherwise — natively outputs a coherent 1-minute
  video from one prompt.** Every option tops out at somewhere between 5 and 20
  seconds per generation. Getting to a minute is fundamentally a **pipeline problem**
  (stitching shorter clips together while keeping characters consistent), not a
  single-model-choice problem. That reframing matters more than which base model you
  pick.
- **There is a project purpose-built for exactly your goal, sitting on top of the
  LTX model you already have installed**: `JoyAI-Echo-1.5` (JD.com). It's built to
  generate 1-5 minute multi-shot stories with a memory mechanism specifically for
  keeping character appearance and voice consistent across shots, with sound
  included, and it runs on **one** GPU instead of MAGI's eight. Recommended as the
  next thing to actually test — see §4.
- Everything below is explained in more detail, with the honest caveats each option
  comes with.

---

## 1. The one thing that matters most: "1 minute" is a pipeline problem

Every open video model surveyed — MAGI-2, LTX-2.3, Wan 2.2, MOVA, HunyuanVideo, all
of them — generates a **single short clip** per call: MAGI-2 is fixed at 10 seconds,
LTX-2.3 goes up to about 20 seconds, most others land in the same 5-10 second range.
None of them take a prompt and hand back a finished 60-second video in one pass.

Every real-world approach to "make it 1 minute" that turned up in this research does
the same three things, regardless of which base model sits underneath:

1. **Generate (or provide) one reference image of each character first.**
2. **Generate each subsequent shot as image-to-video starting from a previous
   frame** (either the last frame of the previous clip, or the character reference),
   instead of pure text-to-video every time — because the character's appearance
   then comes from a real pixel, not a re-description in words.
3. **Stitch the clips together**, optionally with a hard cut/scene change every few
   shots, because identity drift is a known, named problem — one source measured
   "six to nine scenes before cumulative drift becomes noticeable" as a rough
   real-world ceiling for plain frame-chaining.

So the real question isn't just "which model" — it's "which model, plus what
chaining/consistency mechanism." That's why §4 below is the most important part of
this document.

---

## 2. Head-to-head: MAGI-2 vs. what you already have vs. the alternatives

| | **LTX-2.3** (have) | **MAGI-2 Preview** (proposed) | **JoyAI-Echo-1.5** | **MOVA** | **Wan 2.2** |
|---|---|---|---|---|---|
| Native sound? | ✅ Yes, jointly | ✅ Yes, jointly | ✅ Yes, jointly | ✅ Yes, jointly | ❌ Silent |
| Built for multi-shot/long stories? | ❌ No | ❌ No | ✅ **Yes — its whole purpose** | ❌ No | ❌ No (but see §3, Doc 2) |
| Max single clip | ~20s | 10s (fixed) | 10s per shot, **up to 5 min story** | 8s | ~5s |
| GPUs needed | 1 | **8 (whole node)** | 1 | 1 (heavy) | 1 |
| Params | 22B | 114B (6B active) | Built on LTX-2.3's 22B | 32B (18B active) | 14B or 27B |
| Open weights? | ✅ | ✅ Apache 2.0 | ⚠️ Yes, but see license note below | ✅ | ✅ Apache 2.0 |
| ComfyUI / community tooling? | ✅ Extensive | ❌ None found | Growing (ComfyUI nodes exist) | ❌ Minimal | ✅ Extensive |
| Independent quality reviews | Mixed/positive | Thin; mixed leaderboard position | **None found yet — too new** | Mixed-to-negative early reactions | Generally positive |
| Already running in this workspace? | ✅ | ❌ | ❌ | ❌ | ❌ |

**Leaderboard reality check** (from [Artificial Analysis](https://artificialanalysis.ai/video/leaderboard/text-to-video),
a real independent benchmark site, checked mid/late September 2026): MAGI-2 Preview
scores **~1,156-1,162 Elo** in the text-to-video-with-audio category — 6th place
overall, and reportedly 1st among *open-weight* models in that specific category.
That's a genuine, real point in MAGI-2's favor. But it's an overall generation-quality
score, not a "good at long consistent-character video" score — and it comes at 8x
the hardware cost of LTX with none of LTX's ecosystem around it (no ComfyUI, no
LoRA culture, no third-party long-video projects built on top of it, unlike LTX —
see §4).

**A caution about "Wan 3.0" specifically:** several comparison sites casually cite
Wan 3.0 as the current best open-source model. Checked directly against Alibaba's
own GitHub/HuggingFace/Model Studio pages: **this is false as of this research.**
Wan 3.0 is a hosted-API-only preview (application + approval required, billed per
video); no weights, checkpoint, or runnable local package exists yet. The actual
open-weight Wan line stops at **Wan 2.2** (Apache 2.0, real, downloadable). Don't
trust a comparison article's "open source" label without checking the primary repo.

---

## 3. Why MAGI-2 specifically doesn't look worth it for this goal

Putting together everything above, MAGI-2's case is weak specifically **for what you
said you want** (long, consistent-character, GPU-not-a-constraint):

- It doesn't do multi-shot/story generation any better than LTX — both are
  single-short-clip generators that need external chaining for a 1-minute result.
- It costs 8 GPUs per clip vs. LTX's 1 — since GPU isn't your constraint, this
  matters less than it would otherwise, but it's still 8x the resource for no
  demonstrated advantage on your actual goal.
- It has effectively no supporting ecosystem: no ComfyUI, no LoRA/fine-tuning
  culture found, no third-party "long video" or "consistent character" projects
  built on top of it in this research — compare that to LTX, which has JoyAI-Echo
  (long video), at least one published identity-consistency technique built
  specifically for it (ST-DRC, see Doc 2's companion research), and full ComfyUI
  support.
- Its one clear win (slightly higher leaderboard Elo, "best open-weight in the audio
  category") is a general quality signal, not a "solves your specific problem"
  signal.

**This doesn't mean MAGI-2 is bad** — it means it doesn't clearly justify the extra
integration work (see `MAGI_WIREUP_PLAN.md`) *for this particular goal*, when
something more directly purpose-built already sits on infrastructure you have.

---

## 4. The strongest candidate found: JoyAI-Echo-1.5

**What it is:** an open-weight model from JD.com's "Joy Future Academy," explicitly
built for "minute-level multi-shot audio-video generation" — built directly on top
of **LTX-2.3**, the model already installed in this workspace.

**Why it matches your goal almost exactly:**
- **Multi-shot stories up to 5 minutes**, not just one clip.
- A **"cross-modal memory bank"** that specifically carries character appearance
  *and voice timbre* across shots — i.e., built-in machinery for the "consistent
  characters and sound" part of your goal, rather than something you'd have to
  bolt on yourself via manual frame-chaining.
- **Runs on a single GPU** (~46-50GB VRAM at default settings — fits comfortably on
  one of this cluster's H100 80GB cards), with smaller FP8/FP4 variants available
  too. No 8-GPU requirement at all.
- Ships real, downloadable weights (BF16/FP8/FP4) and real inference code today —
  not a paper-only preview.

**Honest caveats — please read before getting excited:**
- **This is a small/niche project, not from a major lab.** It hasn't been through
  the kind of broad independent scrutiny LTX, Wan, or MAGI have.
- **No independent (non-self-reported) quality reviews were found.** Every claim
  about its quality traces back to JD's own model card/paper. Treat "it works great"
  as unverified until we actually try it.
- **Licensing needs a real check before relying on this for anything beyond
  research/internal use.** It's released under LTX-2's own Community License, but
  the GitHub repo's own license section adds: *"This project is not intended for
  commercial use. For commercial use of LTX-2 or its derivatives, please contact
  Lightricks Ltd."* That's a real, specific restriction worth resolving (e.g. by
  actually contacting Lightricks, or a proper legal read of both license texts
  together) before this becomes anything beyond an internal experiment.
- It's very new (the paper/release is from within the last month or two of this
  research), so expect rough edges.

**Suggested next step, if this sounds worth pursuing:** a small, contained pilot —
download it, run one multi-shot story on our own GPUs, and actually look at the
result — before writing any integration code for it. That's a much cheaper way to
validate the "no independent reviews yet" gap than trusting the model card alone.

---

## 5. Other options worth knowing about (briefly)

- **MOVA** (OpenMOSS) — also native audio+video, fully open including training code.
  On paper, a real MAGI-2/LTX competitor. In practice: early community reaction
  (Reddit, checked directly) was mixed-to-skeptical — reports of lip-sync mismatches,
  visible artifacts, and one comment noting it's "seems very very slow compared to
  LTX." Also capped at 8 seconds per clip, so it doesn't help with "1 minute" any
  more than LTX does. Worth watching, not worth switching to today.
- **NAVA** (Baidu ERNIE team) — much smaller (6.3B) native audio-video model, fully
  open with a Gradio demo. Efficient, but nothing found suggesting it's aimed at
  long/multi-shot generation specifically. A "lightweight alternative" to know about,
  not a long-video specialist.
- **LongLive / LongLive 2.0** (NVIDIA labs) — a real, actively-developed
  long/interactive video specialist (built on Wan2.2), with a clever "multi-shot
  attention sink" mechanism for coherence. Worth knowing about, but **no audio** —
  it solves the "long and consistent" half of your goal, not the "and sound" half.
  Could be worth a look if JoyAI-Echo doesn't pan out.
- **HunyuanVideo, CogVideoX, Mochi 1** — solid, well-known open T2V models, but all
  silent (no audio) and none specifically built for multi-shot consistency. Not a
  fit for this particular goal; mentioned only so you know they were considered and
  set aside for a specific reason (no audio).

---

## 6. Recommendation

1. **Don't build the MAGI-2 wire-up next** (pause `MAGI_WIREUP_PLAN.md`) — the
   evidence doesn't show it earning its 8-GPU cost for this specific goal.
2. **Pilot-test JoyAI-Echo-1.5 on our own hardware** before writing any integration
   code for it — cheap to try, directly targets your stated goal, but unverified by
   anyone outside JD so far.
3. **Resolve the commercial-use license question** for JoyAI-Echo early, before
   investing real engineering time, if this is ever more than an internal
   experiment.
4. Keep LTX-2.3's own API (already built) as the fallback/building block regardless
   — JoyAI-Echo is literally built on it, so nothing from that work is wasted either
   way.
