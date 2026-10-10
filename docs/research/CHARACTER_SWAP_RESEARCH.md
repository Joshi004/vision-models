# Swapping a Character (or Just an Outfit) Into an Existing Video

_Research document for a specific use case: "I have one dance video and I want
another character in it, in another dress or similar." Not a decision record —
nothing here has been built. Confidence notes included per source, since this is a
fast-moving research area._

---

## TL;DR

- This is a **different task from everything in `VIDEO_MODEL_COMPARISON.md`.**
  Those models generate new video from a prompt. This use case starts from a video
  that already exists and edits *who's in it* — a genuinely different model family.
- There are two distinct versions of this request, and they have different best
  answers:
  1. **Swap the whole person** (different actor, different body, possibly different
     clothes too) while keeping the original motion, timing, and background →
     **Wan-Animate / Wan-Animate-2** (Alibaba). This is a mature, actually-released,
     open-weight, purpose-built answer — not a research curiosity.
  2. **Keep the same person, just change their outfit** → a narrower, separate
     model family called **video virtual try-on** (e.g. `ViViD`, `MagicTryOn`).
- The good news on sound: in this use case, the video's **original audio track
  usually doesn't need to be touched at all** — you're replacing what's on screen,
  not what's playing. Keep the source audio, mux it onto the new video, done (the
  same "keep the audio, only redo the picture" idea LTX's own `retake` endpoint
  already uses).

---

## 1. The right tool for "swap the whole character": Wan-Animate

**What it is:** an open-weight model from Alibaba's Wan team (the same org behind
the Wan video models), built for exactly two things:

| Mode | What you give it | What you get |
|---|---|---|
| **Animate** | A character reference image + a driving video | A new video of *that character* performing the driving video's motion, on a fresh background |
| **Replace** | A character reference image + an existing video | The **same existing video**, but with the original person replaced by the reference character — background, lighting, and camera motion preserved |

**"Replace" mode is precisely the dance-video use case you described.** Give it your
dance video and a reference image of the character you want instead, and it
outputs the same dance, same camera, same background — different performer. It
even includes a dedicated **"Relighting LoRA"** specifically so the swapped-in
character's lighting and color tone match the original scene instead of looking
pasted in.

**Why this is a solid recommendation, not a speculative one:**
- Actually released: real weights (`Wan-AI/Wan2.2-Animate-14B` on HuggingFace and
  ModelScope), real inference code, real documentation with working example
  commands.
- **Native ComfyUI support**, with the community already using it in practice —
  found real, specific usage discussion (not just marketing) of exactly this
  replace-vs-animate distinction on ComfyUI's own workflow docs and GitHub issue
  threads.
- Actively maintained: a second generation, **Wan-Animate-2**, released just last
  month (August 2026) — improves motion fidelity, adds camera-angle control
  independent of the driving video, and adds a "Lite" variant fast enough for live,
  real-time use. Both a full and a distilled (faster, fewer-steps) checkpoint are
  already published.
- Open license: built on Wan2.2, which is Apache 2.0.

**What it needs from you:** a driving video (your dance clip) and one clear
reference image of the new character. The official pipeline also uses a body-pose
extraction step (skeleton tracking) and a face-motion extraction step under the
hood — this is handled by the model's own preprocessing script, not something you
build yourself.

**Honest caveats:**
- Hardware requirements weren't pinned down precisely in this research — one
  official example command distributes across 8 GPUs, but that may well be for
  speed/throughput rather than a hard requirement for a 14B model; this needs a
  direct check (or a small test) rather than assuming either way.
- Best-documented for full-body/dance-style motion with a fairly clear view of the
  person — very fast, complex, or heavily occluded motion is a reasonable place to
  expect rougher results, though nothing specific on that was found either way.
- Not sound-related — this model only ever touches video. See the TL;DR on why
  that's usually fine here.

---

## 2. If it's really "same person, different outfit": virtual try-on instead

If the actual goal is narrower than a full character swap — same dancer, just
change what they're wearing — that's a different, more specialized task called
**video virtual try-on**, with its own dedicated models:

| Model | License | Status |
|---|---|---|
| **ViViD** (Alibaba) | Apache 2.0 — permissive | Real, released, straightforward to run |
| **MagicTryOn** | CC BY-NC-SA — **non-commercial only** | Real, released, actively updated (as recently as April 2026) |

Both need roughly the same inputs: a video of the person, a photo of the target
garment, and some automatically-derivable helper inputs (a body-pose map, a mask of
where the garment goes) — produced by preprocessing tools the projects point to,
not something to build from scratch.

Several newer research papers (`UniVVT`, `InstructVVT`) are trying to remove the
need for those helper inputs entirely (describe the swap in plain instructions
instead) — promising direction, but these read as recent research releases rather
than proven, widely-used tools yet. Worth a periodic check, not a first choice today.

**A note on `SwapAnyone`** (a paper/project name that comes up in this exact search
space, positioned as a full body-swap method with its own dataset): at last check,
its own repository listed code/training/dataset release as **not yet done**
(unchecked items in its own to-do list) despite the paper being public. Treat it as
"a name to watch," not a currently-usable tool — verify its repo's current state
before counting on it.

---

## 3. Answering the "and sound" part of this use case

Unlike the long-video generation goal in the companion document, this task rarely
needs new audio at all:

- If you're only changing *who's on screen* (character swap or outfit swap), the
  original video's audio track — music, ambient sound, whatever it was — is still
  correct and can simply be kept and re-muxed onto the new visual output, exactly
  like how LTX's own `retake` endpoint already preserves everything outside the
  edited window untouched.
- Native audio-generation models (LTX-2.3, MAGI-2, the ones compared in the other
  document) are solving a **different problem** — generating sound that doesn't
  exist yet for a brand-new clip. They're not the right tool here, and mixing them
  into this use case would be solving a problem you don't actually have.
- The one case where you *would* need real audio work: if the new character is
  also supposed to speak different words than the original performer (dubbing/lip
  re-sync), which is a third, separate task again (closer to what LTX's own
  audio-to-video endpoint or a dedicated lip-sync/dubbing tool would handle) — worth
  flagging only if that turns out to be part of what you actually want.

---

## 4. Recommendation

1. For "put a different character into this existing video, doing the same thing"
   → **Wan-Animate / Wan-Animate-2, replace mode.** This looks like a genuinely
   good, low-risk fit — mature, open, actively maintained, purpose-built for this
   exact request.
2. For "same person, different clothes" specifically → **ViViD** first (permissive
   license), **MagicTryOn** only if you're fine with non-commercial use and want to
   compare quality.
3. Don't reach for a generation model (LTX/MAGI/etc.) for this task — it's solving
   a different problem than the one you have here.
4. Same suggestion as the other document: a small hands-on pilot (one dance clip,
   one reference character) before any real integration work, since — as with
   everything in this fast-moving space — the model card's claims haven't been
   independently verified by us yet.
