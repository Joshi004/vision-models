# Wiring up MAGI-2.2 — What Needs To Be Done

_This is a **planning document only** — nothing here has been built yet. It's a task
list to work from, not a changelog. Written in plain language on purpose; see the
linked source files for the technical detail behind each point._

Qwen3-VL is intentionally not mentioned anywhere below — out of scope, as discussed.

---

## The short version

Getting MAGI-2.2 behind the shared API takes work in **three separate places**:

1. **New code in `model-api/`** — a `services/magi/` module, the same shape as
   `services/ltx/`, plus one genuinely new piece: a small **queue** that can bundle
   several people's requests into one GPU job instead of always starting a fresh one.
2. **A handful of small, well-understood code changes inside MAGI's own project**
   (`ModelService_MAGI-2.2/repo/`) — not huge rewrites, but real changes, not just
   settings.
3. **A few product decisions** about tradeoffs that don't have a single "correct"
   answer (how long to wait for a batch to fill up, what happens if one request in a
   batch fails, etc.) — flagged clearly below so they can be decided rather than
   guessed at.

---

## 1. Why this isn't just "copy what LTX did"

LTX-2.3 and MAGI-2.2 are shaped very differently, and that shape drives everything
below:

| | LTX-2.3 | MAGI-2.2 |
|---|---|---|
| Runs as | A plain Python program in its own environment | A program sealed inside a "container" (a pre-packaged copy of its software), launched a different way |
| GPUs per job | 1 | **8 — an entire machine** |
| Warm-up cost | ~3-4 minutes, every job, unavoidable | ~20-25 minutes for the *first* clip in a job, but only ~4-5 more minutes for each *additional* clip in that **same** job |
| Can it do several clips without re-loading? | No (not built to; not needed, warm-up is small) | **Yes — this is the whole opportunity here** |
| Told its output location as | An exact file name | A folder (it picks the file name itself) |

That warm-up row is the important one. If every request becomes its own fresh MAGI
job, you pay the full ~20-25 minutes every single time, on an entire 8-GPU machine,
for one 10-second clip. If several requests can share **one** job, everyone after
the first only adds ~4-5 minutes each. That's the "leverage the model load once"
idea from the request, and it's the centerpiece of this plan (§3).

---

## 2. New API surface needed in `model-api/`

A new `services/magi/` folder (mirroring `services/ltx/`):

- `config.py` — MAGI's own paths, container image, 8-GPU Slurm shape.
- `schemas.py` — the request format for a MAGI clip.
- `dispatch.py` — turns a request into an actual job (see §3 for why this one is
  bigger than LTX's version).
- `router.py` — the actual HTTP endpoint(s).

**Proposed endpoint:** `POST /v1/magi/videos/generate`

| Field | Meaning | Caveat |
|---|---|---|
| `prompt` | Text description | MAGI expects long, structured descriptions, not a one-liner — see §4's prompt-enhancement note |
| `image_asset_id` | Optional — makes it image-to-video | Must end up mounted into the container as a full disk path, not a shortcut/relative one — a real, previously-hit failure mode (documented in this project's own `STORY_TO_VIDEO_INTEGRATION_SPEC.md`) |
| `orientation` | landscape / portrait | Everyone sharing one batch job must use the **same** orientation (see §3) |
| `seed` | For reproducibility | **Only actually honored once §5.3 below is done.** Until then, accept the field but be upfront that it's not guaranteed — see §3's "honest limitations" note |
| `negative_prompt` | What to avoid | Same caveat as `seed` — see §5.3 |
| `partition` | Same as LTX's partition override | Reuses what already exists — no new work |

No new endpoints needed for status/download/cancel — `/v1/jobs/...` already works for
any backend, MAGI included, without changes.

---

## 3. The queueing / batching design (the main ask)

**The goal:** if 4 people ask for a clip within a short window, MAGI should load
its model **once** and render all 4, instead of cold-starting 4 separate times.

**How, without touching MAGI's own code first (Phase 1):**

MAGI's existing batch mode already accepts a JSON list of prompts and renders them
one after another *without* reloading the model between them — that's exactly what
`generate_batch.sh` already demonstrates today. So Phase 1 is entirely new logic
inside `model-api`, not a MAGI code change:

1. A request lands on `POST /v1/magi/videos/generate`. Instead of submitting a GPU
   job immediately, it's placed on a **waiting list**.
2. A small background task (same pattern as the existing job-status poller) watches
   that list and decides when to actually submit a job, once **either**:
   - the oldest waiting request has waited past a configurable limit (e.g. 45-60
     seconds), **or**
   - the waiting list has grown to a configurable size (e.g. 8 requests) —
     whichever happens first.
3. When it flushes, it builds one combined prompt list, submits **one** 8-GPU job
   covering everyone currently waiting, and remembers which position in that list
   belongs to which original request.
4. Once the job finishes, each original request gets matched back to its own output
   file (`sample_000.mp4`, `sample_001.mp4`, ...) and can be downloaded exactly like
   before — the person who asked for it never needs to know it was bundled with
   anyone else's.
5. If nobody else is waiting, a lone request still goes out as its own "batch of
   one" once its wait limit is hit — so this never makes a single request slower
   than today, only sometimes a little slower in exchange for being much cheaper
   when several requests overlap.

**Two honest limitations of Phase 1** (fixable in §5.3, not before):

- **Only requests asking for the same shape (landscape/portrait) and the same
  "avoid this" text can be bundled together.** Both are settings for MAGI's whole
  job, not per-clip, today. A request with unusual settings just runs alone — never
  broken, just not sped up.
- **A custom `seed` can't truly be honored yet.** MAGI decides each clip's seed by
  its position in the batch, not by a value passed in per-clip. Until §5.3 is done,
  the API should be upfront about this rather than silently ignoring what was asked
  for.

**A bigger, later option (Phase 2, not recommended to start with):** keep one
8-GPU machine permanently loaded and waiting for work, instead of starting fresh
each time. This removes the wait-for-a-batch-to-fill tradeoff entirely, but means
reserving a whole machine on a shared cluster **all the time**, even when nobody's
using it — a real cost, and a bigger change (MAGI has no "wait for the next request"
mode today; someone would have to build one). Worth considering only if usage
becomes frequent enough to justify permanently reserving hardware.

**A third option, not ours to build:** Sand.ai's community is already working on
plugging MAGI-2 into [SGLang](https://github.com/sgl-project/sglang) (see
[this PR](https://github.com/sgl-project/sglang/issues/35011)), a serving system
built specifically for efficient batching like this. As of this research, that
support is **an open pull request, not yet merged, not usable today** — but if it
lands, it may make everything in this document unnecessary and worth replacing.
Worth a periodic check, not worth waiting for.

---

## 4. Other MAGI capabilities worth knowing about (found while researching)

- **It already has a "make my short prompt better" feature — just switched off.**
  MAGI is trained to expect long, detailed descriptions (hundreds of words,
  structured into sections). It ships a built-in step that expands a short prompt
  into that format automatically using an external AI model — but it's off by
  default because no API key is configured. Turning this on could be a big
  usability win (people could send a one-line prompt instead of learning MAGI's
  detailed format) — see §5.4 for what that takes.
- **Resolution is more flexible than one might assume**, though only 1080p is
  currently used in this project's own scripts — lower tiers exist and cost less.
- **Sound can silently go missing.** Per MAGI's own documentation: if a certain
  audio tool (`ffmpeg`) isn't available inside the container, the video still gets
  made — just without sound, and with **no error or warning**. Worth a one-time
  check that this project's container setup actually has it (§5.5).
- **No official or community HTTP API exists for MAGI-2 anywhere** (checked
  publicly) — the only supported way to run it today is the exact command-line /
  container approach this project already uses. So there's no shortcut being missed
  by building this ourselves.

---

## 5. Changes needed inside MAGI's own project (`ModelService_MAGI-2.2/repo/`)

These are small, targeted changes — not rewrites — but they are real code changes,
not configuration.

- [ ] **5.1 — Don't let one bad request ruin everyone else's batch.**
  Today, if one clip in a batch hits an error, the whole run stops right there —
  clips later in that batch never get their turn, even though earlier ones already
  finished fine. Fix: make each clip's turn independent, so one failure doesn't take
  the rest down with it. This matters a lot once batching (§3) is in use — it
  protects other people's requests from someone else's bad prompt.
- [ ] **5.2 — Let each clip in a batch say where its finished file should be checked.**
  (Mechanical bookkeeping to match §3's design — matching output files back to the
  right original request.)
- [ ] **5.3 — Let each clip in a batch carry its own seed and its own "avoid this"
  text**, instead of one setting for the whole batch. This is what removes the two
  "honest limitations" flagged in §3 and makes batching invisible to the people
  using it.
- [ ] **5.4 — Make the prompt-enhancement feature's access key a setting, not
  something hardcoded to "off."** Small, safe change; unlocks a real feature that's
  already built (§4).
- [ ] **5.5 — Confirm the audio-muxing tool is actually present** in the exact
  container/environment this project uses, so sound never silently goes missing.

None of these require deep MAGI expertise to make — they're all localized to one or
two files each — but they are genuine code edits, not settings to flip.

---

## 6. Decisions worth making explicitly (not guessing at)

- How long should a batch wait for more requests before giving up and running with
  what it has? (Longer = cheaper on average, but slower for the first person in.)
- How many requests should one batch hold at most?
- If a custom `seed`/`negative_prompt` is requested and can't be honored yet
  (§3), should the API reject the request, warn but proceed, or just quietly run it
  alone (never batched, always honored, just never sped up)?
- Is it worth reserving one 8-GPU machine permanently (Phase 2, §3) at any point, or
  is "cold start per batch" acceptable indefinitely?

---

## 7. Suggested build order

1. Basic `services/magi/` wiring with **no** batching yet — one request, one job,
   same pattern LTX already uses. Proves the container/mount/output-folder mechanics
   work at all.
2. §5.1 and §5.2 (safety + bookkeeping) inside MAGI's own code.
3. The batching queue itself (§3, Phase 1).
4. §5.3 (per-clip seed / negative prompt) to remove the last two honest limitations.
5. §5.4 (prompt enhancement) and §5.5 (audio check) — independent of the above,
   can happen anytime.
6. Revisit Phase 2 (always-warm machine) and the SGLang project only if real usage
   patterns justify it.
