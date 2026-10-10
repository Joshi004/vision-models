## RVC voice conversion

**What this is for:** converting an existing recording so it sounds like one of a small, curated,
pre-installed set of voices, while keeping the recording's own delivery -- timing, pauses, emphasis,
emotion -- intact. This is **voice conversion**, not text-to-speech: there's no `text` field,
because nothing here is generated -- the words, performance, and pacing all come from the recording
you provide.

**Use this when** you have a real recording and want it to sound like a different (pre-installed)
voice. It is not a general text-to-speech endpoint, and it cannot clone an arbitrary new voice from
a sample on the fly -- see `GET /v1/rvc/voices` for the fixed set of voices actually installed on
this deployment right now.

**Inputs:** a source recording (`source_audio_asset_id`), uploaded first via `POST /v1/uploads`, and
a `voice` chosen from `GET /v1/rvc/voices`. A wide range of real-world recording formats is accepted
(see the endpoint's own description below for the exact list) -- including common phone voice-note
formats -- not just a narrow set of audio containers.

**Determinism:** unlike LTX-2.3 or Wan-Animate's diffusion-based sampling, this conversion is
deterministic given the same inputs and settings -- there is no `seed` field.

**Converting several recordings to the same voice?** See `POST /v1/rvc/batch-convert` instead of
calling `/convert` once per file -- one job (and one queue wait) for the whole set, returned as a
single zip. It's all-or-nothing (one bad file fails the whole batch) and only one batch runs at a
time on this deployment -- see that endpoint's own description below for the exact limits.

**Responsible use:** only convert recordings you have the right to use, and only distribute output
in a voice you have the right to use that voice for -- some installed voices may be subject to
legal restrictions on commercial or public use in some jurisdictions. Whoever operates this
deployment is responsible for which voices are installed; this API does not enforce usage
restrictions itself.