## Breeze TTS 2 text-to-speech

**What this is for:** generating speech from text (English and Chinese) with BreezeBlue's Breeze TTS 2.
This is real **text-to-speech** -- unlike RVC (which converts an existing recording) and Parakeet (which
transcribes one).

**Three modes, one endpoint** (`POST /v1/breeze-tts/synthesize`), chosen by which fields you send:
- **Voice Design** -- `text` (+ `instruction`): invents a voice from a natural-language description. No audio input.
- **Voice Clone** -- `text` + `reference_audio_asset_id` + `reference_text`: speaks `text` in the reference
  speaker's voice. `reference_text` must be the exact transcript of the reference recording; the two fields
  must be sent together.
- **Voice Direction** -- Voice Clone plus `instruction`: keeps the reference speaker's identity while steering
  tone, emotion and pace.

**Inputs:** a reference recording is uploaded first via `POST /v1/uploads` and referenced by `asset_id`. Common
formats are accepted, including phone voice notes (`.m4a`). Use a clean recording with minimal background noise.

**Tips:** write `instruction` in the same language as `text`; use `cfg_scale: 4` whenever you set `instruction`.
Inline vocal events: `(laugh)`, `(sigh)`, `(clears throat)` in English; `[笑]`, `[叹气]` in Chinese.

**Output:** a mono 24 kHz 16-bit WAV via `GET /v1/jobs/{job_id}/result`. Deterministic for the same inputs and `seed`.

**Limits:** text is limited per request (see the endpoint's description for the exact numbers); one request
generates about two minutes of audio at most, so split long scripts across several requests. Each request is its
own one-shot GPU job; there is no live progress and no streaming.

**License and responsible use:** the Breeze TTS 2 weights and self-hosted outputs are licensed for **research and
non-commercial use only**. Only clone voices you have the rights and consent to use. Whoever operates this
deployment is responsible for how it is used; this API does not enforce those restrictions itself.
