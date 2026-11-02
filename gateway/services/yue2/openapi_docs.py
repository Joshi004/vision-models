"""Long-form OpenAPI/Swagger documentation text for YuE2's endpoints, kept out
of router.py so the routes themselves stay readable. Docs shared by every
backend (uploads, jobs, app-level description) live in common/openapi_docs.py
instead.
"""

from __future__ import annotations

from common import openapi_docs as common_docs
from services.yue2 import config

_COMMON_NOTES = f"""
**Result:** `output.flac` (48 kHz stereo, 24-bit FLAC), downloaded with
`GET /v1/jobs/{{job_id}}/result`. The song's length follows the lyrics (typically
several minutes). The same inputs and `seed` give the same audio on the same
hardware. One request makes **one candidate**: send several seeds and choose by ear.

**Model:** YuE2-3B with the YuE2-Vae decoder, bf16, no quantization.

**Timing:** requests are run by long-lived GPU jobs that load the model once.
A request that has to start one takes minutes; one that finds a loaded, free
worker starts at once. A worker idle for {config.IDLE_MINUTES:g} minutes is stopped. No live
progress: `progress` is always null.

**License: non-commercial.** The YuE2 weights are **CC BY-NC 4.0**, with creator
permission required and a commercial licence required for companies. Whoever
operates this deployment is responsible for how it is used; this API does not
enforce that restriction itself. Generated music can resemble existing works:
check originality and disclose AI involvement where required.
"""

GENERATE = f"""
Generates a **full song with sung vocals** from a `style` line and `lyrics` with
**YuE2**, planning a melody-and-chords score first (`cot: full`). This is the
backend to pick for the best-sounding complete English or Chinese song.

- `style` -- one line, **language first**, then genre, instruments, vocal character
  and tempo (there is no BPM field): `English, warm piano pop, expressive female
  voice, acoustic piano, light drums, 88 BPM`.
- `lyrics` -- the words, with `[Verse]` / `[Chorus]` / `[Bridge]` tags on their own
  lines and a blank line between sections.
- `cot` -- `full` (default), `melody`, or `off` (no plan: a little faster, clearly
  worse in upstream's comparison).
{_COMMON_NOTES}
"""

COVER = f"""
Generates a song that **follows a melody score you supply** (`abc`), in the
`style` you describe, with your `lyrics`. Use it to cover a known melody in a new
style. Use `cot: melody` (default) so the accompaniment has freedom; `full`
suits a score with fixed harmony.

`abc` must be ABC text in the model's native format (two voices `V: Vocal` and
`V: Ins`, no chord symbols for a melody-only score), and `lyrics` must follow the
score's sections in order. **This API does not turn a recording into a score**;
upstream does that with a separate tool, SheetSage2. Transcription errors carry
into the cover, so review the score first.
{_COMMON_NOTES}
"""

GENERATE_EXAMPLES = {
    "song": {
        "summary": "A pop song",
        "value": {
            "style": "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM",
            "lyrics": "[Verse]\nStreetlights hum along the avenue\nI keep your number in my shoe\n\n[Chorus]\nStay with me until the morning light\nHold the quiet of the night",
            "cot": "full",
            "seed": 831001,
        },
    },
}

COVER_EXAMPLES = {
    "cover": {
        "summary": "Follow a score, new style",
        "value": {
            "style": "English, upbeat funk-jazz, male vocal, electric piano, slap bass, tight drums, 110 BPM",
            "lyrics": "[Verse]\nStreetlights hum along the avenue\n\n[Chorus]\nStay with me until the morning light",
            "abc": 'X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=88\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nV: Ins clef=treble name="Ins Melody" snm="Inst."\nK:G\n% verse\nV: Vocal\nB8d8c8A8|F16G16|\nV: Ins\nZ2|',
            "cot": "melody",
        },
    },
}

GENERATE_RESPONSES = {**common_docs.SUBMIT_RESPONSES}
COVER_RESPONSES = {**common_docs.SUBMIT_RESPONSES}
