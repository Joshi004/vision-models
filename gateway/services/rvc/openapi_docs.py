"""Long-form OpenAPI/Swagger documentation text for RVC's own endpoints,
kept out of router.py so the routes themselves stay readable. Docs shared
by every backend (uploads, jobs, app-level description) live in
common/openapi_docs.py instead.
"""

from __future__ import annotations

from common import openapi_docs as common_docs
from services.rvc import config

_ACCEPTED_FORMATS_LIST = ", ".join(sorted(ext.lstrip(".") for ext in config.INPUT_EXTENSIONS))

VOICES_LIST = """
Returns the names of every currently-installed voice model, alphabetically
-- exactly the values `convert`'s own `voice` field will accept right now.
This list only changes when this deployment's operator installs a new voice
and restarts the API process -- it is not a live/dynamic catalog you can add
to at request time through this API.
"""

CONVERT = f"""
Converts an existing recording so it sounds like one of the pre-installed
voices, while keeping the recording's own delivery -- timing, pauses,
emphasis, emotion -- intact. This is **voice conversion**, not
text-to-speech voice cloning: there's no `text` field, because nothing here
is generated; the words, performance, and pacing all come from
`source_audio_asset_id` itself. See `GET /v1/rvc/voices` for which target
voices are currently available.

**Pipeline, run as a single job:** the source recording is first decoded to
a WAV (see the accepted-formats note below), then run through the RVC v2
pipeline: self-supervised content extraction from the source recording,
retrieval-augmented timbre transfer from the selected voice's pre-trained
model, then a HiFi-GAN vocoder pass.

**Accepted source-recording formats:** {", ".join(sorted(ext.lstrip(".") for ext in config.INPUT_EXTENSIONS))}.
Decoding to WAV first (rather than handing the file to the conversion
pipeline directly) means real-world recordings work out of the box --
including phone voice notes (`.m4a`/`.aac`) -- not just the narrower set
that pipeline's own audio loading can read natively. An unsupported file is
rejected immediately (400) before any job is submitted.

**Voices are pre-installed, not uploaded per request.** Unlike
`video_asset_id`/`image_asset_id` on other endpoints, `voice` selects from a
small, curated, server-side set of already-trained RVC models -- see
`GET /v1/rvc/voices`. This is a deliberate difference in shape: an RVC voice
model is meant to be reused across many requests, not re-uploaded each time.

**A real measured run on this deployment** (a ~10.7s speech clip, default
settings): the conversion itself took ~20 seconds; the full job (including
Slurm queueing and node allocation) took well under two minutes. Much
longer source audio will take proportionally longer -- not yet measured.

Output format follows `export_format` (default WAV). There is no `seed`
field: RVC's conversion is deterministic given the same inputs and
settings, unlike LTX/Wan-Animate's diffusion-based sampling.

**If you have several recordings to convert to the same voice**, see
`POST /v1/rvc/batch-convert` instead -- one Slurm job (and one queue wait)
for the whole set, rather than one of each per file.
"""

CONVERT_EXAMPLES = {
    "basic_conversion": {
        "summary": "Convert a recording to sound like one of the installed voices",
        "value": {
            "source_audio_asset_id": "<asset_id of the recording to convert, from POST /v1/uploads>",
            "voice": "<voice name, from GET /v1/rvc/voices>",
        },
    },
    "higher_quality_settings": {
        "summary": "Same conversion, prioritizing quality over speed",
        "value": {
            "source_audio_asset_id": "<asset_id of the recording to convert, from POST /v1/uploads>",
            "voice": "<voice name, from GET /v1/rvc/voices>",
            "f0_method": "crepe",
            "index_rate": 0.85,
        },
    },
}

BATCH_CONVERT = f"""
Converts every recording in `sources` to the same pre-installed voice, with
the same settings, as a **single job** -- one queue wait and one node
allocation for the whole set, instead of submitting `POST /v1/rvc/convert`
once per file. Same underlying conversion as `/convert` otherwise (same
voice registry, same settings, same "keeps the original delivery" voice-
conversion behavior, not text-to-speech) -- this endpoint only changes how
many files one job handles.

**Pipeline:** every source is first decoded to a WAV (see the
accepted-formats note below), then converted one at a time through the
same RVC v2 pipeline `/convert` uses, with an extra verification step that
confirms every expected output file actually exists before packaging
everything into one zip.

**Result: a single output.zip**, one converted file per source, each name
prefixed with that source's 1-based position in `sources` (e.g.
`001_<name>_output.<ext>`, `002_<name>_output.<ext>`, ...) so the mapping
back to your request is unambiguous. There is no per-file download --
`GET /v1/jobs/{{job_id}}/result` returns the whole zip.

**All-or-nothing.** If any one file in the batch fails to convert, the
whole job fails and no zip is produced -- there is no partial-success
result and no per-file status. For a batch large enough that this matters,
consider a few smaller batches instead of one very large one.

**Accepted input files:** {", ".join(sorted(ext.lstrip(".") for ext in config.INPUT_EXTENSIONS))}
-- the same set `POST /v1/rvc/convert` accepts, decoded to WAV before the
conversion pipeline sees them, so real-world recordings (including phone
voice notes) work directly. Every source is checked before the job is
submitted; a request containing any unsupported file is rejected
immediately (400, naming every offending source), not discovered partway
through a running batch.

**`export_format` is WAV/MP3/FLAC/OGG** -- the same set `/convert` offers.
M4A isn't offered on either endpoint: the WAV-to-M4A encoding step isn't
reliable on this deployment (no M4A encoder in the installed audio
library), and can silently produce a file that's named `.m4a` but actually
contains plain WAV bytes.

**Up to {config.BATCH_MAX_FILES} files per request.**

**At most one batch runs at a time on this deployment.** Submitting
several batches back to back is safe -- each is queued and will eventually
run -- but they execute one after another, never concurrently (the
underlying batch-conversion step isn't safe to run twice at once). Plan for
total throughput accordingly: three batches of 20 files each will take
roughly as long as one batch of 60.

**No live progress.** Unlike Wan-Animate's replace endpoint, a queued/
running batch job's `progress` field is always null -- poll
`GET /v1/jobs/{{job_id}}` for status only (queued/running/succeeded/failed),
same as `/convert`.

**Default time limit: {config.BATCH_JOB_TIME_LIMIT}** -- an unmeasured
guess since no batch has run on this deployment yet; revisit once real
batch timings exist. The default partition here is pre-emptible, and
losing a long-running batch to pre-emption means starting over from the
first file, not resuming -- for a large or important batch, consider
passing `partition: "main"` instead.
"""

BATCH_CONVERT_EXAMPLES = {
    "basic_batch": {
        "summary": "Convert several recordings to the same installed voice",
        "value": {
            "sources": [
                {"asset_id": "<asset_id of the 1st recording, from POST /v1/uploads>"},
                {"asset_id": "<asset_id of the 2nd recording, from POST /v1/uploads>"},
                {"asset_id": "<asset_id of the 3rd recording, from POST /v1/uploads>"},
            ],
            "voice": "<voice name, from GET /v1/rvc/voices>",
        },
    },
    "large_batch_on_main_partition": {
        "summary": "A larger batch, submitted to the non-preemptible partition",
        "value": {
            "sources": [
                {"asset_id": "<asset_id of the 1st recording, from POST /v1/uploads>"},
                {"asset_id": "<asset_id of the 2nd recording, from POST /v1/uploads>"},
            ],
            "voice": "<voice name, from GET /v1/rvc/voices>",
            "export_format": "MP3",
            "partition": "main",
        },
    },
}

# ---------------------------------------------------------------------------
# 400 responses -- RVC's own, richer than common_docs's generic
# _ASSET_ERROR_RESPONSE, because an unsupported source file type is a real,
# distinct, common 400 cause here that other backends don't have (every
# other backend's file inputs go through PIL/standard decoders with much
# broader native format support). Two named examples under the same 400
# status: a human/LLM reading /docs sees both causes at a glance.
# ---------------------------------------------------------------------------

CONVERT_RESPONSES = {
    **common_docs.SUBMIT_RESPONSES,
    400: {
        "description": "The referenced asset doesn't exist, or its file isn't one of the accepted formats.",
        "content": {
            "application/json": {
                "examples": {
                    "unknown_asset": {
                        "summary": "Unknown or missing asset_id",
                        "value": {"detail": "Invalid asset reference: 'Unknown asset_id: abc123'"},
                    },
                    "unsupported_format": {
                        "summary": "Unsupported source file type",
                        "value": {
                            "detail": (
                                f"Unsupported file extension '.xyz' (accepted: {_ACCEPTED_FORMATS_LIST}): "
                                "recording.xyz"
                            )
                        },
                    },
                }
            }
        },
    },
}

BATCH_CONVERT_RESPONSES = {
    **common_docs.SUBMIT_RESPONSES,
    400: {
        "description": (
            "One or more entries in `sources` is invalid (unknown asset_id, or an unsupported file "
            "format) -- every problem found is reported together, not just the first."
        ),
        "content": {
            "application/json": {
                "example": {
                    "detail": (
                        "2 of 3 source(s) rejected: source #2 (abc123): unknown asset_id; "
                        f"source #3 (recording.xyz): unsupported extension '.xyz' "
                        f"(accepted: {_ACCEPTED_FORMATS_LIST})"
                    )
                }
            }
        },
    },
}
