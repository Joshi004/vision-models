"""Long-form OpenAPI/Swagger documentation text for the Qwen3.8 text
endpoints, kept out of router.py so the routes stay readable. Docs shared by
every backend live in common/openapi_docs.py.
"""

from __future__ import annotations

from common import openapi_docs as common_docs
from services.qwen38 import config

GENERATE = f"""
Sends a prompt to a text model (**Qwen3.8-27B**, FP8) and returns its answer. Text in, text or JSON out:
there is no image, audio or video input. Use it for planning, rewriting, classification, extraction -- anything
where the answer is text. It is a reasoning model: with `thinking: true` (the default) it works the problem out
before it answers, which helps on hard planning and costs time and tokens.

**Structured output.** Pass a JSON Schema in `json_schema` and the answer is *forced* to match it: the result's
`json` field is always parseable and has the fields you asked for. The one exception is an answer cut off by
`max_tokens` (`finish_reason: "length"`; `json` is then `null`). Reasoning tokens count toward `max_tokens`, so give
a thinking request room (the default {config.DEFAULT_MAX_TOKENS:,}, at most {config.MAX_OUTPUT_TOKENS:,}). A long
answer with many parts (for example a dozen or more scenes, each with several prompts) can use well over 20,000
tokens with thinking on, most of it reasoning: send a high `max_tokens` for such requests. Exact length rules in the
prompt ("80 to 180 words") make the model count words in its reasoning and use far more tokens; describe lengths as
approximate where you can.

**A shared model server, not one GPU job per request.** One server holds the model on one GPU and answers every
request. It is started by the first request that needs it and stops by itself after
{config.IDLE_MINUTES:g} minutes without one (see `GET /v1/qwen38/server`). So the first request after a quiet period
waits for the server to start -- the job stays `queued` meanwhile -- while later requests are answered in seconds.
`POST /v1/qwen38/server/warm` starts the server ahead of time. Several requests run on the server at the same time.

**The result is a JSON file** -- download it from `GET /v1/jobs/{{job_id}}/result` once `GET /v1/jobs/{{job_id}}`
reports `status: "succeeded"`:
```json
{{
  "content": "{{\\"title\\": \\"Trail Running Shoe\\", \\"shots\\": [...]}}",
  "json": {{"title": "Trail Running Shoe", "shots": ["..."]}},
  "reasoning": null,
  "finish_reason": "stop",
  "usage": {{"prompt_tokens": 80, "completion_tokens": 376, "total_tokens": 456}},
  "timings": {{"server_wait_seconds": 0.1, "generation_seconds": 6.3}},
  "model": {{"name": "Qwen/Qwen3.8-27B-FP8", "revision": "...", "precision": "fp8"}}
}}
```
- `content`: the answer text (without the reasoning). `null` if the model used all of `max_tokens` thinking.
- `json`: `content` parsed, only when `json_schema` was given and the answer is complete; otherwise `null`.
- `reasoning`: the model's reasoning text, only when `include_reasoning` is true (and `thinking` is on).
- `finish_reason`: `"stop"` (finished) or `"length"` (cut off by `max_tokens`).
- `timings.server_wait_seconds`: time from submit until the request reached the server, mostly server start-up for a
  cold server. `timings.generation_seconds`: time the model took.

**A request the model server refuses fails the job** with its message in `error` (for example a `json_schema` the
server cannot use, or a prompt too long for the {config.MAX_MODEL_LEN:,}-token window). Total message length is
limited to {config.MAX_PROMPT_CHARS:,} characters as a rough guard.

**Timing** (measured on this cluster; see `typical_run_seconds` on `GET /v1/jobs/{{job_id}}` for a live estimate):
a warm server answers a short request in a few seconds and generates about 75 tokens per second, so a 16,000-token
reasoning answer takes about 3.5 minutes and a full {config.MAX_OUTPUT_TOKENS:,}-token one about 11; a cold start
takes several minutes (queue, loading about 28 GiB of weights, one-time kernel compilation). The pre-emptible
`background` partition can stop the server at any moment, which fails the requests in flight; the next request
starts a new server.

The `incognito` option works as everywhere else. The model server itself keeps no copy of your prompts or answers
(request logging is off).
"""

GENERATE_EXAMPLES = {
    "free_text": {
        "summary": "Free-text answer, no reasoning",
        "value": {
            "messages": [{"role": "user", "content": "Rewrite this as one vivid sentence: a dog runs on a beach."}],
            "thinking": False,
            "max_tokens": 300,
        },
    },
    "scene_plan_json": {
        "summary": "Scene plan as JSON that follows a schema",
        "value": {
            "messages": [
                {"role": "system", "content": "You plan short video scenes. Answer only with JSON."},
                {"role": "user", "content": "A 12-second ad for a trail running shoe. 3 shots."},
            ],
            "json_schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "shots"],
                "properties": {
                    "title": {"type": "string"},
                    "shots": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["index", "duration_seconds", "description"],
                            "properties": {
                                "index": {"type": "integer"},
                                "duration_seconds": {"type": "number"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "thinking": True,
            "max_tokens": 4096,
        },
    },
}

_GENERATE_422_AND_429 = {
    422: common_docs.SUBMIT_RESPONSES[422],
    429: common_docs.SUBMIT_RESPONSES[429],
}

GENERATE_RESPONSES = dict(_GENERATE_422_AND_429)

SERVER = f"""
State of the shared model server behind `POST /v1/qwen38/generate`: `stopped` (no server; the next request starts
one), `starting` (queued on the cluster or loading the model) or `ready`. While it is `ready` and idle,
`idle_shutdown_at` says when it stops itself ({config.IDLE_MINUTES:g} minutes after the last request). Read-only and
cheap to poll.
"""

WARM = """
Starts the model server now if it is not already running, so a request sent a few minutes later is answered at once
instead of waiting for the start-up. Returns the server's state straight away (usually `starting`); it does not wait
until the server is ready. Calling it while the server is running only postpones its idle shutdown.
"""

WARM_RESPONSES = {
    502: {
        "description": "The cluster's job scheduler (Slurm) refused to start the server.",
        "content": {
            "application/json": {
                "example": {"detail": "Could not start the model server (sbatch failed): <scheduler error detail>"}
            }
        },
    },
}
