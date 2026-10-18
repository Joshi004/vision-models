## Qwen3.8 text model (text and JSON)

**What this is for:** a general text model (Qwen3.8-27B, FP8) for reasoning over text only: planning
scenes, writing shot or prompt descriptions, rewriting, classifying, extracting. It takes no image,
audio or video input. Give it a `json_schema` and the answer is always JSON that follows that schema.

**Use this when** you need a decision or a piece of text written from text, for example turning a
one-line idea into a list of scenes before you call a video backend. For anything that must look at
pictures or video, this is the wrong endpoint.

### How it differs from the other backends

Every other backend starts a Slurm job per request. This one keeps **one shared model server** (one
H100) running and sends each request to it, so a request is answered in seconds once the server is up.

- The server starts on demand. If it is stopped, your job stays `queued` while the server starts:
  **3 to 10 minutes** (measured; Slurm queue wait not included).
- The server stops itself after **20 minutes without a request** (an operator setting) and starts again
  on the next one. To avoid the wait, call `POST /v1/qwen38/server/warm` ahead of time.
- `GET /v1/qwen38/server` tells you the state: `stopped`, `starting` or `ready`.
- Requests are still normal jobs: submit, poll `GET /v1/jobs/{job_id}`, download
  `GET /v1/jobs/{job_id}/result`. The job goes `running` when the server picks the request up, so
  `queued` time includes any server start. `DELETE /v1/jobs/{job_id}` cancels a request without
  stopping the server.
- Several requests run at the same time on the one server (up to 8).
  More wait in `queued`.
- The gateway holds the waiting requests in memory. If the gateway restarts, requests that were still
  waiting or running fail with a message saying so; submit them again.

### Request

`POST /v1/qwen38/generate` with:

- `messages` (required): chat messages, oldest first. Optional `system` first, last one must be `user`.
- `json_schema` (optional): a JSON Schema object the answer must follow.
- `thinking` (default `true`): let the model reason first. Better for hard planning, slower, and the
  reasoning counts toward `max_tokens`. Use `false` for quick direct answers.
- `include_reasoning` (default `false`): return the reasoning text in the result.
- `max_tokens` (default 4096, at most 49,152): includes reasoning. If it is reached, `finish_reason` is
  `"length"` and the answer is cut off (with a schema, `json` is then `null`; if the reasoning used every token,
  `content` is empty).

**Long answers need a high `max_tokens`.** With thinking on, the model often drafts the whole answer in its
reasoning and checks it before writing it out. A 17-scene plan with first frame, last frame and video prompt per
scene used all of 27,000 tokens on reasoning alone; send 40,000 or more for requests like that, or split them into
smaller batches of scenes. Exact length rules ("80 to 180 words") make the model count words one by one in its
reasoning; say "about 120 words" instead. Prefer sending a `json_schema` for structured answers: with thinking off
and no schema, the same request once stopped after the first JSON block.
- `temperature` (0.6), `top_p` (0.95), `seed` (optional).

### Result

The downloaded result is JSON:

```json
{
  "content": "{\"scenes\": [...]}",
  "json": {"scenes": []},
  "reasoning": null,
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 61, "completion_tokens": 376, "total_tokens": 437},
  "timings": {"server_wait_seconds": 0.1, "generation_seconds": 6.3},
  "model": {"name": "Qwen/Qwen3.8-27B-FP8", "revision": "...", "precision": "fp8"}
}
```

`content` is the raw answer text. `json` is `content` parsed, set only when you sent a `json_schema`
and the answer was complete. `server_wait_seconds` is how long the request waited for the server.

### Workflow

**curl:**
```bash
BASE="http://localhost:8012"

# 1. Submit (add "json_schema" to force a JSON shape)
JOB_ID=$(curl -s -X POST "$BASE/v1/qwen38/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You plan video scenes. Answer only with JSON."},
      {"role": "user", "content": "A 20-second ad for a trail running shoe."}
    ],
    "json_schema": {
      "type": "object",
      "properties": {
        "scenes": {"type": "array", "items": {
          "type": "object",
          "properties": {"seconds": {"type": "number"}, "description": {"type": "string"}},
          "required": ["seconds", "description"]
        }}
      },
      "required": ["scenes"]
    }
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# 2. Poll until status is succeeded or failed
curl -s "$BASE/v1/jobs/$JOB_ID"

# 3. Download
curl -s "$BASE/v1/jobs/$JOB_ID/result" -o answer.json
```

**Python:**
```python
import time
import requests

BASE = "http://localhost:8012"

job_id = requests.post(f"{BASE}/v1/qwen38/generate", json={
    "messages": [{"role": "user", "content": "Name three colours."}],
    "thinking": False,
}).json()["job_id"]

while True:
    status = requests.get(f"{BASE}/v1/jobs/{job_id}").json()
    if status["status"] in ("succeeded", "failed"):
        break
    time.sleep(3)   # seconds-long answers once the server is up; the first request may wait minutes

if status["status"] == "failed":
    raise RuntimeError(status["error"])

result = requests.get(f"{BASE}/v1/jobs/{job_id}/result").json()
print(result["content"])      # or result["json"] when you sent a json_schema
```

Poll every 2-3 seconds for this backend once the server is `ready` (answers take seconds); poll every
10-15 seconds while it is `starting`.

### Notes

- With `thinking: true` and a `json_schema`, the model reasons first and the final answer still
  follows the schema.
- Keep schemas simple: objects, arrays, strings, numbers, enums, `required`. A very large or deeply
  nested schema slows the first token.
- Prompt and messages together are limited to 32,000 characters.
- The model is FP8 (about 29 GB of weights). Its answers are in line with the full-precision model.
- Measured on one H100 with the server up: most answers take 1 to 6 seconds, a long reasoned plan about 20.
