# Qwen3.8-27B-FP8: text and JSON backend

A text-only reasoning model behind `POST /v1/qwen38/generate`. Planning scenes, writing shot
descriptions, classifying, extracting; with an optional JSON Schema the answer always matches it. It
takes no image, audio or video input (the server runs with `--language-model-only`).

Callers read `gateway/services/qwen38/GUIDE.md` (or `GET /v1/guide`). This page is for operators.

## How it runs

Unlike every other backend there is **no Slurm job per request**. One shared vLLM server runs as a
Slurm job on one H100 and every request is sent to it.

1. A request arrives. The gateway creates a normal job (`queued`) and starts an asyncio task for it.
2. The task makes sure a server job exists (`server_manager.ensure_server`): if none, it writes a fresh
   API key and submits `models/qwen38/runner/serve.sh` with `sbatch`. It then waits until the server
   says it is ready.
3. The job goes `running`, the task sends one chat completion to the server, writes `output.json`, and
   marks the job `succeeded` (or `failed` with the server's error).
4. After `QWEN38_IDLE_MINUTES` (default 20) with no request in flight, the gateway's idle watcher
   `scancel`s the server job.

State lives in `$VISION_DATA_DIR/llm/qwen38/`:

| File | Meaning |
|---|---|
| `api_key` | Random key (mode 0600) the server requires. The compute node's port is not exposed beyond the cluster, the key stops other cluster users from using it. |
| `slurm_job_id` | The server job the gateway started. |
| `server.json` | Written by `serve.sh` when `/health` answers: `host`, `port`, `slurm_job_id`, `ready_at`. The server is "ready" only if this names the current Slurm job; `serve.sh` removes it on exit. |
| `server.log` | vLLM's own log. First place to look when a start fails. |

`GET /v1/qwen38/server` reports `stopped` / `starting` / `ready`; `POST /v1/qwen38/server/warm`
starts the server without a prompt.

If the gateway restarts, requests that were in flight fail ("gateway restarted ... submit again").
The server job itself keeps running and is picked up again (its files are on disk).

Cancelling a job (`DELETE /v1/jobs/{id}`) closes its connection to the server, so vLLM stops
generating; the server stays up.

## Setup

```bash
scripts/build_env.sh qwen38           # envs/qwen38 from models/qwen38/env/requirements.lock.txt
models/qwen38/download_weights.sh     # ~29 GB into weights/qwen38/Qwen3.8-27B-FP8 (pinned revision)
```

Restart the gateway (`gateway/run.sh`). The settings are the `QWEN38_*` variables listed in
`gateway/run.sh` and `vision.env.example`. **Set `QWEN38_PARTITION` to a partition that cannot
pre-empt jobs** for steady use: on the default fallback partition (`background`) the server can be
killed at any time; the next request then starts a new one, and requests in flight fail.

### Why vLLM 0.26.0

The node driver supports CUDA 12.8 at most, so torch is `2.11.0+cu128`. vLLM 0.28 and later need a
newer torch than the cu128 index has. vLLM 0.26.0 is installed from its CUDA 12.9 GitHub wheel (it
runs on the 12.8 driver; the wheel only declares a `torchcodec>=0.14` that is not published for
cu128, so `torchcodec` is pinned to 0.11.1 in the lock, with a header comment saying so).
`pip check` warns about that torchcodec pin; it is expected and unused (no video input).

### Server flags (`serve.sh`)

`--reasoning-parser qwen3 --language-model-only --limit-mm-per-prompt '{"image":0,"video":0}'
--max-model-len 65536 --max-num-seqs 8 --gpu-memory-utilization 0.90`. Thinking is switched per
request with `chat_template_kwargs.enable_thinking`; JSON Schema with `response_format:
json_schema` (works with thinking on). The reasoning comes back in `message.reasoning`.

## FP8 vs full precision

The FP8 checkpoint is the model's own published quantization (fine-grained block FP8). Published
and community benchmarks put the quality loss at noise level against BF16 (a fraction of a point on
reasoning and instruction benchmarks). This was **not re-measured here**; if you rely on a task
where exact wording matters, compare a few of your own prompts. Weights are ~29 GB, so one H100
(80 GB) holds them with ~40 GiB of KV cache (about 612k tokens) at 0.90 utilization.

## Measured on this cluster

One H100 on the `main` partition, 2026-10-18. Numbers come from real requests through the gateway.

**Server start (submit to ready):**

| Start | Time |
|---|---|
| Bring-up run, first start ever | about 9.5 minutes (engine init 437 s, of which ~3 min is DeepGEMM warm-up and 41 s torch.compile; weights loaded in 12 s from page cache) |
| Second start, first request through the gateway | 571 s (engine init 405 s, torch.compile 35 s) |
| Third start | about 5.5 minutes |
| Fourth start | about 2.6 minutes (13:31:54 to 13:34:31) |

The starts got faster each time, probably because of the compile and kernel caches in `cache/vllm` and the node's page cache; this was not isolated. Plan for **3 to 10 minutes** for a request that arrives while the server is stopped,
plus any Slurm queue wait. `POST /v1/qwen38/server/warm` hides it.

**Answers with the server warm** (8 requests, queue-to-done including the gateway's 1 s job poll):

| What | Tokens | Time |
|---|---|---|
| Tiny prompt, thinking off | 8 | 0.2 s generating, ~1 s end to end |
| JSON Schema, thinking off | 333 | 5.4 s generating |
| JSON Schema, thinking on (4 scenes) | 291 | 3.8 s generating (reasoning returned with `include_reasoning`) |
| Free text, thinking on | 125 | 1.7 s |
| JSON Schema, thinking on, 10 scenes | 1443 | 18.6 s |

**A long client request (0.16.1, 2026-10-18).** A 17-scene MiniMax H3 plan (5,471-token prompt with exact
word ranges per field, thinking on, no `json_schema`) hit the old 16,384 cap four times with an empty answer: the
model drafts the whole JSON in its reasoning and counts words one by one (65 "Count:" passes). At 27,000 it was
still reasoning. With `max_tokens` 49,152 it finished: 45,334 tokens (about 39k of them reasoning), 624 s of
generation, valid JSON with all 17 scenes and every field inside its word range. With thinking off and no schema the
same request stopped after the `"world"` block (282 tokens). Telling the model not to count words cut the counting
but still used all 16,384 tokens. So the limit went to 49,152 with a 65,536 window and a 30-minute request timeout.
That request used 92% of the new cap, so a rerun can still go over.

Generation ran at about 72-78 tokens/s for a single request. Same `seed` gave the same answer twice.
`max_tokens` reached gives `finish_reason: "length"`.

**Behaviours checked:**

- Cancel (`DELETE /v1/jobs/{id}`) while generating: the job fails with "Cancelled by client request"
  and vLLM drops to 0 running requests within seconds; the server stays up.
- Idle shutdown: with `QWEN38_IDLE_MINUTES=1` the server job was cancelled by the gateway about one
  minute after the last request.
- Gateway restart with the server running: the server is picked up again, no new start. A request
  that was in flight fails with the "gateway restarted" message.
- Two defects found and fixed while testing: cancel did not stop vLLM (closing the socket from
  another thread was not enough; it is now shut down first), and the idle watcher could cancel a server
  that was still starting.
- Not tested: pre-emption of the server on the `background` partition, and several requests in
  parallel beyond one at a time.

## Not covered

- Images, audio, video as input.
- Streaming tokens (a request is one job with one result file).
- More than one GPU (a 27B FP8 model fits one H100).
