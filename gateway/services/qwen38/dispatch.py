"""Runs a Qwen3.8 request: unlike every other backend there is no Slurm job
of its own. Each request is an asyncio task inside the gateway process that

  1. makes sure the shared vLLM server is running and waits until it is ready
     (the job stays `queued` meanwhile),
  2. marks the job `running` and sends one chat completion to the server,
  3. writes output.json, or a .failed marker + error.txt, and marks the job
     finished.

Job rows from this backend have no slurm_job_id. The shared poller
(common/poller.py) therefore never asks Slurm about them; it still finishes a
job from output.json / .failed alone, which covers the case where this
process dies between writing the file and updating the database.
output_path.txt is written first for that reason.

A task lives only as long as the gateway process: fail_orphaned_jobs() fails
whatever a restart interrupted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from common import config as common_config
from common import job_store, retention
from services.qwen38 import config, server_manager
from services.qwen38.schemas import Qwen38GenerateRequest

logger = logging.getLogger("model-api.qwen38")

PIPELINE = "qwen38:generate"

_tasks: dict[str, asyncio.Task] = {}
_calls: dict[str, server_manager.ChatCall] = {}
_semaphore: asyncio.Semaphore | None = None
_loop: asyncio.AbstractEventLoop | None = None


class CapacityError(RuntimeError):
    """Too many jobs already in flight (see common.config.MAX_INFLIGHT_JOBS)."""


def _check_capacity() -> None:
    if (
        common_config.MAX_INFLIGHT_JOBS is not None
        and job_store.count_inflight_jobs() >= common_config.MAX_INFLIGHT_JOBS
    ):
        raise CapacityError(
            f"At capacity: {common_config.MAX_INFLIGHT_JOBS} job(s) already in flight. Try again shortly."
        )


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    return _semaphore


def build_payload(req: Qwen38GenerateRequest) -> dict:
    """The chat completion body for vLLM's OpenAI-compatible API."""
    payload: dict = {
        "model": config.SERVED_MODEL_NAME,
        "messages": [m.model_dump() for m in req.messages],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "chat_template_kwargs": {"enable_thinking": req.thinking},
    }
    if req.seed is not None:
        payload["seed"] = req.seed
    if req.json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": req.json_schema, "strict": True},
        }
    return payload


def _build_result(req: Qwen38GenerateRequest, answer: dict, server_wait_seconds: float, generation_seconds: float) -> dict:
    choice = answer["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content")
    parsed = None
    if req.json_schema is not None and content:
        try:
            parsed = json.loads(content)
        except ValueError:
            parsed = None  # cut off by max_tokens; finish_reason says so
    usage = answer.get("usage") or {}
    return {
        "content": content,
        "json": parsed,
        "reasoning": (message.get("reasoning") or message.get("reasoning_content")) if req.include_reasoning else None,
        "finish_reason": choice.get("finish_reason"),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "timings": {
            "server_wait_seconds": round(server_wait_seconds, 2),
            "generation_seconds": round(generation_seconds, 2),
        },
        "model": {"name": config.MODEL_ID, "revision": config.MODEL_REVISION, "precision": "fp8"},
    }


def _write_output(output_path: Path, result: dict) -> None:
    partial = output_path.with_name("output.partial.json")
    partial.write_text(json.dumps(result, ensure_ascii=False))
    os.replace(partial, output_path)


def _fail(job_id: str, job_dir: Path, message: str) -> None:
    try:
        (job_dir / "error.txt").write_text(message)
        (job_dir / ".failed").touch()
    except OSError:
        logger.warning("Could not write the failure marker for job %s", job_id)
    job_store.mark_failed(job_id, message)
    retention.on_job_failed(job_id)


async def _run(job_id: str, job_dir: Path, req: Qwen38GenerateRequest) -> None:
    output_path = job_dir / "output.json"
    server_manager.request_started()
    started = time.monotonic()
    try:
        await asyncio.to_thread(server_manager.ensure_server)
        await server_manager.wait_until_ready(config.STARTUP_TIMEOUT_SECONDS)
        async with _get_semaphore():
            job_store.mark_running(job_id)
            waited = time.monotonic() - started
            call = server_manager.ChatCall()
            _calls[job_id] = call
            generation_started = time.monotonic()
            answer = await asyncio.to_thread(call.run, build_payload(req))
            generation_seconds = time.monotonic() - generation_started
        _write_output(output_path, _build_result(req, answer, waited, generation_seconds))
        job_store.mark_succeeded(job_id, output_path)
        retention.on_job_succeeded(job_id)
    except asyncio.CancelledError:
        call = _calls.get(job_id)
        if call is not None:
            call.abort()
        raise  # DELETE /v1/jobs/{id} marks the job failed itself
    except server_manager.RequestRejected as e:
        _fail(job_id, job_dir, f"The model server rejected the request: {e}")
    except server_manager.ServerError as e:
        _fail(job_id, job_dir, str(e))
    except Exception:
        logger.exception("Qwen3.8 job %s failed unexpectedly", job_id)
        _fail(job_id, job_dir, "Unexpected error while running the request. See the gateway log.")
    finally:
        _calls.pop(job_id, None)
        server_manager.request_finished()


async def submit_generate(req: Qwen38GenerateRequest) -> str:
    """Validate capacity, create the job directory and row, and start the task. Returns the job id."""
    _check_capacity()
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "output_path.txt").write_text(str(job_dir / "output.json"))
    job_store.create_job(
        job_id,
        PIPELINE,
        job_dir,
        req.model_dump(mode="json"),
        config.PARTITION or common_config.FALLBACK_PARTITION,
        incognito=req.incognito,
    )
    global _loop
    _loop = asyncio.get_running_loop()
    task = _loop.create_task(_run(job_id, job_dir, req))
    _tasks[job_id] = task
    task.add_done_callback(lambda _t: _tasks.pop(job_id, None))
    return job_id


def cancel(job_id: str) -> None:
    """Stop a running or waiting request (its connection is closed, so vLLM stops generating).
    The shared server keeps running."""
    call = _calls.get(job_id)
    if call is not None:
        call.abort()
    task = _tasks.get(job_id)
    if task is not None and _loop is not None:
        # Called from a worker thread (DELETE /v1/jobs/{id} is a plain def), so hand the cancel to the loop.
        _loop.call_soon_threadsafe(task.cancel)


def fail_orphaned_jobs() -> None:
    """At gateway start: a request that was still waiting or running died with the old process.
    One whose output.json was already written is left for the poller to finish."""
    for row in job_store.list_inflight_jobs():
        if not row["pipeline"].startswith("qwen38:"):
            continue
        if (Path(row["job_dir"]) / "output.json").exists():
            continue
        job_store.mark_failed(row["job_id"], "The gateway restarted while this request was in progress. Submit it again.")
        retention.on_job_failed(row["job_id"])
        logger.info("Failed orphaned Qwen3.8 job %s", row["job_id"])
