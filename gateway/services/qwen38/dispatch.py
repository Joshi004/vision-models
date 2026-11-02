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

A task lives only as long as the gateway process: resume_orphaned_jobs() starts
again, from the request stored with the job, whatever a restart interrupted.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import os
import time
from pathlib import Path

from common import config as common_config
from common import capacity, dedup, job_store, retention
from services.qwen38 import config, server_manager
from services.qwen38.schemas import Qwen38GenerateRequest

logger = logging.getLogger("model-api.qwen38")

PIPELINE = "qwen38:generate"

# The one shared model server holds 1 GPU while it is up. It has no limit of its own on the
# capacity page (how many requests it serves at once is fixed when it starts); its GPU still
# counts toward the global limit.
capacity.register("qwen38", "Qwen3.8 language model", capacity.FIXED, planning_gpus=1, gpus_held=server_manager.gpus_held)

_tasks: dict[str, asyncio.Task] = {}
_calls: dict[str, server_manager.ChatCall] = {}
_gate: "_PriorityGate | None" = None
_loop: asyncio.AbstractEventLoop | None = None

# Pause before a request whose server was lost tries again, so the pool of attempts is not
# used up while the lost server is still being cleaned up.
RETRY_DELAY_SECONDS = 10.0


class CapacityError(RuntimeError):
    """Too many jobs already queued or running for this model (see common.config.MAX_QUEUED_PER_BACKEND)."""


class IdempotencyConflict(RuntimeError):
    """An Idempotency-Key that was already used for a different request."""


def _without_priority(request: dict) -> dict:
    return {key: value for key, value in request.items() if key != "priority"}


def _check_capacity() -> None:
    if job_store.count_inflight_for((PIPELINE,)) >= common_config.MAX_QUEUED_PER_BACKEND:
        raise CapacityError(
            f"This model already has {common_config.MAX_QUEUED_PER_BACKEND} job(s) queued or running. "
            "Try again shortly."
        )


class _PriorityGate:
    """Like an asyncio.Semaphore, except that when a slot frees up it goes to the most urgent waiting
    request instead of the one that has waited longest. A waiting request's priority is read from the
    job table at that moment, so a change made from the UI counts; requests of the same priority
    are served in the order they arrived."""

    def __init__(self, slots: int) -> None:
        self._free = slots
        self._arrivals = itertools.count()
        self._waiters: list[tuple[int, str, asyncio.Future]] = []  # (arrival number, job_id, future)

    @contextlib.asynccontextmanager
    async def slot(self, job_id: str):
        await self._acquire(job_id)
        try:
            yield
        finally:
            self._free += 1
            self._wake()

    async def _acquire(self, job_id: str) -> None:
        if self._free > 0 and not self._waiters:
            self._free -= 1
            return
        future = asyncio.get_running_loop().create_future()
        waiter = (next(self._arrivals), job_id, future)
        self._waiters.append(waiter)
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self._free += 1  # the slot was already handed to this request: give it back
                self._wake()
            elif waiter in self._waiters:
                self._waiters.remove(waiter)
            raise

    def _wake(self) -> None:
        while self._free > 0 and self._waiters:
            priorities = job_store.queued_priorities([job_id for _, job_id, _ in self._waiters])
            waiter = min(self._waiters, key=lambda w: (priorities.get(w[1], job_store.DEFAULT_PRIORITY), w[0]))
            self._waiters.remove(waiter)
            self._free -= 1
            waiter[2].set_result(None)


def _get_gate() -> _PriorityGate:
    global _gate
    if _gate is None:
        _gate = _PriorityGate(config.MAX_CONCURRENT_REQUESTS)
    return _gate


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
        # Waiting for a free slot comes first: only the few requests that hold one wait for the server
        # (which polls Slurm), however many more are queued behind them.
        async with _get_gate().slot(job_id):
            while True:
                await asyncio.to_thread(server_manager.ensure_server)
                await server_manager.wait_until_ready(config.STARTUP_TIMEOUT_SECONDS)
                job_store.mark_running(job_id)
                waited = time.monotonic() - started
                call = server_manager.ChatCall()
                _calls[job_id] = call
                generation_started = time.monotonic()
                try:
                    answer = await asyncio.to_thread(call.run, build_payload(req))
                    break
                except server_manager.ServerLost as e:
                    # The server went away under the request: not its fault, so try again (a new
                    # server is started if needed) until the attempts are used up.
                    if not job_store.requeue(job_id):
                        raise
                    logger.warning("Qwen3.8 job %s: %s Trying again.", job_id, e)
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
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


def _start_task(job_id: str, job_dir: Path, req: Qwen38GenerateRequest) -> None:
    global _loop
    _loop = asyncio.get_running_loop()
    task = _loop.create_task(_run(job_id, job_dir, req))
    _tasks[job_id] = task
    task.add_done_callback(lambda _t: _tasks.pop(job_id, None))


async def submit_generate(req: Qwen38GenerateRequest, client_key: str | None = None) -> dedup.Submission:
    """Validate capacity, create the job directory and row, and start the task. Returns the submission.

    Qwen3.8 answers are not repeatable, so a request is never matched by its content. With a
    `client_key` (the Idempotency-Key header) an earlier request with the same key is found again
    instead: its job is returned if it is still queued or running, or has succeeded and its answer
    is still there. The same key with a different request is refused (IdempotencyConflict). A
    request with `incognito: true` ignores the key."""
    body = req.model_dump(mode="json")
    dedup_key = dedup.idempotency_key(PIPELINE, client_key) if client_key and not req.incognito else None
    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    with dedup.lock:
        if dedup_key is not None:
            earlier_jobs = job_store.find_by_dedup_key(dedup_key)
            # `priority` is left out of the comparison: asking again with another priority is the
            # same request (dedup.reuse then raises the earlier job's priority if it is more urgent).
            if earlier_jobs and _without_priority(json.loads(earlier_jobs[0]["request_json"])) != _without_priority(body):
                raise IdempotencyConflict("This Idempotency-Key was already used for a different request.")
            earlier = dedup.pick_reusable(earlier_jobs)
            if earlier is not None:
                return dedup.reuse(earlier, req.priority)
        _check_capacity()
        job_dir.mkdir(parents=True)
        (job_dir / "output_path.txt").write_text(str(job_dir / "output.json"))
        job_store.create_job(
            job_id,
            PIPELINE,
            job_dir,
            body,
            config.PARTITION or common_config.FALLBACK_PARTITION,
            incognito=req.incognito,
            dedup_key=dedup_key,
        )
    _start_task(job_id, job_dir, req)
    return dedup.Submission(job_id, "queued")


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


async def resume_orphaned_jobs() -> None:
    """At gateway start: a request that was still waiting or running died with the old process.
    Each is started again from the request stored with its job, in the order they were submitted.
    One that was already running counts as another attempt, and fails once it has used them up;
    one whose output.json was already written is left for the poller to finish."""
    rows = [row for row in job_store.list_inflight_jobs() if row["pipeline"].startswith("qwen38:")]
    for row in sorted(rows, key=lambda r: r["created_at"]):
        job_id = row["job_id"]
        job_dir = Path(row["job_dir"])
        if (job_dir / "output.json").exists():
            continue
        if row["status"] == "running" and not job_store.requeue(job_id):
            _fail(
                job_id,
                job_dir,
                f"The gateway restarted while this request was in progress, and it was already tried "
                f"{row['attempts']} time(s). Submit it again.",
            )
            logger.info("Failed orphaned Qwen3.8 job %s (attempts used up)", job_id)
            continue
        try:
            req = Qwen38GenerateRequest(**json.loads(row["request_json"]))
        except Exception:
            _fail(job_id, job_dir, "The gateway restarted and this request could not be read back. Submit it again.")
            logger.exception("Could not resume Qwen3.8 job %s", job_id)
            continue
        _start_task(job_id, job_dir, req)
        logger.info("Resumed Qwen3.8 job %s after a restart", job_id)
