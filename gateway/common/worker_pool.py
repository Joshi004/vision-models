"""Warm worker pools: a few long-lived Slurm jobs per backend that each load a
model once and then serve queued requests one at a time.

Used by the Qwen-Image-2.1 and MiniMax H3 backends instead of "one cold Slurm
job per request". The gateway is the only process that decides anything:

  * A request is put in an in-memory queue (enqueue). It has no Slurm job of its own.
  * Whenever a worker of a matching kind is idle, the oldest matching request is
    written into that worker's inbox.json and the job's slurm_job_id is set to
    the worker's Slurm job, so the shared poller (common/poller.py) and
    DELETE /v1/jobs/{id} keep working unchanged.
  * When there are more queued requests than workers still starting, and the pool
    is below its cap, another worker is submitted. A request therefore never waits
    longer than it would for its own cold job.
  * A worker that stays idle for idle_seconds is cancelled; so is a worker that is
    still starting once no request needs it.

The worker side of the file protocol is models/common/pool_worker.py. Files, all
under <DATA_DIR>/pools/<pool name>/<worker id>/:

  worker.json   written here after sbatch (slurm_job_id, key, partition, gpus);
                lets a restarted gateway find its workers again
  inbox.json    written here: {"seq", "job_id", "job_dir"}; a request is new to the worker
                when its seq is higher than the last one the worker handled
  state.json    written by the worker: {"state": starting|loading|idle|busy|exiting,
                "job_id", "seq", "started_at", ...}
  Both are created before the worker starts and only ever rewritten in place: see
  _write_json_inplace for why (NFS caching).
  worker.log    the Slurm job's output (worker events only; a request's own output
                goes to its job folder's run.log)

A request that was queued when the gateway stopped is found again from the
pool_request.json that enqueue writes into its job folder.

Everything here that touches Slurm or files is blocking; async callers run it
with asyncio.to_thread. Stdlib only, like the rest of common/.
"""

from __future__ import annotations

import asyncio
import getpass
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from common import config as common_config
from common import job_store, retention
from common.slurm import cancel_slurm_job

logger = logging.getLogger("model-api.pool")

POOLS_DIR = common_config.DATA_DIR / "pools"

# squeue states that mean the worker's Slurm job still exists.
_LIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING"}

# A worker is not given a request that could run past the end of its own Slurm time limit.
TIME_MARGIN_SECONDS = 60
# Workers that die before ever becoming ready, in a row, before queued requests are failed
# (instead of submitting worker after worker that all fail the same way).
MAX_START_FAILURES = 3
# How often the watch loop looks at Slurm and the worker state files.
TICK_SECONDS = 5.0
# squeue must miss a worker this many times in a row before it is treated as gone.
_MISSES_BEFORE_GONE = 2
# Old worker folders (their worker.log) are kept this long after the worker is gone.
_WORKER_DIR_KEEP_SECONDS = 24 * 3600

_REQUEST_FILE = "pool_request.json"
_ASSIGNMENT_FILE = "pool_assignment.json"


class PoolError(RuntimeError):
    """A worker could not be submitted, or the request can never be served."""


@dataclass(frozen=True)
class PoolRequest:
    """What the pool needs to know about one queued request."""

    job_id: str
    key: str  # requests with different keys never share a worker (e.g. image mode, H3 transformer checkpoint)
    partition: str
    gpus: int  # a worker with at least this many GPUs can serve it
    limit_seconds: int  # longest the request may run once a worker has it

    def to_json(self) -> dict:
        return {
            "job_id": self.job_id,
            "key": self.key,
            "partition": self.partition,
            "gpus": self.gpus,
            "limit_seconds": self.limit_seconds,
        }


@dataclass
class Worker:
    worker_id: str
    slurm_job_id: str
    key: str
    partition: str
    gpus: int
    dir: Path
    submitted_at: float
    squeue_state: str = "PENDING"
    state: dict = field(default_factory=dict)  # the last state.json read
    current_job: str | None = None  # the request given to this worker and not yet finished
    assigned_at: float = 0.0
    limit_seconds: int = 0
    jobs_served: int = 0
    seq: int = 0  # sequence number of the last request written to the inbox
    idle_since: float | None = None
    ever_ready: bool = False
    running_since: float | None = None
    misses: int = 0
    retire: bool = False  # too little Slurm time left for the next request

    def matches(self, request: PoolRequest) -> bool:
        return self.key == request.key and self.partition == request.partition and self.gpus >= request.gpus

    def is_free(self) -> bool:
        """Idle and done with the request it was given. Both halves must hold: state.json can lag
        behind a fresh assignment (still showing the previous job), and only the worker's own
        report that it finished *this* job makes it free again."""
        if self.squeue_state != "RUNNING" or self.state.get("state") != "idle":
            return False
        return self.current_job is None or self.state.get("job_id") == self.current_job

    def is_starting(self) -> bool:
        return self.state.get("state") not in ("idle", "busy", "exiting")


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json_atomic(path: Path, data: dict) -> None:
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(data))
    os.replace(partial, path)


def _write_json_inplace(path: Path, data: dict) -> None:
    """Rewrite a file that the other side polls, without creating, renaming or deleting anything.

    The data directory is on NFS, whose client caches a failed lookup ("no such file") for up to a
    minute and can keep serving a replaced file's old version: measured, a worker that polled for an
    inbox file the gateway had just created picked it up 30-60 s late. Opening a file that already
    exists always re-checks it, so the polled files are created before the worker starts and only
    ever rewritten here, in one write (a reader that catches it half-written fails to parse and
    simply looks again)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, json.dumps(data).encode())
    finally:
        os.close(fd)


def _squeue_states() -> dict[str, str] | None:
    """Slurm job id -> state, for every job of this user; None if squeue could not answer.
    `-a` is required: without it squeue hides the jobs of hidden partitions such as this
    cluster's `background` (measured: the plain listing was empty while the jobs were running)."""
    try:
        result = subprocess.run(
            ["squeue", "-a", "-h", "-u", getpass.getuser(), "-o", "%i %T"], capture_output=True, text=True, timeout=20
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("squeue failed while checking the workers: %s", e)
        return None
    if result.returncode != 0:
        logger.warning("squeue exited %d: %s", result.returncode, result.stderr.strip())
        return None
    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            states[parts[0]] = parts[1]
    return states


def _squeue_job_state(slurm_job_id: str) -> str | None:
    """One job's Slurm state, "" if Slurm has no such job any more, None if squeue could not answer.
    Asked before a worker is given up on, so a wrong or partial list can never make the pool forget a
    live worker (which would leave its GPU job running outside the pool's cap)."""
    try:
        result = subprocess.run(
            ["squeue", "-a", "-h", "-j", slurm_job_id, "-o", "%T"], capture_output=True, text=True, timeout=20
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("squeue failed for worker job %s: %s", slurm_job_id, e)
        return None
    if result.returncode == 0:
        return result.stdout.strip().split("\n")[0].strip()
    if "Invalid job id" in result.stderr:
        return ""
    logger.warning("squeue exited %d for job %s: %s", result.returncode, slurm_job_id, result.stderr.strip())
    return None


def _log_tail(path: Path, max_lines: int = 15, max_chars: int = 1500) -> str:
    try:
        lines = [line for line in path.read_text(errors="replace").splitlines() if line.strip()]
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])[-max_chars:]


_POOLS: list["WorkerPool"] = []


class WorkerPool:
    """The workers and the request queue of one backend.

    build_sbatch_command(worker_id, worker_dir, key, partition, gpus) returns the whole `sbatch`
    command (including --parsable) that starts one worker; the worker must run
    models/common/pool_worker.py's loop on worker_dir.
    """

    # How long a worker may take to load its model once Slurm has started it. Generous: a cold
    # H3 load took up to ~10 minutes in the usage log.
    startup_timeout_seconds = 30 * 60

    def __init__(
        self,
        name: str,
        *,
        pipelines: tuple[str, ...],
        max_workers: int,
        idle_seconds: float,
        worker_time_seconds: int,
        build_sbatch_command: Callable[[str, Path, str, str, int], list[str]],
    ) -> None:
        self.name = name
        self.pipelines = pipelines
        self.max_workers = max(1, max_workers)
        self.idle_seconds = idle_seconds
        self.worker_time_seconds = worker_time_seconds
        self._build_sbatch_command = build_sbatch_command
        self.dir = POOLS_DIR / name
        self._lock = threading.RLock()
        self._queue: list[PoolRequest] = []
        self._workers: dict[str, Worker] = {}
        self._start_failures = 0
        self._last_failure_log = ""
        _POOLS.append(self)

    # --- public ------------------------------------------------------------

    def enqueue(self, request: PoolRequest, job_dir: Path) -> None:
        """Queue a request whose job row and job folder already exist. Raises PoolError if no worker
        can be started for it (the request is then not queued)."""
        if request.limit_seconds + TIME_MARGIN_SECONDS > self.worker_time_seconds:
            raise PoolError(
                f"The worker time limit ({self.worker_time_seconds} s) is shorter than this request's "
                f"limit ({request.limit_seconds} s); raise the worker time setting."
            )
        _write_json_atomic(job_dir / _REQUEST_FILE, request.to_json())
        with self._lock:
            # The worker files only (no squeue call): a worker that finished a request a moment ago
            # must count as free, or this request would start a new worker for nothing.
            self._refresh(check_slurm=False)
            self._queue.append(request)
            failures = self._schedule()
        own = None
        for failed_request, message in failures:
            if failed_request.job_id == request.job_id:
                own = message
            else:
                self._fail_request(failed_request, message)
        if own is not None:
            raise PoolError(own)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a request of this pool. A queued one is just removed; one a worker is running stops
        that worker (its loaded weights are lost). False if the pool does not hold the job."""
        with self._lock:
            for request in self._queue:
                if request.job_id == job_id:
                    self._queue.remove(request)
                    return True
            for worker in list(self._workers.values()):
                if worker.current_job == job_id:
                    self._stop_worker(worker, f"request {job_id} cancelled")
                    return True
        return False

    def owns_slurm_job(self, slurm_job_id: str) -> bool:
        with self._lock:
            return any(w.slurm_job_id == slurm_job_id for w in self._workers.values())

    def recover(self) -> None:
        """At gateway start: find the workers that are still running and the requests still queued."""
        with self._lock:
            states = _squeue_states()
            now = time.time()
            if self.dir.is_dir():
                for worker_dir in sorted(self.dir.iterdir()):
                    self._recover_worker(worker_dir, states, now)
            queued = [
                row
                for row in job_store.list_inflight_jobs()
                if row["pipeline"] in self.pipelines and row["status"] == "queued" and not row["slurm_job_id"]
            ]
            for row in sorted(queued, key=lambda r: r["created_at"]):
                data = _read_json(Path(row["job_dir"]) / _REQUEST_FILE)
                try:
                    request = PoolRequest(
                        job_id=data["job_id"],  # type: ignore[index]
                        key=data["key"],  # type: ignore[index]
                        partition=data["partition"],  # type: ignore[index]
                        gpus=int(data["gpus"]),  # type: ignore[index]
                        limit_seconds=int(data["limit_seconds"]),  # type: ignore[index]
                    )
                except (TypeError, KeyError, ValueError):
                    self._fail_job_id(row["job_id"], "The gateway restarted before this request was queued. Submit it again.")
                    continue
                self._queue.append(request)
            logger.info("Pool %s recovered %d worker(s) and %d queued request(s).", self.name, len(self._workers), len(self._queue))
            failures = self._schedule()
        for request, message in failures:
            self._fail_request(request, message)

    def tick(self) -> None:
        """One pass of the watch loop: read Slurm and the worker files, enforce limits, assign, scale."""
        with self._lock:
            self._refresh()
            self._enforce_limits()
            failures = self._schedule()
        for request, message in failures:
            self._fail_request(request, message)

    async def watch_forever(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.tick)
            except Exception:
                logger.exception("Pool %s check failed", self.name)
            await asyncio.sleep(TICK_SECONDS)

    # --- recovery ----------------------------------------------------------

    def _recover_worker(self, worker_dir: Path, states: dict[str, str] | None, now: float) -> None:
        meta = _read_json(worker_dir / "worker.json")
        if meta is None:
            return
        slurm_job_id = str(meta.get("slurm_job_id", ""))
        squeue_state = states.get(slurm_job_id, "") if states is not None else "RUNNING"
        if squeue_state not in _LIVE_STATES:
            confirmed = _squeue_job_state(slurm_job_id)  # not in the list: ask about this job itself
            squeue_state = "RUNNING" if confirmed is None else confirmed
        if squeue_state not in _LIVE_STATES:
            try:
                if now - worker_dir.stat().st_mtime > _WORKER_DIR_KEEP_SECONDS:
                    shutil.rmtree(worker_dir, ignore_errors=True)
            except OSError:
                pass
            return
        worker = Worker(
            worker_id=worker_dir.name,
            slurm_job_id=slurm_job_id,
            key=str(meta.get("key", "")),
            partition=str(meta.get("partition", "")),
            gpus=int(meta.get("gpus", 1)),
            dir=worker_dir,
            submitted_at=float(meta.get("submitted_at", now)),
            squeue_state=squeue_state,
            state=_read_json(worker_dir / "state.json") or {},
        )
        worker.ever_ready = worker.state.get("state") in ("idle", "busy")
        # The request this worker was given: one in the inbox it has not started yet (its sequence number
        # is newer than the last one the worker reports), or the one it reports as running.
        inbox = _read_json(worker_dir / "inbox.json") or {}
        worker.seq = int(inbox.get("seq", 0))
        worker.jobs_served = int(worker.state.get("seq", 0))  # requests it has started: it is warm if any
        if worker.seq > int(worker.state.get("seq", 0)):
            worker.current_job = inbox.get("job_id")
        elif worker.state.get("state") == "busy":
            worker.current_job = worker.state.get("job_id")
        if worker.current_job:
            request = _read_json(common_config.JOBS_DIR / worker.current_job / _REQUEST_FILE) or {}
            worker.assigned_at = now
            worker.limit_seconds = int(request.get("limit_seconds", self.worker_time_seconds))
        elif worker.state.get("state") == "idle":
            worker.idle_since = now
        self._workers[worker.worker_id] = worker

    # --- the scheduling pass -----------------------------------------------

    def _refresh(self, check_slurm: bool = True) -> None:
        states = _squeue_states() if check_slurm else None
        now = time.time()
        for worker in list(self._workers.values()):
            if states is not None:
                squeue_state = states.get(worker.slurm_job_id, "")
                if squeue_state not in _LIVE_STATES:
                    squeue_state = _squeue_job_state(worker.slurm_job_id)  # confirm before giving up on it
                    if squeue_state is None:
                        continue  # cannot tell right now
                if squeue_state not in _LIVE_STATES:
                    worker.misses += 1
                    if worker.misses >= _MISSES_BEFORE_GONE:
                        self._worker_gone(worker)
                    continue
                worker.misses = 0
                worker.squeue_state = squeue_state
                if squeue_state == "RUNNING" and worker.running_since is None:
                    worker.running_since = now
            worker.state = _read_json(worker.dir / "state.json") or worker.state
            if worker.state.get("state") in ("idle", "busy"):
                worker.ever_ready = True
                self._start_failures = 0
            if worker.current_job is not None and worker.is_free():
                worker.current_job = None
            if worker.is_free() and worker.idle_since is None:
                worker.idle_since = now
            elif not worker.is_free():
                worker.idle_since = None

    def _worker_gone(self, worker: Worker) -> None:
        """A worker's Slurm job ended on its own (not stopped by us)."""
        self._workers.pop(worker.worker_id, None)
        if worker.ever_ready:
            logger.info("Pool %s: worker %s (Slurm job %s) ended.", self.name, worker.worker_id, worker.slurm_job_id)
            return
        self._start_failures += 1
        self._last_failure_log = _log_tail(worker.dir / "worker.log")
        logger.warning(
            "Pool %s: worker %s (Slurm job %s) ended before it was ready (%d in a row).",
            self.name,
            worker.worker_id,
            worker.slurm_job_id,
            self._start_failures,
        )

    def _enforce_limits(self) -> None:
        """Stop a worker that is stuck: a request running past its limit, or a start that never finishes."""
        now = time.time()
        for worker in list(self._workers.values()):
            if worker.current_job is not None and not worker.is_free():
                if worker.assigned_at and now - worker.assigned_at > worker.limit_seconds:
                    job_id = worker.current_job
                    self._stop_worker(worker, f"request {job_id} ran past its {worker.limit_seconds} s limit")
                    self._fail_job_id(job_id, f"The request ran longer than its limit of {worker.limit_seconds} seconds and was stopped.")
            elif worker.is_starting() and worker.running_since is not None:
                if now - worker.running_since > self.startup_timeout_seconds:
                    self._stop_worker(worker, "it did not become ready in time")

    def _schedule(self) -> list[tuple[PoolRequest, str]]:
        """Hand queued requests to idle workers, start workers for the rest, stop workers nobody needs.
        Returns the requests that cannot be served, with the reason; the caller fails them."""
        failures: list[tuple[PoolRequest, str]] = []
        now = time.time()
        self._assign_to_free_workers(now, failures)
        unserved = self._claim_starting_workers()
        gave_up_starting = False
        for request in unserved:
            if self._start_failures >= MAX_START_FAILURES:
                gave_up_starting = True
                failures.append(
                    (
                        request,
                        f"The model worker failed to start {self._start_failures} times in a row."
                        + (f" Last log lines:\n{self._last_failure_log}" if self._last_failure_log else ""),
                    )
                )
                continue
            if len(self._workers) >= self.max_workers and not self._evict_idle_worker():
                break  # at the cap and every worker is busy or still needed: wait
            try:
                self._start_worker(request)
            except PoolError as e:
                failures.append((request, str(e)))
        if gave_up_starting:
            self._start_failures = 0  # the next request gets a fresh try
        for request, _ in failures:
            if request in self._queue:
                self._queue.remove(request)
        self._stop_idle_workers(now)
        return failures

    def _assign_to_free_workers(self, now: float, failures: list[tuple[PoolRequest, str]]) -> None:
        for request in list(self._queue):
            row = job_store.get_job(request.job_id)
            if row is None or row["status"] != "queued":
                self._queue.remove(request)  # cancelled or failed while it waited
                continue
            candidates = sorted(
                (w for w in self._workers.values() if w.is_free() and not w.retire and w.matches(request)),
                # The smallest worker that fits, and of those the one that went idle last, so older idle
                # workers run out their idle time.
                key=lambda w: (w.gpus, -(w.idle_since or 0.0)),
            )
            for worker in candidates:
                if self._seconds_left(worker, now) < request.limit_seconds:
                    worker.retire = True  # cannot finish another request of this size: replace it
                    continue
                try:
                    self._assign(request, worker, now)
                except OSError as e:
                    logger.warning("Pool %s: could not give job %s to worker %s: %s", self.name, request.job_id, worker.worker_id, e)
                    failures.append((request, "Could not hand the request to a model worker (file error)."))
                self._queue.remove(request)
                break

    def _seconds_left(self, worker: Worker, now: float) -> float:
        started = worker.state.get("started_at") or worker.submitted_at
        return float(started) + self.worker_time_seconds - now - TIME_MARGIN_SECONDS

    def _assign(self, request: PoolRequest, worker: Worker, now: float) -> None:
        job_dir = common_config.JOBS_DIR / request.job_id
        warm = worker.jobs_served > 0
        _write_json_atomic(
            job_dir / _ASSIGNMENT_FILE,
            {"worker_id": worker.worker_id, "slurm_job_id": worker.slurm_job_id, "warm": warm},
        )
        job_store.set_slurm_job_id(request.job_id, worker.slurm_job_id)
        job_store.mark_running(request.job_id)
        worker.seq += 1
        _write_json_inplace(worker.dir / "inbox.json", {"seq": worker.seq, "job_id": request.job_id, "job_dir": str(job_dir)})
        worker.current_job = request.job_id
        worker.assigned_at = now
        worker.limit_seconds = request.limit_seconds
        worker.jobs_served += 1
        worker.idle_since = None
        logger.info(
            "Pool %s: job %s -> worker %s (%s start, %d queued).",
            self.name,
            request.job_id,
            worker.worker_id,
            "warm" if warm else "cold",
            len(self._queue) - 1,
        )

    def _claim_starting_workers(self) -> list[PoolRequest]:
        """Match every queued request with at most one worker that is still starting. Starting
        workers nobody claimed are stopped. Returns the requests left without one."""
        starting = [w for w in self._workers.values() if w.is_starting()]
        unserved: list[PoolRequest] = []
        for request in self._queue:
            claimed = next((w for w in starting if w.matches(request)), None)
            if claimed is None:
                unserved.append(request)
            else:
                starting.remove(claimed)
        for worker in starting:
            self._stop_worker(worker, "no request is waiting for it any more")
        return unserved

    def _evict_idle_worker(self) -> bool:
        """At the cap: stop the longest-idle worker to make room. Only idle workers are candidates; at
        this point none of them can serve a queued request."""
        idle = [w for w in self._workers.values() if w.is_free()]
        if not idle:
            return False
        victim = min(idle, key=lambda w: w.idle_since or 0.0)
        self._stop_worker(victim, "making room for a worker of another kind")
        return True

    def _stop_idle_workers(self, now: float) -> None:
        for worker in list(self._workers.values()):
            if not worker.is_free():
                continue
            if worker.retire:
                self._stop_worker(worker, "too little Slurm time left")
            elif worker.idle_since is not None and now - worker.idle_since >= self.idle_seconds:
                self._stop_worker(worker, f"idle for {self.idle_seconds / 60:g} minutes")

    # --- Slurm ---------------------------------------------------------------

    def _start_worker(self, request: PoolRequest) -> None:
        worker_id = uuid.uuid4().hex[:12]
        worker_dir = self.dir / worker_id
        worker_dir.mkdir(parents=True)
        # Both polled files exist before the worker does (see _write_json_inplace).
        _write_json_inplace(worker_dir / "inbox.json", {"seq": 0})
        _write_json_inplace(worker_dir / "state.json", {"state": "starting"})
        command = self._build_sbatch_command(worker_id, worker_dir, request.key, request.partition, request.gpus)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except (subprocess.SubprocessError, OSError) as e:
            shutil.rmtree(worker_dir, ignore_errors=True)
            raise PoolError(f"Could not start a model worker (sbatch could not run: {e}).") from e
        if result.returncode != 0:
            shutil.rmtree(worker_dir, ignore_errors=True)
            raise PoolError(f"Could not start a model worker (sbatch failed): {result.stderr.strip() or result.stdout.strip()}")
        slurm_job_id = result.stdout.strip().split(";")[0]
        now = time.time()
        _write_json_atomic(
            worker_dir / "worker.json",
            {
                "slurm_job_id": slurm_job_id,
                "key": request.key,
                "partition": request.partition,
                "gpus": request.gpus,
                "submitted_at": now,
            },
        )
        self._workers[worker_id] = Worker(
            worker_id=worker_id,
            slurm_job_id=slurm_job_id,
            key=request.key,
            partition=request.partition,
            gpus=request.gpus,
            dir=worker_dir,
            submitted_at=now,
        )
        logger.info(
            "Pool %s: started worker %s (Slurm job %s, %d GPU(s), partition %s, key %s); %d of %d.",
            self.name,
            worker_id,
            slurm_job_id,
            request.gpus,
            request.partition,
            request.key,
            len(self._workers),
            self.max_workers,
        )

    def _stop_worker(self, worker: Worker, reason: str) -> None:
        logger.info("Pool %s: stopping worker %s (Slurm job %s): %s.", self.name, worker.worker_id, worker.slurm_job_id, reason)
        cancel_slurm_job(worker.slurm_job_id)
        self._workers.pop(worker.worker_id, None)

    # --- failing requests --------------------------------------------------

    def _fail_request(self, request: PoolRequest, message: str) -> None:
        self._fail_job_id(request.job_id, message)

    def _fail_job_id(self, job_id: str, message: str) -> None:
        job_dir = common_config.JOBS_DIR / job_id
        try:
            (job_dir / "error.txt").write_text(message)
            (job_dir / ".failed").touch()
        except OSError:
            logger.warning("Could not write the failure marker for job %s", job_id)
        job_store.mark_failed(job_id, message)
        retention.on_job_failed(job_id)


# --- helpers for the gateway ---------------------------------------------------


def cancel_job(job_id: str, slurm_job_id: str | None) -> bool:
    """Used by DELETE /v1/jobs/{id}. True if a pool dealt with the job (removed it from a queue, or
    stopped the worker running it, or found that its worker has moved on), so the caller must NOT
    scancel `slurm_job_id`: for a pooled job that is a worker's id, and that worker may already be
    running somebody else's request."""
    for pool in _POOLS:
        if pool.cancel_job(job_id):
            return True
    if slurm_job_id:
        return any(pool.owns_slurm_job(slurm_job_id) for pool in _POOLS)
    return False


def pools() -> list[WorkerPool]:
    return list(_POOLS)
