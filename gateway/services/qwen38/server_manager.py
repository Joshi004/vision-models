"""The shared vLLM server behind the Qwen3.8 backend: starting it, finding
out whether it is ready, sending it chat requests, and stopping it when idle.

One Slurm job runs the server (models/qwen38/runner/serve.sh). The gateway
and that job share only files under config.STATE_DIR:

  api_key       written here (mode 0600); serve.sh gives it to vLLM
  slurm_job_id  written here when the job is submitted; lets a restarted
                gateway find a server that is still running
  server.json   written by serve.sh once vLLM answers /health: host, port and
                the Slurm job id it belongs to. "Ready" means this file exists
                and names the current job.
  server.log    the Slurm job's output (vLLM's own log; request logging is off,
                so it holds no prompts or answers)

Everything here that touches Slurm, files or the network is blocking; async
callers run it with asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import logging
import os
import secrets
import shlex
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from common import config as common_config
from common.slurm import cancel_slurm_job, resolve_partition
from services.qwen38 import config
from services.qwen38.schemas import ServerStatus

logger = logging.getLogger("model-api.qwen38")

_STATE_FILE = config.STATE_DIR / "server.json"
_JOB_ID_FILE = config.STATE_DIR / "slurm_job_id"
_KEY_FILE = config.STATE_DIR / "api_key"
_LOG_FILE = config.STATE_DIR / "server.log"

# squeue states that mean the server job still exists (COMPLETING is already
# shutting down, so it counts as gone).
_LIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING"}

_lock = threading.RLock()
_requests_in_flight = 0
_last_activity = time.time()


class ServerError(RuntimeError):
    """The server could not be started, stopped being reachable, or went away."""


class RequestRejected(RuntimeError):
    """The server answered, but refused the request (for example an invalid schema or a prompt
    that does not fit). The message is the server's own."""


# --- Slurm and state files -------------------------------------------------


@dataclass
class _Snapshot:
    state: str  # "stopped" | "starting" | "ready"
    slurm_job_id: str | None
    info: dict | None  # server.json, when ready


def _read_json(path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _read_job_id() -> str | None:
    try:
        return _JOB_ID_FILE.read_text().strip() or None
    except OSError:
        return None


def _squeue_state(slurm_job_id: str) -> str | None:
    """The job's Slurm state, "" if the job no longer exists, None if squeue could not answer."""
    try:
        result = subprocess.run(
            ["squeue", "-j", slurm_job_id, "-h", "-o", "%T"], capture_output=True, text=True, timeout=10
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("squeue failed for the model server job %s: %s", slurm_job_id, e)
        return None
    if result.returncode == 0:
        return result.stdout.strip().split("\n")[0].strip()
    if "Invalid job id" in result.stderr:
        return ""  # finished long enough ago that Slurm forgot it
    logger.warning("squeue exited %d for job %s: %s", result.returncode, slurm_job_id, result.stderr.strip())
    return None


def _clear_state() -> None:
    for path in (_STATE_FILE, _JOB_ID_FILE):
        path.unlink(missing_ok=True)


def _snapshot() -> _Snapshot:
    slurm_job_id = _read_job_id()
    if slurm_job_id is None:
        return _Snapshot("stopped", None, None)
    slurm_state = _squeue_state(slurm_job_id)
    if slurm_state is not None and slurm_state not in _LIVE_STATES:
        _clear_state()
        return _Snapshot("stopped", None, None)
    info = _read_json(_STATE_FILE)
    if info is not None and info.get("slurm_job_id") == slurm_job_id and slurm_state in ("RUNNING", None):
        return _Snapshot("ready", slurm_job_id, info)
    return _Snapshot("starting", slurm_job_id, None)


def _ensure_api_key() -> None:
    """Create the key file (readable only by this user) if it does not exist yet."""
    if _KEY_FILE.exists():
        return
    fd = os.open(_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_urlsafe(32))


def _submit_server_job(partition: str) -> str:
    config.STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    _ensure_api_key()
    _STATE_FILE.unlink(missing_ok=True)
    serve_argv = [
        str(config.SERVE_SCRIPT),
        str(config.STATE_DIR),
        str(config.MODEL_DIR),
        str(config.ENV_DIR),
        str(config.CACHE_DIR),
        str(config.MAX_MODEL_LEN),
        str(config.MAX_CONCURRENT_REQUESTS),
        config.SERVED_MODEL_NAME,
    ]
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        "--gres=gpu:1",
        f"--cpus-per-task={config.CPUS_PER_GPU}",
        f"--mem-per-gpu={config.MEM_PER_GPU}",
        f"--time={config.SERVER_TIME_LIMIT}",
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}qwen38-server",
        f"--output={_LOG_FILE}",
        f"--wrap={shlex.join(serve_argv)}",
    ]
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ServerError(f"Could not start the model server (sbatch failed): {result.stderr.strip() or result.stdout.strip()}")
    slurm_job_id = result.stdout.strip().split(";")[0]
    _JOB_ID_FILE.write_text(slurm_job_id)
    logger.info("Started the Qwen3.8 server as Slurm job %s on partition %s.", slurm_job_id, partition)
    return slurm_job_id


def ensure_server() -> str:
    """Make sure a server job exists (queued, loading or ready); start one if not. Returns its Slurm job id."""
    with _lock:
        touch()
        snap = _snapshot()
        if snap.state != "stopped":
            return snap.slurm_job_id  # type: ignore[return-value]
        return _submit_server_job(resolve_partition(config.PARTITION))


def _log_tail(max_lines: int = 15, max_chars: int = 1500) -> str:
    try:
        lines = [line for line in _LOG_FILE.read_text(errors="replace").splitlines() if line.strip()]
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])[-max_chars:]


async def wait_until_ready(timeout_seconds: float) -> dict:
    """Wait for the server to be ready; start-up (queue, weights, compile) is the slow part."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        snap = await asyncio.to_thread(_snapshot)
        if snap.state == "ready":
            return snap.info  # type: ignore[return-value]
        if snap.state == "stopped":
            tail = await asyncio.to_thread(_log_tail)
            raise ServerError("The model server stopped before it was ready." + (f" Last log lines:\n{tail}" if tail else ""))
        if time.monotonic() > deadline:
            raise ServerError(f"The model server was not ready after {int(timeout_seconds)} seconds.")
        await asyncio.sleep(3)


# --- Activity and idle shutdown --------------------------------------------


def touch() -> None:
    global _last_activity
    _last_activity = time.time()


def request_started() -> None:
    global _requests_in_flight
    with _lock:
        _requests_in_flight += 1
        touch()


def request_finished() -> None:
    global _requests_in_flight
    with _lock:
        _requests_in_flight = max(0, _requests_in_flight - 1)
        touch()


def _idle_since(snap: _Snapshot) -> float:
    ready_at = 0.0
    if snap.info and snap.info.get("ready_at"):
        try:
            ready_at = datetime.fromisoformat(snap.info["ready_at"].replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return max(_last_activity, ready_at)


def _stop_if_idle() -> None:
    with _lock:
        snap = _snapshot()
        # Only a ready server can be idle; one that is still starting (queued or loading) must be left alone,
        # whatever the time since the last request.
        if snap.state != "ready" or _requests_in_flight > 0:
            return
        if time.time() - _idle_since(snap) < config.IDLE_MINUTES * 60:
            return
        logger.info("Stopping the Qwen3.8 server (Slurm job %s): idle for %s minutes.", snap.slurm_job_id, config.IDLE_MINUTES)
        cancel_slurm_job(snap.slurm_job_id)  # type: ignore[arg-type]
        _clear_state()


async def idle_watch_forever() -> None:
    while True:
        try:
            await asyncio.to_thread(_stop_if_idle)
        except Exception:
            logger.exception("Qwen3.8 idle check failed")
        await asyncio.sleep(config.IDLE_CHECK_SECONDS)


def get_status() -> ServerStatus:
    with _lock:
        snap = _snapshot()
        idle_shutdown_at = None
        if snap.state == "ready" and _requests_in_flight == 0:
            idle_shutdown_at = datetime.fromtimestamp(
                _idle_since(snap) + config.IDLE_MINUTES * 60, tz=timezone.utc
            ).isoformat()
        return ServerStatus(
            state=snap.state,  # type: ignore[arg-type]
            slurm_job_id=snap.slurm_job_id,
            partition=config.PARTITION or common_config.FALLBACK_PARTITION,
            requests_in_flight=_requests_in_flight,
            idle_shutdown_at=idle_shutdown_at,
            idle_minutes=config.IDLE_MINUTES,
            model=config.MODEL_ID,
        )


# --- Chat requests ----------------------------------------------------------


def _error_message(body: bytes) -> str:
    try:
        data = json.loads(body)
    except ValueError:
        return body.decode(errors="replace")[:500]
    error = data.get("error", data) if isinstance(data, dict) else data
    message = error.get("message") if isinstance(error, dict) else None
    return str(message or error)[:500]


class ChatCall:
    """One chat completion request. abort() closes its connection from another thread, which makes
    vLLM stop generating for it."""

    def __init__(self) -> None:
        self._conn: http.client.HTTPConnection | None = None
        self._aborted = threading.Event()

    def abort(self) -> None:
        self._aborted.set()
        conn = self._conn
        if conn is None:
            return
        # close() alone does not end a connection another thread is blocked reading: the response keeps a
        # reference to the socket. shutdown() does, and that is what tells vLLM to stop generating.
        sock = conn.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # already closed
        conn.close()

    def run(self, payload: dict) -> dict:
        info = _read_json(_STATE_FILE)
        if info is None:
            raise ServerError("The model server is not running (it stopped, or was stopped while idle). Try again.")
        try:
            api_key = _KEY_FILE.read_text().strip()
        except OSError as e:
            raise ServerError(f"Cannot read the server's API key: {e}") from e
        conn = http.client.HTTPConnection(info["host"], int(info["port"]), timeout=config.REQUEST_TIMEOUT_SECONDS)
        self._conn = conn
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            )
            response = conn.getresponse()
            body = response.read()
        except (OSError, http.client.HTTPException) as e:
            if self._aborted.is_set():
                raise ServerError("Cancelled.") from e
            raise ServerError(
                f"Lost the connection to the model server ({type(e).__name__}: {e}). "
                "It may have been stopped or pre-empted; submit the request again."
            ) from e
        finally:
            conn.close()
        if 400 <= response.status < 500:
            raise RequestRejected(_error_message(body))
        if response.status != 200:
            raise ServerError(f"The model server failed (HTTP {response.status}): {_error_message(body)}")
        try:
            return json.loads(body)
        except ValueError as e:
            raise ServerError("The model server returned an answer that is not valid JSON.") from e
