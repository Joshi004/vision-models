"""The worker side of a warm worker pool (the gateway side is gateway/common/worker_pool.py).

A model's worker script loads its model once, then calls serve(). serve() waits for
the gateway to hand it requests, one at a time, and runs each through a handler the
script provides. It follows the same conventions as gateway/common/run_pipeline_job.py,
so the gateway's poller finishes the job from the files, as for any other job:

  <job_dir>/args.json         the full argv of the request (interpreter, script, flags);
                              its --output-path entry is the *partial* output file
  <job_dir>/output_path.txt   the final output path
  success                     the partial output is renamed to the final path
  failure                     error.txt and a .failed marker are written, and the worker
                              ends (a failed request can leave the GPU in a bad state, so
                              the next request gets a new worker)

Files in the worker's own folder (see the gateway module for the full list):

  inbox.json   written by the gateway: {"seq", "job_id", "job_dir"}. A request is new when its
               seq is higher than the last one handled here. Never deleted or replaced: the data
               directory is on NFS, which caches a failed lookup and a replaced file's old version
               for up to a minute, so both control files exist before this process starts and are
               only rewritten in place (see gateway/common/worker_pool.py::_write_json_inplace).
  state.json   written here: {"state": loading|idle|busy|exiting, "job_id", "seq", "started_at", ...}

While a request runs, everything the process prints (Python and native code alike) goes
to the job folder's run.log, so a request's text never reaches the worker's own log and
an incognito job's folder holds all of its output.

Stdlib only: the model's environment has no use for the gateway's packages.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Iterator

# When this worker process started. The gateway uses it to know how much of the Slurm time limit is left.
STARTED_AT = time.time()

POLL_SECONDS = 1.0
_LOG_TAIL_LINES = 40


class WorkerState:
    """The worker's state.json."""

    def __init__(self, worker_dir: Path) -> None:
        self._path = worker_dir / "state.json"
        self.seq = 0  # the inbox sequence number of the last request started

    def write(self, state: str, job_id: str | None = None) -> None:
        data = {
            "state": state,
            "job_id": job_id,
            "seq": self.seq,
            "started_at": STARTED_AT,
            "updated_at": time.time(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }
        # Rewritten in place, in one write (see the module docstring). A gateway that catches it
        # half-written fails to parse it and reads it again on its next pass.
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, json.dumps(data).encode())
        finally:
            os.close(fd)


@contextlib.contextmanager
def _output_to(path: Path) -> Iterator[None]:
    """Point file descriptors 1 and 2 at `path` for the duration, so native libraries' output is captured too."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        for fd in (log_fd, saved_out, saved_err):
            os.close(fd)


def _fail(job_dir: Path, partial_path: Path | None, reason: str) -> None:
    """The same files run_pipeline_job.py leaves for a failed job."""
    if partial_path is not None:
        partial_path.unlink(missing_ok=True)
    tail = ""
    try:
        tail = "\n".join((job_dir / "run.log").read_text(errors="replace").splitlines()[-_LOG_TAIL_LINES:])
    except OSError:
        pass
    (job_dir / "error.txt").write_text(f"{reason}\n\n--- last lines of run.log ---\n{tail}\n")
    (job_dir / ".failed").touch()


def _run_request(job_dir: Path, handle: Callable[[list[str]], None]) -> bool:
    """Run one request. True if its output was written."""
    partial_path: Path | None = None
    try:
        argv: list[str] = json.loads((job_dir / "args.json").read_text())
        output_path = Path((job_dir / "output_path.txt").read_text().strip())
        partial_path = Path(argv[argv.index("--output-path") + 1])
        with _output_to(job_dir / "run.log"):
            try:
                handle(argv)
            except (Exception, SystemExit) as error:  # SystemExit: argparse rejecting the arguments
                traceback.print_exc()
                raise RuntimeError(f"{type(error).__name__}: {error}") from error
        if not partial_path.exists():
            raise RuntimeError("The model finished but wrote no output.")
        partial_path.rename(output_path)
        return True
    except Exception as error:
        try:
            _fail(job_dir, partial_path, str(error))
        except OSError:
            pass  # the poller still ends the job when the worker's Slurm job ends
        return False


def serve(
    worker_dir: Path,
    handle: Callable[[list[str]], None],
    idle_exit_seconds: float,
    state: WorkerState,
) -> int:
    """Serve requests until idle for idle_exit_seconds (a safety net: the gateway normally stops an idle
    worker sooner) or until a request fails. `handle(argv)` runs one request from its full argv and
    writes the partial output file; it raises on failure. Returns the process exit code."""
    inbox = worker_dir / "inbox.json"
    last_job_id: str | None = None
    state.write("idle", last_job_id)
    idle_since = time.monotonic()
    while True:
        try:
            request = json.loads(inbox.read_text())
            new_seq = int(request["seq"])
            is_new = new_seq > state.seq and "job_id" in request
        except (OSError, ValueError, KeyError, TypeError):
            is_new = False  # unreadable (the gateway is mid-write) or malformed: look again
        if not is_new:
            if time.monotonic() - idle_since > idle_exit_seconds:
                state.write("exiting", last_job_id)
                return 0
            time.sleep(POLL_SECONDS)
            continue
        state.seq = new_seq
        job_id = str(request["job_id"])
        state.write("busy", job_id)
        if not _run_request(Path(request["job_dir"]), handle):
            state.write("exiting", job_id)
            return 1
        last_job_id = job_id
        state.write("idle", last_job_id)
        idle_since = time.monotonic()
