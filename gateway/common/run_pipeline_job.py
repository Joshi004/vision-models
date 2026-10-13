#!/usr/bin/env python3
"""Runs on the Slurm compute node for exactly one API-submitted job, for any
backend (LTX-2.3 today, potentially others later) -- nothing here is
backend-specific; each backend's own dispatch module builds the argv this
script executes.

Invoked as: python3 run_pipeline_job.py <job_dir>

  <job_dir>/args.json          JSON list: the full argv (including the
                               interpreter path) to execute. Its --output-path
                               entry already points at the *partial* output
                               file, not the final one.
  <job_dir>/output_path.txt    The final output path to rename the partial
                               file to on success.
  <job_dir>/post_step.json     Optional (written only when a request asked for
                               lip sync): a second command to run after the
                               main one succeeds, with its own interpreter --
                               see _run_lip_sync_step below.

Deliberately stdlib-only (no fastapi/torch import) so it runs fine under the
plain system python3 that Slurm invokes it with -- the actual heavy pipeline
process is a *subprocess*, using the interpreter path baked into args.json
(e.g. LTX-2's own torch/CUDA venv), not this script's own interpreter.

On success: renames the partial output to its final name.
On failure: removes any partial output and writes error.txt + touches
.failed, so a half-written file can never be mistaken for a finished
result, and the API's poller (common/poller.py) has a simple marker to
look for.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

# Signal exit codes that almost always mean "killed by Slurm preemption",
# not a real failure -- mirrors generate_worker.sh's rc == 137/143 handling.
_PREEMPTION_EXIT_CODES = (137, 143)

_LIP_SYNC_ERROR_LOG_LINES = 8
_LIP_SYNC_ERROR_MAX_CHARS = 1500


def _lip_sync_failure_detail(log_path: Path) -> str:
    """The cause of a failed lip-sync step: the last "ERROR: ..." line the lip-sync runner
    printed (it puts the root cause there), else the last few lines of its log."""
    try:
        lines = [line.strip() for line in log_path.read_text(errors="replace").splitlines() if line.strip()]
    except OSError:
        return ""
    errors = [line for line in lines if line.startswith("ERROR:")]
    if errors:
        return errors[-1][len("ERROR:"):].strip()
    return "\n".join(lines[-_LIP_SYNC_ERROR_LOG_LINES:])


def _write_lip_sync_result(job_dir: Path, applied: bool, error: str | None) -> None:
    """The poller (common/poller.py) reads this once the output file exists,
    so it is always written before the final rename."""
    (job_dir / "lip_sync.json").write_text(json.dumps({"requested": True, "applied": applied, "error": error}))


def _run_lip_sync_step(job_dir: Path, partial_path: Path, output_path: Path, spec: dict) -> None:
    """Second step of a request that asked for lip sync. The main step's video
    is kept as `spec["input_path"]` (the un-synced base video) and the post
    command is run on it. Its result becomes the job's output.

    A lip-sync failure is deliberately NOT a job failure: the un-synced base
    video is copied to the output instead and the reason is recorded in
    lip_sync.json, so the client still gets a video and can retry the sync
    alone (POST /v1/tbdub/videos/lip-sync with source_job_id).

    spec: {"argv": [...], "input_path": ..., "output_path": ..., "timeout_seconds": N}
    """
    base_path = Path(spec["input_path"])
    result_path = Path(spec["output_path"])
    partial_path.rename(base_path)

    log_path = job_dir / "run_lipsync.log"
    returncode: int | None = None
    timed_out = False
    spawn_error = ""
    with open(log_path, "w") as log_file:
        try:
            # Own process group, so a timeout also stops the runner's child (TBDub).
            proc = subprocess.Popen(spec["argv"], stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:  # e.g. the interpreter path does not exist
            spawn_error = str(error)
        else:
            try:
                returncode = proc.wait(timeout=spec.get("timeout_seconds"))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                timed_out = True

    if returncode == 0 and result_path.exists():
        _write_lip_sync_result(job_dir, applied=True, error=None)
        result_path.rename(output_path)
        return

    result_path.unlink(missing_ok=True)
    if spawn_error:
        reason = f"Lip sync could not be started ({spawn_error})."
    elif timed_out:
        reason = f"Lip sync timed out after {spec.get('timeout_seconds')} s."
    elif returncode == 0:
        reason = "Lip sync finished but wrote no output."
    else:
        reason = f"Lip sync failed (exit {returncode})."
    detail = _lip_sync_failure_detail(log_path)
    error = (
        f"{reason}{' ' + detail if detail else ''}\n"
        f"The result is the video WITHOUT lip sync. To retry only the sync, call "
        f"POST /v1/tbdub/videos/lip-sync with source_job_id={job_dir.name}."
    )[:_LIP_SYNC_ERROR_MAX_CHARS]
    _write_lip_sync_result(job_dir, applied=False, error=error)
    shutil.copy2(base_path, partial_path)  # partial_path was renamed away above, so the name is free
    partial_path.rename(output_path)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_pipeline_job.py <job_dir>", file=sys.stderr)
        return 2

    job_dir = Path(sys.argv[1])
    argv: list[str] = json.loads((job_dir / "args.json").read_text())
    output_path = Path((job_dir / "output_path.txt").read_text().strip())
    partial_path = Path(argv[argv.index("--output-path") + 1])

    log_path = job_dir / "run.log"
    with open(log_path, "w") as log_file:
        proc = subprocess.run(argv, stdout=log_file, stderr=subprocess.STDOUT)

    if proc.returncode == 0 and partial_path.exists():
        post_step_path = job_dir / "post_step.json"
        if post_step_path.exists():
            _run_lip_sync_step(job_dir, partial_path, output_path, json.loads(post_step_path.read_text()))
        else:
            partial_path.rename(output_path)
        return 0

    partial_path.unlink(missing_ok=True)

    tail = ""
    try:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
    except OSError:
        pass

    if proc.returncode in _PREEMPTION_EXIT_CODES:
        reason = (
            f"Process was killed (exit {proc.returncode}) -- likely preempted by a "
            "higher-priority job on this shared partition. Resubmitting the same "
            "request should work; this is not a real failure."
        )
    else:
        reason = f"Exit code: {proc.returncode}"

    (job_dir / "error.txt").write_text(f"{reason}\n\n--- last lines of run.log ---\n{tail}\n")
    (job_dir / ".failed").touch()
    return proc.returncode if proc.returncode != 0 else 1


if __name__ == "__main__":
    sys.exit(main())
