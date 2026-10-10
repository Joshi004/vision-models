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
import subprocess
import sys
from pathlib import Path

# Signal exit codes that almost always mean "killed by Slurm preemption",
# not a real failure -- mirrors generate_worker.sh's rc == 137/143 handling.
_PREEMPTION_EXIT_CODES = (137, 143)


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
