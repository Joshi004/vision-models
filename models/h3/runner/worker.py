#!/usr/bin/env python3
"""A warm MiniMax H3 worker: it holds a 4 or 8 GPU Slurm allocation and runs the gateway's
queued requests in it, one after another (see `gateway/services/h3/dispatch.py` and
`gateway/common/worker_pool.py`, which start and feed it, and `models/common/pool_worker.py`,
the request loop).

Usage (started by the gateway, under the system python3):
    worker.py --worker-dir DIR --idle-exit-seconds N

Unlike the image worker this does NOT keep the model in a process. Every request is a fresh
`run_h3.py` run, exactly the command a cold job would run (the request's args.json), so that
script, its multi-process layout and its GPU memory handling are untouched. What a second
request saves is the loading: the checkpoints it reads were left in this node's page cache by the
first one, so loading takes seconds instead of minutes (measured on this cluster for a small
request: 264 s -> 13 s for the transformer, 308 s -> 12 s for the encoder, 441 s -> 77 s in total).

A request that fails ends the worker, as for the image worker: the next request gets a new one.

Stdlib only, like run_pipeline_job.py: the heavy process is the subprocess, run with the
interpreter path stored in args.json.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # models/common/pool_worker.py

import pool_worker  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker-dir", required=True, type=Path)
    p.add_argument("--idle-exit-seconds", required=True, type=float)
    worker_args = p.parse_args()

    state = pool_worker.WorkerState(worker_args.worker_dir)

    def handle(argv: list[str]) -> None:
        # pool_worker has pointed this process's stdout and stderr at the job's run.log; the
        # subprocess inherits them. The message names the exit code only: argv holds the prompt.
        returncode = subprocess.run(argv).returncode
        if returncode != 0:
            raise RuntimeError(f"Exit code: {returncode}")

    return pool_worker.serve(worker_args.worker_dir, handle, worker_args.idle_exit_seconds, state)


if __name__ == "__main__":
    sys.exit(main())
