#!/usr/bin/env python3
"""A warm Qwen-Image-2.1 worker: loads one checkpoint onto the GPU once, then serves
image requests from the gateway one after another (see
`gateway/services/qwen_image21/dispatch.py` and `gateway/common/worker_pool.py`, which
start and feed it, and `models/common/pool_worker.py`, the request loop).

Usage (started by the gateway, with this model's own environment):
    worker.py --worker-dir DIR --model-dir DIR --mode fast|quality --idle-exit-seconds N

Each request is a job folder whose args.json holds the same argv `run_generate.py` takes
(that script's own `parse_args` and `generate` do the work). A request that fails ends
the worker: the next request gets a new one, so a failure such as running out of GPU
memory never leaves a damaged process serving.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNNER_DIR))
sys.path.insert(0, str(RUNNER_DIR.parents[1] / "common"))  # models/common/pool_worker.py

import pool_worker  # noqa: E402
import run_generate  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker-dir", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=["fast", "quality"])
    p.add_argument("--idle-exit-seconds", required=True, type=float)
    worker_args = p.parse_args()

    state = pool_worker.WorkerState(worker_args.worker_dir)
    state.write("loading")
    run_generate.configure_environment()
    pipe = run_generate.load_pipeline(worker_args.model_dir, worker_args.mode)

    import torch

    def handle(argv: list[str]) -> None:
        # argv is [interpreter, run_generate.py, flags...], as for a cold job.
        started = time.time()
        args = run_generate.parse_args(argv[2:])
        if args.mode != worker_args.mode or args.model_dir != worker_args.model_dir:
            raise RuntimeError(f"This worker serves {worker_args.mode} mode only; the request asked for {args.mode}.")
        try:
            if run_generate.generate(pipe, args, started) != 0:
                raise RuntimeError("The image was not written.")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    return pool_worker.serve(worker_args.worker_dir, handle, worker_args.idle_exit_seconds, state)


if __name__ == "__main__":
    sys.exit(main())
