#!/usr/bin/env python3
"""A warm Parakeet worker: loads the transcription model onto the GPU once, then
serves transcription requests from the gateway one after another (see
`gateway/services/parakeet/dispatch.py` and `gateway/common/worker_pool.py`, which
start and feed it, and `models/common/pool_worker.py`, the request loop).

Usage (started through worker.sh, which sets the CUDA library path first):
    worker.py --worker-dir DIR --model-path FILE --idle-exit-seconds N

Each request is a job folder whose args.json holds the same argv a one-shot job runs
([run_transcribe.sh, flags...]); `run_transcribe.py`'s own `_parse_args` and `run` do the
work. A request that fails ends the worker: the next request gets a new one, so a failure
such as running out of GPU memory never leaves a damaged process serving.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNNER_DIR))
sys.path.insert(0, str(RUNNER_DIR.parents[1] / "common"))  # models/common/pool_worker.py

import pool_worker  # noqa: E402
import run_transcribe  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker-dir", required=True, type=Path)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--idle-exit-seconds", required=True, type=float)
    worker_args = p.parse_args()

    state = pool_worker.WorkerState(worker_args.worker_dir)
    state.write("loading")
    model = run_transcribe.load_model(worker_args.model_path)

    import torch

    def loaded_model(model_path: Path):
        if model_path != worker_args.model_path:
            raise run_transcribe.TranscribeError("The request points at a different checkpoint than this worker loaded.")
        return model

    def handle(argv: list[str]) -> None:
        # argv is [run_transcribe.sh, flags...], as for a one-shot job.
        args = run_transcribe._parse_args(argv[1:])
        try:
            if run_transcribe.run(args, loaded_model) != 0:
                raise RuntimeError("The transcription failed (see the lines above).")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    return pool_worker.serve(worker_args.worker_dir, handle, worker_args.idle_exit_seconds, state)


if __name__ == "__main__":
    sys.exit(main())
