"""ACE-Step 1.5 music generation configuration: project paths, request limits,
and this backend's Slurm resource shape.

API-wide settings (host/port, job/upload storage, retention, partition
fallback) live in common/config.py instead -- this module only has what's
specific to ACE-Step 1.5 itself.

Zero third-party dependencies (stdlib only): nothing here should require
fastapi/uvicorn.
"""

from __future__ import annotations

import os
from pathlib import Path

from common import config as common_config
from common.slurm import slurm_time_seconds

# ---------------------------------------------------------------------------
# Project layout. This model's own code (runner scripts) lives in
# models/ace_step15/; its environment and weights live under the shared data
# directory (see common/config.py). The upstream code is the pinned submodule
# third_party/ACE-Step-1.5, installed editable into the model's environment.
# ---------------------------------------------------------------------------
ACE_STEP15_PROJECT_ROOT = common_config.REPO_ROOT / "models" / "ace_step15"
ACE_STEP15_PYTHON = common_config.ENVS_DIR / "ace_step15" / "bin" / "python"
RUN_GENERATE_SCRIPT = ACE_STEP15_PROJECT_ROOT / "runner" / "run_generate.py"
WORKER_SCRIPT = ACE_STEP15_PROJECT_ROOT / "runner" / "worker.py"

# Holds checkpoints/ (acestep-v15-xl-sft, acestep-5Hz-lm-4B, vae, Qwen3-Embedding-0.6B),
# fetched by models/ace_step15/download_weights.sh. Loaded from disk, so a job never
# needs network access.
MODEL_DIR = Path(os.environ.get("ACE_STEP15_MODEL_DIR", str(common_config.WEIGHTS_DIR / "ace_step15")))

# ---------------------------------------------------------------------------
# Request limits, from upstream's documented ranges (docs/en/INFERENCE.md).
# ---------------------------------------------------------------------------
MAX_CAPTION_CHARS = 512
MAX_LYRICS_CHARS = 4096
MIN_DURATION_SECONDS = 10.0
MAX_DURATION_SECONDS = 600.0
DEFAULT_STEPS = 50  # upstream: 32-100 for non-turbo models
MAX_STEPS = 100
DEFAULT_GUIDANCE_SCALE = 7.0
MAX_SOURCE_SECONDS = 600.0  # longest source recording for cover / repaint

# ---------------------------------------------------------------------------
# Slurm allocation shape. Partition is deliberately not configured here -- see
# services/wan_animate/config.py's own comment. The DiT (~20 GB) and the 4B
# language model (~8 GB) are loaded fully onto one GPU in bf16, no offload.
# TYPICAL_RUN_SECONDS is a cold start measured on 2026-10-23 (165 s from submit to finished) --
# server.py::_typical_run_seconds prefers real recent job history.
# ---------------------------------------------------------------------------
MEM_PER_GPU = os.environ.get("ACE_STEP15_MEM_PER_GPU", "64G")
CPUS_PER_GPU = os.environ.get("ACE_STEP15_CPUS", "8")
JOB_TIME_LIMIT = os.environ.get("ACE_STEP15_JOB_TIME", "00:30:00")
TYPICAL_RUN_SECONDS = 165

# Part of the key that recognises a repeated request (common/dedup.py). Change it by hand when the
# weights or the generation code change what a request produces, so earlier results are not reused.
RESULT_VERSION = "1"

# ---------------------------------------------------------------------------
# Warm workers (see common/worker_pool.py): long-lived jobs (one GPU each) load
# the models once and serve queued requests one at a time. How many may run at
# once is this model's limit on the GPU capacity page (common/capacity.py), not
# a setting here.
# A worker with nothing to do for IDLE_MINUTES is stopped. JOB_TIME_LIMIT is
# the longest one request may run; WORKER_TIME_LIMIT is the Slurm time limit of
# a worker, and must be longer than that.
# ---------------------------------------------------------------------------
IDLE_MINUTES = float(os.environ.get("ACE_STEP15_IDLE_MINUTES", "10"))
WORKER_TIME_LIMIT = os.environ.get("ACE_STEP15_WORKER_TIME", "12:00:00")

def job_time_limit_seconds() -> int:
    """The longest one request may run, in seconds."""
    return slurm_time_seconds(JOB_TIME_LIMIT, "ACE_STEP15_JOB_TIME")


def worker_time_limit_seconds() -> int:
    return slurm_time_seconds(WORKER_TIME_LIMIT, "ACE_STEP15_WORKER_TIME")
