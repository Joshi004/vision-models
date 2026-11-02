"""Stable Audio 3 Medium configuration: project paths, request limits, and this
backend's Slurm resource shape.

API-wide settings (host/port, job/upload storage, retention, partition
fallback) live in common/config.py instead -- this module only has what's
specific to Stable Audio 3 itself.

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
# models/stable_audio3/; its environment and weights live under the shared
# data directory (see common/config.py). The upstream code is the pinned
# submodule third_party/stable-audio-3, installed editable into the model's
# environment.
# ---------------------------------------------------------------------------
STABLE_AUDIO3_PROJECT_ROOT = common_config.REPO_ROOT / "models" / "stable_audio3"
STABLE_AUDIO3_PYTHON = common_config.ENVS_DIR / "stable_audio3" / "bin" / "python"
RUN_GENERATE_SCRIPT = STABLE_AUDIO3_PROJECT_ROOT / "runner" / "run_generate.py"
WORKER_SCRIPT = STABLE_AUDIO3_PROJECT_ROOT / "runner" / "worker.py"

# Holds hf/, a private Hugging Face cache with stabilityai/stable-audio-3-medium (which also
# carries the T5Gemma text encoder), fetched by models/stable_audio3/download_weights.sh. Loaded
# from disk, so a job never needs network access.
MODEL_DIR = Path(os.environ.get("STABLE_AUDIO3_MODEL_DIR", str(common_config.WEIGHTS_DIR / "stable_audio3")))

# ---------------------------------------------------------------------------
# Request limits. Upstream's README says 380 s for Medium, its launch material
# says ~285 s. A 380 s clip was generated on this cluster on 2026-10-23
# (docs/models/stable_audio3/README.md), so 380 s is the default; it is also the
# runner's own hard bound. STABLE_AUDIO3_MAX_DURATION can lower it.
# ---------------------------------------------------------------------------
MAX_PROMPT_CHARS = 2000  # upstream's examples are long descriptive paragraphs
MAX_DURATION_SECONDS = float(os.environ.get("STABLE_AUDIO3_MAX_DURATION", "380"))
if not 0 < MAX_DURATION_SECONDS <= 380:
    raise RuntimeError("STABLE_AUDIO3_MAX_DURATION must be between 0 and 380 seconds.")
DEFAULT_DURATION_SECONDS = 30.0
DEFAULT_STEPS = 8  # the post-trained Medium checkpoint's own default
MAX_STEPS = 50
DEFAULT_INIT_NOISE_LEVEL = 0.5
MAX_SOURCE_SECONDS = MAX_DURATION_SECONDS  # longest input recording for transform / inpaint

# ---------------------------------------------------------------------------
# Slurm allocation shape. Partition is deliberately not configured here -- see
# services/wan_animate/config.py's own comment. The model (1.4B, ~10 GB of
# files; upstream peak 6.5 GB VRAM) is loaded fully onto one GPU.
# TYPICAL_RUN_SECONDS is a cold start measured on 2026-10-23 (65-140 s from submit to finished) --
# server.py::_typical_run_seconds prefers real recent job history.
# ---------------------------------------------------------------------------
MEM_PER_GPU = os.environ.get("STABLE_AUDIO3_MEM_PER_GPU", "32G")
CPUS_PER_GPU = os.environ.get("STABLE_AUDIO3_CPUS", "8")
JOB_TIME_LIMIT = os.environ.get("STABLE_AUDIO3_JOB_TIME", "00:20:00")
TYPICAL_RUN_SECONDS = 100

# Part of the key that recognises a repeated request (common/dedup.py). Change it by hand when the
# weights or the generation code change what a request produces, so earlier results are not reused.
RESULT_VERSION = "1"

# ---------------------------------------------------------------------------
# Warm workers (see common/worker_pool.py): long-lived jobs (one GPU each) load
# the model once and serve queued requests one at a time. How many may run at
# once is this model's limit on the GPU capacity page (common/capacity.py), not
# a setting here.
# A worker with nothing to do for IDLE_MINUTES is stopped. JOB_TIME_LIMIT is the
# longest one request may run; WORKER_TIME_LIMIT is the Slurm time limit of a
# worker, and must be longer than that.
# ---------------------------------------------------------------------------
IDLE_MINUTES = float(os.environ.get("STABLE_AUDIO3_IDLE_MINUTES", "10"))
WORKER_TIME_LIMIT = os.environ.get("STABLE_AUDIO3_WORKER_TIME", "12:00:00")

def job_time_limit_seconds() -> int:
    """The longest one request may run, in seconds."""
    return slurm_time_seconds(JOB_TIME_LIMIT, "STABLE_AUDIO3_JOB_TIME")


def worker_time_limit_seconds() -> int:
    return slurm_time_seconds(WORKER_TIME_LIMIT, "STABLE_AUDIO3_WORKER_TIME")
