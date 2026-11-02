"""Stable Audio Open 1.0 configuration: project paths, request limits, and this
backend's Slurm resource shape.

API-wide settings (host/port, job/upload storage, retention, partition
fallback) live in common/config.py instead -- this module only has what's
specific to Stable Audio Open itself.

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
# models/stable_audio_open/; its environment and weights live under the shared
# data directory (see common/config.py). The upstream code is the pinned
# submodule third_party/stable-audio-tools, installed editable into the
# model's environment.
# ---------------------------------------------------------------------------
STABLE_AUDIO_OPEN_PROJECT_ROOT = common_config.REPO_ROOT / "models" / "stable_audio_open"
STABLE_AUDIO_OPEN_PYTHON = common_config.ENVS_DIR / "stable_audio_open" / "bin" / "python"
RUN_GENERATE_SCRIPT = STABLE_AUDIO_OPEN_PROJECT_ROOT / "runner" / "run_generate.py"
WORKER_SCRIPT = STABLE_AUDIO_OPEN_PROJECT_ROOT / "runner" / "worker.py"

# Holds model/ (model_config.json + model.safetensors) and hf/ (the T5 text encoder's cache),
# fetched by models/stable_audio_open/download_weights.sh. Loaded from disk, so a job never
# needs network access.
MODEL_DIR = Path(os.environ.get("STABLE_AUDIO_OPEN_MODEL_DIR", str(common_config.WEIGHTS_DIR / "stable_audio_open")))

# ---------------------------------------------------------------------------
# Request limits. The model's fixed window is 2,097,152 samples = 47.55 s at
# 44.1 kHz; the runner enforces the same 47 s.
# ---------------------------------------------------------------------------
MAX_PROMPT_CHARS = 1000
MAX_DURATION_SECONDS = 47.0
DEFAULT_DURATION_SECONDS = 30.0
DEFAULT_STEPS = 100  # the stable-audio-tools example; the model card example uses 200
MAX_STEPS = 250
DEFAULT_CFG_SCALE = 7.0
MIN_INIT_NOISE_LEVEL = 0.05
MAX_INIT_NOISE_LEVEL = 20.0
DEFAULT_INIT_NOISE_LEVEL = 0.5
MAX_SOURCE_SECONDS = 120.0  # longest input recording; only its first MAX_DURATION_SECONDS are used

# ---------------------------------------------------------------------------
# Slurm allocation shape. Partition is deliberately not configured here -- see
# services/wan_animate/config.py's own comment. The model (~5 GB in float32,
# plus a T5 text encoder) is loaded fully onto one GPU. TYPICAL_RUN_SECONDS (measured on 2026-10-23: about 2 minutes from submit to finished on a cold start) is
# a placeholder for the first requests -- server.py::_typical_run_seconds
# prefers real recent job history.
# ---------------------------------------------------------------------------
MEM_PER_GPU = os.environ.get("STABLE_AUDIO_OPEN_MEM_PER_GPU", "32G")
CPUS_PER_GPU = os.environ.get("STABLE_AUDIO_OPEN_CPUS", "8")
JOB_TIME_LIMIT = os.environ.get("STABLE_AUDIO_OPEN_JOB_TIME", "00:20:00")
TYPICAL_RUN_SECONDS = 120

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
IDLE_MINUTES = float(os.environ.get("STABLE_AUDIO_OPEN_IDLE_MINUTES", "10"))
WORKER_TIME_LIMIT = os.environ.get("STABLE_AUDIO_OPEN_WORKER_TIME", "12:00:00")

def job_time_limit_seconds() -> int:
    """The longest one request may run, in seconds."""
    return slurm_time_seconds(JOB_TIME_LIMIT, "STABLE_AUDIO_OPEN_JOB_TIME")


def worker_time_limit_seconds() -> int:
    return slurm_time_seconds(WORKER_TIME_LIMIT, "STABLE_AUDIO_OPEN_WORKER_TIME")
