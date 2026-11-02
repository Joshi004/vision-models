"""YuE2 song generation configuration: project paths, request limits, and this
backend's Slurm resource shape.

API-wide settings (host/port, job/upload storage, retention, partition
fallback) live in common/config.py instead -- this module only has what's
specific to YuE2 itself.

Zero third-party dependencies (stdlib only): nothing here should require
fastapi/uvicorn.
"""

from __future__ import annotations

import os
from pathlib import Path

from common import config as common_config
from common.slurm import slurm_time_seconds

# ---------------------------------------------------------------------------
# Project layout. This model's own code (runner scripts) lives in models/yue2/;
# its environment and weights live under the shared data directory (see
# common/config.py). The upstream code is the pinned submodule third_party/YuE,
# installed editable into the model's environment.
# ---------------------------------------------------------------------------
YUE2_PROJECT_ROOT = common_config.REPO_ROOT / "models" / "yue2"
YUE2_PYTHON = common_config.ENVS_DIR / "yue2" / "bin" / "python"
RUN_GENERATE_SCRIPT = YUE2_PROJECT_ROOT / "runner" / "run_generate.py"
WORKER_SCRIPT = YUE2_PROJECT_ROOT / "runner" / "worker.py"

# Holds YuE2-3B/ and YuE2-Vae/, fetched by models/yue2/download_weights.sh. Loaded from
# disk, so a job never needs network access.
MODEL_DIR = Path(os.environ.get("YUE2_MODEL_DIR", str(common_config.WEIGHTS_DIR / "yue2")))

# ---------------------------------------------------------------------------
# Request limits. Upstream documents no hard character limits; these keep one
# request well inside the model's 24,576-token context and are env-overridable.
# ---------------------------------------------------------------------------
MAX_STYLE_CHARS = int(os.environ.get("YUE2_MAX_STYLE_CHARS", "1000"))
MAX_LYRICS_CHARS = int(os.environ.get("YUE2_MAX_LYRICS_CHARS", "6000"))
MAX_ABC_CHARS = int(os.environ.get("YUE2_MAX_ABC_CHARS", "20000"))
MAX_SEED = 2**63 - 1  # upstream accepts [0, 2**63)

# ---------------------------------------------------------------------------
# Slurm allocation shape. Partition is deliberately not configured here -- see
# services/wan_animate/config.py's own comment. YuE2-3B (~7 GB) and its VAE are
# loaded fully onto one GPU in bf16 (upstream: about 11 GB peak, one request at
# a time). TYPICAL_RUN_SECONDS is a cold start measured on 2026-10-23 (150 s from submit to finished) --
# server.py::_typical_run_seconds prefers real recent job history.
# ---------------------------------------------------------------------------
MEM_PER_GPU = os.environ.get("YUE2_MEM_PER_GPU", "64G")
CPUS_PER_GPU = os.environ.get("YUE2_CPUS", "8")
JOB_TIME_LIMIT = os.environ.get("YUE2_JOB_TIME", "00:45:00")
TYPICAL_RUN_SECONDS = 150

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
IDLE_MINUTES = float(os.environ.get("YUE2_IDLE_MINUTES", "10"))
WORKER_TIME_LIMIT = os.environ.get("YUE2_WORKER_TIME", "12:00:00")

def job_time_limit_seconds() -> int:
    """The longest one request may run, in seconds."""
    return slurm_time_seconds(JOB_TIME_LIMIT, "YUE2_JOB_TIME")


def worker_time_limit_seconds() -> int:
    return slurm_time_seconds(WORKER_TIME_LIMIT, "YUE2_WORKER_TIME")
