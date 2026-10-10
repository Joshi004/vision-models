"""Wan-Animate v1 (Wan2.2-Animate-14B, replace mode) configuration: project
paths and this backend's Slurm resource shape.

API-wide settings (host/port, job/upload storage, retention,
partition fallback) live in common/config.py instead -- this module only
has what's specific to the Wan-Animate v1 project itself.

Zero third-party dependencies (stdlib only), same reasoning as
common/config.py and services/ltx/config.py: nothing here should require
fastapi/uvicorn to be installed.
"""

from __future__ import annotations

import os
from pathlib import Path

from common import config as common_config

# ---------------------------------------------------------------------------
# Project layout. Wan-Animate v1's own code lives in models/wan_animate/ in
# the repository (see docs/models/wan_animate/README.md for why the original
# project folder had a "v1" subfolder, and models/wan_animate/runner/
# run_replace.py for what this backend actually runs). The upstream Wan2.2
# source is third_party/Wan2.2/; the environment and weights live under the
# shared data directory (see common/config.py). v2 (when it exists) gets its
# own sibling services/ module and its own model folder, not a branch of
# this one -- the two are different pipelines with different environments.
# ---------------------------------------------------------------------------
WAN_ANIMATE_PROJECT_ROOT = common_config.REPO_ROOT / "models" / "wan_animate"

# The heavy torch/CUDA venv run_replace.py needs to actually run (flash-attn,
# sam2, decord, onnxruntime, ...). The API server's own venv (gateway env)
# never imports any of this.
WAN_ANIMATE_PYTHON = common_config.ENVS_DIR / "wan_animate" / "bin" / "python"
RUN_REPLACE_SCRIPT = WAN_ANIMATE_PROJECT_ROOT / "runner" / "run_replace.py"

RUN_PIPELINE_JOB_SCRIPT = common_config.MODEL_API_DIR / "common" / "run_pipeline_job.py"

# ---------------------------------------------------------------------------
# Slurm allocation shape for one replace-mode job. Partition is deliberately
# not configured here: every request may pick its own, falling back to
# common.config.FALLBACK_PARTITION -- see common/slurm.py::resolve_partition.
#
# These defaults come from a real measured smoke-test run on this cluster
# (single H100, the official repo's own replace-mode example: a 6.8s,
# 1280x720 driving video) -- see ModelService_Wan-Animate-2/README.md for
# the full breakdown. That run: ~25 min wall clock total, ~40.6 GB peak host
# RSS, ~74.6 GB peak GPU VRAM (out of 80 GB -- offload_model is on
# automatically for single-GPU jobs, and there isn't a lot of headroom left;
# a much longer or higher-resolution driving video could plausibly OOM). A
# much longer driving video also costs more wall-clock time (the model
# processes it in ~2.5s/77-frame segments, each ~6.5 min to denoise) and may
# need a longer JOB_TIME_LIMIT than this default covers; there's no
# per-request override for that today.
# ---------------------------------------------------------------------------
MEM_PER_GPU = os.environ.get("WAN_ANIMATE_MEM_PER_GPU", "64G")
CPUS_PER_GPU = os.environ.get("WAN_ANIMATE_CPUS", "8")
JOB_TIME_LIMIT = os.environ.get("WAN_ANIMATE_API_TIME", "02:00:00")

# Fallback "typically takes about..." estimate shown alongside a queued/
# running job when there isn't enough recent history yet (see
# common/job_store.py::recent_run_seconds and
# server.py::_typical_run_seconds) -- this backend's own documented,
# measured example run (see openapi_docs.py / STATUS.md). Real requests
# scale with driving-video length, so this is only ever a rough starting
# point until real history accumulates.
TYPICAL_RUN_SECONDS = 25 * 60
