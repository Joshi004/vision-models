"""LTX-2.3-specific configuration: model/checkpoint paths and this backend's
Slurm resource shape.

API-wide settings (host/port, job/upload storage, retention,
partition fallback) live in common/config.py instead -- this module only
has what's specific to the LTX-2.3 project itself.

Zero third-party dependencies (stdlib only), same reasoning as
common/config.py: nothing here should require fastapi/uvicorn to be
installed.
"""

from __future__ import annotations

import os
from pathlib import Path

from common import config as common_config

# ---------------------------------------------------------------------------
# Project layout. LTX-2.3's own scripts live in models/ltx23/ in the
# repository, the upstream LTX-2 source in third_party/ltx-2.3/ (pinned
# submodule), and the environment and weights under the shared data
# directory (see common/config.py).
# ---------------------------------------------------------------------------
LTX_PROJECT_ROOT = common_config.REPO_ROOT / "models" / "ltx23"
LTX_REPO = common_config.REPO_ROOT / "third_party" / "ltx-2.3"
MODEL_DIR = common_config.WEIGHTS_DIR / "ltx23" / "ltx-2.3"
GEMMA_DIR = common_config.WEIGHTS_DIR / "ltx23" / "gemma-3-12b"

# The heavy torch/CUDA venv generate.sh already uses to actually run a
# pipeline. The API server's own venv (gateway env) never imports torch.
PIPELINE_PYTHON = common_config.ENVS_DIR / "ltx23" / "bin" / "python"

# Model checkpoints (same files generate.sh already requires).
DISTILLED_CKPT = MODEL_DIR / "ltx-2.3-22b-distilled-1.1.safetensors"
DEV_CKPT = MODEL_DIR / "ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA = MODEL_DIR / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
DISTILLED_LORA_STRENGTH = 0.8  # matches generate.sh's --quality recipe
UPSCALER = MODEL_DIR / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

RUN_PIPELINE_JOB_SCRIPT = common_config.MODEL_API_DIR / "common" / "run_pipeline_job.py"

# Audio-to-video entrypoint, run with PIPELINE_PYTHON: converts the input
# audio to stereo (the LTX-2.3 audio encoder rejects mono) and then runs
# ltx_pipelines.a2vid_two_stage -- see that script's own header comment.
RUN_A2VID_SCRIPT = LTX_PROJECT_ROOT / "runner" / "run_a2vid.py"

# ---------------------------------------------------------------------------
# Video/audio defaults (mirrors generate.sh)
# ---------------------------------------------------------------------------
DEFAULT_FRAME_RATE = 24.0
LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT = 1920, 1088
PORTRAIT_WIDTH, PORTRAIT_HEIGHT = 1088, 1920
DEFAULT_NUM_FRAMES = 121  # ~5s @ 24fps -- the pipelines' own default

# ---------------------------------------------------------------------------
# Slurm allocation shape for one LTX job. Partition is deliberately not
# configured here: every request may pick its own, falling back to
# common.config.FALLBACK_PARTITION -- see common/slurm.py::resolve_partition.
# Everything else about the allocation is not per-request today, so it
# stays as an operator-tunable env var, same names generate.sh already uses.
# ---------------------------------------------------------------------------
MEM_PER_GPU = os.environ.get("LTX_MEM_PER_GPU", "96G")
CPUS_PER_GPU = os.environ.get("LTX_CPUS", "8")
# Per-job wall-clock limit. Generous default: even --quality plus a cold
# Slurm queue wait comfortably fits in 30 minutes; a request that needs
# longer (e.g. a long --duration video) can't override this per-request
# today -- see services/ltx/dispatch.py.
JOB_TIME_LIMIT = os.environ.get("LTX_API_TIME", "00:30:00")

# Fallback "typically takes about..." estimate shown alongside a queued/
# running job when there isn't enough recent history yet for that exact
# pipeline (see common/job_store.py::recent_run_seconds and
# server.py::_typical_run_seconds) -- the top of this backend's own
# documented 5-8 minute budget (see GUIDE.md). One number for every
# LTX recipe, same simplification server.py's fallback table makes.
TYPICAL_RUN_SECONDS = 8 * 60
