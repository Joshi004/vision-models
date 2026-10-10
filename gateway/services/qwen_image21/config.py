"""Qwen-Image-2.1 image generation configuration: project paths, accepted
reference-image formats, the aspect-ratio presets, request limits, and this
backend's Slurm resource shape.

API-wide settings (host/port, job/upload storage, retention, partition
fallback) live in common/config.py instead -- this module only has what's
specific to Qwen-Image-2.1 itself.

Zero third-party dependencies (stdlib only), same reasoning as
services/breeze_tts/config.py: nothing here should require fastapi/uvicorn.
"""

from __future__ import annotations

import os
from pathlib import Path

from common import config as common_config

# ---------------------------------------------------------------------------
# Project layout. This model's own code (the Slurm entrypoint) lives in
# models/qwen_image21/ in the repository; its environment and weights live
# under the shared data directory (see common/config.py). There is no
# third_party/ checkout: the model runs through diffusers' own
# QwenImage21Pipeline.
# ---------------------------------------------------------------------------
QWEN_IMAGE21_PROJECT_ROOT = common_config.REPO_ROOT / "models" / "qwen_image21"

# The project's own torch/CUDA environment. The API server's own env (gateway)
# never imports any of this.
QWEN_IMAGE21_PYTHON = common_config.ENVS_DIR / "qwen_image21" / "bin" / "python"

# Slurm job entrypoint, run with QWEN_IMAGE21_PYTHON.
RUN_GENERATE_SCRIPT = QWEN_IMAGE21_PROJECT_ROOT / "runner" / "run_generate.py"

# Local diffusers checkpoints, fetched by models/qwen_image21/download_weights.sh.
# Loaded from disk, so a job never needs network access (every Slurm job this
# API submits already runs with HF_HUB_OFFLINE=1).
_WEIGHTS_ROOT = common_config.WEIGHTS_DIR / "qwen_image21"
BASE_MODEL_DIR = Path(os.environ.get("QWEN_IMAGE21_BASE_DIR", str(_WEIGHTS_ROOT / "base")))
TURBO_MODEL_DIR = Path(os.environ.get("QWEN_IMAGE21_TURBO_DIR", str(_WEIGHTS_ROOT / "turbo")))

# "fast" = Turbo checkpoint (its own saved 8-step schedule);
# "quality" = base checkpoint (default 40 steps).
MODEL_DIR_BY_MODE = {"fast": TURBO_MODEL_DIR, "quality": BASE_MODEL_DIR}

RUN_PIPELINE_JOB_SCRIPT = common_config.MODEL_API_DIR / "common" / "run_pipeline_job.py"

# ---------------------------------------------------------------------------
# Output sizes. Upstream's recommended sizes for the model's native 2K
# resolution (README "Supported Aspect Ratios"); each is a multiple of 16.
# ---------------------------------------------------------------------------
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

# ---------------------------------------------------------------------------
# Reference images (edit endpoint). Checked up front by dispatch.py so an
# unsupported file gets an immediate 400 instead of a GPU-job failure.
# ---------------------------------------------------------------------------
REFERENCE_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
MAX_REFERENCE_IMAGES = 10  # the model's documented limit

# ---------------------------------------------------------------------------
# Request limits. Upstream's own example prompts run to ~6,000 characters (a
# fully specified poster), so the cap is generous; env-overridable.
# ---------------------------------------------------------------------------
MAX_PROMPT_CHARS = int(os.environ.get("QWEN_IMAGE21_MAX_PROMPT_CHARS", "8000"))
DEFAULT_QUALITY_STEPS = 40
MAX_QUALITY_STEPS = 100

# ---------------------------------------------------------------------------
# Slurm allocation shape for one image job. Partition is deliberately not
# configured here -- see services/wan_animate/config.py's own comment.
#
# Both checkpoints are ~31 GB in bf16 and are loaded fully onto the GPU (no
# CPU offload), which fits one 80 GB H100. TYPICAL_RUN_SECONDS is a
# placeholder for the first requests -- server.py::_typical_run_seconds
# prefers real recent job history the moment any exists.
# ---------------------------------------------------------------------------
MEM_PER_GPU = os.environ.get("QWEN_IMAGE21_MEM_PER_GPU", "64G")
CPUS_PER_GPU = os.environ.get("QWEN_IMAGE21_CPUS", "8")
JOB_TIME_LIMIT = os.environ.get("QWEN_IMAGE21_JOB_TIME", "00:30:00")
TYPICAL_RUN_SECONDS = 300
