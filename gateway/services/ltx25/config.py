"""LTX-2.5-specific configuration: model/checkpoint paths and this backend's
Slurm resource shape.

API-wide settings (host/port, job/upload storage, retention,
partition fallback) live in common/config.py instead -- this module only
has what's specific to the LTX-2.5 project itself.

LTX-2.5 ships as a *split* pack (one .safetensors per component) rather than
LTX-2.3's single fat checkpoint + separate Gemma directory, so every
component below is its own path. It also runs from its own project
directory and its own venv (ltx-2.5/), not LTX-2.3's: the 2.5 code needs a
newer transformers (Gemma 4 text encoder) than 2.3's pinned environment.

Zero third-party dependencies (stdlib only), same reasoning as
common/config.py: nothing here should require fastapi/uvicorn to be
installed.
"""

from __future__ import annotations

import os
from pathlib import Path

from common import config as common_config
from services.ltx import config as ltx_config

# ---------------------------------------------------------------------------
# Project layout. LTX-2.5's own scripts live in models/ltx25/ in the
# repository, the upstream LTX-2 source in third_party/ltx-2.5/ (pinned
# submodule), and the environment and weights under the shared data
# directory (see common/config.py).
# ---------------------------------------------------------------------------
LTX25_PROJECT_ROOT = common_config.REPO_ROOT / "models" / "ltx25"
LTX25_REPO = common_config.REPO_ROOT / "third_party" / "ltx-2.5"
MODEL_DIR = common_config.WEIGHTS_DIR / "ltx25" / "ltx-2.5"

# The torch/CUDA venv that actually runs a pipeline. The API server's own
# venv (gateway env) never imports torch.
PIPELINE_PYTHON = common_config.ENVS_DIR / "ltx25" / "bin" / "python"

# ---------------------------------------------------------------------------
# Model components (see ltx-2.5/download_weights.sh for where each comes from).
# ---------------------------------------------------------------------------
DISTILLED_TRANSFORMER = MODEL_DIR / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
TEXT_ENCODER = MODEL_DIR / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
AUDIO_VAE = MODEL_DIR / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"
SPATIAL_UPSCALER = MODEL_DIR / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
DURATION_HEAD = MODEL_DIR / "model_patches" / "ltx-2.5-duration-head-bf16.safetensors"
# Stage-2 detailing IC-LoRA used only by the "quality" (DFR) recipe.
DETAILING_LORA = MODEL_DIR / "loras" / "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"

# Two interchangeable video VAEs ship with 2.5: "diffusion" (upstream's
# higher-quality decoder) and "conv" (lighter, no extra dependencies).
# Operator-selectable, not per-request.
#
# Default is "conv", from the pilot on this cluster (see STATUS.md): this
# cluster's CUDA 12.8 driver can't run upstream's preferred torch/natten
# build, so the diffusion decoder runs on its Triton fallback here -- and a
# 10-second (241-frame) clip decoded that way came out with its final ~0.25s
# replaced by a flat grey block, while the identical clip through the conv
# VAE was clean. 5-second clips were clean with both, and the two looked
# nearly identical side by side, so "conv" costs almost nothing in quality
# and is reliable at every length tried.
_VIDEO_VAE_FILES = {
    "diffusion": MODEL_DIR / "vae" / "ltx-2.5-video-vae-bf16.safetensors",
    "conv": MODEL_DIR / "vae" / "ltx-2.5-video-vae-conv-bf16.safetensors",
}
VIDEO_VAE_KIND = os.environ.get("LTX25_VIDEO_VAE", "conv")
if VIDEO_VAE_KIND not in _VIDEO_VAE_FILES:
    raise ValueError(f"LTX25_VIDEO_VAE must be one of {sorted(_VIDEO_VAE_FILES)}, got {VIDEO_VAE_KIND!r}")
VIDEO_VAE = _VIDEO_VAE_FILES[VIDEO_VAE_KIND]

# --enhance-prompt on the 2.5 pipelines needs a *generative* instruct Gemma
# in addition to the bundled Gemma 4 text encoder (which can only encode, not
# generate). LTX-2.3's Gemma 3 12B instruct directory already on disk is
# exactly that, so it's reused instead of downloading a second copy.
ENHANCER_GEMMA_DIR = Path(
    os.environ.get("LTX25_ENHANCER_GEMMA_DIR", str(ltx_config.GEMMA_DIR))
)

RUN_PIPELINE_JOB_SCRIPT = common_config.MODEL_API_DIR / "common" / "run_pipeline_job.py"

# Video/audio shape defaults (24 fps, 1920x1088 / 1088x1920, 121 frames, the
# 8k+1 frame rule) are deliberately not repeated here: LTX-2.5 uses exactly
# LTX-2.3's, so this backend reuses services/ltx/video_shape.py and
# VideoShapeMixin instead of keeping a second copy that could drift.

# ---------------------------------------------------------------------------
# Slurm allocation shape for one LTX-2.5 job. Partition is deliberately not
# configured here: every request may pick its own, falling back to
# common.config.FALLBACK_PARTITION -- see common/slurm.py::resolve_partition.
# ---------------------------------------------------------------------------
MEM_PER_GPU = os.environ.get("LTX25_MEM_PER_GPU", "96G")
CPUS_PER_GPU = os.environ.get("LTX25_CPUS", "8")
# Per-job wall-clock limit. The "quality" (DFR) recipe is slower than
# "fast", so this is more generous than LTX-2.3's 30 minutes.
JOB_TIME_LIMIT = os.environ.get("LTX25_API_TIME", "00:45:00")

# Fallback "typically takes about..." estimate shown alongside a queued/
# running job when there isn't enough recent history yet for that exact
# pipeline (see server.py::_typical_run_seconds). One number for every
# LTX-2.5 recipe: set to the top of the measured range (~4.5-10.5 min across
# fast/quality/retake, depending mostly on how cold the weight cache was --
# see STATUS.md's LTX-2.5 section), so a job isn't shown an optimistic
# estimate. Replaced automatically by the median of real recent runs once
# any exist.
TYPICAL_RUN_SECONDS = 10 * 60
