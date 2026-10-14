"""MiniMax H3 configuration: project paths, the model's video shape limits, and
this backend's Slurm resource shape.

API-wide settings (host/port, job/upload storage, retention, partition
fallback) live in common/config.py instead -- this module only has what is
specific to H3.

Zero third-party dependencies (stdlib only), same reasoning as
common/config.py: nothing here should require fastapi/uvicorn to be installed.
"""

from __future__ import annotations

import os
from pathlib import Path

from common import config as common_config

# ---------------------------------------------------------------------------
# Project layout. H3 runs through diffusers' own MiniMax-H3 modular pipeline
# (diffusers 0.41), so there is no third_party/ checkout: this API's wrapper
# is models/h3/runner/run_h3.py, and the environment and weights live under
# the shared data directory (see common/config.py;
# models/h3/download_weights.sh fetches the weights).
# ---------------------------------------------------------------------------
H3_PROJECT_ROOT = common_config.REPO_ROOT / "models" / "h3"

# The torch/CUDA venv H3 needs (diffusers 0.41, transformers 5.19, torch 2.9.1).
# The API server's own venv (gateway env) never imports any of it.
H3_PYTHON = common_config.ENVS_DIR / "h3" / "bin" / "python"
RUN_H3_SCRIPT = H3_PROJECT_ROOT / "runner" / "run_h3.py"

# Local copy of MiniMaxAI/MiniMax-H3 in diffusers layout (transformer/,
# transformer_ref/, text_encoder/, vae/, audio_vae/, ...), fetched by
# models/h3/download_weights.sh. Loaded from disk, so a job never needs network
# access (every Slurm job this API submits already runs with HF_HUB_OFFLINE=1).
MODEL_DIR = Path(os.environ.get("H3_MODEL_DIR", str(common_config.WEIGHTS_DIR / "h3" / "MiniMax-H3")))

RUN_PIPELINE_JOB_SCRIPT = common_config.MODEL_API_DIR / "common" / "run_pipeline_job.py"

# ---------------------------------------------------------------------------
# Video shape. All fixed by the released model (diffusers' MiniMax-H3 docs):
# 24 fps; frame counts are 17 * n + 5 (what the video VAE decodes); 5 to 15
# seconds, which on that grid means 124 (5.17 s) to 345 (14.4 s) frames; a
# 768-pixel short edge, with the area capped at 768 x 1344.
# ---------------------------------------------------------------------------
FPS = 24
FRAMES_PER_CHUNK = 17
FRAMES_OFFSET = 5
MIN_FRAMES = 124
MAX_FRAMES = 345
DEFAULT_FRAMES = 124
MIN_DURATION_SECONDS = 5.0
MAX_DURATION_SECONDS = 15.0

LANDSCAPE_HEIGHT, LANDSCAPE_WIDTH = 768, 1344
PORTRAIT_HEIGHT, PORTRAIT_WIDTH = 1344, 768
SIZE_MULTIPLE = 32
# Explicit height/width bounds: the area the model was trained at is 768 x 1344;
# smaller canvases are the documented speed lever, larger ones are untested here
# and need more GPU memory than one 80 GB card has left after the transformer.
MIN_SIDE = 256
MAX_SIDE = 1536

# Reference limits (ref2va): 12 in total, of which at most 9 images, 3 videos, 3 audio.
# The audio-to-video endpoint takes exactly one audio reference.
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3

# ---------------------------------------------------------------------------
# Slurm allocation shape. TWO GPUs on one node per job: the Qwen3-VL-32B
# encoder (62 GiB of bf16 weights) on one, the 33B transformer (61.7 GiB) on the
# other -- see models/h3/runner/run_h3.py. Partition is deliberately not
# configured here: every request may pick its own, falling back to
# common.config.FALLBACK_PARTITION -- see common/slurm.py::resolve_partition.
#
# NONE of the numbers below has been measured on this cluster yet; they are
# estimates, to be replaced once real jobs have run (see docs/models/h3/README.md).
#   * Host RAM: both halves load into RAM before moving onto their GPU, about
#     144 GB at the same time (62 GiB encoder + 72 GiB transformer and VAEs), so
#     2 x 112G leaves a wide margin.
#   * CPUs: two processes read and convert weights in parallel.
#   * Time: about 5 min of loading (the two halves in parallel), plus 50
#     denoising steps. diffusers' docs: 960x544 runs ~2.3x faster per step than
#     the trained 1344x768, and an independent benchmark on one H100 measured
#     ~3 s/step at 960x544 for a 5 s clip. A 15 s clip has ~3x the tokens.
# ---------------------------------------------------------------------------
GPUS_PER_JOB = 2
MEM_PER_GPU = os.environ.get("H3_MEM_PER_GPU", "112G")
CPUS = os.environ.get("H3_CPUS", "16")
JOB_TIME_LIMIT = os.environ.get("H3_API_TIME", "01:30:00")

# Optional diffusers attention backend for the transformer, e.g. "_flash_3_hub"
# (FlashAttention-3 on Hopper, roughly 3x faster per diffusers' docs). Unset =
# diffusers' default. Operator-selectable, not per-request. The hub kernels need
# the `kernels` package and a pre-filled Hugging Face cache, because jobs run
# offline -- see docs/models/h3/README.md before turning this on.
ATTENTION_BACKEND = os.environ.get("H3_ATTENTION_BACKEND") or None

# Fallback "typically takes about..." estimate shown alongside a queued/running
# job when there is not enough recent history yet for that exact pipeline (see
# server.py::_typical_run_seconds). A deliberately pessimistic guess (unmeasured)
# that is replaced automatically by the median of real recent runs once any exist.
TYPICAL_RUN_SECONDS = 20 * 60
