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
from common.slurm import slurm_time_seconds

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
WORKER_SCRIPT = H3_PROJECT_ROOT / "runner" / "worker.py"

# ---------------------------------------------------------------------------
# Video shape. Fixed by the released model (diffusers' MiniMax-H3 docs): 24 fps;
# frame counts are 17 * n + 5 (what the video VAE decodes); a 768-pixel short
# edge, with the area capped at 768 x 1344.
#
# The model is documented for 5 to 15 seconds (124 to 345 frames on that grid),
# but that window is NO LONGER ENFORCED: any length is rounded up to the next
# 17 * n + 5 frames, and the runner switches off diffusers' own 5-15 s check
# (run_h3.py::allow_any_length). Outside 5-15 s is EXPERIMENTAL: quality is
# unknown and long clips may run out of GPU memory. The only remaining ceiling
# is MAX_DURATION_SECONDS, a safety guard so one request cannot hold its GPUs
# for hours; raise it with H3_MAX_SECONDS.
# ---------------------------------------------------------------------------
FPS = 24
FRAMES_PER_CHUNK = 17
FRAMES_OFFSET = 5
DEFAULT_FRAMES = 124
MAX_DURATION_SECONDS = float(os.environ.get("H3_MAX_SECONDS", "60"))

LANDSCAPE_HEIGHT, LANDSCAPE_WIDTH = 768, 1344
PORTRAIT_HEIGHT, PORTRAIT_WIDTH = 1344, 768
SIZE_MULTIPLE = 32
# Explicit height/width bounds: the area the model was trained at is 768 x 1344;
# smaller canvases are the documented speed lever, larger ones are untested here
# and may not fit in the memory left for activations on the job's GPUs (see the
# Slurm allocation comment below). The sizer (services/h3/sizing.py) rejects a request
# whose sequence does not fit.
MIN_SIDE = 256
MAX_SIDE = 1536

# Reference limits (ref2va), the ones diffusers' pipeline enforces: at most 9 images, 3 videos and 3 audio clips,
# and 12 references of any kind in total. MiniMax also documents each video/audio clip as 2 to 15 s, with the
# videos adding up to at most 15 s and the audio clips likewise; those length rules are not enforced here (the
# pipeline cuts every reference to the output's length instead).
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_REFERENCES = 12

# ---------------------------------------------------------------------------
# Slurm allocation shape. Every job holds N GPUs on one node (N = 4 or 8, chosen
# per request by services/h3/sizing.py) and always runs "ulysses":
#   * While the prompt and references are encoded, the Qwen3-VL-32B encoder
#     (62 GiB of bf16 weights) is split across two GPUs (layers 0-31 and 32-63) on a
#     4-GPU job, and across four GPUs (16 layers each, g1 .. g4) on an 8-GPU job,
#     and the 33B transformer (61.7 GiB) sits on the first GPU.
#   * Then ALL N GPUs denoise together: the transformer is copied to every GPU
#     (61.7 GiB each) and each GPU computes every step on 1/N of the token
#     sequence (diffusers' "Ulysses" context parallelism), so the activation
#     memory per GPU shrinks as N grows. About 15 GiB per GPU is left for it.
# "split" (the transformer's blocks divided over two GPUs) stays in the runner as an
# operator fallback only. See models/h3/runner/run_h3.py.
# Partition is deliberately not configured here: every request may pick its own,
# falling back to common.config.FALLBACK_PARTITION -- see common/slurm.py::resolve_partition.
#
# Host RAM: the encoder's and the transformer's weights are read into RAM before they
# move onto their GPUs, about 144 GB at the same time, so the default of 256G for the
# whole job leaves a margin (it is per job now, not per GPU, because a job's size varies).
# ---------------------------------------------------------------------------
MEM = os.environ.get("H3_MEM", "256G")
CPUS_PER_GPU = int(os.environ.get("H3_CPUS_PER_GPU", "8"))
# The longest a job may be given, and so the longest estimate a request may have. Slurm's HH:MM:SS.
JOB_TIME_LIMIT = os.environ.get("H3_API_TIME", "01:30:00")

# ---------------------------------------------------------------------------
# Warm workers (see common/worker_pool.py). A request is not given a Slurm job of
# its own: long-lived jobs (4 or 8 GPUs each, as sized) each run queued requests
# one after another (models/h3/runner/worker.py); how many may run at once is
# H3's limit on the GPU capacity page (common/capacity.py), not a setting here. Only the first
# request on a worker pays for reading the checkpoints from shared storage; the
# next ones find them in the node's page cache (about 6 minutes saved, see the
# worker script). A worker with nothing to do for IDLE_MINUTES is stopped.
# Workers are separate per transformer checkpoint (`transformer` for t2va and
# fl2va, `transformer_ref` for ref2va) and per GPU count. Each request still has
# its own time limit (the sizer's, at most JOB_TIME_LIMIT); WORKER_TIME_LIMIT is
# the Slurm time limit of a worker and must be longer than that.
# ---------------------------------------------------------------------------
IDLE_MINUTES = float(os.environ.get("H3_IDLE_MINUTES", "10"))
WORKER_TIME_LIMIT = os.environ.get("H3_WORKER_TIME", "12:00:00")

# Optional diffusers attention backend for the transformer, e.g. "_flash_3_hub"
# (FlashAttention-3 on Hopper, roughly 2x faster than the default per measurements
# on this cluster). Unset = diffusers' default. Operator-selectable, not per-request.
# The hub kernels need the `kernels` package and a pre-filled Hugging Face cache,
# because jobs run offline -- see docs/models/h3/README.md before turning this on.
ATTENTION_BACKEND = os.environ.get("H3_ATTENTION_BACKEND") or None

# How the job's GPUs share the denoising (see the Slurm allocation comment above).
# Always "ulysses" for now (the runner's --parallelism flag); "split" only works
# with two GPUs, which the sizer never asks for, so it is not offered here.
PARALLELISM = "ulysses"

# ---------------------------------------------------------------------------
# Per-request sizing (services/h3/sizing.py). EVERY NUMBER BELOW IS AN INITIAL
# GUESS from the failed and the successful jobs seen so far, to be refitted from
# the usage log (DATA_DIR/metrics/h3_jobs.jsonl, see services/h3/usage_log.py).
# ---------------------------------------------------------------------------
# GPU counts a job may use, smallest first; the sizer takes the smallest that fits.
# "4" pins every job to four GPUs. Only 4 and 8 divide the transformer's 56 heads
# evenly while leaving room for the two encoder GPUs.
GPU_OPTIONS = tuple(sorted({int(n) for n in os.environ.get("H3_GPU_OPTIONS", "4,8").split(",")}))
if not GPU_OPTIONS or any(n not in (4, 8) for n in GPU_OPTIONS):
    raise RuntimeError(f"H3_GPU_OPTIONS must list 4 and/or 8, got {GPU_OPTIONS}.")

# A 4-GPU job that runs out of memory in the encoder or in the denoising is run once more on 8 GPUs
# (services/h3/oom_retry.py; the encoder then gets four GPUs instead of two). H3_OOM_RETRY=0 turns that off.
# Has no effect unless 8 is one of GPU_OPTIONS.
OOM_RETRY = os.environ.get("H3_OOM_RETRY", "1") != "0" and 8 in GPU_OPTIONS

# Largest share of the denoising sequence one GPU may hold (rows of the packed
# sequence: text + references + audio + video). Activations are about 0.18 MiB per
# row and about 15 GiB are free next to the weights, so ~80000 would just fit; 40000
# leaves a wide margin until real peaks are known.
ROWS_PER_GPU = int(os.environ.get("H3_ROWS_PER_GPU", "40000"))

# Largest token count the encoder may be given. It keeps every layer's output, about
# 0.63 MiB per token (60000 tokens = ~38 GiB, spread over its two GPUs, which each
# have ~48 GiB left after their half of the weights), plus attention and MLP
# workspace on top; the margin is deliberate. The failed 2-image request had 13,673.
MAX_ENCODER_TOKENS = int(os.environ.get("H3_MAX_ENCODER_TOKENS", "60000"))

# Prompt tokens: characters / 3.5 is a deliberately high ratio (structured English
# prompts are ~4 characters per token). Each reference adds a label and markers.
PROMPT_CHARS_PER_TOKEN = 3.5
TOKENS_PER_REFERENCE_LABEL = 16

# Seconds of one job, at 50 steps, for a 73000-row sequence (243 frames at 1344x768,
# no references), the baseline the denoising time is scaled from.
LOAD_SECONDS = 330        # reading the weights from shared storage
COPY_SECONDS = 60         # copying the transformer to the other GPUs
DECODE_SECONDS = 120      # video + audio VAE decode and mux
REFERENCE_ENCODE_SECONDS_PER_ROW = 0.0004   # VAE-encoding references, per row they add (a pure placeholder: never measured)
BASELINE_DENOISE_SECONDS = 862
BASELINE_ROWS = 73000
DENOISE_ROWS_EXPONENT = 1.5   # between linear (MLP) and quadratic (attention)
DEFAULT_STEPS = 50
# How many times faster than one GPU the denoising runs on N GPUs.
DENOISE_SPEEDUP = {4: 3.0, 8: 5.0}
# Without FlashAttention-3 the denoising takes about twice as long.
NO_FA3_SLOWDOWN = 2.0
# Slurm's --time is this many times the estimate plus a margin, within [MIN, JOB_TIME_LIMIT].
TIME_LIMIT_FACTOR = 2.0
TIME_LIMIT_MARGIN_SECONDS = 10 * 60
MIN_TIME_LIMIT_SECONDS = 30 * 60

# Fallback "typically takes about..." estimate shown alongside a queued/running
# job when there is not enough recent history yet for that exact pipeline (see
# server.py::_typical_run_seconds). Only used for jobs without their own estimate
# (older jobs); a job sized by sizing.py carries `estimated_run_seconds` instead.
TYPICAL_RUN_SECONDS = 20 * 60

# Part of the key that recognises a repeated request (common/dedup.py). Change it by hand when the
# weights or the generation code change what a request produces, so earlier results are not reused.
RESULT_VERSION = "1"


def job_time_limit_seconds() -> int:
    """JOB_TIME_LIMIT (Slurm's HH:MM:SS, as every service config here writes it) in seconds."""
    parts = JOB_TIME_LIMIT.split(":")
    if len(parts) != 3:
        raise RuntimeError(f"H3_API_TIME must look like HH:MM:SS, got {JOB_TIME_LIMIT!r}.")
    hours, minutes, seconds = (int(part) for part in parts)
    return hours * 3600 + minutes * 60 + seconds


def worker_time_limit_seconds() -> int:
    return slurm_time_seconds(WORKER_TIME_LIMIT, "H3_WORKER_TIME")
