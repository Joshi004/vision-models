"""TBDub (lip sync, 30-step Teacher) configuration: project paths and this
backend's Slurm resource shape.

API-wide settings (host/port, job/upload storage, retention,
partition fallback) live in common/config.py instead -- this module only
has what's specific to the TBDub project itself.

Zero third-party dependencies (stdlib only), same reasoning as
common/config.py and services/ltx/config.py: nothing here should require
fastapi/uvicorn to be installed.
"""

from __future__ import annotations

import os

from common import config as common_config

# ---------------------------------------------------------------------------
# Project layout. TBDub (github.com/TaoLiveAIGC/TBDub, Apache-2.0): the
# upstream source is third_party/TBDub/ (pinned submodule), this API's own
# wrapper is models/tbdub/runner/run_lipsync.py, and the environment and
# weights live under the shared data directory (see common/config.py;
# models/tbdub/download_weights.sh fetches the weights).
# ---------------------------------------------------------------------------
TBDUB_PROJECT_ROOT = common_config.REPO_ROOT / "models" / "tbdub"
TBDUB_REPO = common_config.REPO_ROOT / "third_party" / "TBDub"

# The torch/CUDA venv TBDub needs (Python 3.10, numpy 1.x, mediapipe, ...).
# It is deliberately separate from the LTX venvs, whose pins conflict with it:
# every lip-sync step runs as its own subprocess with this interpreter.
TBDUB_PYTHON = common_config.ENVS_DIR / "tbdub" / "bin" / "python"
WEIGHTS_DIR = common_config.WEIGHTS_DIR / "tbdub"
RUN_LIPSYNC_SCRIPT = TBDUB_PROJECT_ROOT / "runner" / "run_lipsync.py"

RUN_PIPELINE_JOB_SCRIPT = common_config.MODEL_API_DIR / "common" / "run_pipeline_job.py"

# TBDub assumes 25 fps video everywhere (its own checkpoint manifest enforces
# it). A request that asks for lip sync on an LTX job is generated at this
# rate, so the sync step needs no re-encoding; the standalone endpoint converts
# any other frame rate itself (see run_lipsync.py).
LIP_SYNC_FRAME_RATE = 25.0

# What an LTX audio-to-video job calls its own, un-synced video once lip sync
# has been requested; kept in the job folder after the job, so the sync can be
# redone (POST /v1/tbdub/videos/lip-sync with source_job_id) without
# regenerating the video.
BASE_VIDEO_NAME = "lipsync_base.mp4"

# Pipelines whose finished jobs can be re-synced through `source_job_id`.
SOURCE_PIPELINES = frozenset({"ltx:audio-to-video", "ltx25:audio-to-video"})

# ---------------------------------------------------------------------------
# Slurm allocation shape.
#
# The standalone endpoint: one H100. Measured 2026-10-12 on this cluster
# (portrait 1088x1920, 5 s / 126 frames): peak GPU memory ~29 GB, peak host RAM
# ~16 GB, 3m07s-4m11s per job, of which 30 denoising steps take ~34 s per
# 77-frame chunk (chunks advance 72 frames), i.e. roughly 12 s of GPU time per
# second of video plus ~2 min of loading and face tracking. The time limit is
# generous because that cost grows with video length.
# ---------------------------------------------------------------------------
MEM_PER_GPU = os.environ.get("TBDUB_MEM_PER_GPU", "64G")
CPUS_PER_GPU = os.environ.get("TBDUB_CPUS", "8")
JOB_TIME_LIMIT = os.environ.get("TBDUB_API_TIME", "02:00:00")

# The combined case: lip sync as the second step of an LTX audio-to-video job.
# The LTX job's own time limit is extended by this many seconds, and the lip
# sync step is given exactly this long before it is stopped (see
# run_pipeline_job.py) -- so a slow lip-sync step can never use up the time the
# already-finished video needs to be handed back. Measured step: 181-189 s for a
# 5 s clip (about 7 min expected for the longest, 20 s, LTX-2.5 clip), so this
# is several times what is needed.
LIPSYNC_EXTRA_SECONDS = int(os.environ.get("TBDUB_LIPSYNC_EXTRA_SECONDS", str(30 * 60)))

# Fallback "typically takes about..." estimate for a queued/running standalone
# lip-sync job until real history exists (see server.py::_typical_run_seconds):
# about what a ~10 s clip takes, extrapolated from the 5 s measurements above.
TYPICAL_RUN_SECONDS = 5 * 60


def extend_time_limit(limit: str, extra_seconds: int) -> str:
    """Add `extra_seconds` to a Slurm `--time` value in HH:MM:SS form (the
    form every service config here uses). Raises ValueError for any other
    form rather than guessing."""
    parts = limit.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected a time limit like HH:MM:SS, got {limit!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    total = hours * 3600 + minutes * 60 + seconds + extra_seconds
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
