"""Parakeet TDT transcription configuration: project paths, accepted input
formats, and this backend's Slurm resource shape.

API-wide settings (host/port, job/upload storage, retention,
partition fallback) live in common/config.py instead -- this module only
has what's specific to the Parakeet project itself.

Zero third-party dependencies (stdlib only), same reasoning as
common/config.py and services/rvc/config.py: nothing here should require
fastapi/uvicorn to be installed.
"""

from __future__ import annotations

import os
from pathlib import Path

from common import config as common_config

# ---------------------------------------------------------------------------
# Project layout. This model's own code lives in models/parakeet/ in the
# repository; its environment and weights live under the shared data
# directory (see common/config.py).
# ---------------------------------------------------------------------------
PARAKEET_PROJECT_ROOT = common_config.REPO_ROOT / "models" / "parakeet"

# The Slurm job entrypoint -- a small bash wrapper (not the venv's python
# directly) because it needs to set LD_LIBRARY_PATH *before* python starts,
# pointing at this venv's own PyTorch-bundled CUDA libraries (12.4) ahead of
# the system's CUDA 12.9 toolkit -- without it, cuda-python's bindings fail
# against the driver's CUDA 12.8 ("CUDA failure! 35", cudaErrorInsufficientDriver;
# see ModelService_Parakeet/CUDA_FIX_SUMMARY.md). See that project's own
# run_transcribe.sh/run_transcribe.py for the full story.
RUN_TRANSCRIBE_SCRIPT = PARAKEET_PROJECT_ROOT / "runner" / "run_transcribe.sh"

# The local .nemo checkpoint every transcription job loads via
# ASRModel.restore_from() -- never from_pretrained(), so a job never needs
# network access to fetch it (every Slurm job this API submits already runs
# with HF_HUB_OFFLINE=1 regardless -- see dispatch.py's _submit_slurm_job).
MODEL_PATH = Path(
    os.environ.get(
        "PARAKEET_MODEL_PATH",
        str(common_config.WEIGHTS_DIR / "parakeet" / "parakeet-tdt-0.6b-v3.nemo"),
    )
)

RUN_PIPELINE_JOB_SCRIPT = common_config.MODEL_API_DIR / "common" / "run_pipeline_job.py"

# ---------------------------------------------------------------------------
# Accepted input formats -- checked up front by dispatch.py before a job is
# ever submitted, so an unsupported file gets an immediate 400 instead of a
# confusing GPU-job failure partway through. Every entry confirmed directly
# against this venv's own bundled ffmpeg binary (`ffmpeg -demuxers`, version
# 7.0.2-static) on this cluster -- same binary architecture
# ModelService_RVC-Applio's own normalize_audio.py already uses (see that
# project's INPUT_EXTENSIONS for the audio half of this list, confirmed
# there). Video containers (.mp4/.mov/.mkv/.webm/.avi) are accepted too --
# run_transcribe.py's ffmpeg decode step only ever reads the first audio
# stream (`-map 0:a:0`), so a video works exactly like a plain audio file.
# ---------------------------------------------------------------------------
INPUT_EXTENSIONS = frozenset({
    # Audio (same set confirmed working for ModelService_RVC-Applio)
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".aiff", ".aif",
    ".m4a", ".aac", ".wma", ".caf", ".3gp", ".amr",
    # Video containers -- only the first audio stream is used
    ".mp4", ".mov", ".mkv", ".webm", ".avi",
})

# ---------------------------------------------------------------------------
# Slurm allocation shape for one transcription job. Partition is
# deliberately not configured here -- see services/wan_animate/config.py's
# own comment on the same pattern.
#
# These numbers come from 5 real jobs submitted through this API itself
# (the "verify" step): a 10s clip (two formats of the same recording), a
# 17s video, and a 35-minute audiobook chapter that exercised chunking
# (4 chunks). The 4 that succeeded took 63-117s end to end (queueing +
# node allocation + the run itself), median ~94s -- TYPICAL_RUN_SECONDS
# below is a round number in that range. Model loading, not audio length,
# dominated: the 35-minute chunked file (71s) finished faster than the two
# 10-second clips (117s each) simply because it landed on a node with a
# warmer filesystem cache for the checkpoint -- node-to-node variance
# outweighed the 200x difference in audio duration here. This fallback
# only matters until real job history exists -- server.py's
# _typical_run_seconds already prefers that the moment any exists, and it
# does now.
# ---------------------------------------------------------------------------
MEM_PER_GPU = os.environ.get("PARAKEET_MEM_PER_GPU", "32G")
CPUS_PER_GPU = os.environ.get("PARAKEET_CPUS", "4")
JOB_TIME_LIMIT = os.environ.get("PARAKEET_JOB_TIME", "01:00:00")
TYPICAL_RUN_SECONDS = 90
