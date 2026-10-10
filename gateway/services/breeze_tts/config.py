"""Breeze TTS 2 (BreezeBlue) text-to-speech configuration: project paths,
accepted reference-audio formats, request limits, and this backend's Slurm
resource shape.

API-wide settings (host/port, job/upload storage, retention, partition
fallback) live in common/config.py instead -- this module only has what's
specific to the Breeze TTS 2 project itself.

Zero third-party dependencies (stdlib only), same reasoning as
services/rvc/config.py: nothing here should require fastapi/uvicorn.
"""

from __future__ import annotations

import os
from pathlib import Path

from common import config as common_config

# ---------------------------------------------------------------------------
# Project layout. This model's own code (the Slurm entrypoint) lives in
# models/breeze_tts/ in the repository, the upstream project in
# third_party/breeze-tts/, and its environment and weights under the shared
# data directory (see common/config.py).
# ---------------------------------------------------------------------------
BREEZE_TTS_PROJECT_ROOT = common_config.REPO_ROOT / "models" / "breeze_tts"

# The project's own torch/CUDA venv. The API server's own venv
# (gateway env) never imports any of this.
BREEZE_TTS_PYTHON = common_config.ENVS_DIR / "breeze_tts" / "bin" / "python"

# Slurm job entrypoint, run with BREEZE_TTS_PYTHON (needs torch, qwen-tts and
# imageio-ffmpeg from that venv). Wraps upstream's infer.py and decodes any
# reference recording to WAV first -- see that script's own header comment.
RUN_SYNTHESIZE_SCRIPT = BREEZE_TTS_PROJECT_ROOT / "runner" / "run_synthesize.py"

# Local checkpoint directory (tokenizer, model, bundled audio_tokenizer).
# Loaded from disk, so a job never needs network access (every Slurm job this
# API submits already runs with HF_HUB_OFFLINE=1).
MODEL_DIR = Path(
    os.environ.get("BREEZE_TTS_MODEL_DIR", str(common_config.WEIGHTS_DIR / "breeze_tts" / "breeze-tts-2"))
)

RUN_PIPELINE_JOB_SCRIPT = common_config.MODEL_API_DIR / "common" / "run_pipeline_job.py"

# ---------------------------------------------------------------------------
# Accepted reference-recording extensions (voice clone / direction). Checked
# up front by dispatch.py so an unsupported file gets an immediate 400
# instead of a GPU-job failure. Same set as services/rvc/config.py: the
# project's run_synthesize.py decodes every reference to WAV via the ffmpeg
# binary bundled in its own venv (normalize_audio.py) before Breeze sees it.
# ---------------------------------------------------------------------------
REFERENCE_EXTENSIONS = frozenset({
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".aiff", ".aif",
    ".m4a", ".aac", ".mp4", ".webm", ".wma", ".caf", ".3gp", ".amr",
})

# ---------------------------------------------------------------------------
# Request limits. Upstream's infer.py caps generation at 1500 new tokens per
# request (MAX_NEW_TOKENS), so text far beyond one short passage would be cut
# off. MAX_TEXT_CHARS is a conservative guard, env-overridable; see GUIDE.md.
# Measured: 1000 English characters -> ~92 s of audio (job ~170 s cold). That
# is the only measured point; Chinese speech is far denser per character, so
# ~500 Chinese characters is an unmeasured estimate of the same ceiling.
# ---------------------------------------------------------------------------
MAX_TEXT_CHARS = int(os.environ.get("BREEZE_TTS_MAX_TEXT_CHARS", "1000"))
MAX_INSTRUCTION_CHARS = 500
MAX_REFERENCE_TEXT_CHARS = 1000

# ---------------------------------------------------------------------------
# Slurm allocation shape for one synthesis job. Partition is deliberately not
# configured here -- see services/wan_animate/config.py's own comment.
#
# Eager inference needs ~7.7 GiB of GPU memory (upstream README); every GPU on
# this cluster is an H100 80GB. TYPICAL_RUN_SECONDS is a placeholder for the
# first requests -- server.py::_typical_run_seconds prefers real recent job
# history the moment any exists (see the measured numbers in STATUS.md).
# ---------------------------------------------------------------------------
MEM_PER_GPU = os.environ.get("BREEZE_TTS_MEM_PER_GPU", "32G")
CPUS_PER_GPU = os.environ.get("BREEZE_TTS_CPUS", "4")
JOB_TIME_LIMIT = os.environ.get("BREEZE_TTS_JOB_TIME", "00:15:00")
TYPICAL_RUN_SECONDS = 90
