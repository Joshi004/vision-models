"""RVC (via the Applio fork) voice conversion configuration: project paths,
the curated voice registry, and this backend's Slurm resource shape.

API-wide settings (host/port, job/upload storage, retention,
partition fallback) live in common/config.py instead -- this module only
has what's specific to the RVC/Applio project itself.

Zero third-party dependencies (stdlib only), same reasoning as
common/config.py and services/ltx/config.py: nothing here should require
fastapi/uvicorn to be installed.
"""

from __future__ import annotations

import os
from pathlib import Path

from common import config as common_config

# ---------------------------------------------------------------------------
# Project layout. RVC via Applio (github.com/IAHispano/Applio, MIT): this
# model's own wrapper scripts live in models/rvc/ in the repository, the
# upstream Applio source in third_party/Applio/ (pinned submodule), and the
# environment and voices under the shared data directory (see
# common/config.py).
# ---------------------------------------------------------------------------
RVC_PROJECT_ROOT = common_config.REPO_ROOT / "models" / "rvc"

# The heavy torch/CUDA venv core.py needs to actually run. The API server's
# own venv (gateway env) never imports any of this.
RVC_PYTHON = common_config.ENVS_DIR / "rvc" / "bin" / "python"
CORE_SCRIPT = common_config.REPO_ROOT / "third_party" / "Applio" / "core.py"

# Wrapper scripts, both run with RVC_PYTHON (Applio's own venv interpreter,
# not whatever system python3 a Slurm job would otherwise get) so their own
# `import normalize_audio` (which needs imageio-ffmpeg) and their
# subprocess call into core.py both resolve correctly. Each fixes the same
# two real gaps between Applio's own CLI and what running it as a Slurm job
# needs:
#   1. core.py's own module-level code resolves an internal file
#      (rvc/lib/tools/tts_voices.json) relative to the process's cwd, not
#      its own __file__, so it fails outright unless invoked with Applio's
#      project root as cwd -- a Slurm job doesn't start there by default.
#   2. This deployment's libsndfile (1.2.2, no ffmpeg fallback) can't
#      decode AAC/M4A and other real-world recording formats that Applio's
#      own load_audio_infer()/convert_audio_batch() only ever hand straight
#      to soundfile.read(). Both wrappers decode every input to WAV first
#      (normalize_audio.py's to_wav(), via the ffmpeg binary already
#      bundled in this venv through imageio-ffmpeg -- no system ffmpeg
#      exists on this cluster) before core.py ever sees it.
# See each script's own header comment for the fuller story -- run_batch.py
# also covers two further gaps specific to batch-infer, plus a real naming
# trap (`batch-infer`, hyphenated, differs from its own Python function's
# name).
RUN_CONVERT_SCRIPT = RVC_PROJECT_ROOT / "runner" / "run_convert.py"
RUN_BATCH_SCRIPT = RVC_PROJECT_ROOT / "runner" / "run_batch.py"

RUN_PIPELINE_JOB_SCRIPT = common_config.MODEL_API_DIR / "common" / "run_pipeline_job.py"

# ---------------------------------------------------------------------------
# Curated voice registry. Unlike LTX/Wan-Animate's per-request assets, an
# RVC voice model (a .pth + .index pair) is meant to be reused across many
# requests, not uploaded fresh each time -- see VOICE_CONVERSION_RESEARCH.md
# section 1 for the fuller reasoning. Every subfolder of VOICES_DIR
# containing both a model.pth and a model.index becomes one selectable
# voice, named after its folder.
#
# Scanned once at import time -- same "restart to pick up changes"
# convention every other config in this API already follows (see
# STATUS.md section 6). Add a new voice by dropping
# voices/<name>/{model.pth,model.index} in and restarting the API process.
# ---------------------------------------------------------------------------
VOICES_DIR = Path(os.environ.get("RVC_VOICES_DIR", str(common_config.WEIGHTS_DIR / "rvc" / "voices")))


def _scan_voice_registry(voices_dir: Path) -> dict[str, dict[str, Path]]:
    registry: dict[str, dict[str, Path]] = {}
    if not voices_dir.is_dir():
        return registry
    for entry in sorted(voices_dir.iterdir()):
        if not entry.is_dir():
            continue
        pth_path = entry / "model.pth"
        index_path = entry / "model.index"
        if pth_path.is_file() and index_path.is_file():
            registry[entry.name] = {"pth_path": pth_path, "index_path": index_path}
    return registry


VOICE_REGISTRY: dict[str, dict[str, Path]] = _scan_voice_registry(VOICES_DIR)

# ---------------------------------------------------------------------------
# Accepted source-recording extensions, shared by both /convert and
# /batch-convert -- checked up front by dispatch.py before a job is ever
# submitted, so an unsupported file gets an immediate 400 instead of a
# confusing GPU-job failure partway through. Used to be batch-only and
# narrower (this deployment's libsndfile 1.2.2, no ffmpeg fallback, could
# only decode WAV/MP3/FLAC/OGG/Opus/AIFF directly) -- both endpoints' own
# Slurm-side wrappers (run_convert.py / run_batch.py) now decode every
# input to WAV first, via the ffmpeg binary already bundled in this venv
# through imageio-ffmpeg (no system ffmpeg exists on this cluster) -- see
# normalize_audio.py. This list is every format actually confirmed to
# demux with that binary (checked directly against a real `ffmpeg
# -demuxers` on this cluster), not exhaustive of everything ffmpeg could
# ever support.
# ---------------------------------------------------------------------------
INPUT_EXTENSIONS = frozenset({
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".aiff", ".aif",
    ".m4a", ".aac", ".mp4", ".webm", ".wma", ".caf", ".3gp", ".amr",
})

# ---------------------------------------------------------------------------
# Slurm allocation shape for one convert job. Partition is deliberately not
# configured here -- see services/wan_animate/config.py's own comment on
# the same pattern.
#
# These numbers come from a real measured smoke-test run on this cluster
# (single GPU, a ~10.7s speech clip, default rmvpe pitch extraction):
# `core.py infer` itself reported "Conversion completed... in 19.83
# seconds"; the full srun round-trip (queueing + node allocation + the run
# itself) was ~70s. TYPICAL_RUN_SECONDS below is a rough blend of the two
# until real job history accumulates through the API itself (see
# server.py::_typical_run_seconds, which prefers real recent history over
# this fallback the moment any exists) -- expect this to self-correct
# after a handful of real requests.
# ---------------------------------------------------------------------------
MEM_PER_GPU = os.environ.get("RVC_MEM_PER_GPU", "16G")
CPUS_PER_GPU = os.environ.get("RVC_CPUS", "4")
JOB_TIME_LIMIT = os.environ.get("RVC_JOB_TIME", "00:15:00")
TYPICAL_RUN_SECONDS = 60

# ---------------------------------------------------------------------------
# Batch conversion (POST /v1/rvc/batch-convert): many recordings, one
# installed voice, one Slurm job, one output.zip. Same GPU/mem/CPU shape as
# a single convert job above (still exactly one GPU doing one file at a
# time internally -- see run_batch.py) -- only the time limit and job name
# differ.
# ---------------------------------------------------------------------------

# A fixed name (not per-job, unlike every other backend's job-name) plus
# --dependency=singleton at submission time (see dispatch.py) makes Slurm
# run at most one batch at a time -- convert_audio_batch() reads and then
# deletes one shared, hardcoded path (assets/infer_pid.txt) in Applio's own
# project root for the *whole* batch's duration; two batches overlapping
# would race on that same file and spuriously fail whichever one finishes
# second. Single-file /convert jobs never touch that file and are
# unaffected -- only batch jobs share this job name.
BATCH_SLURM_JOB_NAME = "rvc-api-batch"

# Unmeasured guesses (no batch has actually run on this cluster yet,
# unlike TYPICAL_RUN_SECONDS above) -- both are env-overridable for that
# reason. Revisit once real batch job history exists.
BATCH_MAX_FILES = int(os.environ.get("RVC_BATCH_MAX_FILES", "50"))
BATCH_JOB_TIME_LIMIT = os.environ.get("RVC_BATCH_JOB_TIME", "02:00:00")
