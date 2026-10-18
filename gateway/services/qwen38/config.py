"""Qwen3.8-27B-FP8 text backend configuration: project paths, the shared
vLLM server's Slurm shape, and the request limits.

Unlike every other backend, this one does not run one Slurm job per
request. One vLLM server runs as a single Slurm job on one GPU; the gateway
starts it when a request needs it, sends the requests to it, and stops it
after QWEN38_IDLE_MINUTES without any (see server_manager.py).

API-wide settings live in common/config.py. Zero third-party dependencies
(stdlib only), same reasoning as the other services' config modules.
"""

from __future__ import annotations

import os
from pathlib import Path

from common import config as common_config

# ---------------------------------------------------------------------------
# Project layout. The code is in models/qwen38/; the environment and the
# weights live under the shared data directory (see common/config.py).
# ---------------------------------------------------------------------------
PROJECT_ROOT = common_config.REPO_ROOT / "models" / "qwen38"
SERVE_SCRIPT = PROJECT_ROOT / "runner" / "serve.sh"
ENV_DIR = common_config.ENVS_DIR / "qwen38"

MODEL_DIR = Path(
    os.environ.get("QWEN38_MODEL_DIR", str(common_config.WEIGHTS_DIR / "qwen38" / "Qwen3.8-27B-FP8"))
)
# Name clients see in results, and the revision models/qwen38/download_weights.sh pins.
SERVED_MODEL_NAME = "qwen3.8-27b"
MODEL_ID = "Qwen/Qwen3.8-27B-FP8"
MODEL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"

# Where the running server leaves its files: api_key (written here, read by
# serve.sh), server.json (written by serve.sh once vLLM is ready),
# slurm_job_id (written here, so a restarted gateway finds the server again)
# and server.log. Must be on storage every compute node sees.
STATE_DIR = Path(os.environ.get("QWEN38_STATE_DIR", str(common_config.DATA_DIR / "llm" / "qwen38")))
# vLLM's compile cache: a later cold start reuses it instead of compiling again.
CACHE_DIR = Path(os.environ.get("QWEN38_CACHE_DIR", str(common_config.DATA_DIR / "cache" / "vllm")))

# ---------------------------------------------------------------------------
# The server's Slurm job. One H100 holds the 27.7 GiB of FP8 weights plus the
# KV cache. Unset partition = the API's default (common_config.FALLBACK_PARTITION);
# on the pre-emptible `background` partition the server can be killed at any
# time, which fails the requests in flight -- use `main` to avoid that.
# ---------------------------------------------------------------------------
PARTITION = os.environ.get("QWEN38_PARTITION") or None
MEM_PER_GPU = os.environ.get("QWEN38_MEM_PER_GPU", "64G")
CPUS_PER_GPU = os.environ.get("QWEN38_CPUS", "8")
# Backstop: Slurm ends the server after this long even if it never goes idle;
# the next request starts a new one.
SERVER_TIME_LIMIT = os.environ.get("QWEN38_SERVER_TIME", "12:00:00")

IDLE_MINUTES = float(os.environ.get("QWEN38_IDLE_MINUTES", "20"))
IDLE_CHECK_SECONDS = 30.0
# How long a request waits for the server to become ready (queue + load +
# compile) before the job fails.
STARTUP_TIMEOUT_SECONDS = float(os.environ.get("QWEN38_STARTUP_TIMEOUT", "1800"))
# How long one chat completion may take before the job fails. A full
# MAX_OUTPUT_TOKENS answer takes about 11 minutes alone (~75 tokens/s) and longer
# while other requests share the GPU.
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("QWEN38_REQUEST_TIMEOUT", "1800"))
# Requests sent to the server at once; further jobs stay `queued` until one finishes.
MAX_CONCURRENT_REQUESTS = int(os.environ.get("QWEN38_MAX_CONCURRENT", "8"))

# ---------------------------------------------------------------------------
# Request limits.
# ---------------------------------------------------------------------------
MAX_MODEL_LEN = 65536  # tokens, prompt + answer (passed to vLLM as --max-model-len)
# Long multi-part answers with thinking (e.g. 17 scenes of frame and video prompts) used more than
# 27,000 tokens, mostly reasoning. 49,152 leaves 16,384 tokens of the window for the prompt, which
# covers MAX_PROMPT_CHARS. The KV cache (~612k tokens) still fits MAX_NUM_SEQS full-length requests.
MAX_OUTPUT_TOKENS = 49152
DEFAULT_MAX_TOKENS = 4096
# Total characters over all messages. A rough guard that keeps an ordinary
# prompt plus its answer inside MAX_MODEL_LEN; vLLM's own exact check still
# applies and its message is passed back to the client.
MAX_PROMPT_CHARS = 32000
MAX_SCHEMA_BYTES = 64 * 1024

# Warm-server estimate shown until real jobs exist (replaced by the median of
# recent runs). Measured with the server warm: 1-6 s for most answers (see
# docs/models/qwen38/README.md); excludes a server start.
TYPICAL_RUN_SECONDS = 5
