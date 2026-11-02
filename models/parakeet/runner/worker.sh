#!/bin/bash
# Slurm entrypoint of a warm Parakeet worker, started by the gateway (see
# gateway/services/parakeet/dispatch.py and gateway/common/worker_pool.py). A wrapper for
# the same reason as run_transcribe.sh: the CUDA library path must be set before Python
# starts. Every flag is forwarded to worker.py unchanged.
#
# Usage: worker.sh --worker-dir DIR --model-path FILE --idle-exit-seconds N
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cuda_env.sh
source "$SERVICE_DIR/cuda_env.sh"

exec "$VENV_PATH/bin/python" "$SERVICE_DIR/worker.py" "$@"
