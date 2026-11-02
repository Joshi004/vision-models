#!/bin/bash
# Slurm job entrypoint for one on-demand Parakeet transcription request,
# submitted by model-api (see model-api/services/parakeet/dispatch.py, which
# builds the argv forwarded below, and model-api/common/run_pipeline_job.py,
# which actually invokes this script on the compute node -- that wrapper
# owns the partial-path-rename-on-success / .failed-marker-on-failure
# convention; this script (and run_transcribe.py underneath it) only need
# to write --output-path and exit non-zero on any real failure).
#
# Only exists to set the same LD_LIBRARY_PATH override start_service.sh
# already uses for the legacy always-on service (see CUDA_FIX_SUMMARY.md):
# this venv's own PyTorch-bundled NVIDIA CUDA libraries (12.4) must come
# before the system's CUDA 12.9 toolkit on the library search path, or
# cuda-python's bindings fail with "CUDA failure! 35"
# (cudaErrorInsufficientDriver) against the driver's CUDA 12.8. A plain
# `sbatch --wrap="<venv>/bin/python run_transcribe.py ..."` would miss this
# env var entirely, since it's set here, not baked into the venv itself.
#
# Usage: run_transcribe.sh --input-path FILE --work-dir DIR \
#            --model-path FILE --output-path FILE
# Every flag is forwarded to run_transcribe.py unchanged -- this script
# parses none of them itself.
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Sets VENV_PATH and the LD_LIBRARY_PATH fix described above (shared with worker.sh).
# shellcheck source=cuda_env.sh
source "$SERVICE_DIR/cuda_env.sh"

# exec, not a plain call: replaces this shell process with python entirely,
# so python's own exit code is reported directly to Slurm/run_pipeline_job.py
# with no intermediate bash wrapper translation.
exec "$VENV_PATH/bin/python" "$SERVICE_DIR/run_transcribe.py" "$@"
