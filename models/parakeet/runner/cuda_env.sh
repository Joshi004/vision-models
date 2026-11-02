# Sourced (not run) by run_transcribe.sh and worker.sh, which set SERVICE_DIR (this
# folder) first. Sets VENV_PATH to the Parakeet environment and puts that environment's own
# PyTorch-bundled NVIDIA CUDA libraries (12.4) ahead of the system's CUDA 12.9 toolkit on the
# library search path: without it, cuda-python's bindings fail with "CUDA failure! 35"
# (cudaErrorInsufficientDriver) against the driver's CUDA 12.8. See run_transcribe.sh's header.

REPO_ROOT="$(cd "$SERVICE_DIR/../../.." && pwd)"  # <repo>/models/parakeet/runner
DATA_DIR="${VISION_DATA_DIR:-$REPO_ROOT/../vision-data}"
VENV_PATH="$DATA_DIR/envs/parakeet"

NVIDIA_LIB_PATH="$VENV_PATH/lib/python3.10/site-packages/nvidia"
if [ -d "$NVIDIA_LIB_PATH" ]; then
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$NVIDIA_LIB_PATH/cublas/lib:$NVIDIA_LIB_PATH/cuda_cupti/lib:$NVIDIA_LIB_PATH/cuda_nvrtc/lib:$NVIDIA_LIB_PATH/cuda_runtime/lib:$NVIDIA_LIB_PATH/cudnn/lib:$NVIDIA_LIB_PATH/cufft/lib:$NVIDIA_LIB_PATH/curand/lib:$NVIDIA_LIB_PATH/cusolver/lib:$NVIDIA_LIB_PATH/cusparse/lib:$NVIDIA_LIB_PATH/nccl/lib:$NVIDIA_LIB_PATH/nvtx/lib:${LD_LIBRARY_PATH:-}"
fi
