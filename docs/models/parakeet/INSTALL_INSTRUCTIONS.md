# Installation Instructions for Parakeet Service

## Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA driver 12.8 or higher
- Virtual environment (recommended)

## Installation Steps

### 1. Create and Activate Virtual Environment

```bash
python3 -m venv /home/naresh/venvs/parakeet-service
source /home/naresh/venvs/parakeet-service/bin/activate
```

### 2. Install PyTorch with CUDA 12.4

**Important:** Due to CUDA driver compatibility (driver supports CUDA 12.8, but system has CUDA toolkit 12.9), you must install PyTorch built for CUDA 12.4:

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Other Dependencies

```bash
pip install nemo_toolkit[asr]>=1.22.0
pip install fastapi>=0.104.0
pip install uvicorn[standard]>=0.24.0
pip install python-multipart>=0.0.6
pip install pydantic>=2.5.0
pip install requests>=2.31.0
pip install librosa>=0.10.1
pip install soundfile>=0.12.1
pip install omegaconf>=2.3.0
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### CUDA Error 35 (cudaErrorInsufficientDriver)

This error occurs when there's a mismatch between CUDA runtime version and driver version:

- **Cause**: PyTorch built with CUDA 12.9, but driver only supports CUDA 12.8
- **Solution**: Use PyTorch built for CUDA 12.4 (as shown above)

### Verification Commands

Check your CUDA setup:
```bash
# Check NVIDIA driver version
nvidia-smi

# Check CUDA toolkit version
nvcc --version

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

## Starting the Service

Once installed, start the service with:

```bash
./start_service.sh
```

Or manually:

```bash
source /home/naresh/venvs/parakeet-service/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8006
```


