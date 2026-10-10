#!/bin/bash

# Script to restart Parakeet service after PyTorch upgrade
# This should be run on the worker node with GPU allocation

echo "🔄 Restarting Parakeet Service..."
echo ""

# Find and kill existing uvicorn processes
echo "Stopping existing service..."
pkill -f "uvicorn app:app" || echo "No existing service found"
sleep 2

# Navigate to service directory
cd /home/naresh/Vision/ModelService_Parakeet

# Activate virtual environment
source /home/naresh/venvs/parakeet-service/bin/activate

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA: {torch.version.cuda}'); print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Start the service
echo "Starting service..."
./start_service.sh


