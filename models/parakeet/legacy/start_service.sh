#!/bin/bash

# Parakeet Large ASR Service Startup Script
# This script starts the FastAPI server for Parakeet Large ASR model

# Exit on any error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}Starting Parakeet Large ASR Service...${NC}"

# Load environment variables from config.env if it exists
if [ -f "$SERVICE_DIR/config.env" ]; then
    export $(cat "$SERVICE_DIR/config.env" | grep -v '^#' | xargs)
fi

# Set CUDA environment variables
# Using PyTorch's bundled NVIDIA CUDA libraries instead of system CUDA
# System has CUDA 12.9, but driver supports 12.8 - PyTorch bundled libraries match the driver

# Check and create virtual environment if it doesn't exist
VENV_PATH="/home/naresh/venvs/parakeet-service"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating it...${NC}"
    # Ensure venvs directory exists
    mkdir -p /home/naresh/venvs
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}Virtual environment created${NC}"
    
    # Activate and install dependencies
    source "$VENV_PATH/bin/activate"
    echo -e "${YELLOW}Installing dependencies (this may take a few minutes)...${NC}"
    pip install --upgrade pip
    cd "$SERVICE_DIR"
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed${NC}"
else
    # Activate existing virtual environment
    source "$VENV_PATH/bin/activate"
    echo -e "${YELLOW}Virtual environment activated${NC}"
fi

# Set LD_LIBRARY_PATH to include CUDA driver and PyTorch's bundled NVIDIA CUDA libraries
# This allows PyTorch to find CUDA without using system CUDA 12.9
NVIDIA_LIB_PATH="$VENV_PATH/lib/python3.10/site-packages/nvidia"
if [ -d "$NVIDIA_LIB_PATH" ]; then
    # Add CUDA driver path first, then all NVIDIA library paths
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$NVIDIA_LIB_PATH/cublas/lib:$NVIDIA_LIB_PATH/cuda_cupti/lib:$NVIDIA_LIB_PATH/cuda_nvrtc/lib:$NVIDIA_LIB_PATH/cuda_runtime/lib:$NVIDIA_LIB_PATH/cudnn/lib:$NVIDIA_LIB_PATH/cufft/lib:$NVIDIA_LIB_PATH/curand/lib:$NVIDIA_LIB_PATH/cusolver/lib:$NVIDIA_LIB_PATH/cusparse/lib:$NVIDIA_LIB_PATH/nccl/lib:$NVIDIA_LIB_PATH/nvtx/lib:${LD_LIBRARY_PATH:-}"
    echo -e "${GREEN}Using PyTorch bundled CUDA libraries with system CUDA driver${NC}"
fi

# Datasets server configuration
DATASETS_DIR="/home/naresh/datasets"
DATASETS_PORT=8080

# Check if port 8080 is already in use
if lsof -Pi :$DATASETS_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}HTTP server already running on port $DATASETS_PORT - reusing existing server${NC}"
    echo "  Media files accessible at: http://localhost:$DATASETS_PORT/"
else
    echo -e "${GREEN}Starting HTTP server for datasets folder...${NC}"
    echo "  Directory: $DATASETS_DIR"
    echo "  Port: $DATASETS_PORT"
    
    # Check if datasets directory exists
    if [ ! -d "$DATASETS_DIR" ]; then
        echo -e "${YELLOW}Warning: Directory $DATASETS_DIR does not exist. Creating it...${NC}"
        mkdir -p "$DATASETS_DIR"
    fi
    
    # Start Python HTTP server in background with nohup for proper detachment
    cd "$DATASETS_DIR"
    nohup python3 -m http.server $DATASETS_PORT > /tmp/datasets_server.log 2>&1 &
    DATASETS_SERVER_PID=$!
    echo $DATASETS_SERVER_PID > /tmp/datasets_server.pid
    echo -e "${GREEN}HTTP server started with PID: $DATASETS_SERVER_PID${NC}"
    echo "  Media URLs: http://localhost:$DATASETS_PORT/<filename>"
    
    # Return to service directory
    cd "$SERVICE_DIR"
    
    # Give the HTTP server a moment to start and verify it's running
    sleep 3
    
    # Verify the server is actually running
    if lsof -Pi :$DATASETS_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${GREEN}HTTP server verified running on port $DATASETS_PORT${NC}"
    else
        echo -e "${YELLOW}Warning: HTTP server may not have started properly. Check /tmp/datasets_server.log${NC}"
        if [ -f /tmp/datasets_server.log ]; then
            echo "  Last few lines of server log:"
            tail -5 /tmp/datasets_server.log | sed 's/^/    /'
        fi
    fi
fi

echo ""

# Create logs directory if it doesn't exist
mkdir -p "$SERVICE_DIR/logs"

# Service configuration
PORT=${PORT:-8006}
HOST=${HOST:-0.0.0.0}

echo -e "${GREEN}Starting FastAPI server with following configuration:${NC}"
echo "  Port: $PORT"
echo "  Host: $HOST"
echo ""
echo -e "${YELLOW}Service will be accessible at: http://localhost:$PORT${NC}"
echo -e "${YELLOW}API documentation at: http://localhost:$PORT/docs${NC}"
echo -e "${YELLOW}Logs will be saved to: $SERVICE_DIR/logs/service.log${NC}"
echo ""

# Start uvicorn server with logging (output to both console and log file)
uvicorn app:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info \
    2>&1 | tee "$SERVICE_DIR/logs/service.log"

