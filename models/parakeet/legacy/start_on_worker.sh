#!/bin/bash

# Start Parakeet service on worker node (within SLURM job)
# This script should be run inside a SLURM job allocation

echo "Starting Parakeet service on $(hostname)..."

# Change to service directory
cd /home/naresh/Vision/ModelService_Parakeet

# Start the service
./start_service.sh



