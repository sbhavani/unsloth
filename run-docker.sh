#!/bin/bash

# Helper script to run Unsloth examples in Docker
# Usage: ./run-docker.sh [fp8] [script_name]
# Example: ./run-docker.sh fp8 test_fp8_quick.py

set -e

# Determine which image to use
if [ "$1" == "fp8" ]; then
    IMAGE="unsloth-fp8:latest"
    shift
else
    IMAGE="unsloth:latest"
fi

# Check if image exists
if ! docker image inspect "$IMAGE" &> /dev/null; then
    echo "Error: Docker image '$IMAGE' not found."
    echo "Please build the image first:"
    if [ "$IMAGE" == "unsloth-fp8:latest" ]; then
        echo "  ./build-docker.sh fp8"
    else
        echo "  ./build-docker.sh"
    fi
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# If a script name is provided, run it
if [ -n "$1" ]; then
    SCRIPT_NAME="$1"
    echo "Running: $SCRIPT_NAME in Docker container..."
    docker run --gpus all -it --rm \
        -v "$SCRIPT_DIR:/workspace/host" \
        "$IMAGE" \
        bash -c "cd /workspace/unsloth/examples && python $SCRIPT_NAME"
else
    # Interactive mode
    echo "Starting interactive Docker container..."
    echo "You'll be in /workspace/unsloth/examples"
    echo "Run your scripts with: python test_fp8_quick.py"
    docker run --gpus all -it --rm \
        -v "$SCRIPT_DIR:/workspace/host" \
        "$IMAGE"
fi
