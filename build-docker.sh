#!/bin/bash

# Build script for Unsloth Docker image
# This ensures the build happens from the correct directory with the right context

set -e  # Exit on error

echo "========================================"
echo "Building Unsloth Docker Image"
echo "========================================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory (repository root)
cd "$SCRIPT_DIR"

# Verify we're in the right place
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found!"
    echo "Make sure you're running this script from the repository root."
    exit 1
fi

if [ ! -d "unsloth" ]; then
    echo "ERROR: unsloth/ directory not found!"
    echo "Make sure you're running this script from the repository root."
    exit 1
fi

echo "✓ Found pyproject.toml"
echo "✓ Found unsloth/ directory"
echo ""

# Check which Dockerfile to use
if [ "$1" == "fp8" ]; then
    DOCKERFILE="Dockerfile.fp8"
    TAG="unsloth-fp8:latest"
    echo "Building with FP8 support using $DOCKERFILE"
else
    DOCKERFILE="Dockerfile"
    TAG="unsloth:latest"
    echo "Building standard image using $DOCKERFILE"
fi

if [ ! -f "$DOCKERFILE" ]; then
    echo "ERROR: $DOCKERFILE not found!"
    exit 1
fi

echo "✓ Found $DOCKERFILE"
echo ""
echo "Building Docker image: $TAG"
echo "This may take several minutes..."
echo ""

# Build the image
docker build -f "$DOCKERFILE" -t "$TAG" .

echo ""
echo "========================================"
echo "✓ Build completed successfully!"
echo "========================================"
echo ""
echo "To run the container:"
echo "  docker run --gpus all -it --rm $TAG"
echo ""
echo "To run with mounted workspace:"
echo "  docker run --gpus all -it --rm -v \$(pwd):/workspace/host $TAG"
echo ""
