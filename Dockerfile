# Dockerfile for Unsloth with FP8 Mixed Precision Training
# Base: NVIDIA PyTorch container with Transformer Engine support
#
# Build from the repository root:
#   docker build -t unsloth:latest .
#
# Or for FP8 specifically:
#   docker build -f Dockerfile.fp8 -t unsloth-fp8:latest .
#
# Run (requires NVIDIA GPU):
#   docker run --gpus all -it --rm unsloth:latest

FROM nvcr.io/nvidia/pytorch:25.10-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

LABEL maintainer="Unsloth AI"
LABEL description="Unsloth with FP8 mixed precision training support"
LABEL version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Note: Transformer Engine is already included in the base image

# Install HuggingFace ecosystem and dependencies
RUN pip install --no-cache-dir \
    transformers>=4.35.0 \
    accelerate>=0.26.0 \
    datasets>=2.14.0 \
    tokenizers>=0.15.0 \
    peft>=0.7.0 \
    trl>=0.7.0

# Install other training dependencies
RUN pip install --no-cache-dir \
    bitsandbytes>=0.41.0 \
    scipy \
    sentencepiece \
    protobuf

# Install unsloth_zoo (required by unsloth)
RUN pip install --no-cache-dir unsloth_zoo>=2025.11.4

# Set up workspace
WORKDIR /workspace

# Copy the entire unsloth directory to the container
# The build context should be the repository root
COPY . /workspace/unsloth/

# Install Unsloth in development mode
WORKDIR /workspace/unsloth
RUN pip install -e .

# Create directory for outputs
RUN mkdir -p /workspace/outputs

# Verify installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import transformer_engine; print(f'Transformer Engine: {transformer_engine.__version__}')" && \
    python -c "from unsloth import FastLanguageModel; print('Unsloth: OK')"

# Set default working directory
WORKDIR /workspace/unsloth

CMD ["bash"]
