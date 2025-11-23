# Docker Setup for FP8 Training Benchmarks

This guide explains how to use Docker to run FP8 mixed precision training benchmarks with Unsloth.

## Prerequisites

1. **NVIDIA GPU** with compute capability 8.0+
   - Optimal: H100, H200
   - Supported: A100, RTX 4090

2. **NVIDIA Driver** installed on host
   ```bash
   nvidia-smi  # Should show your GPU
   ```

3. **Docker** with NVIDIA Container Toolkit
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

4. **Verify GPU access in Docker**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
   ```

## Quick Start

### Option 1: Using Docker directly

```bash
# Build the image
docker build -f Dockerfile.fp8 -t unsloth-fp8:latest .

# Run quick test
docker run --gpus all --rm unsloth-fp8:latest python test_fp8_quick.py

# Run full benchmark
docker run --gpus all --rm unsloth-fp8:latest python benchmark_fp8_vs_bf16.py

# Interactive shell
docker run --gpus all -it --rm unsloth-fp8:latest bash
```

### Option 2: Using Docker Compose (recommended)

```bash
# Start container
docker-compose -f docker-compose.fp8.yml up -d

# Enter container
docker-compose -f docker-compose.fp8.yml exec unsloth-fp8 bash

# Inside container:
python test_fp8_quick.py
python benchmark_fp8_vs_bf16.py

# Stop container
docker-compose -f docker-compose.fp8.yml down
```

## Container Details

### Base Image
- **Image**: `nvcr.io/nvidia/pytorch:25.10-py3`
- **Includes**: PyTorch, CUDA, Transformer Engine
- **CUDA Version**: 12.x
- **Python**: 3.10

### Installed Packages
- ✅ PyTorch (from base image)
- ✅ Transformer Engine (FP8 support)
- ✅ HuggingFace Transformers, Accelerate, Datasets
- ✅ TRL, PEFT
- ✅ Unsloth (development install)

### Working Directory
```
/workspace/unsloth/examples/
```

### Available Scripts
- `test_fp8_quick.py` - Quick FP8 validation
- `fp8_finetuning_example.py` - Full finetuning example
- `benchmark_fp8_vs_bf16.py` - Performance benchmark

## Usage Examples

### 1. Quick FP8 Test

```bash
docker run --gpus all --rm unsloth-fp8:latest python test_fp8_quick.py
```

**Expected output**:
```
✅ FP8 Training Test PASSED!
Final loss: 1.2345
Memory used: 3.45 GB
```

### 2. Run Benchmark

```bash
# Full benchmark (FP8 + BF16)
docker run --gpus all --rm \
  -v $(pwd)/outputs:/workspace/outputs \
  unsloth-fp8:latest python benchmark_fp8_vs_bf16.py

# Only FP8
docker run --gpus all --rm \
  unsloth-fp8:latest python benchmark_fp8_vs_bf16.py --mode fp8

# Custom steps
docker run --gpus all --rm \
  unsloth-fp8:latest python benchmark_fp8_vs_bf16.py --steps 200
```

Results will be in `outputs/benchmark_results.json`

### 3. Full Finetuning Example

```bash
docker run --gpus all --rm \
  -v $(pwd)/outputs:/workspace/outputs \
  unsloth-fp8:latest python fp8_finetuning_example.py
```

Model will be saved to `outputs/fp8_finetuned_model/`

### 4. Interactive Development

```bash
# Start container with mounted code
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/unsloth \
  unsloth-fp8:latest bash

# Inside container:
cd /workspace/unsloth/examples
python benchmark_fp8_vs_bf16.py
```

## Docker Compose Usage

### Start Container
```bash
docker-compose -f docker-compose.fp8.yml up -d
```

### Check Status
```bash
docker-compose -f docker-compose.fp8.yml ps
```

### View Logs
```bash
docker-compose -f docker-compose.fp8.yml logs -f
```

### Execute Commands
```bash
# Enter shell
docker-compose -f docker-compose.fp8.yml exec unsloth-fp8 bash

# Run script directly
docker-compose -f docker-compose.fp8.yml exec unsloth-fp8 \
  python benchmark_fp8_vs_bf16.py
```

### Stop Container
```bash
docker-compose -f docker-compose.fp8.yml down
```

### Clean Up (including volumes)
```bash
docker-compose -f docker-compose.fp8.yml down -v
```

## Volume Mounts

### Code (Development)
```yaml
volumes:
  - .:/workspace/unsloth
```
Local code changes are reflected in the container.

### Outputs
```yaml
volumes:
  - ./outputs:/workspace/outputs
```
Training outputs and benchmarks saved to local `outputs/` directory.

### Model Cache
```yaml
volumes:
  - huggingface_cache:/workspace/cache
```
Downloaded models are cached between runs (speeds up subsequent runs).

## Environment Variables

Configure in `docker-compose.fp8.yml`:

```yaml
environment:
  # Use specific GPU (0, 1, 2, etc.)
  - CUDA_VISIBLE_DEVICES=0

  # Cache directories
  - TRANSFORMERS_CACHE=/workspace/cache
  - HF_HOME=/workspace/cache

  # PyTorch memory management
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Multi-GPU Setup

### Select Specific GPUs
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
```

### Use All GPUs
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all  # Use all available GPUs
          capabilities: [gpu]
```

## Troubleshooting

### "nvidia-smi not found" in container
**Issue**: NVIDIA runtime not configured

**Solution**:
```bash
# Check Docker daemon configuration
cat /etc/docker/daemon.json

# Should include:
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

# Restart Docker
sudo systemctl restart docker
```

### "Out of memory" errors
**Solution**:
```bash
# Reduce batch size in scripts
# Or specify GPU with more memory
docker run --gpus '"device=1"' ...
```

### "FP8 not supported"
**Issue**: GPU doesn't support FP8

**Solution**:
- Check GPU: `nvidia-smi`
- Need H100/H200 (optimal) or A100+ (supported)
- Verify Transformer Engine is installed:
  ```bash
  docker run --gpus all --rm unsloth-fp8:latest \
    python -c "import transformer_engine; print(transformer_engine.__version__)"
  ```

### Slow first run
**Explanation**: First run downloads models (~4GB for Llama-3.2-1B)

**Solution**: Model cache persists between runs via Docker volume

### Permission issues with outputs
**Solution**:
```bash
# Create outputs directory with correct permissions
mkdir -p outputs
chmod 777 outputs

# Or run with user ID
docker run --gpus all --rm -u $(id -u):$(id -g) ...
```

## Best Practices

### 1. Use Docker Compose for Development
```bash
docker-compose -f docker-compose.fp8.yml up -d
docker-compose -f docker-compose.fp8.yml exec unsloth-fp8 bash
```

### 2. Mount Local Code
Enables iterative development without rebuilding:
```yaml
volumes:
  - .:/workspace/unsloth
```

### 3. Persist Outputs
```yaml
volumes:
  - ./outputs:/workspace/outputs
```

### 4. Cache Models
```yaml
volumes:
  - huggingface_cache:/workspace/cache
```

### 5. Clean Up After Benchmarks
```bash
docker-compose -f docker-compose.fp8.yml down
docker system prune -f  # Clean up unused containers/images
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: FP8 Benchmark

on: [push]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu, h100]
    steps:
      - uses: actions/checkout@v2

      - name: Build Docker image
        run: docker build -f Dockerfile.fp8 -t unsloth-fp8:latest .

      - name: Run FP8 test
        run: docker run --gpus all --rm unsloth-fp8:latest python test_fp8_quick.py

      - name: Run benchmark
        run: |
          docker run --gpus all --rm \
            -v $(pwd)/outputs:/workspace/outputs \
            unsloth-fp8:latest python benchmark_fp8_vs_bf16.py --steps 50

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: outputs/benchmark_results.json
```

## Resource Requirements

### Minimum
- GPU: A100 (40GB VRAM)
- RAM: 32GB
- Disk: 50GB (for images + models)

### Recommended
- GPU: H100 (80GB VRAM)
- RAM: 64GB
- Disk: 100GB (for caching multiple models)

## Performance Tips

1. **Pre-download models** before benchmarking:
   ```bash
   docker run --gpus all --rm \
     -v huggingface_cache:/workspace/cache \
     unsloth-fp8:latest \
     python -c "from transformers import AutoModel; AutoModel.from_pretrained('unsloth/Llama-3.2-1B-Instruct')"
   ```

2. **Use SSD for Docker volumes** for faster I/O

3. **Increase shared memory** if needed:
   ```bash
   docker run --gpus all --shm-size=16g ...
   ```

## Additional Resources

- [NVIDIA PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [Unsloth Documentation](https://docs.unsloth.ai)
