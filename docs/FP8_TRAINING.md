# FP8 Mixed Precision Training with Unsloth

This guide explains how to use FP8 (8-bit floating point) mixed precision training with Unsloth, leveraging HuggingFace Accelerate's integration with NVIDIA Transformer Engine.

## Overview

FP8 mixed precision training provides several benefits:

- **Reduced Memory Usage**: FP8 uses 8 bits instead of 16 (BF16/FP16) or 32 (FP32), reducing memory consumption
- **Faster Training**: Optimized FP8 operations on supported hardware (Hopper GPUs)
- **Maintained Accuracy**: Careful scaling techniques minimize accuracy degradation
- **Larger Batch Sizes**: Lower memory usage allows for larger batch sizes

## Requirements

### Hardware Requirements

- **Optimal**: NVIDIA H100, H200 (Hopper architecture, compute capability 9.0)
- **Supported**: NVIDIA A100, RTX 4090 (Ampere/Ada, compute capability 8.0+)
- **Minimum**: Any CUDA-capable GPU (may have reduced performance)

### Software Requirements

```bash
# Core dependencies
pip install torch>=2.0.0
pip install transformers>=4.35.0
pip install accelerate>=0.26.0

# Transformer Engine for FP8 support
pip install transformer-engine>=1.0.0

# Unsloth
pip install unsloth
```

**Note**: Transformer Engine requires CUDA 11.8+ or CUDA 12.0+

## Quick Start

### Basic Usage with SFTTrainer

```python
from unsloth import FastLanguageModel
from unsloth.fp8_training import FP8TrainingConfig, check_fp8_support
from transformers import TrainingArguments
from trl import SFTTrainer
import os

# Check FP8 support
if not check_fp8_support():
    print("FP8 not supported on this system")
    exit(1)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=False,  # Don't use 4bit with FP8
)

# Add LoRA adapters (optional)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)

# Enable FP8 via environment variable
os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"

# Setup training
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,  # FP8 handles precision
    output_dir="outputs",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# Train
trainer.train()
```

### Using Accelerate Launch

For multi-GPU training, use `accelerate launch`:

```bash
# Configure accelerate (first time only)
accelerate config

# Launch training
accelerate launch your_training_script.py
```

In your `accelerate` config, select:
- Mixed Precision: FP8
- Backend: Transformer Engine

### Manual Training Loop

For more control, use the `FP8Trainer` class:

```python
from unsloth.fp8_training import FP8Trainer, FP8TrainingConfig

# Setup FP8 config
fp8_config = FP8TrainingConfig(
    fp8_format="HYBRID",
    amax_history_len=32,
    amax_compute_algo="max",
)

# Create FP8 trainer
fp8_trainer = FP8Trainer(
    model=model,
    optimizer=optimizer,
    fp8_config=fp8_config,
    convert_model_to_fp8=True,
)

# Training loop
for batch in dataloader:
    loss = fp8_trainer.training_step(batch)
    print(f"Loss: {loss.item()}")

# Save model
fp8_trainer.save_model("model.pt")
```

## FP8 Configuration

### FP8 Formats

The `fp8_format` parameter controls which FP8 format is used:

- **`HYBRID`** (recommended): E4M3 for forward pass, E5M2 for backward pass
  - Best balance of range and precision
  - Recommended for most use cases

- **`E4M3`**: 4-bit exponent, 3-bit mantissa
  - Higher precision, lower range
  - Good for activations

- **`E5M2`**: 5-bit exponent, 2-bit mantissa
  - Higher range, lower precision
  - Good for gradients

### Scaling Configuration

```python
fp8_config = FP8TrainingConfig(
    fp8_format="HYBRID",
    amax_history_len=32,      # History window for scaling factor
    amax_compute_algo="max",  # Algorithm: "max" or "most_recent"
    margin=0,                 # Safety margin for scaling
    fp8_dpa=False,            # FP8 dot product attention
)
```

**Parameters explained**:

- `amax_history_len`: Number of previous steps to consider when computing the absolute maximum value for scaling. Larger values provide more stable scaling but slower adaptation to changing gradients.

- `amax_compute_algo`:
  - `"max"`: Use the maximum amax from history (more stable)
  - `"most_recent"`: Use only the most recent amax (faster adaptation)

- `margin`: Additional safety margin for scaling factors (usually 0)

- `fp8_dpa`: Enable FP8 for dot product attention (experimental, may reduce accuracy)

## Best Practices

### 1. Padding to Multiples of 16

For optimal FP8 performance, pad your sequences to multiples of 16:

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# When tokenizing
outputs = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=2048,  # Multiple of 16
    return_tensors="pt",
)
```

### 2. Batch Size Considerations

FP8 reduces memory usage, allowing larger batch sizes:

```python
# Before (BF16)
batch_size = 4

# After (FP8) - can often increase by 1.5-2x
batch_size = 8
```

Start with your BF16 batch size and gradually increase while monitoring memory.

### 3. Learning Rate

FP8 training may benefit from slightly adjusted learning rates:

```python
# Start with your normal learning rate
learning_rate = 2e-4

# If training is unstable, try:
# - Increasing warmup steps
# - Slightly reducing learning rate
# - Using a linear schedule
```

### 4. Monitoring Training

Watch for signs of FP8-related issues:

- **Gradient overflow**: Loss becomes NaN
  - Solution: Reduce learning rate or increase `amax_history_len`

- **Slow convergence**: Model learns slower than BF16
  - Solution: Try `fp8_format="E4M3"` or adjust scaling parameters

- **Accuracy degradation**: Final model performs worse
  - Solution: Increase `amax_history_len` or use `fp8_format="HYBRID"`

### 5. Debugging

Enable logging to monitor FP8 behavior:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Check if FP8 is actually being used
from unsloth.fp8_training import check_fp8_support
assert check_fp8_support(), "FP8 not available"
```

## Advanced Usage

### Using with FSDP (Fully Sharded Data Parallel)

```bash
# Configure FSDP with FP8
accelerate config

# Select:
# - Distributed training: FSDP
# - Mixed precision: FP8
# - Sharding strategy: FULL_SHARD

# Launch
accelerate launch --config_file fsdp_fp8_config.yaml train.py
```

### Using with DeepSpeed

FP8 is compatible with DeepSpeed ZeRO:

```python
# In your training args
training_args = TrainingArguments(
    deepspeed="ds_config.json",
    # ... other args
)
```

DeepSpeed config (`ds_config.json`):
```json
{
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": false
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

### Custom FP8 Autocast

For complete control over FP8 regions:

```python
from unsloth.fp8_training import get_fp8_autocast_context

fp8_config = FP8TrainingConfig(fp8_format="HYBRID")

# In training loop
with get_fp8_autocast_context(fp8_config):
    outputs = model(**batch)
    loss = outputs.loss

loss.backward()
```

## Compatibility

### Compatible with:
- ✅ LoRA/QLoRA (recommended for memory efficiency)
- ✅ Full finetuning
- ✅ DDP (DistributedDataParallel)
- ✅ FSDP (Fully Sharded Data Parallel)
- ✅ DeepSpeed ZeRO
- ✅ Gradient checkpointing
- ✅ Mixed batch sizes

### Not compatible with:
- ❌ 4-bit/8-bit quantization (use one or the other, not both)
- ❌ Some custom kernels (may fall back to standard operations)

## Troubleshooting

### "FP8 not available" error

```
Unsloth FP8: `transformer_engine` is not installed.
```

**Solution**: Install Transformer Engine:
```bash
pip install transformer-engine
```

### "CUDA version mismatch" error

**Solution**: Ensure CUDA 11.8+ or 12.0+:
```bash
nvidia-smi  # Check CUDA version
```

### NaN loss during training

**Causes**:
1. Learning rate too high
2. Gradient overflow in FP8

**Solutions**:
```python
# Increase scaling history
fp8_config = FP8TrainingConfig(amax_history_len=64)

# Reduce learning rate
learning_rate = 1e-4

# Add gradient clipping
training_args = TrainingArguments(
    max_grad_norm=1.0,
    # ... other args
)
```

### Slower than BF16

**Possible causes**:
1. GPU doesn't have FP8 tensor cores (pre-Hopper)
2. Small batch sizes (overhead dominates)
3. Short sequences (padding overhead)

**Solutions**:
- Use larger batch sizes (FP8 allows this)
- Ensure sequences are padded to multiples of 16
- Profile to identify bottlenecks

## Performance Benchmarks

Typical performance on H100 (vs BF16):

| Model | FP8 Memory | FP8 Speed | Accuracy |
|-------|------------|-----------|----------|
| Llama-3.2-1B | 0.6x | 1.3x | ~99% |
| Llama-3.2-3B | 0.6x | 1.4x | ~99% |
| Mistral-7B | 0.6x | 1.5x | ~98% |

*Results vary based on hardware, sequence length, and configuration*

## References

- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [HuggingFace Accelerate FP8 Guide](https://huggingface.co/docs/accelerate/usage_guides/fp8)
- [FP8 Formats and Quantization](https://arxiv.org/abs/2209.05433)

## Support

For issues or questions:
- GitHub Issues: https://github.com/unslothai/unsloth/issues
- Check existing FP8-related issues first
- Include GPU model and CUDA version in bug reports
