# FP8 Mixed Precision Training Examples

This directory contains examples and benchmarks for FP8 mixed precision training in Unsloth using NVIDIA Transformer Engine.

## Quick Links

- [Unsloth Llama 3 Tutorial](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama)
- [Alpaca Dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- [HF Accelerate FP8 Guide](https://huggingface.co/docs/accelerate/usage_guides/fp8)
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)

## Scripts

### 1. Quick Test (`test_fp8_quick.py`)

**Purpose**: Validate FP8 training setup works correctly

**Usage**:
```bash
python test_fp8_quick.py
```

**What it does**:
- Checks FP8 support (GPU, dependencies)
- Loads Llama 3.2-1B model
- Runs 10 training steps with FP8
- Validates everything works

**Expected output**:
```
✅ FP8 Training Test PASSED!
Final loss: 1.2345
Memory used: 3.45 GB
```

**Time**: ~1-2 minutes

---

### 2. Full Example (`fp8_finetuning_example.py`)

**Purpose**: Complete FP8 finetuning example with Llama 3 + Alpaca

**Usage**:
```bash
python fp8_finetuning_example.py
```

**What it does**:
- Enables FP8 mixed precision training
- Loads Llama 3.2-1B-Instruct
- Adds LoRA adapters
- Trains on Alpaca dataset
- Saves finetuned model

**Configuration**:
- Model: Llama-3.2-1B-Instruct
- Dataset: Alpaca (1000 examples)
- LoRA rank: 16
- Batch size: 4
- Max seq length: 2048

**Time**: ~10-20 minutes (depends on GPU)

---

### 3. Benchmark (`benchmark_fp8_vs_bf16.py`)

**Purpose**: Compare FP8 vs BF16 training performance

**Usage**:
```bash
# Run both FP8 and BF16
python benchmark_fp8_vs_bf16.py

# Only FP8
python benchmark_fp8_vs_bf16.py --mode fp8

# Only BF16
python benchmark_fp8_vs_bf16.py --mode bf16

# Custom number of steps
python benchmark_fp8_vs_bf16.py --steps 200
```

**What it measures**:
- ⚡ **Speed**: Steps/second, samples/second
- 💾 **Memory**: Peak VRAM usage
- 📉 **Loss**: Training loss convergence
- 🎯 **Accuracy**: Loss difference between FP8 and BF16

**Output**:
```
📊 SPEED
  BF16:  0.1234 steps/sec
  FP8:   0.1850 steps/sec
  Speedup: 1.50x ✅

📈 THROUGHPUT
  BF16:  0.99 samples/sec
  FP8:   1.48 samples/sec
  Gain: 1.50x ✅

💾 MEMORY
  BF16:  8.45 GB
  FP8:   5.07 GB
  Reduction: 40.0% ✅

📉 LOSS
  BF16:  1.2345
  FP8:   1.2367
  Difference: 0.0022 (0.18%)

🎯 SUMMARY
  FP8 is 1.50x faster than BF16
  FP8 uses 40.0% less memory
  FP8 maintains 99.8% of BF16 loss accuracy
```

Results are saved to `benchmark_results.json`

**Time**: ~10-30 minutes (depends on steps and GPU)

---

## Hardware Requirements

### FP8 Training
- **Optimal**: H100, H200 (Hopper GPUs, compute capability 9.0)
  - Native FP8 tensor cores
  - 1.3-1.5x speedup
  - ~40% memory savings
- **Supported**: RTX 4090, L4 (Ada Lovelace, compute capability 8.9)
  - Native FP8 support
  - 1.2-1.3x speedup
  - ~40% memory savings
- **Not Supported**: A100 (compute capability 8.0 - below 8.9 minimum)

### BF16 Baseline
- Any CUDA GPU with compute capability 8.0+

## Software Requirements

```bash
# Install dependencies
pip install torch>=2.0.0
pip install transformers>=4.35.0
pip install accelerate>=0.26.0
pip install trl
pip install datasets
pip install unsloth

# For FP8 training
pip install transformer-engine>=1.0.0
```

**Note**: Transformer Engine requires CUDA 11.8+ or 12.0+

## Expected Performance

Based on Accelerate benchmarks and literature:

| Metric | H100 | A100 |
|--------|------|------|
| Speed | 1.3-1.5x | 1.1-1.3x |
| Memory | ~40% reduction | ~40% reduction |
| Accuracy | ~99% maintained | ~99% maintained |

## Troubleshooting

### "FP8 training not supported"
```
ERROR: FP8 training not supported on this system
```

**Solution**:
1. Check GPU: `nvidia-smi` (need H100/H200/A100)
2. Install Transformer Engine: `pip install transformer-engine`
3. Check CUDA version: Should be 11.8+ or 12.0+

### "Out of memory"
**Solution**:
- Reduce batch size in the script
- Reduce max_seq_length
- Use gradient checkpointing (already enabled)

### NaN loss
**Solution**:
- Increase `amax_history_len` in `setup_fp8_mixed_precision_training()`
- Reduce learning rate
- Add gradient clipping

## Multi-GPU Training

For multi-GPU setups, use `accelerate launch`:

```bash
# Configure accelerate (first time only)
accelerate config

# Select:
# - Mixed Precision: FP8
# - Backend: Transformer Engine

# Launch
accelerate launch fp8_finetuning_example.py
accelerate launch benchmark_fp8_vs_bf16.py
```

## Advanced Usage

### Custom FP8 Configuration

```python
from unsloth import setup_fp8_mixed_precision_training

# Custom settings
setup_fp8_mixed_precision_training(
    backend="te",
    fp8_format="HYBRID",     # or "E4M3", "E5M2"
    amax_history_len=64,     # Larger = more stable
    amax_compute_algo="max", # or "most_recent"
)
```

### Combining FP8 + LoRA + QLoRA

```python
# Maximum memory efficiency
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    load_in_4bit=False,  # Don't use 4bit with FP8 training
)

# FP8 training + LoRA adapters
setup_fp8_mixed_precision_training()
model = FastLanguageModel.get_peft_model(model, r=16, ...)
```

## References

- [Unsloth Llama 3 Finetuning Tutorial](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama)
- [Alpaca Dataset (cleaned)](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- [HF Accelerate FP8 Examples](https://github.com/huggingface/accelerate/tree/main/benchmarks/fp8/transformer_engine)
- [Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [FP8 Training Paper](https://arxiv.org/abs/2209.05433)

## Support

For issues or questions:
- [Unsloth GitHub Issues](https://github.com/unslothai/unsloth/issues)
- Include GPU model, CUDA version, and error messages
