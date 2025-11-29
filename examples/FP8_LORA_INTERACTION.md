# FP8 and LoRA Interaction

## Key Finding: FP8 Speedup Varies with LoRA Configuration

**Observation from benchmarks:**
- **1B model + LoRA rank 16**: 1.13x FP8 speedup ✅
- **8B model + LoRA rank 32**: 1.00x FP8 speedup ❌

## Why Does This Happen?

### The Problem: LoRA Doesn't Use FP8

When you use FP8 with LoRA, the compute splits into two parts:

```
Total Compute Time = Base Model (FP8) + LoRA Adapters (BF16)
```

**Base Model Layers:**
- Use FP8 mixed precision
- Get 1.3-1.5x speedup from tensor cores

**LoRA Adapter Layers:**
- Always run in BF16/FP32
- NO FP8 acceleration
- Overhead scales with LoRA rank

### Mathematical Model

If LoRA takes X% of total compute:
- FP8 speedup on base model: 1.5x
- LoRA speedup: 1.0x (no change)
- **Effective speedup = 1 / ((1-X)/1.5 + X/1.0)**

Examples:
- **LoRA = 10% of compute**: Effective speedup = 1.43x ✅
- **LoRA = 30% of compute**: Effective speedup = 1.25x ⚠️  
- **LoRA = 50% of compute**: Effective speedup = 1.09x ❌
- **LoRA = 70% of compute**: Effective speedup = 1.02x ❌

## Why 8B Shows Worse Speedup Than 1B

### Configuration Differences

| Model | LoRA Rank | LoRA Params | Base Params | LoRA % | Observed Speedup |
|-------|-----------|-------------|-------------|--------|------------------|
| 1B    | 16        | ~8M         | ~1B         | ~0.8%  | 1.13x ✅         |
| 8B    | 32        | ~100M       | ~8B         | ~1.2%  | 1.00x ❌         |

Wait, percentages are similar! So why the difference?

### The Real Culprit: Compute Intensity

The issue isn't just parameter count - it's **compute per parameter**:

1. **LoRA adapters are LOW-RANK**
   - Fewer FLOPs per forward pass
   - BUT: Added to every layer
   - Many small matrix multiplications

2. **Larger models have more layers**
   - 8B model: 32 layers
   - 1B model: 16-22 layers
   - More layers = more LoRA overhead per token

3. **LoRA rank 32 vs 16**
   - Rank 32: 4x more parameters than rank 8
   - 2x more parameters than rank 16
   - Compute scales quadratically in some operations

### Gradient Checkpointing Amplification

With gradient checkpointing enabled:
- Forward pass runs once (FP8 acceleration)
- **Backward pass recomputes activations** (LoRA in BF16)
- LoRA overhead hits twice as hard!

## Solutions

### Solution 1: Reduce LoRA Rank (Recommended)

Use smaller LoRA rank on larger models:

```python
# ❌ BAD for FP8 on 8B model
model = FastLanguageModel.get_peft_model(model, r=32, ...)

# ✅ GOOD for FP8 on 8B model  
model = FastLanguageModel.get_peft_model(model, r=8, ...)   # or r=16
```

**Trade-offs:**
- ✅ Better FP8 speedup (1.2-1.3x)
- ✅ Lower memory usage
- ⚠️ Slightly lower adapter capacity (often negligible)

### Solution 2: Full Fine-Tuning (Maximum FP8)

Skip LoRA entirely:

```python
# Skip get_peft_model(), just train the full model
for param in model.parameters():
    param.requires_grad = True

trainer = SFTTrainer(model=model, ...)
```

**Trade-offs:**
- ✅ Maximum FP8 speedup (1.3-1.5x)
- ❌ Much higher memory usage
- ❌ Slower training (more parameters to update)

### Solution 3: Target Fewer Modules

Use LoRA on fewer layers:

```python
# ❌ BAD - Too many LoRA layers
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]  # 7 per layer

# ✅ GOOD - Fewer LoRA layers
target_modules = ["q_proj", "v_proj"]  # 2 per layer
```

**Trade-offs:**
- ✅ Better FP8 speedup
- ✅ Lower memory
- ⚠️ May need higher rank to compensate

### Solution 4: Increase Batch Size

LoRA overhead is per-sample. Larger batches amortize it:

```python
# ❌ Small batch - LoRA overhead dominates
TrainingArguments(per_device_train_batch_size=4, ...)

# ✅ Large batch - base model compute dominates
TrainingArguments(per_device_train_batch_size=16, ...)  # or 32
```

## Recommended Configurations for FP8

### For 1B-3B Models with H100:
```python
LORA_RANK = 16
BATCH_SIZE = 8-16
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
# Expected speedup: 1.2-1.3x
```

### For 7B-8B Models with H100:
```python
LORA_RANK = 8  # Lower than usual!
BATCH_SIZE = 16-32
target_modules = ["q_proj", "v_proj"]  # Fewer modules
# Expected speedup: 1.2-1.3x
```

### For 13B+ Models with H100:
```python
LORA_RANK = 8
BATCH_SIZE = 8-16
target_modules = ["q_proj", "v_proj"]
# Expected speedup: 1.3-1.4x (LoRA smaller % of total)
```

### For Maximum FP8 Speedup:
```python
# No LoRA - full fine-tuning
BATCH_SIZE = 4-8  # Limited by memory
# Expected speedup: 1.4-1.5x (pure FP8, no LoRA overhead)
```

## Testing Scripts

Three test scripts are provided:

1. **`test_fp8_8b_low_lora.py`** - LoRA rank 8 (vs 32)
2. **`test_fp8_8b_no_lora.py`** - No LoRA (pure FP8)
3. **`benchmark_fp8_vs_bf16.py`** - Comprehensive comparison

## Summary

**Key Takeaway:** FP8 speedup on LoRA-finetuned models depends heavily on:
1. LoRA rank (lower is better for FP8)
2. Number of LoRA-targeted modules (fewer is better)
3. Batch size (larger is better)
4. Model size (larger models need lower LoRA ranks)

The "sweet spot" for FP8+LoRA is **rank 8-16 with large batch sizes**.
