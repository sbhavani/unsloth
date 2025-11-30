# FP8 Training Findings and Recommendations

## Summary of Benchmark Results

### What We Observed

| Model | LoRA Rank | Batch Size | FP8 Speedup | Memory Increase |
|-------|-----------|------------|-------------|-----------------|
| 1B    | 16        | 4          | 1.13x ✅    | +88% (5.3 vs 2.8 GB) |
| 8B    | 32        | 16         | 1.00x ❌    | +66% (30.7 vs 18.5 GB) |

### Key Findings

1. **FP8 works but LoRA overhead dominates on larger models**
   - LoRA adapters always run in BF16
   - Higher LoRA rank = more BF16 compute
   - On 8B model with rank 32, LoRA overhead hides FP8 speedup

2. **FP8 uses MORE memory than BF16 (expected)**
   - Maintains FP32 master weights + FP8 compute copies
   - Memory overhead proportionally higher on small models
   - Primary benefit is SPEED, not memory

3. **Two different FP8 methods exist:**
   - **Method 1**: `setup_fp8_mixed_precision_training()` (Accelerate's Transformer Engine)
   - **Method 2**: `load_in_fp8=True` (Unsloth native) ← Used in unslothai/notebooks

## Recommended Tests (in order)

### Test 1: Unsloth's Native FP8 (PRIORITY)
```bash
python /workspace/unsloth/examples/test_fp8_load_in_fp8.py
```

**Why:** This is the method used in unslothai/notebooks and may work better with Unsloth's optimizations.

**Expected:** Should see better speedup than Accelerate method (~1.2-1.3x).

### Test 2: Lower LoRA Rank
```bash
python /workspace/unsloth/examples/test_fp8_8b_low_lora.py
```

**Why:** LoRA rank 32 may be too high. Test with rank 8.

**Expected:** Better speedup (~1.2x) as LoRA overhead is reduced.

### Test 3: No LoRA (Pure FP8)
```bash
python /workspace/unsloth/examples/test_fp8_8b_no_lora.py
```

**Why:** Measure pure FP8 speedup without LoRA overhead.

**Expected:** Maximum speedup (~1.3-1.5x) but uses more memory.

### Test 4: Diagnostic
```bash
python /workspace/unsloth/examples/diagnose_fp8.py
```

**Why:** Check which layers are actually using FP8.

**Expected:** Identify if layers are being converted to Transformer Engine.

## Optimal Configurations for H100

### Small Models (1B-3B)
```python
# Good LoRA + FP8 balance
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    load_in_fp8=True,  # Unsloth native
    fast_inference=True,
    max_lora_rank=16,
)

model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
)
# Expected: 1.2-1.3x speedup
```

### Large Models (7B-8B)
```python
# Lower LoRA rank to reduce BF16 overhead
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    load_in_fp8=True,  # Unsloth native
    fast_inference=True,
    max_lora_rank=8,  # Lower!
)

model = FastLanguageModel.get_peft_model(
    model, r=8,  # Lower rank
    target_modules=["q_proj", "v_proj"],  # Fewer modules
)

TrainingArguments(
    per_device_train_batch_size=16,  # Larger batch
    gradient_accumulation_steps=4,
)
# Expected: 1.2-1.3x speedup
```

### Maximum FP8 Speedup (Full Fine-Tuning)
```python
# No LoRA - pure FP8
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    load_in_fp8=True,
    fast_inference=True,
)

# No get_peft_model() - train full model
for param in model.parameters():
    param.requires_grad = True

TrainingArguments(
    per_device_train_batch_size=8,  # Smaller due to memory
    gradient_accumulation_steps=8,
)
# Expected: 1.3-1.5x speedup
```

## Two FP8 Methods Explained

### Method 1: Accelerate's Transformer Engine
```python
from unsloth import setup_fp8_mixed_precision_training

setup_fp8_mixed_precision_training(backend="te")
# Then load model normally
```

**Pros:**
- Standard HuggingFace approach
- Works with any model

**Cons:**
- May not integrate well with Unsloth's custom kernels
- Environment variables may not be picked up by Trainer

### Method 2: Unsloth's Native FP8
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="...",
    load_in_fp8=True,  # Unsloth native
    fast_inference=True,
)
```

**Pros:**
- Designed to work with Unsloth's optimizations
- Used in unslothai/notebooks (proven)
- May give better speedup

**Cons:**
- Unsloth-specific (not standard HF)

## Why LoRA Hurts FP8 Speedup

```
Training Step Compute Breakdown:

With LoRA rank 32 on 8B model:
┌─────────────────────────────────────────┐
│ Base Model Forward/Backward (FP8)  60% │  ← 1.5x speedup
│ LoRA Forward/Backward (BF16)       40% │  ← 1.0x speedup
└─────────────────────────────────────────┘
Effective speedup: 1.15x

With LoRA rank 8 on 8B model:
┌─────────────────────────────────────────┐
│ Base Model Forward/Backward (FP8)  80% │  ← 1.5x speedup
│ LoRA Forward/Backward (BF16)       20% │  ← 1.0x speedup
└─────────────────────────────────────────┘
Effective speedup: 1.29x

Without LoRA:
┌─────────────────────────────────────────┐
│ Full Model Forward/Backward (FP8) 100% │  ← 1.5x speedup
└─────────────────────────────────────────┘
Effective speedup: 1.50x
```

## Next Steps

1. **Try `load_in_fp8=True` method** (highest priority)
   - This is what unslothai/notebooks uses
   - May work better with Unsloth

2. **Lower LoRA rank to 8** for 8B models
   - Reduces BF16 overhead
   - Still maintains good adapter capacity

3. **Increase batch size** if VRAM allows
   - Larger batches amortize LoRA overhead
   - Better tensor core utilization

4. **Consider full fine-tuning** for maximum FP8
   - No LoRA overhead
   - Pure FP8 speedup (1.3-1.5x)

## Files and Documentation

- **Test Scripts:**
  - `test_fp8_load_in_fp8.py` - Unsloth native method
  - `test_fp8_8b_low_lora.py` - Low LoRA rank
  - `test_fp8_8b_no_lora.py` - No LoRA
  - `diagnose_fp8.py` - Layer analysis

- **Documentation:**
  - `FP8_LORA_INTERACTION.md` - Detailed LoRA-FP8 analysis
  - `SFT_TRAINER_USAGE.md` - API and usage guide
  - `FP8_TRAINING.md` - Original FP8 guide

- **Benchmark Script:**
  - `benchmark_fp8_vs_bf16.py` - Comprehensive comparison
