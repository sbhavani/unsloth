# FP8 Performance Failure Analysis

## Results: FP8 is SLOWER than BF16

### Full Fine-Tuning Benchmark Results

| Metric | BF16 | FP8 | Ratio |
|--------|------|-----|-------|
| **Time** | 86.49s | 105.23s | 0.82x (18% SLOWER) ❌ |
| **Steps/sec** | 1.1561 | 0.9503 | 0.82x |
| **Memory** | 20.29 GB | 28.97 GB | 1.43x (43% MORE) |
| **Loss** | 1.5959 | 1.5975 | Similar |

### Expected vs Actual

**Expected:**
- FP8: 1.3-1.5x FASTER than BF16
- Memory: 15-30% higher (acceptable)

**Actual:**
- FP8: 18% SLOWER than BF16 ❌
- Memory: 43% higher (as expected)

## Hypothesis: FP8 Not Being Applied

The slowdown suggests FP8 layers are **not being used** for computation. Possible causes:

### 1. Accelerate's FP8 Not Integrated with Trainer

**Issue:** Setting environment variables may not be enough. The `Trainer` might not automatically pick up FP8 settings.

**Evidence:**
- No visible FP8 wrapping in logs
- Slowdown suggests extra overhead without FP8 benefit

**Solution:** May need to use `accelerate launch` or manually create `Accelerator` with `FP8RecipeKwargs`.

### 2. Unsloth's Custom Kernels Conflict with FP8

**Issue:** Unsloth applies custom CUDA kernels that may prevent Accelerate from wrapping layers with Transformer Engine.

**Evidence:**
- Unsloth patches models heavily
- TE needs to wrap specific layer types
- Unsloth's patched layers may not be recognized

**Solution:** May need Unsloth-specific FP8 integration.

### 3. FP8 Overhead Without Benefit

**Issue:** FP8 metadata/scaling is being applied, but actual compute stays in BF16.

**Evidence:**
- Memory increase matches FP8 (dual precision)
- But performance is worse (overhead without speedup)

**Solution:** Verify TE layers are actually being used.

## Diagnostic Steps

### Step 1: Check Environment Variables

```python
import os
print("ACCELERATE_MIXED_PRECISION:", os.environ.get("ACCELERATE_MIXED_PRECISION"))
print("ACCELERATE_FP8_BACKEND:", os.environ.get("ACCELERATE_FP8_BACKEND"))
print("ACCELERATE_FP8_FORMAT:", os.environ.get("ACCELERATE_FP8_FORMAT"))
```

### Step 2: Check for Transformer Engine Layers

```python
from transformer_engine.pytorch import Linear as TELinear

te_count = sum(1 for m in model.modules() if isinstance(m, TELinear))
print(f"Transformer Engine layers: {te_count}")
# Expected: >0 if FP8 is working
# Actual: Likely 0
```

### Step 3: Try accelerate launch

Instead of Python script, use:
```bash
accelerate launch --mixed_precision fp8 llama_8b_full_finetune_fp8.py
```

## Possible Root Cause

**HuggingFace Trainer does not automatically support FP8 via environment variables.**

From Accelerate docs, FP8 requires:
1. Creating `Accelerator` with `FP8RecipeKwargs`
2. Calling `accelerator.prepare(model, optimizer)`
3. Using the prepared model

The `Trainer` class may not do this automatically even with env vars set.

## Next Steps

### Option 1: Use accelerate launch (Recommended)

Create an `accelerate` config with FP8:
```bash
accelerate config
# Select: fp8, transformer_engine
```

Then launch:
```bash
accelerate launch llama_8b_full_finetune_fp8.py
```

### Option 2: Manual Accelerator Integration

Modify the script to use `Accelerator` directly:
```python
from accelerate import Accelerator
from accelerate.utils import FP8RecipeKwargs

fp8_kwargs = FP8RecipeKwargs(backend="te", fp8_format="HYBRID")
accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=[fp8_kwargs])

# Prepare model
model = accelerator.prepare(model)

# Then use with Trainer
```

### Option 3: Check if Unsloth Supports FP8

Contact Unsloth team or check if there's native FP8 support that integrates with their optimizations.

## Conclusion

**Current Implementation:** FP8 environment variables are set but not being used by Trainer.

**Result:** Overhead without speedup (0.82x slower).

**Fix Needed:** Proper Accelerate integration, likely via `accelerate launch` or manual `Accelerator` setup.

**Alternative:** This may indicate that supervised fine-tuning with FP8 via Transformer Engine is **not currently compatible** with Unsloth's optimizations. The GRPO/RL examples use `load_in_fp8=True` (quantization), not Transformer Engine (compute).

## Recommendation

Given these results, for now:
1. **Stick with BF16** for LoRA and full fine-tuning
2. Focus on Unsloth's existing optimizations (which already provide 2x speedup)
3. Wait for official Unsloth FP8 support for supervised fine-tuning
4. Current FP8 implementation (via Accelerate) may not be compatible with Unsloth

The 1.13x speedup you saw on 1B+LoRA may have been measurement noise, not real FP8 benefit.
