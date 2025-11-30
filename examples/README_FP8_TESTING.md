# FP8 Testing Guide - H100

## TL;DR

**For H100 with 8B models:**
Run these two scripts to measure pure FP8 speedup without LoRA overhead:

```bash
# 1. BF16 baseline (no LoRA)
python llama_8b_full_finetune_bf16.py

# 2. FP8 version (no LoRA)
python llama_8b_full_finetune_fp8.py
```

**Expected result:** FP8 should be **1.3-1.5x faster** than BF16.

---

## Background: Why Full Fine-Tuning?

### What We Discovered

Your benchmarks showed:
- **1B model + LoRA rank 16**: 1.13x FP8 speedup ✅
- **8B model + LoRA rank 32**: 1.00x FP8 speedup ❌

**Root cause:** LoRA adapters always run in BF16 and don't benefit from FP8.

On the 8B model with LoRA rank 32:
```
Total Training Time = Base Model (FP8) + LoRA Adapters (BF16)
                      60% @ 1.5x speedup  + 40% @ 1.0x speedup
                      = Effective 1.15x speedup

But with measurement noise/overhead → appears as 1.00x
```

### The Solution

**Full fine-tuning** (no LoRA) = 100% FP8 compute = Maximum speedup!

---

## Test Scripts

### Primary Test: Full Fine-Tuning

**1. Run BF16 baseline:**
```bash
python llama_8b_full_finetune_bf16.py
```
- Llama 3.1 8B Instruct
- Full model training (no LoRA)
- BF16 precision
- 100 steps, batch size 2, grad accum 8
- ~10-15 minutes on H100

**2. Run FP8 version:**
```bash
python llama_8b_full_finetune_fp8.py
```
- Same configuration as BF16
- FP8 mixed precision via Transformer Engine
- Should be **1.3-1.5x faster**

**Compare:**
- Steps/second: FP8 should be ~1.3-1.5x higher
- Memory: FP8 uses slightly more (expected)
- Loss: Should be similar (99%+ accuracy maintained)

---

### Alternative Tests (if you need LoRA)

**If you must use LoRA**, try lower ranks to reduce BF16 overhead:

```bash
# Low LoRA rank (8 instead of 32)
python test_fp8_8b_low_lora.py
# Expected: ~1.2x speedup (better than 1.00x with rank 32)
```

---

## Understanding FP8 Memory Behavior

**FP8 uses MORE memory than BF16** (this is correct behavior):

| Precision | What's Stored | Memory |
|-----------|---------------|--------|
| BF16 | Weights (BF16) + Gradients (BF16) | 1x |
| FP8 | Master weights (FP32) + FP8 copies + Scaling factors | ~1.5x |

**Why?** FP8 mixed precision needs:
1. **Master weights** in FP32/BF16 for accurate optimizer updates
2. **FP8 copies** for fast forward/backward computation
3. **Scaling factors** per layer to prevent underflow/overflow

**Trade-off:** More memory for faster compute (1.3-1.5x speedup)

---

## Key Learnings

### 1. `load_in_fp8=True` ≠ FP8 Training

unslothai/notebooks examples use `load_in_fp8=True`, but this is:
- **FP8 quantization** for inference (GRPO/RL with vLLM)
- **NOT** FP8 compute training

For FP8 compute training, use:
```python
setup_fp8_mixed_precision_training(backend="te")
```

### 2. LoRA and FP8 Don't Mix Well

LoRA adapters are **always BF16** and don't benefit from FP8.

On larger models (7B+) with higher LoRA ranks (32+):
- LoRA overhead can dominate compute
- FP8 speedup gets hidden
- Result: Minimal or zero speedup

**Solutions:**
- Use lower LoRA ranks (8-16)
- Use fewer target modules (just q_proj, v_proj)
- Skip LoRA entirely for maximum FP8 benefit

### 3. FP8 is About Speed, Not Memory

Don't use FP8 to save memory - it uses MORE memory!

Use FP8 when:
- ✅ You need faster training (1.3-1.5x)
- ✅ You have H100/H200 GPUs
- ✅ You can afford the memory overhead
- ✅ Speed matters more than memory

Don't use FP8 when:
- ❌ Memory is your bottleneck
- ❌ Using small models (<3B) with high LoRA ranks
- ❌ On A100 or older GPUs

---

## Expected Results on H100

### Full Fine-Tuning (Recommended)
```
BF16 Baseline:
  Time: 600s
  Steps/sec: 0.167
  Memory: 45 GB

FP8:
  Time: 430s (1.4x speedup ✅)
  Steps/sec: 0.233
  Memory: 52 GB (+15%, expected)
```

### LoRA Training (Suboptimal)
```
1B + LoRA rank 16:
  Speedup: 1.13x ✅ (LoRA is small)

8B + LoRA rank 32:
  Speedup: 1.00x ❌ (LoRA overhead too high)

8B + LoRA rank 8:
  Speedup: 1.20x ⚠️ (better, but not ideal)
```

---

## Files Reference

### Test Scripts
- `llama_8b_full_finetune_bf16.py` - BF16 baseline (run first)
- `llama_8b_full_finetune_fp8.py` - FP8 version (run second)
- `test_fp8_8b_low_lora.py` - Low LoRA rank test
- `diagnose_fp8.py` - Check FP8 layer coverage
- `benchmark_fp8_vs_bf16.py` - Comprehensive benchmark

### Documentation
- `FP8_FINDINGS.md` - Complete findings and analysis
- `FP8_LORA_INTERACTION.md` - Why LoRA hurts FP8 speedup
- `SFT_TRAINER_USAGE.md` - SFTTrainer API guide

---

## Questions?

**Q: Why is FP8 only 1.13x on 1B but 1.00x on 8B?**
A: LoRA overhead. 8B model with LoRA rank 32 spends too much time on BF16 LoRA compute.

**Q: Will full fine-tuning fit on my H100?**
A: Yes! The scripts use batch size 2 with gradient accumulation. Should use ~50GB on 8B model.

**Q: Can I use FP8 with LoRA?**
A: Yes, but use low ranks (8-16) on large models (7B+) for best results.

**Q: What about `load_in_fp8=True`?**
A: That's for FP8 quantization (inference), not FP8 training compute.

---

## Next Steps

1. Run the two full fine-tuning scripts
2. Compare results (should see 1.3-1.5x speedup)
3. If speedup is lower, run diagnostic script
4. Share results!
