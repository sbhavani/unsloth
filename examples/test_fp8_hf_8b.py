#!/usr/bin/env python3
"""
Test FP8 with standard HuggingFace 8B model (NO Unsloth patching).
Uses BOTH torch.amp (BF16) AND fp8_autocast together - the correct setup.
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
import time
print("=" * 80)
print("FP8 Test - HuggingFace 8B with BF16 + FP8 Autocast")
print("=" * 80)

# Load WITHOUT Unsloth
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as te_recipe
from transformer_engine.pytorch import fp8_autocast
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils.transformer_engine import convert_model

# Initialize Accelerator (required for convert_model)
if not AcceleratorState._shared_state:
    _ = Accelerator()

print("\n[1] Loading HuggingFace 8B model (NO Unsloth)...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")

print(f"  Model config: hidden={model.config.hidden_size}, intermediate={model.config.intermediate_size}")

# Count nn.Linear before
nn_count = sum(1 for m in model.modules() if type(m).__name__ == 'Linear' and 'torch.nn' in type(m).__module__)
print(f"  nn.Linear layers: {nn_count}")

# Convert to FP8
print("\n[2] Converting to FP8 (te.Linear layers)...")
with torch.no_grad():
    convert_model(model, to_transformer_engine=True, _convert_linear=True, _convert_ln=False)

te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  TE layers: {te_count}")

# Disable gradient checkpointing if enabled
model.gradient_checkpointing_disable()

# Create FP8 recipe
print("\n[3] Creating FP8 recipe...")
fp8_recipe = te_recipe.DelayedScaling(
    fp8_format=te_recipe.Format.HYBRID,
    amax_history_len=32,
    amax_compute_algo="max",
)

# Test inputs - use batch=4, seq=256 to fit in memory
input_ids = torch.randint(0, 32000, (4, 256), device="cuda")
attention_mask = torch.ones_like(input_ids)

# Warmup with both autocasts
print("\n[4] Warmup...")
model.eval()
with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            for _ in range(3):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()

# Benchmark BF16 ONLY (baseline)
print("\n[5] Benchmark BF16 ONLY (baseline):")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(20):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()
bf16_time = time.perf_counter() - start
bf16_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"  Time: {bf16_time*1000:.1f} ms")
print(f"  Memory: {bf16_mem:.2f} GB")

# Benchmark BF16 + FP8 (the correct setup)
print("\n[6] Benchmark BF16 + FP8 autocast (GEMMs in FP8):")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            for _ in range(20):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()
fp8_time = time.perf_counter() - start
fp8_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"  Time: {fp8_time*1000:.1f} ms")
print(f"  Memory: {fp8_mem:.2f} GB")

# Results
print("\n" + "=" * 80)
print("RESULTS (HuggingFace 8B - BF16 baseline vs BF16+FP8)")
print("=" * 80)
speedup = bf16_time / fp8_time
print(f"BF16 only:  {bf16_time*1000:.1f} ms, {bf16_mem:.2f} GB")
print(f"BF16 + FP8: {fp8_time*1000:.1f} ms, {fp8_mem:.2f} GB")
print(f"Speedup: {speedup:.2f}x")

if speedup > 1.1:
    print("\n✅ FP8 IS faster with BF16+FP8 setup!")
    print("   This is the correct configuration for training.")
elif speedup < 0.9:
    print("\n⚠️  FP8 is SLOWER - unexpected for 8B model")
else:
    print("\n⚠️  No significant difference")
