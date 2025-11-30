#!/usr/bin/env python3
"""
Test FP8 with manual fp8_autocast context on Unsloth model.
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
import time
print("=" * 80)
print("FP8 Manual Context Test")
print("=" * 80)

from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training, convert_to_fp8
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as te_recipe
from transformer_engine.pytorch import fp8_autocast

# Setup
setup_fp8_mixed_precision_training()

# Load model
print("\n[1] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    load_in_16bit=True,
)

model = FastLanguageModel.for_training(model)

# Convert to FP8
print("\n[2] Converting to FP8...")
model = convert_to_fp8(model)

te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  TE layers: {te_count}")

# Disable gradient checkpointing
for module in model.modules():
    if hasattr(module, 'gradient_checkpointing'):
        module.gradient_checkpointing = False

# Create FP8 recipe
print("\n[3] Creating FP8 recipe...")
fp8_recipe = te_recipe.DelayedScaling(
    fp8_format=te_recipe.Format.HYBRID,
    amax_history_len=32,
    amax_compute_algo="max",
)

# Test inputs
input_ids = torch.randint(0, 32000, (4, 256), device="cuda")
attention_mask = torch.ones_like(input_ids)

# Warmup
print("\n[4] Warmup...")
model.train()
with torch.no_grad():
    for _ in range(3):
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()

# Benchmark WITHOUT fp8_autocast (should use BF16)
print("\n[5] Benchmark WITHOUT fp8_autocast:")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    for _ in range(20):
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()
bf16_time = time.perf_counter() - start
bf16_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"  Time: {bf16_time*1000:.1f} ms")
print(f"  Memory: {bf16_mem:.2f} GB")

# Benchmark WITH manual fp8_autocast
print("\n[6] Benchmark WITH manual fp8_autocast:")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
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
print("RESULTS")
print("=" * 80)
speedup = bf16_time / fp8_time
print(f"BF16: {bf16_time*1000:.1f} ms, {bf16_mem:.2f} GB")
print(f"FP8:  {fp8_time*1000:.1f} ms, {fp8_mem:.2f} GB")
print(f"Speedup: {speedup:.2f}x")

if speedup > 1.1:
    print("\n✅ FP8 IS faster when using manual fp8_autocast!")
    print("   Issue: apply_fp8_autocast wrapper isn't working properly")
elif speedup < 0.9:
    print("\n⚠️  FP8 is SLOWER - something is wrong with te.Linear in Unsloth model")
else:
    print("\n⚠️  No significant speedup - FP8 might not be activating")
