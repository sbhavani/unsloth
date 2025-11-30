#!/usr/bin/env python3
"""
FP8 Test following HuggingFace Accelerate's official benchmark pattern.
Reference: https://github.com/huggingface/accelerate/blob/main/benchmarks/fp8/transformer_engine/non_distributed.py
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
import time
import transformer_engine.common.recipe as te_recipe
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling
from transformer_engine.pytorch import fp8_autocast

print("=" * 80)
print("FP8 Test - Following HuggingFace Accelerate Reference")
print("=" * 80)

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate.utils.transformer_engine import convert_model

# ============================================================================
# BASELINE: Raw TransformerEngine (no Accelerate)
# ============================================================================
print("\n" + "=" * 80)
print("BASELINE: Raw TransformerEngine")
print("=" * 80)

print("\n[1] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map=None,
)
print(f"  Model: hidden={model.config.hidden_size}, intermediate={model.config.intermediate_size}")

# Get old params for optimizer mapping
def get_named_parameters(model):
    return {n: p for n, p in model.named_parameters()}

old_named_params = get_named_parameters(model)

# Convert to TE (following HF pattern)
print("\n[2] Converting to TE layers...")
with torch.no_grad():
    convert_model(model)

# Update optimizer params mapping (important!)
new_named_params = get_named_parameters(model)
mapping = {p: new_named_params[n] for n, p in old_named_params.items() if n in new_named_params}

te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  TE layers: {te_count}")

# FP8 Recipe (same as HF benchmark)
FP8_RECIPE_KWARGS = {"fp8_format": te_recipe.Format.HYBRID, "amax_history_len": 32, "amax_compute_algo": "max"}
fp8_recipe = DelayedScaling(**FP8_RECIPE_KWARGS)
print(f"  Recipe: {fp8_recipe}")

# Move to GPU
model.to("cuda")
model.gradient_checkpointing_disable()

# Test inputs
input_ids = torch.randint(0, 32000, (4, 256), device="cuda")
attention_mask = torch.ones_like(input_ids)
labels = input_ids.clone()

# Warmup with HF pattern: FP8 outer, BF16 inner
print("\n[3] Warmup (FP8 outer, BF16 inner - HF pattern)...")
model.train()
for _ in range(3):
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # Do backward to warm up FP8 scaling
    outputs.loss.backward()
model.zero_grad()
torch.cuda.synchronize()

# Benchmark BF16 ONLY (baseline)
print("\n[4] Benchmark BF16 ONLY (no FP8):")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    model.zero_grad()
torch.cuda.synchronize()
bf16_time = time.perf_counter() - start
bf16_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"  Time: {bf16_time*1000:.1f} ms (10 train steps)")
print(f"  Memory: {bf16_mem:.2f} GB")

# Benchmark FP8 + BF16 (HF pattern: FP8 outer, BF16 inner)
print("\n[5] Benchmark FP8 + BF16 (HF pattern):")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    model.zero_grad()
torch.cuda.synchronize()
fp8_time = time.perf_counter() - start
fp8_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"  Time: {fp8_time*1000:.1f} ms (10 train steps)")
print(f"  Memory: {fp8_mem:.2f} GB")

# Results
print("\n" + "=" * 80)
print("RESULTS (Raw TE - HF Reference Pattern)")
print("=" * 80)
speedup = bf16_time / fp8_time
print(f"BF16 only: {bf16_time*1000:.1f} ms, {bf16_mem:.2f} GB")
print(f"FP8 + BF16: {fp8_time*1000:.1f} ms, {fp8_mem:.2f} GB")
print(f"Speedup: {speedup:.2f}x")

if speedup > 1.1:
    print("\n✅ FP8 IS faster with HF reference pattern!")
elif speedup < 0.9:
    print("\n⚠️  FP8 is SLOWER - need to investigate")
else:
    print("\n⚠️  No significant difference")
