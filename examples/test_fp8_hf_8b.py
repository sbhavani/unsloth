#!/usr/bin/env python3
"""
Test FP8 with standard HuggingFace 8B model (NO Unsloth patching).
8B has hidden=4096 which should benefit from FP8.
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
import time
print("=" * 80)
print("FP8 Test - Standard HuggingFace 8B Model (NO Unsloth)")
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
print("\n[2] Converting to FP8...")
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

# Warmup
print("\n[4] Warmup...")
model.eval()  # Use eval mode for inference test
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
print("RESULTS (Standard HuggingFace 8B - NO Unsloth)")
print("=" * 80)
speedup = bf16_time / fp8_time
print(f"BF16: {bf16_time*1000:.1f} ms, {bf16_mem:.2f} GB")
print(f"FP8:  {fp8_time*1000:.1f} ms, {fp8_mem:.2f} GB")
print(f"Speedup: {speedup:.2f}x")

if speedup > 1.1:
    print("\n✅ FP8 IS faster on 8B HuggingFace model!")
    print("   If Unsloth 8B was slower, then Unsloth patching breaks FP8")
elif speedup < 0.9:
    print("\n⚠️  FP8 is SLOWER even on 8B model")
    print("   There may be an issue with TE or the benchmark setup")
else:
    print("\n⚠️  No significant difference on 8B")
