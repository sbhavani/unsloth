#!/usr/bin/env python3
"""
Test FP8 exactly how HuggingFace Accelerate does it:
- Convert nn.Linear -> te.Linear
- Use Accelerate's prepare() with fp8 mixed precision
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"
os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"

import torch
import time
print("=" * 80)
print("FP8 Test - Accelerate Style (te.Linear only)")
print("=" * 80)

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import transformer_engine.pytorch as te

# Create accelerator with FP8
print("\n[1] Creating Accelerator with FP8...")
accelerator = Accelerator(mixed_precision="fp8")
print(f"  mixed_precision: {accelerator.mixed_precision}")
print(f"  state.mixed_precision: {accelerator.state.mixed_precision}")

print("\n[2] Loading HuggingFace 8B model...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map=None,  # Let accelerator handle device
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
print(f"  Model loaded: hidden={model.config.hidden_size}")

# Count layers before
nn_before = sum(1 for m in model.modules() if type(m).__name__ == 'Linear' and 'torch.nn' in type(m).__module__)
te_before = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  Before prepare: nn.Linear={nn_before}, te.Linear={te_before}")

# Create dummy optimizer (required by accelerate for TE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Prepare with accelerator - this converts to TE and wraps forward
print("\n[3] Running accelerator.prepare(model, optimizer)...")
model, optimizer = accelerator.prepare(model, optimizer)

# Count layers after
nn_after = sum(1 for m in model.modules() if type(m).__name__ == 'Linear' and 'torch.nn' in type(m).__module__)
te_after = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  After prepare: nn.Linear={nn_after}, te.Linear={te_after}")

# Disable gradient checkpointing
if hasattr(model, 'gradient_checkpointing_disable'):
    model.gradient_checkpointing_disable()

# Test inputs
input_ids = torch.randint(0, 32000, (4, 256), device="cuda")
attention_mask = torch.ones_like(input_ids)

# Warmup
print("\n[4] Warmup...")
model.train()  # Training mode for FP8
with torch.no_grad():
    for _ in range(3):
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()

# Benchmark - Accelerate handles fp8_autocast internally through prepared model
print("\n[5] Benchmark (Accelerate handles FP8 internally):")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    for _ in range(20):
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()
fp8_time = time.perf_counter() - start
fp8_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"  Time: {fp8_time*1000:.1f} ms (20 iters)")
print(f"  Memory: {fp8_mem:.2f} GB")

# Compare to baseline - reload without FP8
print("\n[6] Loading baseline (BF16 only, no TE)...")
del model, optimizer
torch.cuda.empty_cache()

accelerator_bf16 = Accelerator(mixed_precision="bf16")
model_bf16 = AutoModelForCausalLM.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map=None,
).to("cuda")
optimizer_bf16 = torch.optim.AdamW(model_bf16.parameters(), lr=1e-5)
model_bf16, optimizer_bf16 = accelerator_bf16.prepare(model_bf16, optimizer_bf16)

if hasattr(model_bf16, 'gradient_checkpointing_disable'):
    model_bf16.gradient_checkpointing_disable()

# Warmup baseline
model_bf16.train()
with torch.no_grad():
    for _ in range(3):
        _ = model_bf16(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()

# Benchmark baseline
print("\n[7] Benchmark BF16 baseline:")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    for _ in range(20):
        _ = model_bf16(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()
bf16_time = time.perf_counter() - start
bf16_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"  Time: {bf16_time*1000:.1f} ms (20 iters)")
print(f"  Memory: {bf16_mem:.2f} GB")

# Results
print("\n" + "=" * 80)
print("RESULTS (Accelerate Style)")
print("=" * 80)
speedup = bf16_time / fp8_time
print(f"BF16: {bf16_time*1000:.1f} ms, {bf16_mem:.2f} GB")
print(f"FP8:  {fp8_time*1000:.1f} ms, {fp8_mem:.2f} GB")
print(f"Speedup: {speedup:.2f}x")

if speedup > 1.1:
    print("\n✅ Accelerate-style FP8 shows speedup!")
elif speedup < 0.9:
    print("\n⚠️  FP8 is slower even with Accelerate's native approach")
else:
    print("\n⚠️  No significant difference")
