#!/usr/bin/env python3
"""
Test FP8 directly without Unsloth to verify TE works.
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
print("=" * 80)
print("Direct FP8 Test (without Unsloth wrappers)")
print("=" * 80)

import transformer_engine.pytorch as te
import transformer_engine.common.recipe as te_recipe
from transformer_engine.pytorch import fp8_autocast

# Create a simple model with TE layers - use LARGE dimensions like real LLMs
print("\n[1] Creating simple TE model with LLM-scale dimensions...")
# Llama 8B uses hidden_size=4096, intermediate_size=14336
HIDDEN = 4096
INTERMEDIATE = 14336

class SimpleTEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Simulate MLP: hidden -> intermediate -> hidden
        self.gate_proj = te.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = te.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = te.Linear(INTERMEDIATE, HIDDEN, bias=False)
    
    def forward(self, x):
        # SwiGLU-style MLP
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        return x

model = SimpleTEModel().cuda().bfloat16()
model.train()
print(f"  Hidden: {HIDDEN}, Intermediate: {INTERMEDIATE}")

# Create FP8 recipe
print("\n[2] Creating FP8 recipe...")
fp8_recipe = te_recipe.DelayedScaling(
    fp8_format=te_recipe.Format.HYBRID,
    amax_history_len=32,
    amax_compute_algo="max",
)
print(f"  Recipe: {fp8_recipe}")

# Test forward WITHOUT fp8_autocast
print("\n[3] Forward WITHOUT fp8_autocast:")
# Use realistic batch size and sequence length
BATCH = 4
SEQ_LEN = 512
x = torch.randn(BATCH, SEQ_LEN, HIDDEN, device="cuda", dtype=torch.bfloat16)
print(f"  Input shape: {x.shape} (batch={BATCH}, seq={SEQ_LEN}, hidden={HIDDEN})")
with torch.no_grad():
    y = model(x)
print(f"  Output shape: {y.shape}")

# Test forward WITH fp8_autocast
print("\n[4] Forward WITH fp8_autocast:")
with torch.no_grad():
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        y = model(x)
print(f"  Output shape: {y.shape}")

# Check memory difference
print("\n[5] Memory comparison:")
torch.cuda.reset_peak_memory_stats()

# BF16 forward
with torch.no_grad():
    for _ in range(10):
        y = model(x)
torch.cuda.synchronize()
bf16_mem = torch.cuda.max_memory_allocated() / 1e6
print(f"  BF16 peak memory: {bf16_mem:.1f} MB")

torch.cuda.reset_peak_memory_stats()

# FP8 forward
with torch.no_grad():
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        for _ in range(10):
            y = model(x)
torch.cuda.synchronize()
fp8_mem = torch.cuda.max_memory_allocated() / 1e6
print(f"  FP8 peak memory: {fp8_mem:.1f} MB")

# Speed comparison
import time

print("\n[6] Speed comparison:")

# BF16 speed
torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        y = model(x)
torch.cuda.synchronize()
bf16_time = time.perf_counter() - start
print(f"  BF16: {bf16_time*1000:.2f} ms for 100 iters")

# FP8 speed
torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        for _ in range(100):
            y = model(x)
torch.cuda.synchronize()
fp8_time = time.perf_counter() - start
print(f"  FP8: {fp8_time*1000:.2f} ms for 100 iters")

speedup = bf16_time / fp8_time
print(f"\n  Speedup: {speedup:.2f}x")
if speedup > 1.1:
    print("  ✅ FP8 IS faster on this hardware!")
else:
    print("  ⚠️  FP8 not showing expected speedup")
