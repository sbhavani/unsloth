#!/usr/bin/env python3
"""
Direct FP8 test using TE layers without any HuggingFace or Unsloth code.
Tests BOTH torch.amp (BF16) + fp8_autocast together.
Uses LLM-scale dimensions (4096 hidden, 14336 intermediate).
"""
import torch
import time
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as te_recipe
from transformer_engine.pytorch import fp8_autocast

print("=" * 80)
print("Direct FP8 Test - BF16 + FP8 Autocast (LLM-scale)")
print("=" * 80)

# LLM-scale dimensions (same as Llama 8B)
HIDDEN = 4096
INTERMEDIATE = 14336
BATCH = 4
SEQ_LEN = 512

print(f"\n[1] Creating TE model with LLM-scale dimensions...")
print(f"  Hidden: {HIDDEN}, Intermediate: {INTERMEDIATE}")
print(f"  Input shape: ({BATCH}, {SEQ_LEN}, {HIDDEN})")

class SimpleTEModel(torch.nn.Module):
    """Simple MLP mimicking LLM FFN block"""
    def __init__(self):
        super().__init__()
        self.gate_proj = te.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = te.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = te.Linear(INTERMEDIATE, HIDDEN, bias=False)
    
    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        return x

model = SimpleTEModel().cuda().bfloat16()

print("\n[2] Creating FP8 recipe...")
fp8_recipe = te_recipe.DelayedScaling(
    fp8_format=te_recipe.Format.HYBRID,
    amax_history_len=32,
    amax_compute_algo="max",
)
print(f"  Recipe: {fp8_recipe}")

# Create input
x = torch.randn(BATCH, SEQ_LEN, HIDDEN, dtype=torch.bfloat16, device="cuda")

# Warmup with both autocasts
print("\n[3] Warmup...")
with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            for _ in range(10):
                _ = model(x)
torch.cuda.synchronize()

# Benchmark BF16 ONLY
print("\n[4] Benchmark BF16 ONLY:")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(100):
            _ = model(x)
torch.cuda.synchronize()
bf16_time = (time.perf_counter() - start) * 1000
bf16_mem = torch.cuda.max_memory_allocated() / 1e6
print(f"  Time: {bf16_time:.2f} ms for 100 iters")
print(f"  Memory: {bf16_mem:.1f} MB")

# Benchmark BF16 + FP8
print("\n[5] Benchmark BF16 + FP8 autocast:")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            for _ in range(100):
                _ = model(x)
torch.cuda.synchronize()
fp8_time = (time.perf_counter() - start) * 1000
fp8_mem = torch.cuda.max_memory_allocated() / 1e6
print(f"  Time: {fp8_time:.2f} ms for 100 iters")
print(f"  Memory: {fp8_mem:.1f} MB")

# Results
print("\n" + "=" * 80)
print("RESULTS (Direct TE - BF16 vs BF16+FP8)")
print("=" * 80)
speedup = bf16_time / fp8_time
print(f"BF16 only:  {bf16_time:.2f} ms, {bf16_mem:.1f} MB")
print(f"BF16 + FP8: {fp8_time:.2f} ms, {fp8_mem:.1f} MB")
print(f"Speedup: {speedup:.2f}x")

if speedup > 1.1:
    print("\n✅ FP8 IS faster with BF16+FP8 setup!")
else:
    print("\n⚠️  FP8 not showing expected speedup")
