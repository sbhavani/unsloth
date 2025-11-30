#!/usr/bin/env python3
"""
Test FP8 converting ONLY MLP layers (gate_proj, up_proj, down_proj).
Attention layers stay as nn.Linear to reduce FP8 overhead.
MLP is ~2/3 of compute, attention is already optimized by Flash Attention.
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
import time
print("=" * 80)
print("FP8 Test - MLP Layers Only (Attention stays BF16)")
print("=" * 80)

from transformers import AutoModelForCausalLM, AutoTokenizer
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as te_recipe
from transformer_engine.pytorch import fp8_autocast
from accelerate import Accelerator
from accelerate.state import AcceleratorState

if not AcceleratorState._shared_state:
    _ = Accelerator()

print("\n[1] Loading HuggingFace 8B model...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

print(f"  Model: hidden={model.config.hidden_size}, layers={model.config.num_hidden_layers}")

# Convert ONLY MLP layers to TE
print("\n[2] Converting ONLY MLP layers to FP8...")
mlp_names = ['gate_proj', 'up_proj', 'down_proj']
converted = 0

with torch.no_grad():
    for name, module in model.named_modules():
        # Check if this is an MLP linear layer
        if isinstance(module, torch.nn.Linear):
            layer_name = name.split('.')[-1]
            if layer_name in mlp_names:
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                parent = model.get_submodule(parent_name)
                
                # Create TE Linear
                te_linear = te.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    params_dtype=torch.bfloat16,
                )
                te_linear.weight.copy_(module.weight)
                if module.bias is not None:
                    te_linear.bias.copy_(module.bias)
                te_linear.to(module.weight.device)
                
                # Replace
                setattr(parent, layer_name, te_linear)
                converted += 1

te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
nn_count = sum(1 for m in model.modules() if type(m).__name__ == 'Linear' and 'torch.nn' in type(m).__module__)
print(f"  Converted {converted} MLP layers to te.Linear")
print(f"  te.Linear: {te_count}, nn.Linear: {nn_count}")

model.gradient_checkpointing_disable()

# FP8 recipe
fp8_recipe = te_recipe.DelayedScaling(
    fp8_format=te_recipe.Format.HYBRID,
    amax_history_len=32,
    amax_compute_algo="max",
)

# Test inputs
input_ids = torch.randint(0, 32000, (4, 256), device="cuda")
attention_mask = torch.ones_like(input_ids)

# Warmup
print("\n[3] Warmup...")
model.eval()
with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            for _ in range(3):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()

# Benchmark BF16 ONLY
print("\n[4] Benchmark BF16 ONLY (baseline):")
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

# Benchmark BF16 + FP8 (MLP only)
print("\n[5] Benchmark BF16 + FP8 (MLP layers only):")
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
print("RESULTS (MLP-only FP8)")
print("=" * 80)
speedup = bf16_time / fp8_time
print(f"BF16 only:      {bf16_time*1000:.1f} ms, {bf16_mem:.2f} GB")
print(f"BF16 + FP8 MLP: {fp8_time*1000:.1f} ms, {fp8_mem:.2f} GB")
print(f"Speedup: {speedup:.2f}x")

if speedup > 1.1:
    print("\n✅ MLP-only FP8 shows speedup!")
    print("   Converting only MLP layers reduces FP8 overhead")
elif speedup < 0.9:
    print("\n⚠️  Still slower - FP8 overhead is significant even with fewer layers")
else:
    print("\n⚠️  No significant difference")
