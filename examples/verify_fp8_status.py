#!/usr/bin/env python3
"""
Verify FP8 is actually being used at runtime.
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

print("=" * 80)
print("FP8 Status Verification")
print("=" * 80)

# Check environment variables
print("\n[1] Environment Variables:")
for var in ["ACCELERATE_MIXED_PRECISION", "ACCELERATE_FP8_BACKEND", "ACCELERATE_FP8_FORMAT"]:
    print(f"  {var}: {os.environ.get(var, 'NOT SET')}")

# Setup FP8
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training, convert_to_fp8
setup_fp8_mixed_precision_training(backend="te")

print("\n[2] After setup_fp8_mixed_precision_training():")
for var in ["ACCELERATE_MIXED_PRECISION", "ACCELERATE_FP8_BACKEND", "ACCELERATE_FP8_FORMAT"]:
    print(f"  {var}: {os.environ.get(var, 'NOT SET')}")

# Check Accelerator state
print("\n[3] Accelerator State:")
from accelerate import Accelerator
from accelerate.state import AcceleratorState

# Create accelerator to see what it picks up
acc = Accelerator()
print(f"  mixed_precision: {acc.mixed_precision}")
print(f"  state.mixed_precision: {acc.state.mixed_precision}")
print(f"  fp8_enabled: {getattr(acc, 'fp8_enabled', 'N/A')}")
print(f"  native_amp: {getattr(acc, 'native_amp', 'N/A')}")

# Check if we have FP8 handler
if hasattr(acc, 'fp8_recipe_handler'):
    print(f"  fp8_recipe_handler: {acc.fp8_recipe_handler}")
if hasattr(acc, 'te_recipe_handler'):
    print(f"  te_recipe_handler: {acc.te_recipe_handler}")

# Check TE availability
print("\n[4] Transformer Engine:")
try:
    import transformer_engine.pytorch as te
    print(f"  ✅ transformer_engine available")
    print(f"  Version: {te.__version__ if hasattr(te, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"  ❌ transformer_engine NOT available: {e}")

# Load a small model to check layer types
print("\n[5] Loading model to check layer types...")
import torch
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    load_in_16bit=True,
)

# Count layer types
print("\n[6] Layer Types BEFORE accelerator.prepare():")
te_count = 0
linear_count = 0
for name, module in model.named_modules():
    if hasattr(te, 'Linear') and isinstance(module, te.Linear):
        te_count += 1
    elif isinstance(module, torch.nn.Linear):
        linear_count += 1
print(f"  te.Linear: {te_count}")
print(f"  nn.Linear: {linear_count}")

# Convert to FP8 using convert_to_fp8 (bypasses accelerator.prepare limitation)
print("\n[7] Running convert_to_fp8(model)...")
model = convert_to_fp8(model)

# Count again
print("\n[8] Layer Types AFTER convert_to_fp8():")
te_count = 0
linear_count = 0
for name, module in model.named_modules():
    if hasattr(te, 'Linear') and isinstance(module, te.Linear):
        te_count += 1
        if te_count <= 3:  # Print first few
            print(f"    Found te.Linear: {name}")
    elif isinstance(module, torch.nn.Linear):
        linear_count += 1
print(f"  te.Linear: {te_count}")
print(f"  nn.Linear: {linear_count}")

if te_count == 0:
    print("\n  ⚠️  WARNING: No TE layers found! FP8 may not be active!")
else:
    print(f"\n  ✅ Found {te_count} TE layers - FP8 should be active")

# Check if model forward is wrapped
print("\n[9] Model forward wrapping:")
print(f"  model.forward: {model.forward}")
if hasattr(model.forward, '__wrapped__'):
    print(f"  __wrapped__: {model.forward.__wrapped__}")
