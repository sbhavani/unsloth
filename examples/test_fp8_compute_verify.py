#!/usr/bin/env python3
"""
Verify FP8 compute is actually happening by checking inside forward pass.
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
print("=" * 80)
print("FP8 Compute Verification")
print("=" * 80)

from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training, convert_to_fp8, apply_fp8_autocast
import transformer_engine.pytorch as te

# Setup
setup_fp8_mixed_precision_training()

# Load small model
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

# Check TE layers exist
te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  TE layers: {te_count}")

# Apply autocast wrapper
print("\n[3] Applying FP8 autocast wrapper...")
model = apply_fp8_autocast(model)

# Check if forward is wrapped
print("\n[4] Checking forward wrapper:")
print(f"  model.forward: {model.forward}")
if hasattr(model.forward, '__wrapped__'):
    print(f"  __wrapped__: YES - wrapper is applied")
else:
    print(f"  __wrapped__: NO - wrapper may not be applied!")

# Test forward pass with FP8 context check
print("\n[5] Testing forward pass...")

# Create a hook to check if FP8 is active during forward
fp8_was_active = [False]
fp8_meta_found = [False]

def check_fp8_hook(module, input, output):
    # Check if the module has FP8 metadata (indicates FP8 is being used)
    if hasattr(module, 'fp8_meta') and module.fp8_meta is not None:
        fp8_meta_found[0] = True
        # Check if fp8_meta has scaling factors (indicates active FP8)
        if hasattr(module.fp8_meta, 'scale'):
            fp8_was_active[0] = True
    return output

# Find first TE layer and add hook
for name, module in model.named_modules():
    if isinstance(module, te.Linear):
        module.register_forward_hook(check_fp8_hook)
        print(f"  Added hook to: {name}")
        break

# Run forward
model.train()

# Disable gradient checkpointing for this test
for module in model.modules():
    if hasattr(module, 'gradient_checkpointing'):
        module.gradient_checkpointing = False

input_ids = torch.randint(0, 32000, (2, 64), device="cuda")
attention_mask = torch.ones_like(input_ids)

print("\n[6] Running forward pass...")
with torch.no_grad():
    output = model(input_ids=input_ids, attention_mask=attention_mask)

print(f"\n[7] Results:")
print(f"  FP8 meta found: {fp8_meta_found[0]}")
print(f"  FP8 scaling active: {fp8_was_active[0]}")

if fp8_meta_found[0]:
    print("  ✅ FP8 metadata IS present on TE layers!")
else:
    print("  ❌ FP8 metadata NOT found - FP8 may not be active")

# Also check memory to see if FP8 scaling factors are present
print(f"\n[8] Memory check:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
