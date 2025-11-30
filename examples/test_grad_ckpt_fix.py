#!/usr/bin/env python3
"""
Test script to verify gradient checkpointing fix for full fine-tuning.

This tests that FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
properly sets _gradient_checkpointing_func, which is required by transformers 4.57+.

Previously this would fail with:
    AttributeError: 'LlamaDecoderLayer' object has no attribute '_gradient_checkpointing_func'
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
from unsloth import FastLanguageModel

print("=" * 80)
print("Testing Gradient Checkpointing Fix for Full Fine-tuning")
print("=" * 80)

# Load model
print("\n[1/3] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# Enable gradient checkpointing - this is where the bug was
print("\n[2/3] Enabling gradient checkpointing via for_training()...")
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
tokenizer.pad_token = tokenizer.eos_token

# Verify _gradient_checkpointing_func is set
print("\n[3/3] Verifying gradient checkpointing setup...")

# Check on the base model
base_model = model.model if hasattr(model, 'model') else model
has_func = hasattr(base_model, '_gradient_checkpointing_func')
print(f"  Base model has _gradient_checkpointing_func: {has_func}")

if has_func:
    func = base_model._gradient_checkpointing_func
    print(f"  Function: {func}")

# Also check gradient_checkpointing flag on layers
gc_enabled = False
for name, module in model.named_modules():
    if hasattr(module, 'gradient_checkpointing') and module.gradient_checkpointing:
        gc_enabled = True
        break
print(f"  Gradient checkpointing enabled on layers: {gc_enabled}")

# Test a forward pass
print("\n[4/4] Testing forward pass with gradient checkpointing...")
try:
    inputs = tokenizer("Hello world!", return_tensors="pt").to(model.device)
    inputs["labels"] = inputs["input_ids"].clone()
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**inputs)
    
    print(f"  Forward pass successful!")
    print(f"  Loss: {outputs.loss.item():.4f}")
    
    # Test backward pass
    outputs.loss.backward()
    print(f"  Backward pass successful!")
    
except AttributeError as e:
    if "_gradient_checkpointing_func" in str(e):
        print(f"\n  ❌ FAILED: {e}")
        print("  The fix did not work - _gradient_checkpointing_func is still missing")
        exit(1)
    else:
        raise
except Exception as e:
    print(f"\n  ❌ FAILED with unexpected error: {e}")
    raise

print("\n" + "=" * 80)
print("✅ SUCCESS: Gradient checkpointing is working correctly!")
print("=" * 80)
print(f"\nPeak memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
