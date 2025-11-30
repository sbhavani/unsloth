#!/usr/bin/env python3
"""
FP8 Training with Unsloth + Accelerate

This example shows how to use FP8 mixed precision training with Unsloth.
FP8 provides ~1.3-1.6x speedup on H100 GPUs for compute-bound workloads.

Key points:
- Use setup_fp8_mixed_precision_training() to get an FP8-configured Accelerator
- Call accelerator.prepare(model, optimizer) TOGETHER (required for FP8)
- Use larger batch sizes for best FP8 benefits (compute-bound workloads)
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"
# Disable torch.compile to avoid conflicts with FP8
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"

import torch
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training
from datasets import load_dataset
import time

print("=" * 80)
print("FP8 Training with Unsloth + Accelerate")
print("=" * 80)

# ============================================================================
# Step 1: Get FP8-configured Accelerator
# ============================================================================
print("\n[1/5] Setting up FP8 Accelerator...")
accelerator = setup_fp8_mixed_precision_training(
    fp8_format="HYBRID",      # E4M3 forward, E5M2 backward (recommended)
    amax_history_len=32,      # Scaling factor history length
    amax_compute_algo="max",  # Use max from history (stable)
)

# ============================================================================
# Step 2: Load model with Unsloth
# ============================================================================
print("\n[2/5] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,  # FP8 works best without quantization
)

# Enable training mode (applies Unsloth optimizations)
model = FastLanguageModel.for_training(model)

# ============================================================================
# Step 3: Create optimizer and prepare with Accelerator
# ============================================================================
print("\n[3/5] Preparing model with FP8 Accelerator...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# CRITICAL: Prepare model and optimizer TOGETHER for FP8!
model, optimizer = accelerator.prepare(model, optimizer)

# Disable gradient checkpointing (conflicts with FP8 prepared model)
for module in model.modules():
    if hasattr(module, 'gradient_checkpointing'):
        module.gradient_checkpointing = False

# Check conversion
import transformer_engine.pytorch as te
te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  Converted {te_count} layers to te.Linear")

# ============================================================================
# Step 4: Prepare dataset
# ============================================================================
print("\n[4/5] Preparing dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]")

def format_alpaca(example):
    if example["input"]:
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": text}

dataset = dataset.map(format_alpaca)

# Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None,
    )

dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
dataset.set_format("torch")

# Create dataloader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ============================================================================
# Step 5: Training loop
# ============================================================================
print("\n[5/5] Starting FP8 training...")

model.train()
num_steps = 50
total_loss = 0
start_time = time.perf_counter()

for step, batch in enumerate(dataloader):
    if step >= num_steps:
        break
    
    # Move to device
    input_ids = batch["input_ids"].to(accelerator.device)
    attention_mask = batch["attention_mask"].to(accelerator.device)
    labels = input_ids.clone()
    
    # Forward
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    # Backward
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    
    total_loss += loss.item()
    
    if (step + 1) % 10 == 0:
        print(f"  Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")

elapsed = time.perf_counter() - start_time
avg_loss = total_loss / num_steps

print("\n" + "=" * 80)
print("FP8 Training Complete!")
print("=" * 80)
print(f"Training time: {elapsed:.1f}s")
print(f"Steps/sec: {num_steps/elapsed:.2f}")
print(f"Average loss: {avg_loss:.4f}")
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
