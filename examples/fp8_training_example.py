#!/usr/bin/env python3
"""
FP8 Training with Unsloth

FP8 provides ~1.3-1.6x speedup on H100 GPUs with Unsloth optimizations.

NOTE: SFTTrainer has compatibility issues with Accelerate FP8 + variable seq lengths.
This uses a manual loop with fixed-length padding which works reliably.
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training
from datasets import load_dataset
from torch.utils.data import DataLoader

print("=" * 80)
print("FP8 Training with Unsloth")
print("=" * 80)

# ============================================================================
# Step 1: Get FP8-configured Accelerator
# ============================================================================
print("\n[1/5] Setting up FP8 Accelerator...")
accelerator = setup_fp8_mixed_precision_training()

# ============================================================================
# Step 2: Load model
# ============================================================================
print("\n[2/5] Loading model...")
# Use seq_length divisible by 8
max_seq_length = 512  # Smaller for memory, divisible by 8
dtype = torch.bfloat16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=False,
)

model = FastLanguageModel.for_training(model, use_gradient_checkpointing=False)

# ============================================================================
# Step 3: Prepare with FP8 Accelerator
# ============================================================================
print("\n[3/5] Preparing model with FP8...")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model, optimizer = accelerator.prepare(model, optimizer)

import transformer_engine.pytorch as te
te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  Converted {te_count} layers to te.Linear for FP8")

# ============================================================================
# Step 4: Prepare dataset with FIXED padding
# ============================================================================
print("\n[4/5] Preparing dataset...")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        texts.append(alpaca_prompt.format(inst, inp, out) + tokenizer.eos_token)
    # FIXED length padding - critical for FP8!
    return tokenizer(texts, truncation=True, padding="max_length", max_length=max_seq_length)

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
dataset.set_format("torch")

# Simple collate that preserves fixed padding
def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
    }

# batch_size=8: 8 × 512 = 4096, divisible by 8 ✓
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# ============================================================================
# Step 5: Training loop
# ============================================================================
print("\n[5/5] Starting FP8 training...")
print(f"  Batch size: 8, Seq length: {max_seq_length}")
print(f"  8 × {max_seq_length} = {8 * max_seq_length} (divisible by 8 ✓)")

model.train()
total_loss = 0
num_steps = 60

import time
start_time = time.time()

for step, batch in enumerate(dataloader):
    if step >= num_steps:
        break
    
    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
    
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["input_ids"],
    )
    loss = outputs.loss
    
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    
    total_loss += loss.item()
    
    if (step + 1) % 10 == 0:
        print(f"  Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")

elapsed = time.time() - start_time

# ============================================================================
# Results
# ============================================================================
print("\n" + "=" * 80)
print("FP8 Training Complete!")
print("=" * 80)
print(f"Training time: {elapsed:.1f}s")
print(f"Steps: {num_steps}")
print(f"Avg loss: {total_loss / num_steps:.4f}")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")
