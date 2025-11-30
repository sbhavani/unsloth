#!/usr/bin/env python3
"""
FP8 Training WITH Gradient Checkpointing - experimental
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import torch
import time
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training
from datasets import load_dataset
from torch.utils.data import DataLoader

print("=" * 80)
print("FP8 + Gradient Checkpointing (experimental)")
print("=" * 80)

# Setup FP8
print("\n[1/5] Setting up FP8...")
accelerator = setup_fp8_mixed_precision_training()

# Load model WITHOUT gradient checkpointing first
print("\n[2/5] Loading model...")
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
# Load without grad checkpointing - we'll enable it after prepare
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=False)
tokenizer.pad_token = tokenizer.eos_token

# Prepare with FP8
print("\n[3/5] Preparing with FP8...")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model, optimizer = accelerator.prepare(model, optimizer)

import transformer_engine.pytorch as te
te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  Converted {te_count} layers to te.Linear")

# NOW enable gradient checkpointing on the inner model
print("\n[4/5] Enabling gradient checkpointing...")
try:
    # Try to access the inner model through accelerate's wrapping
    inner_model = model
    while hasattr(inner_model, 'module'):
        inner_model = inner_model.module
    
    # Enable gradient checkpointing using torch's native API
    if hasattr(inner_model, 'gradient_checkpointing_enable'):
        inner_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("  ✅ Gradient checkpointing enabled (use_reentrant=False)")
    else:
        print("  ⚠️ Model doesn't support gradient_checkpointing_enable")
except Exception as e:
    print(f"  ❌ Failed to enable gradient checkpointing: {e}")

# Prepare dataset
print("\n[5/5] Preparing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def tokenize_fn(examples):
    texts = [alpaca_prompt.format(i, inp, o) + tokenizer.eos_token 
             for i, inp, o in zip(examples["instruction"], examples["input"], examples["output"])]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=max_seq_length, return_attention_mask=True)

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
dataset.set_format("torch")

def collate(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
    }

# Try larger batch since we have gradient checkpointing
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)

# Training
print("\n" + "=" * 80)
print("Starting FP8 + Grad Checkpointing Training (batch=8, seq=512)")
print("=" * 80)

model.train()
num_steps = 60
total_loss = 0

start = time.perf_counter()
for step, batch in enumerate(dataloader):
    if step >= num_steps:
        break
    
    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
    
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["input_ids"],
    )
    
    accelerator.backward(outputs.loss)
    optimizer.step()
    optimizer.zero_grad()
    
    total_loss += outputs.loss.item()
    if (step + 1) % 10 == 0:
        print(f"  Step {step+1}/{num_steps}, Loss: {outputs.loss.item():.4f}")

elapsed = time.perf_counter() - start

print("\n" + "=" * 80)
print("FP8 + Gradient Checkpointing Complete!")
print("=" * 80)
print(f"Time: {elapsed:.1f}s")
print(f"Samples/sec: {num_steps * 8 / elapsed:.2f}")
print(f"Avg loss: {total_loss / num_steps:.4f}")
print(f"Peak memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
