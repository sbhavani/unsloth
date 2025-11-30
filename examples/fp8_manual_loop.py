#!/usr/bin/env python3
"""
FP8 Training - Manual loop with plain HuggingFace model
(Unsloth's fused loss causes OOM with FP8 + larger batches)
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import FP8RecipeKwargs

print("=" * 80)
print("FP8 Training - HuggingFace Model (batch=2)")
print("=" * 80)

# Setup FP8 Accelerator
print("\n[1/4] Setting up FP8...")
kwargs_handlers = [FP8RecipeKwargs(backend="TE", fp8_format="HYBRID", amax_history_len=32, amax_compute_algo="max")]
accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=kwargs_handlers)

# Load model - plain HuggingFace (no Unsloth fused loss)
print("\n[2/4] Loading model (HuggingFace, no Unsloth patches)...")
max_seq_length = 512

model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map=None,  # Let accelerator handle device
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Disable gradient checkpointing
if hasattr(model, 'gradient_checkpointing_disable'):
    model.gradient_checkpointing_disable()

# Prepare with FP8
print("\n[3/4] Preparing with FP8...")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model, optimizer = accelerator.prepare(model, optimizer)

import transformer_engine.pytorch as te
te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  Converted {te_count} layers to te.Linear")

# Prepare dataset
print("\n[4/4] Preparing dataset...")
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

dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)

# Training
print("\n" + "=" * 80)
print("Starting FP8 Training (batch=2, seq=512)")
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
print("FP8 Training Complete!")
print("=" * 80)
print(f"Time: {elapsed:.1f}s")
print(f"Samples/sec: {num_steps * 2 / elapsed:.2f}")
print(f"Avg loss: {total_loss / num_steps:.4f}")
print(f"Peak memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
