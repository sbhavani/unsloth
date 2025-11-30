#!/usr/bin/env python3
"""
BF16 Full Fine-tuning with gradient checkpointing enabled
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # Same as FP8

import torch
import time
from unsloth import FastLanguageModel
from datasets import load_dataset
from torch.utils.data import DataLoader

print("=" * 80)
print("BF16 Full Fine-tuning (batch=8, seq=512, WITH grad ckpt)")
print("=" * 80)

# Load model - full fine-tuning (no LoRA)
print("\n[1/4] Loading model...")
max_seq_length = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    full_finetuning=True,  # Proper way to enable full fine-tuning with gradient checkpointing
)

# for_training() just sets training mode flags
model = FastLanguageModel.for_training(model)
tokenizer.pad_token = tokenizer.eos_token

print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Prepare optimizer
print("\n[2/4] Preparing optimizer...")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model = model.cuda()

# Prepare dataset
print("\n[3/4] Preparing dataset...")
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

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)

# Training
print("\n[4/4] Starting training...")
print("=" * 80)

model.train()
num_steps = 60
total_loss = 0

start = time.perf_counter()
for step, batch in enumerate(dataloader):
    if step >= num_steps:
        break
    
    batch = {k: v.cuda() for k, v in batch.items()}
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        )
    
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    total_loss += outputs.loss.item()
    if (step + 1) % 10 == 0:
        print(f"  Step {step+1}/{num_steps}, Loss: {outputs.loss.item():.4f}")

elapsed = time.perf_counter() - start

print("\n" + "=" * 80)
print("BF16 Full Fine-tuning Complete!")
print("=" * 80)
print(f"Time: {elapsed:.1f}s")
print(f"Samples/sec: {num_steps * 8 / elapsed:.2f}")
print(f"Avg loss: {total_loss / num_steps:.4f}")
print(f"Peak memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
