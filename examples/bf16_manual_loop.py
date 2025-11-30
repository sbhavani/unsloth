#!/usr/bin/env python3
"""
BF16 Training with Unsloth - same settings as FP8 for fair comparison
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # Same as FP8 test

import torch
import time
from unsloth import FastLanguageModel
from datasets import load_dataset
from torch.utils.data import DataLoader

# Patch for gradient checkpointing
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
    _orig = GradientCheckpointingLayer.__call__
    def _patched(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            if not hasattr(self, '_gradient_checkpointing_func'):
                self.gradient_checkpointing = False
        return _orig(self, *args, **kwargs)
    GradientCheckpointingLayer.__call__ = _patched
except: pass

print("=" * 80)
print("BF16 Training - Unsloth (fused loss disabled)")
print("=" * 80)

# Load model with Unsloth
print("\n[1/3] Loading model...")
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=False)
tokenizer.pad_token = tokenizer.eos_token

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Prepare dataset
print("\n[2/3] Preparing dataset...")
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

# Same batch=4 as FP8
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate)

# Training
print("\n[3/3] Starting BF16 Training (batch=4, seq=512)")
print("=" * 80)

model.train()
device = "cuda"
model = model.to(device)
num_steps = 60
total_loss = 0

start = time.perf_counter()
for step, batch in enumerate(dataloader):
    if step >= num_steps:
        break
    
    batch = {k: v.to(device) for k, v in batch.items()}
    
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
print("BF16 Training Complete!")
print("=" * 80)
print(f"Time: {elapsed:.1f}s")
print(f"Samples/sec: {num_steps * 4 / elapsed:.2f}")
print(f"Avg loss: {total_loss / num_steps:.4f}")
print(f"Peak memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
