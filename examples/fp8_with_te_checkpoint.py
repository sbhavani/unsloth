#!/usr/bin/env python3
"""
FP8 Training with TE's native checkpoint function
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import torch
import time
from functools import partial

# Import Unsloth first (as recommended)
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformer_engine.pytorch as te

print("=" * 80)
print("FP8 + TE Checkpoint (native FP8 gradient checkpointing)")
print("=" * 80)

# Setup FP8 (this applies Unsloth's patches)
print("\n[1/6] Setting up FP8...")
accelerator = setup_fp8_mixed_precision_training()

# NOW patch GradientCheckpointingLayer AFTER Unsloth's setup
print("\n[2/6] Patching GradientCheckpointingLayer with TE checkpoint...")
from transformers.modeling_layers import GradientCheckpointingLayer

def te_gc_call(self, *args, **kwargs):
    """GradientCheckpointingLayer that uses TE's checkpoint for FP8"""
    if self.gradient_checkpointing and self.training:
        # Use TE's checkpoint instead of torch's - handles FP8 scaling factors
        return te.distributed.checkpoint(
            partial(torch.nn.Module.__call__, self, **kwargs),
            *args,
            use_reentrant=True,
        )
    return torch.nn.Module.__call__(self, *args, **kwargs)

GradientCheckpointingLayer.__call__ = te_gc_call
print("  ✅ Patched GradientCheckpointingLayer")

# Load model
print("\n[3/6] Loading model...")
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
# Enable gradient checkpointing
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
tokenizer.pad_token = tokenizer.eos_token

# Manually set _gradient_checkpointing_func so our TE patch doesn't get bypassed
for module in model.modules():
    if isinstance(module, GradientCheckpointingLayer):
        module._gradient_checkpointing_func = te.distributed.checkpoint

# Prepare with FP8
print("\n[4/6] Preparing with FP8...")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model, optimizer = accelerator.prepare(model, optimizer)

te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  Converted {te_count} layers to te.Linear")

# Verify gradient checkpointing is enabled
gc_enabled = sum(1 for m in model.modules() if hasattr(m, 'gradient_checkpointing') and m.gradient_checkpointing)
print(f"  Layers with gradient_checkpointing=True: {gc_enabled}")

# Prepare dataset
print("\n[5/6] Preparing dataset...")
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
print("\n[6/6] Starting training...")
print("=" * 80)
print("FP8 + TE Checkpoint Training (batch=8, seq=512)")
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
print("FP8 + TE Checkpoint Complete!")
print("=" * 80)
print(f"Time: {elapsed:.1f}s")
print(f"Samples/sec: {num_steps * 8 / elapsed:.2f}")
print(f"Avg loss: {total_loss / num_steps:.4f}")
print(f"Peak memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
