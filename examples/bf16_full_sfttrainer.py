#!/usr/bin/env python3
"""
BF16 Full Fine-tuning with SFTTrainer (baseline comparison)
Matches Unsloth notebook pattern as closely as possible
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # Match FP8 setting

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("=" * 80)
print("BF16 Full Fine-tuning + SFTTrainer (Llama-3.2-3B)")
print("=" * 80)

# Check GPU
gpu_name = torch.cuda.get_device_name(0)
gpu_cap = torch.cuda.get_device_capability(0)
print(f"\nGPU: {gpu_name}")
print(f"Compute capability: {gpu_cap[0]}.{gpu_cap[1]}")

# Load model with Unsloth
# NOTE: NOT using full_finetuning=True for fair comparison with FP8
# (FP8 can't use it due to fused CE loss conflict with Transformer Engine)
print("\n[1/3] Loading model...")
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    # NOT using full_finetuning=True - matching FP8 for fair comparison
)

# For full fine-tuning: skip get_peft_model(), just call for_training()
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=False)

# Manually unfreeze ALL parameters (same as FP8 script)
for param in model.parameters():
    param.requires_grad = True

# Explicitly disable gradient checkpointing
if hasattr(model, 'gradient_checkpointing_disable'):
    model.gradient_checkpointing_disable()
model.config.use_cache = True

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Prepare dataset - pre-tokenize to match FP8 script exactly
print("\n[2/3] Preparing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

# Pre-tokenize with fixed padding (same as FP8 script for fair comparison)
def tokenize_and_format(examples):
    texts = [alpaca_prompt.format(i, inp, o) + EOS_TOKEN 
             for i, inp, o in zip(examples["instruction"], examples["input"], examples["output"])]
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=max_seq_length)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
dataset = dataset.map(tokenize_and_format, batched=True, remove_columns=dataset.column_names)

# Train with SFTTrainer (notebook pattern)
print("\n[3/3] Training...")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # Effective batch = 16
        gradient_checkpointing=False,   # Match FP8 for fair comparison
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-5,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        bf16=True,
        remove_unused_columns=False,  # Pre-tokenized data
    ),
)

trainer_stats = trainer.train()

print("\n" + "=" * 80)
print("BF16 Full Fine-tuning Complete!")
print("=" * 80)
print(f"Time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")
