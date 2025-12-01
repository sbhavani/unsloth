#!/usr/bin/env python3
"""
FP8 Full Fine-tuning with SFTTrainer
Based on working manual loop approach - uses setup_fp8_mixed_precision_training()
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # Disable fused loss (incompatible with FP8)

import torch
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training
from datasets import load_dataset

print("=" * 80)
print("FP8 Full Fine-tuning + Trainer (Llama-3.2-3B)")
print("=" * 80)

# Check GPU
gpu_name = torch.cuda.get_device_name(0)
gpu_cap = torch.cuda.get_device_capability(0)
print(f"\nGPU: {gpu_name}")
print(f"Compute capability: {gpu_cap[0]}.{gpu_cap[1]}")

# Setup FP8 FIRST (like manual loop)
print("\n[1/4] Setting up FP8...")
accelerator = setup_fp8_mixed_precision_training()

# Load model with Unsloth (no full_finetuning=True to avoid fused loss)
print("\n[2/4] Loading model...")
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Ensure consistent padding

# Prepare model with FP8 (converts to te.Linear)
# Note: TE requires model+optimizer together for prepare()
print("\n[3/4] Preparing with FP8...")
_dummy_opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
model, _ = accelerator.prepare(model, _dummy_opt)
del _dummy_opt  # Trainer will create its own optimizer

import transformer_engine.pytorch as te
te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  Converted {te_count} layers to te.Linear")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Prepare dataset
print("\n[4/4] Preparing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

# Pre-tokenize with fixed padding for FP8 alignment (like manual loop)
def tokenize_fn(examples):
    texts = [alpaca_prompt.format(i, inp, o) + EOS_TOKEN 
             for i, inp, o in zip(examples["instruction"], examples["input"], examples["output"])]
    return tokenizer(
        texts, 
        truncation=True, 
        padding="max_length",  # Fixed padding for FP8 alignment
        max_length=max_seq_length, 
        return_tensors=None
    )

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

# Add labels for causal LM
def add_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples
dataset = dataset.map(add_labels, batched=True)

# Train with HF Trainer (pre-tokenized data for FP8 alignment)
print("\nStarting training...")
print("=" * 80)

# Use standard Trainer since we pre-tokenized (more reliable with FP8)
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        per_device_train_batch_size=4,  # 4 × 512 = 2048, divisible by 8 ✓
        gradient_accumulation_steps=4,  # Effective batch = 16
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-5,
        logging_steps=10,
        optim="adamw_8bit",  # Save memory
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        bf16=True,
        remove_unused_columns=False,
    ),
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer_stats = trainer.train()

print("\n" + "=" * 80)
print("FP8 Full Fine-tuning Complete!")
print("=" * 80)
print(f"Time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")
