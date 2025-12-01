#!/usr/bin/env python3
"""
FP8 Full Fine-tuning with SFTTrainer - Standard Configuration
Uses typical SFTTrainer settings that users would normally use.
Only FP8-specific changes are made (no gradient checkpointing, manual param unfreeze).
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("=" * 80)
print("FP8 Full Fine-tuning + SFTTrainer - STANDARD CONFIG (Llama-3.1-8B)")
print("=" * 80)

# Check GPU
gpu_name = torch.cuda.get_device_name(0)
gpu_cap = torch.cuda.get_device_capability(0)
print(f"\nGPU: {gpu_name}")
print(f"Compute capability: {gpu_cap[0]}.{gpu_cap[1]}")

# Setup FP8 FIRST
print("\n[1/4] Setting up FP8...")
accelerator = setup_fp8_mixed_precision_training()

# Load model with full_finetuning=True
# Fused CE loss is auto-disabled when FP8/TE layers are detected
print("\n[2/4] Loading model...")
max_seq_length = 2048  # Standard seq length
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    full_finetuning=True,  # Now works with FP8!
)

model = FastLanguageModel.for_training(model, use_gradient_checkpointing=False)

# FP8 REQUIRED: Disable gradient checkpointing (cuBLAS issues on some GPUs)
if hasattr(model, 'gradient_checkpointing_disable'):
    model.gradient_checkpointing_disable()
model.config.use_cache = True

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Prepare with FP8
print("\n[3/4] Preparing with FP8...")
_dummy_opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
model, _ = accelerator.prepare(model, _dummy_opt)
del _dummy_opt

import transformer_engine.pytorch as te
te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  Converted {te_count} layers to te.Linear")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  After FP8 prep: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Prepare dataset - STANDARD: Use formatting function + dataset_text_field
print("\n[4/4] Preparing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        texts.append(alpaca_prompt.format(inst, inp, out) + EOS_TOKEN)
    return {"text": texts}

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Train with SFTTrainer - STANDARD CONFIG
print("\nStarting training...")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        # Standard batch settings
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        
        # FP8 REQUIRED: No gradient checkpointing
        gradient_checkpointing=False,
        
        # Standard training settings
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=10,
        
        # Standard optimizer settings
        optim="adamw_8bit",
        weight_decay=0.01,  # Standard
        lr_scheduler_type="cosine",  # Standard
        max_grad_norm=1.0,  # Standard gradient clipping
        
        seed=3407,
        output_dir="outputs",
        report_to="none",
        save_strategy="steps",
        save_steps=500,
        
        bf16=True,
        
        # Standard SFT settings
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    ),
)

trainer_stats = trainer.train()

print("\n" + "=" * 80)
print("FP8 Full Fine-tuning Complete! (Standard Config)")
print("=" * 80)
print(f"Time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")
