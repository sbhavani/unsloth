#!/usr/bin/env python3
"""
Typical Unsloth BF16 Training - Standard workflow with all optimizations
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("=" * 80)
print("Typical Unsloth BF16 Training (SFTTrainer + all optimizations)")
print("=" * 80)

# Load model - standard Unsloth pattern
print("\n[1/3] Loading model...")
max_seq_length = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# Standard Unsloth: gradient checkpointing ENABLED (saves memory)
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
print("\n[2/3] Preparing dataset...")
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

# Train with SFTTrainer - standard Unsloth pattern
print("\n[3/3] Starting training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=8,  # Can use larger batch with grad checkpointing
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,  # Standard Unsloth: ENABLED
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-5,
        logging_steps=10,
        optim="adamw_8bit",  # Standard Unsloth: 8-bit optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        bf16=True,
    ),
)

trainer_stats = trainer.train()

print("\n" + "=" * 80)
print("Typical Unsloth BF16 Complete!")
print("=" * 80)
print(f"Time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")
