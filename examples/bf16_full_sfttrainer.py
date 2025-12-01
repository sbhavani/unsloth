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

# Load model with Unsloth (same as notebook)
print("\n[1/3] Loading model...")
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# For full fine-tuning: skip get_peft_model(), just call for_training()
# Disable gradient checkpointing to match FP8 (fair comparison)
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=False)

# Explicitly disable gradient checkpointing on model (Unsloth may enable it)
if hasattr(model, 'gradient_checkpointing_disable'):
    model.gradient_checkpointing_disable()
model.config.use_cache = True  # Enable cache when not using GC

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Debug: show frozen parameters
print("  Frozen parameters:")
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"    - {name}: {param.numel():,} params")

# FIX: Unfreeze all parameters for full fine-tuning
print("  Unfreezing all parameters for full fine-tuning...")
for param in model.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  After unfreeze: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Prepare dataset (notebook pattern)
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

# Train with SFTTrainer (notebook pattern)
print("\n[3/3] Training...")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=4,  # Same as FP8
        gradient_accumulation_steps=4,  # Effective batch = 16
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
        # SFT-specific (TRL 0.24.0)
        dataset_text_field="text",
        max_length=max_seq_length,
        packing=False,
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
