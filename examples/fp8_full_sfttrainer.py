#!/usr/bin/env python3
"""
FP8 Full Fine-tuning with SFTTrainer
Does NOT use full_finetuning=True to avoid fused CE loss conflict with FP8/TE.
Instead, manually unfreezes all parameters.
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("=" * 80)
print("FP8 Full Fine-tuning + SFTTrainer (Llama-3.2-3B)")
print("=" * 80)

# Check GPU
gpu_name = torch.cuda.get_device_name(0)
gpu_cap = torch.cuda.get_device_capability(0)
print(f"\nGPU: {gpu_name}")
print(f"Compute capability: {gpu_cap[0]}.{gpu_cap[1]}")

# Setup FP8 FIRST
print("\n[1/4] Setting up FP8...")
accelerator = setup_fp8_mixed_precision_training()

# Load model WITHOUT full_finetuning=True (to avoid fused CE loss conflict with FP8)
print("\n[2/4] Loading model...")
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    # Note: NOT using full_finetuning=True - we'll manually unfreeze params
)

# For full fine-tuning: skip get_peft_model(), just call for_training()
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=False)

# Manually unfreeze ALL parameters for full fine-tuning
# (This is what full_finetuning=True does, but without the fused CE loss)
for param in model.parameters():
    param.requires_grad = True

# Explicitly disable gradient checkpointing on model (Unsloth may enable it)
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

# Prepare dataset - must pre-tokenize with fixed padding for FP8 alignment
print("\n[4/4] Preparing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

# FP8 requires fixed sequence lengths for dimension alignment
def tokenize_and_format(examples):
    texts = [alpaca_prompt.format(i, inp, o) + EOS_TOKEN 
             for i, inp, o in zip(examples["instruction"], examples["input"], examples["output"])]
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=max_seq_length)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
dataset = dataset.map(tokenize_and_format, batched=True, remove_columns=dataset.column_names)

# Train with SFTTrainer
print("\nStarting training...")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
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
        remove_unused_columns=False,
    ),
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
