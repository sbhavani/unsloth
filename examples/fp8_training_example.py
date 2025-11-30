#!/usr/bin/env python3
"""
FP8 Training with Unsloth + Accelerate

This example shows how to use FP8 mixed precision training with Unsloth.
FP8 provides ~1.3-1.6x speedup on H100 GPUs for compute-bound workloads.

Key points:
- Use setup_fp8_mixed_precision_training() to get an FP8-configured Accelerator
- Call accelerator.prepare(model, optimizer) TOGETHER (required for FP8)
- Use larger batch sizes for best FP8 benefits (compute-bound workloads)
- Keep bf16=True in TrainingArguments (FP8 works WITH BF16 autocast)
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

print("=" * 80)
print("FP8 Training with Unsloth + Accelerate")
print("=" * 80)

# ============================================================================
# Step 1: Get FP8-configured Accelerator
# ============================================================================
print("\n[1/5] Setting up FP8 Accelerator...")
accelerator = setup_fp8_mixed_precision_training(
    fp8_format="HYBRID",      # E4M3 forward, E5M2 backward (recommended)
    amax_history_len=32,      # Scaling factor history length
    amax_compute_algo="max",  # Use max from history (stable)
)

# ============================================================================
# Step 2: Load model with Unsloth
# ============================================================================
print("\n[2/5] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,  # FP8 works best without quantization
)

# Enable training mode (applies Unsloth optimizations)
model = FastLanguageModel.for_training(model)

# ============================================================================
# Step 3: Create optimizer and prepare with Accelerator
# ============================================================================
print("\n[3/5] Preparing model with FP8 Accelerator...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# CRITICAL: Prepare model and optimizer TOGETHER for FP8!
model, optimizer = accelerator.prepare(model, optimizer)

# Check conversion
import transformer_engine.pytorch as te
te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  Converted {te_count} layers to te.Linear")

# ============================================================================
# Step 4: Prepare dataset
# ============================================================================
print("\n[4/5] Preparing dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

def format_alpaca(example):
    if example["input"]:
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": text}

dataset = dataset.map(format_alpaca)

# ============================================================================
# Step 5: Train with SFTTrainer
# ============================================================================
print("\n[5/5] Starting FP8 training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./fp8_output",
        per_device_train_batch_size=8,  # Larger batch = more FP8 benefit!
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=50,
        learning_rate=1e-5,
        bf16=True,  # KEEP bf16=True! FP8 works WITH BF16 autocast
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    ),
    dataset_text_field="text",
    max_seq_length=512,
)

# Train!
result = trainer.train()

print("\n" + "=" * 80)
print("FP8 Training Complete!")
print("=" * 80)
print(f"Training time: {result.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {result.metrics['train_samples_per_second']:.2f}")
print(f"Final loss: {result.metrics['train_loss']:.4f}")
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
