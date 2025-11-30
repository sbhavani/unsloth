#!/usr/bin/env python3
"""
FP8 Training with Unsloth - Following Official Pattern

FP8 provides ~1.3-1.6x speedup on H100 GPUs.

Key requirements:
- batch_size × seq_len must be divisible by 8
- Use fixed padding to ensure consistent dimensions
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorWithPadding

print("=" * 80)
print("FP8 Training with Unsloth (Official Pattern)")
print("=" * 80)

# ============================================================================
# Step 1: Get FP8-configured Accelerator
# ============================================================================
print("\n[1/5] Setting up FP8 Accelerator...")
accelerator = setup_fp8_mixed_precision_training()

# ============================================================================
# Step 2: Load model
# ============================================================================
print("\n[2/5] Loading model...")
max_seq_length = 512  # Divisible by 8
dtype = torch.bfloat16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=False,
)

model = FastLanguageModel.for_training(model, use_gradient_checkpointing=False)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================================
# Step 3: Prepare with FP8 Accelerator
# ============================================================================
print("\n[3/5] Preparing model with FP8...")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model, optimizer = accelerator.prepare(model, optimizer)

import transformer_engine.pytorch as te
te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  Converted {te_count} layers to te.Linear for FP8")

# ============================================================================
# Step 4: Pre-tokenize dataset with FIXED padding
# ============================================================================
print("\n[4/5] Preparing dataset with fixed padding...")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def tokenize_fn(examples):
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        texts.append(alpaca_prompt.format(inst, inp, out) + tokenizer.eos_token)
    # FIXED padding - all sequences exactly max_seq_length
    tokens = tokenizer(
        texts, 
        truncation=True, 
        padding="max_length", 
        max_length=max_seq_length,
        return_attention_mask=True,
    )
    tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
    return tokens

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

print(f"  All sequences padded to {max_seq_length} tokens")
print(f"  8 × {max_seq_length} = {8 * max_seq_length} (divisible by 8 ✓)")

# ============================================================================
# Step 5: Train with SFTTrainer
# ============================================================================
print("\n[5/5] Starting FP8 training...")

# Simple collator that preserves fixed padding
def fixed_collator(features):
    batch = {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
        "labels": torch.stack([torch.tensor(f["labels"]) for f in features]),
    }
    # Add attention_mask if present, otherwise create from input_ids
    if "attention_mask" in features[0]:
        batch["attention_mask"] = torch.stack([torch.tensor(f["attention_mask"]) for f in features])
    else:
        # Create attention mask: 1 for non-pad tokens
        batch["attention_mask"] = (batch["input_ids"] != tokenizer.pad_token_id).long()
    return batch

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=fixed_collator,
    args=SFTConfig(
        per_device_train_batch_size=8,  # 8 × 512 = 4096, divisible by 8 ✓
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-5,
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        bf16=True,
        dataset_text_field=None,  # Using pre-tokenized data
        max_seq_length=max_seq_length,
    ),
)

trainer_stats = trainer.train()

# ============================================================================
# Results
# ============================================================================
print("\n" + "=" * 80)
print("FP8 Training Complete!")
print("=" * 80)
print(f"Training time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")
