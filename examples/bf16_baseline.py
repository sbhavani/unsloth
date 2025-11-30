#!/usr/bin/env python3
"""
BF16 Baseline for comparison with FP8
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("=" * 80)
print("BF16 Baseline Training")
print("=" * 80)

# ============================================================================
# Load model (same as FP8)
# ============================================================================
print("\n[1/3] Loading model...")
max_seq_length = 512
dtype = torch.bfloat16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=False,
)

# Same settings as FP8 (no gradient checkpointing for fair comparison)
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=False)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================================
# Prepare dataset (same as FP8)
# ============================================================================
print("\n[2/3] Preparing dataset...")

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

def fixed_collator(features):
    batch = {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
        "labels": torch.stack([torch.tensor(f["labels"]) for f in features]),
    }
    if "attention_mask" in features[0]:
        batch["attention_mask"] = torch.stack([torch.tensor(f["attention_mask"]) for f in features])
    else:
        batch["attention_mask"] = (batch["input_ids"] != tokenizer.pad_token_id).long()
    return batch

# ============================================================================
# Train (same settings as FP8)
# ============================================================================
print("\n[3/3] Starting BF16 training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=fixed_collator,
    args=SFTConfig(
        per_device_train_batch_size=2,  # Same as FP8
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-5,
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_bf16",
        report_to="none",
        bf16=True,
        dataset_text_field=None,
        max_seq_length=max_seq_length,
    ),
)

trainer_stats = trainer.train()

# ============================================================================
# Results
# ============================================================================
print("\n" + "=" * 80)
print("BF16 Training Complete!")
print("=" * 80)
print(f"Training time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")
