#!/usr/bin/env python3
"""
Test FP8 with accelerate launch (proper way to enable FP8)

Run with:
    accelerate launch --mixed_precision fp8 examples/test_fp8_accelerate_launch.py

This bypasses TrainingArguments and lets Accelerate properly control FP8.
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
import time

# Import unsloth first
from unsloth import FastLanguageModel

from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer

# Check if we're running under accelerate with FP8
from accelerate import Accelerator
accelerator = Accelerator()
print(f"Accelerator mixed_precision: {accelerator.mixed_precision}")
print(f"Accelerator state mixed_precision: {accelerator.state.mixed_precision}")

# Configuration
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 1e-5
NUM_STEPS = 100

print("=" * 80)
print("FP8 Test with accelerate launch")
print("=" * 80)

# Load model
print("\n[1/4] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=True,
)

# Enable training
print("[2/4] Enabling training mode...")
model = FastLanguageModel.for_training(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

# Dataset
print("[3/4] Preparing dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

alpaca_prompt = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

def formatting_prompts_func(examples):
    texts = []
    for i, o, inp in zip(examples["instruction"], examples["output"], examples["input"]):
        texts.append(alpaca_prompt.format(instruction=i, input=inp, output=o))
    return texts

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# Train - DON'T set bf16 or fp16 - let accelerate handle it!
print("[4/4] Training...")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=data_collator,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_steps=NUM_STEPS,
        learning_rate=LEARNING_RATE,
        # DON'T set fp16 or bf16 - let accelerate control precision!
        logging_steps=10,
        output_dir="outputs/fp8_accelerate_launch",
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        warmup_steps=5,
    ),
)

torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
result = trainer.train()
end = time.perf_counter()

total_time = end - start
peak_mem = torch.cuda.max_memory_allocated() / 1e9

print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Time: {total_time:.2f}s")
print(f"Steps/sec: {NUM_STEPS/total_time:.4f}")
print(f"Samples/sec: {(NUM_STEPS * BATCH_SIZE * GRADIENT_ACCUMULATION)/total_time:.2f}")
print(f"Peak memory: {peak_mem:.2f} GB")
print(f"Final loss: {result.training_loss:.4f}")
