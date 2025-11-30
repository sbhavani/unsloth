"""
Llama 3.1 8B Full Fine-Tuning with BF16 (Baseline)

Full fine-tuning (no LoRA) with BF16 precision as baseline.
Compare this to llama_8b_full_finetune_fp8.py to measure FP8 speedup.

Expected: FP8 should be 1.3-1.5x faster than this baseline.
"""

import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
import time

print("=" * 80)
print("Llama 3.1 8B - Full Fine-Tuning with BF16 (BASELINE)")
print("=" * 80)

# Configuration
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2  # Same as FP8 version
GRADIENT_ACCUMULATION = 8
NUM_STEPS = 100
LEARNING_RATE = 1e-5

print(f"\nConfiguration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Mode: FULL fine-tuning (no LoRA)")
print(f"  Precision: BF16 (baseline)")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Training steps: {NUM_STEPS}")

# NO FP8 setup - use BF16

# Load model for full fine-tuning (NO LoRA)
print(f"\n[1/3] Loading model for full fine-tuning...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,
)

# NO get_peft_model() call - train the full model!
print("\n[2/3] Preparing for full fine-tuning (all parameters trainable)...")
model = FastLanguageModel.for_training(model)

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

# Prepare dataset - Alpaca format
print("\n[3/3] Preparing Alpaca dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# Setup tokenizer and data collator
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training
print("\nStarting full fine-tuning with BF16 (baseline)...")
print("=" * 80)

torch.cuda.reset_peak_memory_stats()

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    data_collator=data_collator,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_steps=NUM_STEPS,
        warmup_steps=5,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=True,  # Use BF16
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs/llama_8b_full_ft_bf16",
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    ),
)

start_time = time.time()
result = trainer.train()
end_time = time.time()

# Metrics
total_time = end_time - start_time
steps_per_sec = NUM_STEPS / total_time
samples_per_sec = (NUM_STEPS * BATCH_SIZE * GRADIENT_ACCUMULATION) / total_time
mem_peak = torch.cuda.max_memory_allocated() / 1024**3

print("\n" + "=" * 80)
print("✅ FULL FINE-TUNING WITH BF16 COMPLETED (BASELINE)")
print("=" * 80)
print(f"\nPerformance:")
print(f"  Total time: {total_time:.2f}s")
print(f"  Steps/second: {steps_per_sec:.4f}")
print(f"  Samples/second: {samples_per_sec:.2f}")
print(f"  Final loss: {result.training_loss:.4f}")
print(f"  Peak memory: {mem_peak:.2f} GB")
print(f"\n💡 This is the BF16 baseline for comparison.")
print(f"   Run llama_8b_full_finetune_fp8.py to compare FP8 speedup.")
print("=" * 80)
