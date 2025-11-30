"""
BF16 baseline for 8B using the SAME pattern as the 1B test.

This is the baseline to compare against test_fp8_8b_like_1b.py.
"""

import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

# Import unsloth FIRST before other libraries
from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
import time

print("=" * 80)
print("BF16 8B Baseline - Using SAME pattern as 1B test")
print("=" * 80)

# Configuration - SAME as FP8 test
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 512
LORA_RANK = 8
BATCH_SIZE = 4
NUM_STEPS = 100

print(f"\nConfiguration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Max seq length: {MAX_SEQ_LENGTH}")
print(f"  LoRA rank: {LORA_RANK}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Steps: {NUM_STEPS}")

# NO FP8 setup - BF16 baseline

# Load model
print("\n[1/4] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,
)

# Add LoRA
print("\n[2/4] Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",
)

# Prepare dataset
print("\n[3/4] Preparing dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(text)
    return texts

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Train
print("\n[4/4] Running training...")
print("=" * 80)

torch.cuda.reset_peak_memory_stats()

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=data_collator,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        max_steps=NUM_STEPS,
        fp16=False,
        bf16=True,  # BF16 baseline
        logging_steps=10,
        output_dir="outputs/test_bf16_8b_like_1b",
        report_to="none",
        dataloader_num_workers=0,
    ),
)

start_time = time.time()
result = trainer.train()
end_time = time.time()

# Metrics
total_time = end_time - start_time
steps_per_sec = NUM_STEPS / total_time
mem_peak = torch.cuda.max_memory_allocated() / 1024**3

print("\n" + "=" * 80)
print("✅ 8B BF16 BASELINE COMPLETED")
print("=" * 80)
print(f"\nPerformance:")
print(f"  Total time: {total_time:.2f}s")
print(f"  Steps/second: {steps_per_sec:.4f}")
print(f"  Final loss: {result.training_loss:.4f}")
print(f"  Peak memory: {mem_peak:.2f} GB")
print(f"\n💡 Compare this to test_fp8_8b_like_1b.py to measure FP8 speedup.")
print("=" * 80)
