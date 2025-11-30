"""
Test FP8 on 8B using the SAME pattern as the 1B test that worked.

Key difference from full fine-tuning:
- Uses get_peft_model() (LoRA) instead of for_training()
- Uses lower LoRA rank (8) like the 1B test
- Same code path that showed 1.13x speedup

The hypothesis is that get_peft_model() applies Unsloth patches
that are compatible with FP8, while for_training() does not.
"""

import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

# Import unsloth FIRST before other libraries
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training, check_fp8_training_support

import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
import time

print("=" * 80)
print("FP8 8B Test - Using SAME pattern as 1B test that worked")
print("=" * 80)

# Check FP8 support
if not check_fp8_training_support():
    print("❌ FP8 not supported")
    exit(1)

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# Configuration - SAME as 1B test pattern
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 512  # Same as 1B test
LORA_RANK = 8  # Same as 1B test!
BATCH_SIZE = 4  # Same as 1B test
NUM_STEPS = 100

print(f"\nConfiguration (matching 1B test):")
print(f"  Model: {MODEL_NAME}")
print(f"  Max seq length: {MAX_SEQ_LENGTH}")
print(f"  LoRA rank: {LORA_RANK}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Steps: {NUM_STEPS}")

# Enable FP8 FIRST (same order as 1B test)
print("\n[1/5] Enabling FP8 mixed precision training...")
setup_fp8_mixed_precision_training(backend="te")

# Load model (same as 1B test)
print("\n[2/5] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,
)

# Add LoRA (SAME as 1B test - this is the key!)
print("\n[3/5] Adding LoRA adapters (same as 1B test)...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,  # Same rank as 1B test
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Same modules
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",
)

# Prepare dataset (same pattern)
print("\n[4/5] Preparing dataset...")
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
    pad_to_multiple_of=8,
)

# Train (same pattern)
print("\n[5/5] Running training...")
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
        bf16=True,  # FP8 works WITH BF16 autocast!
        logging_steps=10,
        output_dir="outputs/test_fp8_8b_like_1b",
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
print("✅ 8B TEST (1B PATTERN) COMPLETED")
print("=" * 80)
print(f"\nPerformance:")
print(f"  Total time: {total_time:.2f}s")
print(f"  Steps/second: {steps_per_sec:.4f}")
print(f"  Final loss: {result.training_loss:.4f}")
print(f"  Peak memory: {mem_peak:.2f} GB")
print(f"\n💡 This uses the SAME code path as the 1B test that got 1.13x speedup.")
print(f"   If this works, the issue is with for_training() vs get_peft_model().")
print("=" * 80)
