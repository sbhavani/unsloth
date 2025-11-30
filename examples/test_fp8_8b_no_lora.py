"""
FP8 8B Test - NO LoRA (Full Fine-Tuning)

Test pure FP8 speedup without LoRA adapters.
LoRA adds BF16 computation that doesn't benefit from FP8.
"""

import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training
import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
import time

print("=" * 80)
print("FP8 8B Test - NO LoRA (Pure FP8 on full model)")
print("=" * 80)

# Configuration
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 8  # Smaller due to full fine-tuning
GRADIENT_ACCUMULATION = 8
NUM_STEPS = 50  # Fewer steps (full FT is slower)

print(f"\nConfiguration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Mode: FULL fine-tuning (no LoRA)")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

# Enable FP8
print("\n[1/4] Setting up FP8...")
setup_fp8_mixed_precision_training(backend="te")

# Load model - NO LoRA
print(f"\n[2/4] Loading model (no LoRA)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,
)

# Make model trainable (no LoRA, just regular fine-tuning)
print("\n[3/4] Preparing for full fine-tuning...")
for param in model.parameters():
    param.requires_grad = True

# Dataset
print("\n[4/4] Preparing dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]")  # Smaller dataset

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = f"### Instruction:\n{instruction}\n"
        if input_text:
            text += f"\n### Input:\n{input_text}\n"
        text += f"\n### Response:\n{output}"
        texts.append(text)
    return texts

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# Training
print("\nTraining WITHOUT LoRA (pure FP8)...")
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
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_steps=NUM_STEPS,
        warmup_steps=2,
        learning_rate=2e-5,  # Lower LR for full FT
        fp16=False,
        bf16=True,  # FP8 works WITH BF16 autocast!
        logging_steps=10,
        optim="adamw_torch",  # Regular AdamW for full FT
        weight_decay=0.01,
        lr_scheduler_type="linear",
        output_dir="outputs/test_fp8_8b_no_lora",
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
samples_per_sec = (NUM_STEPS * BATCH_SIZE * GRADIENT_ACCUMULATION) / total_time
mem_peak = torch.cuda.max_memory_allocated() / 1024**3

print("\n" + "=" * 80)
print("✅ NO LORA TEST COMPLETED")
print("=" * 80)
print(f"\nPerformance:")
print(f"  Total time: {total_time:.2f}s")
print(f"  Steps/second: {steps_per_sec:.4f}")
print(f"  Samples/second: {samples_per_sec:.2f}")
print(f"  Final loss: {result.training_loss:.4f}")
print(f"  Peak memory: {mem_peak:.2f} GB")
print(f"\n💡 This shows PURE FP8 speedup without LoRA overhead.")
print(f"   Compare this to BF16 full fine-tuning to see true FP8 benefit.")
print("=" * 80)
