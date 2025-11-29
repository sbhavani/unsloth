"""
FP8 8B Test - Lower LoRA Rank

Test hypothesis: LoRA rank 32 is too high and dominates compute,
hiding FP8 benefits. Try with rank 8 or 16 instead.
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
print("FP8 8B Test - Low LoRA Rank (8 instead of 32)")
print("=" * 80)

# Configuration
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 16  # Large batch for H100
GRADIENT_ACCUMULATION = 4
NUM_STEPS = 100
LORA_RANK = 8  # MUCH smaller than 32!

print(f"\nConfiguration:")
print(f"  Model: {MODEL_NAME}")
print(f"  LoRA rank: {LORA_RANK} (vs 32 in previous test)")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

# Enable FP8
print("\n[1/5] Setting up FP8...")
setup_fp8_mixed_precision_training(backend="te")

# Load model
print(f"\n[2/5] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,
)

# Add LoRA with SMALL rank
print(f"\n[3/5] Adding LoRA (rank {LORA_RANK})...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,  # Small rank!
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Dataset
print("\n[4/5] Preparing dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

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
print("\n[5/5] Training with LOW LoRA rank...")
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
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        output_dir="outputs/test_fp8_8b_low_lora",
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
print("✅ LOW LORA RANK TEST COMPLETED")
print("=" * 80)
print(f"\nPerformance:")
print(f"  Total time: {total_time:.2f}s")
print(f"  Steps/second: {steps_per_sec:.4f}")
print(f"  Samples/second: {samples_per_sec:.2f}")
print(f"  Final loss: {result.training_loss:.4f}")
print(f"  Peak memory: {mem_peak:.2f} GB")
print(f"\n💡 Hypothesis: Lower LoRA rank should show better FP8 speedup")
print(f"   because less compute is spent on BF16 LoRA adapters.")
print("=" * 80)
