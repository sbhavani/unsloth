"""
FP8 Test using load_in_fp8=True

Test Unsloth's native FP8 loading method (load_in_fp8=True)
instead of Accelerate's setup_fp8_mixed_precision_training().

This is the method used in unslothai/notebooks FP8 examples.
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
print("FP8 Test - Using load_in_fp8=True (Unsloth Native Method)")
print("=" * 80)

# Configuration
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 4
NUM_STEPS = 100
LORA_RANK = 16  # Moderate rank

print(f"\nConfiguration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Method: load_in_fp8=True (Unsloth native)")
print(f"  LoRA rank: {LORA_RANK}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")

# Load model with load_in_fp8=True (Unsloth's native FP8)
print(f"\n[1/4] Loading model with load_in_fp8=True...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,
    load_in_fp8=True,  # Unsloth's native FP8!
    fast_inference=True,
    max_lora_rank=LORA_RANK,
)

# Add LoRA
print(f"\n[2/4] Adding LoRA (rank {LORA_RANK})...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK * 2,  # As per FP8 notebook
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Dataset
print("\n[3/4] Preparing dataset...")
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
print("\n[4/4] Training with load_in_fp8=True...")
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
        output_dir="outputs/test_load_in_fp8",
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
print("✅ load_in_fp8=True TEST COMPLETED")
print("=" * 80)
print(f"\nPerformance:")
print(f"  Total time: {total_time:.2f}s")
print(f"  Steps/second: {steps_per_sec:.4f}")
print(f"  Samples/second: {samples_per_sec:.2f}")
print(f"  Final loss: {result.training_loss:.4f}")
print(f"  Peak memory: {mem_peak:.2f} GB")
print(f"\n💡 This uses Unsloth's native FP8 method (load_in_fp8=True)")
print(f"   vs Accelerate's setup_fp8_mixed_precision_training().")
print(f"   Compare this to see which method gives better speedup!")
print("=" * 80)
