#!/usr/bin/env python3
"""
FP8 + LoRA + SFTTrainer (matches notebook pattern exactly)

Run with: accelerate launch --mixed_precision fp8 examples/fp8_lora_sfttrainer.py
Compare with: python examples/bf16_typical_lora.py
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("=" * 80)
print("FP8 + LoRA + SFTTrainer (notebook pattern)")
print("=" * 80)

# Load model - same as notebook
print("\n[1/4] Loading model...")
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# Add LoRA adapters - same as notebook
print("\n[2/4] Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset - same as notebook
print("\n[3/4] Preparing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        texts.append(alpaca_prompt.format(inst, inp, out) + EOS_TOKEN)
    return {"text": texts}

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Train with SFTTrainer - identical to notebook
print("\n[4/4] Starting training...")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        bf16=True,  # Accelerate handles FP8 conversion
    ),
)

trainer_stats = trainer.train()

print("\n" + "=" * 80)
print("FP8 + LoRA + SFTTrainer Complete!")
print("=" * 80)
print(f"Time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")
