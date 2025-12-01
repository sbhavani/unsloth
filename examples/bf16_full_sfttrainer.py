#!/usr/bin/env python3
"""
BF16 Full Fine-tuning with SFTTrainer (baseline comparison)
All parameters trainable (no LoRA)
NOTE: Uses same setup as FP8 version (raw HF, no Unsloth optimizations) for fair comparison
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("=" * 80)
print("BF16 Full Fine-tuning + SFTTrainer (Llama-3.2-3B)")
print("=" * 80)

# Load model directly with HF (same as FP8 version for fair comparison)
print("\n[1/3] Loading model...")
max_seq_length = 512
model_name = "unsloth/Llama-3.2-3B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# Enable standard gradient checkpointing
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Verify all params trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Model loaded: {total:,} params")
print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

# Prepare dataset
print("\n[2/3] Preparing dataset...")
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

# Train with SFTTrainer
print("\n[3/3] Training...")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=8,  # Match FP8 for fair comparison
        gradient_accumulation_steps=2,  # Effective batch = 16
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-5,  # Lower LR for full FT
        logging_steps=10,
        optim="adamw_8bit",  # 8-bit optimizer to save memory
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        bf16=True,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    ),
)

trainer_stats = trainer.train()

print("\n" + "=" * 80)
print("BF16 Full Fine-tuning Complete!")
print("=" * 80)
print(f"Time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")
