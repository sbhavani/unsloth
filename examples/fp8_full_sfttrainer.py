#!/usr/bin/env python3
"""
FP8 Full Fine-tuning with SFTTrainer
All parameters trainable (no LoRA)
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

# Workaround for accelerate/TE compatibility issue
import transformer_engine.pytorch as te
if not hasattr(te, 'fp8'):
    class _FakeFP8:
        @staticmethod
        def check_mxfp8_support():
            return False, "MXFP8 not available"
    te.fp8 = _FakeFP8()

import torch
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("=" * 80)
print("FP8 Full Fine-tuning + SFTTrainer (Llama-3.2-3B)")
print("=" * 80)

# Setup FP8 FIRST
print("\n[1/4] Setting up FP8...")
accelerator = setup_fp8_mixed_precision_training()

# Load model with full_finetuning=True
print("\n[2/4] Loading model...")
max_seq_length = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    full_finetuning=True,  # Enable full fine-tuning with gradient checkpointing
)

# No get_peft_model() - we're doing full fine-tuning
model = FastLanguageModel.for_training(model)
tokenizer.pad_token = tokenizer.eos_token

# Prepare model with FP8
print("\n[3/4] Preparing model with FP8...")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model, optimizer = accelerator.prepare(model, optimizer)

te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  Converted {te_count} layers to te.Linear")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Prepare dataset
print("\n[4/4] Preparing dataset...")
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
print("\nStarting training...")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=1,  # Minimal batch for full FT memory
        gradient_accumulation_steps=16,  # Effective batch = 16
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-5,  # Lower LR for full FT
        logging_steps=10,
        optim="adamw_torch",  # Standard optimizer (already prepared)
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        bf16=True,
    ),
)

trainer_stats = trainer.train()

print("\n" + "=" * 80)
print("FP8 Full Fine-tuning Complete!")
print("=" * 80)
print(f"Time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")
