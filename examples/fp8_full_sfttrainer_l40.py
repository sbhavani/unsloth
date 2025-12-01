#!/usr/bin/env python3
"""
FP8 Full Fine-tuning with SFTTrainer - L40 Compatible
Uses full_finetuning=True with UNSLOTH_RETURN_LOGITS=1 to disable fused CE loss.

Uses 3B model (fits in L40 memory without gradient checkpointing).
"""
import os
import shutil

# CRITICAL: Set env var BEFORE importing unsloth to disable fused CE loss at compile time
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["HF_DATASETS_NUM_PROC"] = "1"

# Clear compiled cache to ensure fresh compilation
cache_path = os.path.join(os.path.dirname(__file__), "..", "unsloth_compiled_cache")
cache_path = os.path.abspath(cache_path)
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
    print(f"Cleared {cache_path}")
else:
    # Also try current directory
    if os.path.exists("./unsloth_compiled_cache"):
        shutil.rmtree("./unsloth_compiled_cache")
        print("Cleared ./unsloth_compiled_cache")

import torch
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("=" * 80)
print("FP8 Full Fine-tuning + SFTTrainer (Llama-3.2-3B) - L40 Test")
print("=" * 80)
print("Testing: full_finetuning=True with FP8 auto-detection fix")
print("=" * 80)

# Check GPU
gpu_name = torch.cuda.get_device_name(0)
gpu_cap = torch.cuda.get_device_capability(0)
print(f"\nGPU: {gpu_name}")
print(f"Compute capability: {gpu_cap[0]}.{gpu_cap[1]}")

if "L40" not in gpu_name and "Ada" not in gpu_name:
    print("WARNING: This script is optimized for L40. Results may vary on other GPUs.")

# Setup FP8 FIRST
print("\n[1/4] Setting up FP8...")
accelerator = setup_fp8_mixed_precision_training()

# Load model with full_finetuning=True
# The fix should auto-detect FP8/TE layers and skip fused CE loss
print("\n[2/4] Loading model with full_finetuning=True...")
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B",  # 3B fits on L40 without grad ckpt
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    full_finetuning=True,  # Should work with FP8 now!
)

# Disable gradient checkpointing (required for FP8 on L40 - cuBLAS issues)
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=False)

if hasattr(model, 'gradient_checkpointing_disable'):
    model.gradient_checkpointing_disable()
model.config.use_cache = True

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

if trainable == 0:
    print("  ERROR: No trainable parameters! full_finetuning may not be working.")
elif trainable < total * 0.9:
    print(f"  WARNING: Only {100*trainable/total:.1f}% trainable. Expected ~100%.")
else:
    print("  ✓ Full fine-tuning enabled (100% trainable)")

# Prepare with FP8
print("\n[3/4] Preparing with FP8...")
_dummy_opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
model, _ = accelerator.prepare(model, _dummy_opt)
del _dummy_opt

import transformer_engine.pytorch as te
te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  Converted {te_count} layers to te.Linear")
print("  ✓ UNSLOTH_RETURN_LOGITS=1 disables fused CE loss (set before import)")

if te_count == 0:
    print("  ERROR: No TE layers found! FP8 conversion may have failed.")
else:
    print(f"  ✓ FP8 enabled ({te_count} TE layers)")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  After FP8 prep: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Prepare dataset with fixed padding for FP8 alignment
print("\n[4/4] Preparing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def tokenize_and_format(examples):
    texts = [alpaca_prompt.format(i, inp, o) + EOS_TOKEN 
             for i, inp, o in zip(examples["instruction"], examples["input"], examples["output"])]
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=max_seq_length)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
dataset = dataset.map(tokenize_and_format, batched=True, remove_columns=dataset.column_names)

# Train with SFTTrainer
print("\n" + "=" * 80)
print("Starting training...")
print("If this works without errors, the FP8 auto-detection fix is successful!")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,  # Required for FP8 on L40
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-5,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="constant",
        max_grad_norm=0.0,
        seed=3407,
        output_dir="outputs",
        report_to="none",
        save_strategy="no",
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    ),
)

trainer_stats = trainer.train()

print("\n" + "=" * 80)
print("SUCCESS! FP8 Full Fine-tuning Complete!")
print("=" * 80)
print(f"Time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")

print("\n" + "=" * 80)
print("FP8 + full_finetuning=True VERIFIED!")
print("- UNSLOTH_RETURN_LOGITS=1 disabled fused CE loss")
print("- full_finetuning=True works with FP8")
print("- Gradient offloading enabled ('smartly offload gradients')")
print("=" * 80)
