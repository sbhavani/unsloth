"""
Quick FP8 Training Test

This is a minimal script to verify FP8 training works correctly.
Runs a few training steps to validate the setup.

Usage:
    python test_fp8_quick.py
"""

import os
# Disable multiprocessing to avoid pickling issues with unsloth
os.environ["HF_DATASETS_NUM_PROC"] = "1"

# Import unsloth FIRST before other libraries
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training, check_fp8_training_support

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

print("=" * 80)
print("Quick FP8 Training Test")
print("=" * 80)

# Check FP8 support
if not check_fp8_training_support():
    print("❌ ERROR: FP8 training not supported")
    print("Requirements:")
    print("  - CUDA GPU (H100/H200 optimal, A100+ supported)")
    print("  - pip install transformer-engine")
    print("  - pip install accelerate>=0.26.0")
    exit(1)

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ CUDA: {torch.version.cuda}")

# Enable FP8
print("\n[1/5] Enabling FP8 mixed precision training...")
setup_fp8_mixed_precision_training(backend="te")

# Load model
print("\n[2/5] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=512,  # Shorter for quick test
    dtype=None,
    load_in_4bit=False,
)

# Add LoRA
print("\n[3/5] Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # Smaller rank for quick test
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=8,
    use_gradient_checkpointing="unsloth",
)

# Prepare dataset
print("\n[4/5] Preparing dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")

def formatting_prompts_func(examples):
    """Format examples - Unsloth expects a list of strings."""
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(text)
    return texts

tokenizer.pad_token = tokenizer.eos_token

# Train
print("\n[5/5] Running training (10 steps)...")
trainer = SFTTrainer(
    model=model,                          # PEFT model with LoRA
    processing_class=tokenizer,           # Tokenizer (renamed from 'tokenizer' in TRL >= 0.18)
    train_dataset=dataset,                # Raw dataset (not pre-tokenized)
    formatting_func=formatting_prompts_func,  # BATCHED: returns list of strings
    args=TrainingArguments(
        per_device_train_batch_size=2,
        max_steps=10,
        fp16=False,                       # Disabled for FP8
        bf16=False,                       # Disabled for FP8
        logging_steps=5,
        output_dir="outputs/test_fp8",
        report_to="none",
        dataloader_num_workers=0,         # Disable dataloader multiprocessing
    ),
)

result = trainer.train()

print("\n" + "=" * 80)
print("✅ FP8 Training Test PASSED!")
print("=" * 80)
print(f"Final loss: {result.training_loss:.4f}")
print(f"Memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print("\nFP8 training is working correctly!")
