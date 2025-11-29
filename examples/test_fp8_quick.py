"""
Quick FP8 Training Test

This is a minimal script to verify FP8 training works correctly.
Runs a few training steps to validate the setup.

Usage:
    python test_fp8_quick.py
"""

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training, check_fp8_training_support

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

def format_prompts(examples):
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names)
tokenizer.pad_token = tokenizer.eos_token

# Train
print("\n[5/5] Running training (10 steps)...")
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    max_seq_length=512,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        max_steps=10,
        fp16=False,
        bf16=False,
        logging_steps=5,
        output_dir="outputs/test_fp8",
        report_to="none",
    ),
)

result = trainer.train()

print("\n" + "=" * 80)
print("✅ FP8 Training Test PASSED!")
print("=" * 80)
print(f"Final loss: {result.training_loss:.4f}")
print(f"Memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print("\nFP8 training is working correctly!")
