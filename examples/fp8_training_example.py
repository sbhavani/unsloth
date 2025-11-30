#!/usr/bin/env python3
"""
FP8 Training with Unsloth - Following Official Pattern

This adapts the official Unsloth Llama Alpaca example for FP8 training.
FP8 provides ~1.3-1.6x speedup on H100 GPUs.

Key differences from standard Unsloth:
- FP8 works best with FULL fine-tuning (no LoRA) for maximum speedup
- Must use accelerator.prepare(model, optimizer) together
- Gradient checkpointing must be disabled (conflicts with FP8)
- Use packing=True with max_seq_length divisible by 8 (FP8 tensor requirement)
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("=" * 80)
print("FP8 Training with Unsloth (Official Pattern)")
print("=" * 80)

# ============================================================================
# Step 1: Get FP8-configured Accelerator
# ============================================================================
print("\n[1/5] Setting up FP8 Accelerator...")
accelerator = setup_fp8_mixed_precision_training()

# ============================================================================
# Step 2: Load model (following Unsloth pattern)
# ============================================================================
print("\n[2/5] Loading model...")
# FP8 requires max_seq_length divisible by 8
max_seq_length = 2048  # Divisible by 8 ✓
dtype = torch.bfloat16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=False,  # FP8 works best with full precision, not 4bit
)

# For FP8: Use for_training() with gradient_checkpointing=False
# FP8 conflicts with gradient checkpointing
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=False)

# ============================================================================
# Step 3: Prepare with FP8 Accelerator
# ============================================================================
print("\n[3/5] Preparing model with FP8...")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# CRITICAL: Prepare model and optimizer TOGETHER for FP8!
model, optimizer = accelerator.prepare(model, optimizer)

# Check TE conversion
import transformer_engine.pytorch as te
te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
print(f"  Converted {te_count} layers to te.Linear for FP8")

# ============================================================================
# Step 4: Prepare dataset (following Unsloth pattern)
# ============================================================================
print("\n[4/5] Preparing dataset...")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
dataset = dataset.map(formatting_prompts_func, batched=True)

# ============================================================================
# Step 5: Train with SFTTrainer (following Unsloth pattern)
# ============================================================================
print("\n[5/5] Starting FP8 training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=True,  # Pack to max_seq_length (divisible by 8 for FP8)
    args=SFTConfig(
        per_device_train_batch_size=4,  # Larger batch = more FP8 benefit
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,   # Must be False for FP8!
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-5,
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        bf16=True,  # FP8 works WITH BF16 autocast
    ),
)

# Train!
trainer_stats = trainer.train()

# ============================================================================
# Results
# ============================================================================
print("\n" + "=" * 80)
print("FP8 Training Complete!")
print("=" * 80)
print(f"Training time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")
