"""
FP8 Large Model Test - H100 Optimized

Test FP8 training on a larger model (7B-8B) to demonstrate real benefits.
On larger models, FP8 shows significant speed improvements while memory
overhead becomes proportionally smaller.

Hardware: H100 (80GB recommended)
Expected: 1.3-1.5x speedup, better memory efficiency than small models
"""

import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training, check_fp8_training_support
import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
import time

print("=" * 80)
print("FP8 Large Model Test - H100")
print("=" * 80)

# Check FP8 support
if not check_fp8_training_support():
    print("❌ ERROR: FP8 training not supported")
    exit(1)

# Configuration
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"  # 8B model
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4  # Reasonable for 8B on H100
GRADIENT_ACCUMULATION = 2
NUM_STEPS = 20  # More steps for better benchmark
LORA_RANK = 32  # Larger rank for 8B model

print(f"\nConfiguration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"  Training steps: {NUM_STEPS}")
print(f"  LoRA rank: {LORA_RANK}")

# Enable FP8
print("\n[1/5] Setting up FP8 mixed precision...")
setup_fp8_mixed_precision_training(backend="te")

# Load model
print(f"\n[2/5] Loading model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,
)

# Add LoRA
print("\n[3/5] Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Prepare dataset
print("\n[4/5] Preparing dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]")

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

# Create data collator for FP8
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# Training
print("\n[5/5] Running FP8 training...")
print("=" * 80)

# Get initial memory
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated() / 1024**3

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
        bf16=True,  # FP8 works WITH BF16 autocast!
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        output_dir="outputs/test_fp8_large",
        report_to="none",
        dataloader_num_workers=0,
    ),
)

# Train and measure
start_time = time.time()
result = trainer.train()
end_time = time.time()

# Get final memory
mem_peak = torch.cuda.max_memory_allocated() / 1024**3

# Calculate metrics
total_time = end_time - start_time
steps_per_sec = NUM_STEPS / total_time
samples_per_sec = (NUM_STEPS * BATCH_SIZE * GRADIENT_ACCUMULATION) / total_time

print("\n" + "=" * 80)
print("✅ FP8 LARGE MODEL TRAINING COMPLETED!")
print("=" * 80)
print(f"\nPerformance Metrics:")
print(f"  Total time: {total_time:.2f}s")
print(f"  Steps/second: {steps_per_sec:.4f}")
print(f"  Samples/second: {samples_per_sec:.2f}")
print(f"  Final loss: {result.training_loss:.4f}")
print(f"\nMemory Usage:")
print(f"  Initial: {mem_before:.2f} GB")
print(f"  Peak: {mem_peak:.2f} GB")
print(f"  Used: {mem_peak - mem_before:.2f} GB")
print(f"\n💡 Tip: Compare this to BF16 by running benchmark_fp8_vs_bf16.py")
print("   with MODEL_NAME updated to this 8B model.")
print("=" * 80)
