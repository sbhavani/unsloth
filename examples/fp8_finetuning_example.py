"""
FP8 Mixed Precision Finetuning Example

This example demonstrates how to use FP8 mixed precision training with Unsloth,
leveraging HuggingFace Accelerate and NVIDIA Transformer Engine for efficient
finetuning on Hopper GPUs (H100, H200, etc.).

FP8 training provides:
- Reduced memory usage compared to BF16/FP16
- Faster training on supported hardware
- Minimal accuracy degradation with proper scaling

Requirements:
- NVIDIA GPU with compute capability 8.0+ (preferably 9.0+ for Hopper)
- CUDA 11.8+
- pip install accelerate>=0.26.0
- pip install transformer-engine

Usage:
    # Single GPU
    python fp8_finetuning_example.py

    # Multi-GPU with DDP
    accelerate launch fp8_finetuning_example.py

    # Multi-GPU with custom config
    accelerate config  # Configure your setup
    accelerate launch fp8_finetuning_example.py
"""

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.fp8_training import (
    FP8TrainingConfig,
    get_fp8_accelerator,
    prepare_model_for_fp8_training,
    check_fp8_support,
)

# Check FP8 support
if not check_fp8_support():
    print("FP8 training is not supported on this system. Exiting...")
    exit(1)

# Model configuration
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # Auto-detect (will use bfloat16 if available)

# FP8 configuration
FP8_CONFIG = FP8TrainingConfig(
    fp8_format="HYBRID",      # HYBRID uses E4M3 for forward, E5M2 for backward
    amax_history_len=32,      # History length for scaling factor computation
    amax_compute_algo="max",  # Use max from history for stability
    fp8_dpa=False,            # FP8 dot product attention (experimental)
)

# Training configuration
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-4
WARMUP_STEPS = 5
LOGGING_STEPS = 10
OUTPUT_DIR = "outputs/fp8_finetuned_model"


def format_prompts(examples):
    """Format dataset examples for instruction finetuning."""
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        # Simple instruction format - adjust based on your model's chat template
        text = f"""### Instruction:
{instruction}

### Response:
{output}"""
        texts.append(text)
    return {"text": texts}


def main():
    print("=" * 80)
    print("Unsloth FP8 Mixed Precision Finetuning Example")
    print("=" * 80)

    # =========================================================================
    # Step 1: Load model in BF16/FP16 (we'll convert to FP8 for training)
    # =========================================================================
    print("\n[1/6] Loading model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=False,  # FP8 training doesn't use 4bit quantization
        load_in_8bit=False,
        load_in_16bit=False,
        # For FP8, we load the model normally and convert it to FP8 layers
    )

    # Add LoRA adapters (optional - you can also do full finetuning with FP8)
    print("\n[2/6] Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                    # LoRA rank
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # =========================================================================
    # Step 2: Setup FP8 Accelerator
    # =========================================================================
    print("\n[3/6] Setting up FP8 accelerator...")

    # Note: When using SFTTrainer or other HF Trainers, they create their own
    # Accelerator instance. For manual training loops, you would use:
    # accelerator = get_fp8_accelerator(fp8_config=FP8_CONFIG)
    # model, optimizer = accelerator.prepare(model, optimizer)

    # For this example with SFTTrainer, we'll use the environment variable
    # approach or pass it through training arguments
    import os
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"

    # =========================================================================
    # Step 3: Prepare dataset
    # =========================================================================
    print("\n[4/6] Loading and preparing dataset...")

    # Load a sample dataset (using Alpaca for demonstration)
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")  # Small subset for demo

    # Format the dataset
    dataset = dataset.map(
        format_prompts,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Important: For FP8, pad sequences to multiples of 16 for optimal performance
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # =========================================================================
    # Step 4: Setup Trainer with FP8 support
    # =========================================================================
    print("\n[5/6] Setting up trainer...")

    # For FP8 with Accelerate, you can either:
    # Option A: Use accelerate launch with a config file
    # Option B: Create custom training loop with FP8Trainer
    # Option C: Use standard HF Trainer with FP8 environment settings (shown here)

    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=False,  # FP8 handles precision
        logging_steps=LOGGING_STEPS,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",  # Change to "wandb" if you want to log to W&B
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=False,  # Can enable for efficiency
    )

    # =========================================================================
    # Step 5: Train with FP8
    # =========================================================================
    print("\n[6/6] Starting FP8 training...\n")
    print(f"FP8 Format: {FP8_CONFIG.fp8_format}")
    print(f"Amax History Length: {FP8_CONFIG.amax_history_len}")
    print(f"Amax Compute Algorithm: {FP8_CONFIG.amax_compute_algo}\n")

    # Train the model
    trainer_stats = trainer.train()

    # =========================================================================
    # Step 6: Save the model
    # =========================================================================
    print("\nSaving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 80)
    print("\nTraining Statistics:")
    print(trainer_stats)


def manual_training_loop_example():
    """
    Alternative example using manual training loop with FP8Trainer.

    This gives you more control over the training process and shows how to use
    the FP8Trainer class directly.
    """
    from unsloth.fp8_training import FP8Trainer
    from torch.utils.data import DataLoader

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=False,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Setup FP8 trainer
    fp8_trainer = FP8Trainer(
        model=model,
        optimizer=optimizer,
        fp8_config=FP8_CONFIG,
        convert_model_to_fp8=True,  # Convert model layers to TE FP8 layers
    )

    # Prepare dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")
    dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names)

    def collate_fn(examples):
        # Tokenize and pad to multiple of 16 for FP8
        texts = [ex["text"] for ex in examples]
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Training loop
    print("Starting manual FP8 training loop...")
    for epoch in range(NUM_TRAIN_EPOCHS):
        for step, batch in enumerate(dataloader):
            loss = fp8_trainer.training_step(batch)

            if step % LOGGING_STEPS == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    # Save model
    fp8_trainer.save_model(f"{OUTPUT_DIR}/manual_loop_model.pt")
    print(f"Model saved to {OUTPUT_DIR}/manual_loop_model.pt")


if __name__ == "__main__":
    # Run the main training example
    main()

    # Uncomment to run the manual training loop example instead
    # manual_training_loop_example()
