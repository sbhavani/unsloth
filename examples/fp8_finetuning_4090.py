"""
FP8 Finetuning Example - Optimized for RTX 4090

This example is optimized for RTX 4090 (24GB VRAM) with:
- Increased batch size (FP8 memory savings)
- Larger model support
- Performance tuning for Ada Lovelace

RTX 4090 Notes:
- Compute capability 8.9 (✅ FP8 supported)
- Native FP8 support (Ada Lovelace architecture)
- Expected speedup: ~1.2-1.3x vs BF16
- Memory savings: ~40%
"""

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training, check_fp8_training_support

# Check GPU
if not check_fp8_training_support():
    print("❌ FP8 training not supported")
    exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {'.'.join(map(str, torch.cuda.get_device_capability()))}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Model configuration - can use larger models with FP8
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"  # Or try 3B with FP8!
MAX_SEQ_LENGTH = 2048

# Training configuration - optimized for 4090's 24GB
BATCH_SIZE = 8  # Increased from 4 (FP8 uses less memory)
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-4

def format_prompts(examples):
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        text = f"""### Instruction:
{instruction}

### Response:
{output}"""
        texts.append(text)
    return {"text": texts}


def main():
    print("\n" + "=" * 80)
    print("FP8 Finetuning on RTX 4090")
    print("=" * 80)

    # Enable FP8
    print("\n[1/5] Enabling FP8 mixed precision training...")
    setup_fp8_mixed_precision_training(
        backend="te",
        fp8_format="HYBRID",
        amax_history_len=32,
        amax_compute_algo="max",
    )

    # Load model
    print("\n[2/5] Loading model...")
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
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Prepare dataset
    print("\n[4/5] Loading dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
    dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names, num_proc=1)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Training
    print("\n[5/5] Starting training...")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Note: RTX 4090 has native FP8 support (Ada Lovelace)")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=5,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=False,
            bf16=False,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs/fp8_4090",
            report_to="none",
            dataloader_num_workers=0,  # Disable dataloader multiprocessing
        ),
        packing=False,
    )

    result = trainer.train()

    # Save
    print("\n[6/6] Saving model...")
    model.save_pretrained("outputs/fp8_4090")
    tokenizer.save_pretrained("outputs/fp8_4090")

    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print(f"Final loss: {result.training_loss:.4f}")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("=" * 80)

    print("\n💡 RTX 4090 Tips:")
    print("  - Native FP8 support (Ada Lovelace architecture)")
    print("  - FP8 speedup: ~1.2-1.3x vs BF16")
    print("  - Memory savings: ~40% (can use larger batches!)")
    print("  - Try larger models: Llama-3.2-3B fits with FP8")


if __name__ == "__main__":
    main()
