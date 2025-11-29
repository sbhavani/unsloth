"""
FP8 vs BF16 Performance Benchmark

This script benchmarks FP8 mixed precision training against BF16 to measure:
- Training speed (steps/second)
- Memory usage (VRAM)
- Training loss convergence

Based on the standard Unsloth Llama 3 + Alpaca finetuning example.

Hardware Requirements:
- For FP8: H100/H200 (optimal) or A100+ (supported)
- For BF16 baseline: Any CUDA GPU

Usage:
    # Run both benchmarks
    python benchmark_fp8_vs_bf16.py

    # Run only FP8
    python benchmark_fp8_vs_bf16.py --mode fp8

    # Run only BF16
    python benchmark_fp8_vs_bf16.py --mode bf16

References:
- Unsloth Llama 3 Tutorial: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama
- Alpaca Dataset: https://huggingface.co/datasets/yahma/alpaca-cleaned
"""

import torch
import time
import argparse
import json
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training, check_fp8_training_support
import gc


# Benchmark Configuration
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_STEPS = 100  # Number of training steps for benchmark
LEARNING_RATE = 2e-4
LORA_RANK = 16

# Dataset Configuration
DATASET_NAME = "yahma/alpaca-cleaned"
DATASET_SPLIT = "train[:1000]"  # Use subset for faster benchmarking


def get_memory_stats():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "max_allocated_gb": round(max_allocated, 2),
        }
    return {}


def format_alpaca(examples):
    """Format Alpaca dataset for instruction finetuning."""
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        # Format: instruction + input (if present) + output
        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}"""

        if input_text:
            text += f"\n\n### Input:\n{input_text}"

        text += f"\n\n### Response:\n{output}"
        texts.append(text)

    return {"text": texts}


def prepare_dataset():
    """Load and prepare the Alpaca dataset."""
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    # Format dataset
    dataset = dataset.map(
        format_alpaca,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1,  # Disable multiprocessing to avoid pickling issues
    )

    print(f"Dataset loaded: {len(dataset)} examples")
    return dataset


def run_benchmark(mode="bf16", num_steps=NUM_TRAIN_STEPS):
    """
    Run training benchmark for specified mode.

    Args:
        mode: "bf16" or "fp8"
        num_steps: Number of training steps to run

    Returns:
        dict: Benchmark results including speed, memory, and loss
    """
    print("\n" + "=" * 80)
    print(f"Running Benchmark: {mode.upper()}")
    print("=" * 80)

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Setup FP8 if needed
    if mode == "fp8":
        if not check_fp8_training_support():
            print("ERROR: FP8 training not supported on this system")
            return None

        print("Enabling FP8 mixed precision training...")
        setup_fp8_mixed_precision_training(
            backend="te",
            fp8_format="HYBRID",
            amax_history_len=32,
            amax_compute_algo="max",
        )

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=False,  # Don't use 4bit for fair comparison
        load_in_8bit=False,
    )

    # Add LoRA adapters
    print("Adding LoRA adapters...")
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
    dataset = prepare_dataset()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Setup training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,
        max_steps=num_steps,
        learning_rate=LEARNING_RATE,
        fp16=False if mode == "fp8" else False,
        bf16=False if mode == "fp8" else True,  # Use BF16 for baseline
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=f"outputs/benchmark_{mode}",
        report_to="none",
        save_strategy="no",  # Don't save during benchmark
        dataloader_num_workers=0,  # Disable dataloader multiprocessing
    )

    print(f"\nTraining Configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Training steps: {num_steps}")
    print(f"  Precision: {mode.upper()}")

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        packing=False,
    )

    # Get memory before training
    mem_before = get_memory_stats()
    print(f"\nMemory before training: {mem_before}")

    # Start training and measure time
    print(f"\nStarting training ({num_steps} steps)...")
    start_time = time.time()

    try:
        result = trainer.train()
        end_time = time.time()

        # Get memory after training
        mem_after = get_memory_stats()

        # Calculate metrics
        total_time = end_time - start_time
        steps_per_second = num_steps / total_time
        samples_per_second = (num_steps * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) / total_time

        results = {
            "mode": mode,
            "model": MODEL_NAME,
            "total_time_seconds": round(total_time, 2),
            "steps_per_second": round(steps_per_second, 4),
            "samples_per_second": round(samples_per_second, 2),
            "final_loss": round(result.training_loss, 4),
            "memory_before_gb": mem_before,
            "memory_after_gb": mem_after,
            "peak_memory_gb": mem_after.get("max_allocated_gb", 0),
            "config": {
                "batch_size": BATCH_SIZE,
                "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "lora_rank": LORA_RANK,
                "max_seq_length": MAX_SEQ_LENGTH,
                "num_steps": num_steps,
            }
        }

        print(f"\n{mode.upper()} Benchmark Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Steps/second: {steps_per_second:.4f}")
        print(f"  Samples/second: {samples_per_second:.2f}")
        print(f"  Final loss: {result.training_loss:.4f}")
        print(f"  Peak memory: {mem_after.get('max_allocated_gb', 0):.2f} GB")

        return results

    except Exception as e:
        print(f"\nERROR during {mode} training: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Cleanup
        del model, tokenizer, trainer
        gc.collect()
        torch.cuda.empty_cache()


def compare_results(bf16_results, fp8_results):
    """Compare FP8 vs BF16 results and print summary."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: FP8 vs BF16")
    print("=" * 80)

    if bf16_results is None or fp8_results is None:
        print("ERROR: Cannot compare - one or both benchmarks failed")
        return

    # Speed comparison
    speedup = fp8_results["steps_per_second"] / bf16_results["steps_per_second"]
    throughput_gain = fp8_results["samples_per_second"] / bf16_results["samples_per_second"]

    # Memory comparison
    bf16_mem = bf16_results["peak_memory_gb"]
    fp8_mem = fp8_results["peak_memory_gb"]
    mem_reduction = (1 - fp8_mem / bf16_mem) * 100 if bf16_mem > 0 else 0

    # Loss comparison
    loss_diff = abs(fp8_results["final_loss"] - bf16_results["final_loss"])
    loss_diff_pct = (loss_diff / bf16_results["final_loss"]) * 100 if bf16_results["final_loss"] > 0 else 0

    print(f"\n📊 SPEED")
    print(f"  BF16:  {bf16_results['steps_per_second']:.4f} steps/sec")
    print(f"  FP8:   {fp8_results['steps_per_second']:.4f} steps/sec")
    print(f"  Speedup: {speedup:.2f}x {'✅' if speedup > 1.0 else '⚠️'}")

    print(f"\n📈 THROUGHPUT")
    print(f"  BF16:  {bf16_results['samples_per_second']:.2f} samples/sec")
    print(f"  FP8:   {fp8_results['samples_per_second']:.2f} samples/sec")
    print(f"  Gain: {throughput_gain:.2f}x {'✅' if throughput_gain > 1.0 else '⚠️'}")

    print(f"\n💾 MEMORY")
    print(f"  BF16:  {bf16_mem:.2f} GB")
    print(f"  FP8:   {fp8_mem:.2f} GB")
    print(f"  Reduction: {mem_reduction:.1f}% {'✅' if mem_reduction > 0 else '⚠️'}")

    print(f"\n📉 LOSS")
    print(f"  BF16:  {bf16_results['final_loss']:.4f}")
    print(f"  FP8:   {fp8_results['final_loss']:.4f}")
    print(f"  Difference: {loss_diff:.4f} ({loss_diff_pct:.2f}%)")

    print(f"\n🎯 SUMMARY")
    print(f"  FP8 is {speedup:.2f}x faster than BF16")
    print(f"  FP8 uses {mem_reduction:.1f}% less memory")
    print(f"  FP8 maintains {100 - loss_diff_pct:.1f}% of BF16 loss accuracy")

    # Save results
    results = {
        "bf16": bf16_results,
        "fp8": fp8_results,
        "comparison": {
            "speedup": round(speedup, 2),
            "throughput_gain": round(throughput_gain, 2),
            "memory_reduction_pct": round(mem_reduction, 1),
            "loss_difference": round(loss_diff, 4),
            "loss_difference_pct": round(loss_diff_pct, 2),
        }
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: benchmark_results.json")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark FP8 vs BF16 training")
    parser.add_argument(
        "--mode",
        choices=["both", "bf16", "fp8"],
        default="both",
        help="Which benchmark to run (default: both)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=NUM_TRAIN_STEPS,
        help=f"Number of training steps (default: {NUM_TRAIN_STEPS})"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FP8 vs BF16 Performance Benchmark")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Training steps: {args.steps}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

    bf16_results = None
    fp8_results = None

    if args.mode in ["both", "bf16"]:
        bf16_results = run_benchmark("bf16", args.steps)

    if args.mode in ["both", "fp8"]:
        fp8_results = run_benchmark("fp8", args.steps)

    if args.mode == "both" and bf16_results and fp8_results:
        compare_results(bf16_results, fp8_results)

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
