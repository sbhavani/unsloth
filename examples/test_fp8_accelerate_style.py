#!/usr/bin/env python3
"""
Test FP8 exactly how HuggingFace Accelerate does it.
Reference: https://github.com/huggingface/accelerate/blob/main/benchmarks/fp8/transformer_engine/non_distributed.py
"""
import os
import sys
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
import time
import json

# Use larger batch for compute-bound workload (FP8 benefits)
BATCH_SIZE = 8
SEQ_LEN = 512

MODE = sys.argv[1] if len(sys.argv) > 1 else "both"

if MODE == "bf16":
    print("=" * 80)
    print("BF16 Baseline Test")
    print("=" * 80)
    
    from transformers import AutoModelForCausalLM
    from accelerate import Accelerator
    
    accelerator = Accelerator(mixed_precision="bf16")
    print(f"  mixed_precision: {accelerator.mixed_precision}")
    
    # Don't manually .to("cuda") - let accelerator handle it
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model, optimizer = accelerator.prepare(model, optimizer)
    
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    print(f"  Using batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}")
    input_ids = torch.randint(0, 32000, (BATCH_SIZE, SEQ_LEN), device=accelerator.device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    # Warmup with training (forward + backward)
    print("  Warmup...")
    model.train()
    for _ in range(3):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        accelerator.backward(outputs.loss)
        optimizer.zero_grad()
    torch.cuda.synchronize()
    
    # Benchmark actual training
    print("  Benchmarking...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        accelerator.backward(outputs.loss)
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\nBF16 Results (10 train steps):")
    print(f"  Time: {elapsed*1000:.1f} ms")
    print(f"  Memory: {mem:.2f} GB")
    
    with open("/tmp/bf16_results.json", "w") as f:
        json.dump({"time": elapsed, "memory": mem}, f)

elif MODE == "fp8":
    print("=" * 80)
    print("FP8 Test (Accelerate Style - with FP8RecipeKwargs)")
    print("=" * 80)
    
    from transformers import AutoModelForCausalLM
    from accelerate import Accelerator
    from accelerate.utils import FP8RecipeKwargs
    import transformer_engine.pytorch as te
    
    # Use FP8RecipeKwargs like HF benchmark does
    FP8_RECIPE_KWARGS = {"fp8_format": "HYBRID", "amax_history_len": 32, "amax_compute_algo": "max"}
    kwargs_handlers = [FP8RecipeKwargs(backend="TE", **FP8_RECIPE_KWARGS)]
    
    accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=kwargs_handlers)
    print(f"  mixed_precision: {accelerator.mixed_precision}")
    print(f"  fp8_backend: {accelerator.fp8_backend}")
    
    # Don't manually .to("cuda") - let accelerator handle it
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    
    nn_before = sum(1 for m in model.modules() if type(m).__name__ == 'Linear' and 'torch.nn' in type(m).__module__)
    te_before = sum(1 for m in model.modules() if isinstance(m, te.Linear))
    print(f"  Before prepare: nn.Linear={nn_before}, te.Linear={te_before}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model, optimizer = accelerator.prepare(model, optimizer)
    
    nn_after = sum(1 for m in model.modules() if type(m).__name__ == 'Linear' and 'torch.nn' in type(m).__module__)
    te_after = sum(1 for m in model.modules() if isinstance(m, te.Linear))
    print(f"  After prepare: nn.Linear={nn_after}, te.Linear={te_after}")
    
    if te_after == 0:
        print("\n⚠️  WARNING: No te.Linear layers! Accelerate didn't convert.")
        print("  Checking why...")
        # Check if model has layers with dimensions not divisible by 16
        for name, m in model.named_modules():
            if type(m).__name__ == 'Linear' and 'torch.nn' in type(m).__module__:
                if m.weight.shape[0] % 16 != 0 or m.weight.shape[1] % 16 != 0:
                    print(f"    {name}: shape {m.weight.shape} - NOT divisible by 16!")
    else:
        print(f"  ✅ {te_after} layers converted to te.Linear")
    
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    print(f"  Using batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}")
    input_ids = torch.randint(0, 32000, (BATCH_SIZE, SEQ_LEN), device=accelerator.device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    # Warmup with training (forward + backward)
    print("  Warmup...")
    model.train()
    for _ in range(3):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        accelerator.backward(outputs.loss)
        optimizer.zero_grad()
    torch.cuda.synchronize()
    
    # Benchmark actual training
    print("  Benchmarking...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        accelerator.backward(outputs.loss)
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\nFP8 Results (10 train steps):")
    print(f"  Time: {elapsed*1000:.1f} ms")
    print(f"  Memory: {mem:.2f} GB")
    
    if os.path.exists("/tmp/bf16_results.json"):
        with open("/tmp/bf16_results.json") as f:
            bf16 = json.load(f)
        
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        speedup = bf16["time"] / elapsed
        print(f"BF16: {bf16['time']*1000:.1f} ms, {bf16['memory']:.2f} GB")
        print(f"FP8:  {elapsed*1000:.1f} ms, {mem:.2f} GB")
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 1.1:
            print("\n✅ FP8 IS faster!")
        elif speedup < 0.9:
            print("\n⚠️  FP8 is slower")
        else:
            print("\n⚠️  No significant difference")

else:
    import subprocess
    print("Running BF16 baseline...")
    subprocess.run([sys.executable, __file__, "bf16"])
    print("\n" + "=" * 80 + "\n")
    print("Running FP8 test...")
    subprocess.run([sys.executable, __file__, "fp8"])
