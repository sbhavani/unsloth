#!/usr/bin/env python3
"""
Test FP8 exactly how HuggingFace Accelerate does it.
Run BF16 baseline first, then FP8 in separate process.
"""
import os
import sys
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
import time
import json

MODE = sys.argv[1] if len(sys.argv) > 1 else "both"

if MODE == "bf16":
    print("=" * 80)
    print("BF16 Baseline Test")
    print("=" * 80)
    
    from transformers import AutoModelForCausalLM
    from accelerate import Accelerator
    
    accelerator = Accelerator(mixed_precision="bf16")
    print(f"  mixed_precision: {accelerator.mixed_precision}")
    
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to("cuda")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model, optimizer = accelerator.prepare(model, optimizer)
    
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    input_ids = torch.randint(0, 32000, (4, 256), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    
    # Warmup
    model.train()
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\nBF16 Results:")
    print(f"  Time: {elapsed*1000:.1f} ms")
    print(f"  Memory: {mem:.2f} GB")
    
    # Save results
    with open("/tmp/bf16_results.json", "w") as f:
        json.dump({"time": elapsed, "memory": mem}, f)

elif MODE == "fp8":
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"
    
    print("=" * 80)
    print("FP8 Test (Accelerate Style)")
    print("=" * 80)
    
    from transformers import AutoModelForCausalLM
    from accelerate import Accelerator
    import transformer_engine.pytorch as te
    
    accelerator = Accelerator(mixed_precision="fp8")
    print(f"  mixed_precision: {accelerator.mixed_precision}")
    
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to("cuda")
    
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
        print("  This might be a version issue or config problem.")
    
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    input_ids = torch.randint(0, 32000, (4, 256), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    
    # Warmup
    model.train()
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\nFP8 Results:")
    print(f"  Time: {elapsed*1000:.1f} ms")
    print(f"  Memory: {mem:.2f} GB")
    
    # Compare with BF16 if available
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

else:
    # Run both in sequence
    import subprocess
    print("Running BF16 baseline...")
    subprocess.run([sys.executable, __file__, "bf16"])
    print("\n" + "=" * 80 + "\n")
    print("Running FP8 test...")
    subprocess.run([sys.executable, __file__, "fp8"])
