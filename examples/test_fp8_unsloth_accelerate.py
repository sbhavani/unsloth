#!/usr/bin/env python3
"""
Test if Unsloth patching breaks Accelerate's FP8 conversion.
Compare: HF model vs Unsloth-patched model with Accelerate FP8.
"""
import os
import sys
os.environ["HF_DATASETS_NUM_PROC"] = "1"

import torch
import time
import json

BATCH_SIZE = 8
SEQ_LEN = 512

MODE = sys.argv[1] if len(sys.argv) > 1 else "both"

if MODE == "hf":
    # ========================================================================
    # HuggingFace model + Accelerate FP8 (baseline - we know this works)
    # ========================================================================
    print("=" * 80)
    print("HuggingFace Model + Accelerate FP8")
    print("=" * 80)
    
    from transformers import AutoModelForCausalLM
    from accelerate import Accelerator
    from accelerate.utils import FP8RecipeKwargs
    import transformer_engine.pytorch as te
    
    FP8_RECIPE_KWARGS = {"fp8_format": "HYBRID", "amax_history_len": 32, "amax_compute_algo": "max"}
    kwargs_handlers = [FP8RecipeKwargs(backend="TE", **FP8_RECIPE_KWARGS)]
    accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=kwargs_handlers)
    
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    
    print(f"  Before prepare: nn.Linear={sum(1 for m in model.modules() if type(m).__name__ == 'Linear')}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model, optimizer = accelerator.prepare(model, optimizer)
    
    te_count = sum(1 for m in model.modules() if isinstance(m, te.Linear))
    print(f"  After prepare: te.Linear={te_count}")
    
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    input_ids = torch.randint(0, 32000, (BATCH_SIZE, SEQ_LEN), device=accelerator.device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    # Warmup
    model.train()
    for _ in range(3):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        accelerator.backward(outputs.loss)
        optimizer.zero_grad()
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(10):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        accelerator.backward(outputs.loss)
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\nHF + Accelerate FP8 Results:")
    print(f"  Time: {elapsed*1000:.1f} ms")
    print(f"  Memory: {mem:.2f} GB")
    print(f"  te.Linear layers: {te_count}")
    
    with open("/tmp/hf_fp8_results.json", "w") as f:
        json.dump({"time": elapsed, "memory": mem, "te_layers": te_count}, f)

elif MODE == "unsloth":
    # ========================================================================
    # Unsloth model + Accelerate FP8 (test if patching breaks conversion)
    # ========================================================================
    print("=" * 80)
    print("Unsloth Model + Accelerate FP8")
    print("=" * 80)
    
    # Import Unsloth FIRST (this patches things)
    from unsloth import FastLanguageModel
    
    from accelerate import Accelerator
    from accelerate.utils import FP8RecipeKwargs
    import transformer_engine.pytorch as te
    
    FP8_RECIPE_KWARGS = {"fp8_format": "HYBRID", "amax_history_len": 32, "amax_compute_algo": "max"}
    kwargs_handlers = [FP8RecipeKwargs(backend="TE", **FP8_RECIPE_KWARGS)]
    accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=kwargs_handlers)
    
    # Load with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length=SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    
    # Apply Unsloth's training optimizations
    model = FastLanguageModel.for_training(model)
    
    nn_before = sum(1 for m in model.modules() if type(m).__name__ == 'Linear' and 'torch.nn' in type(m).__module__)
    te_before = sum(1 for m in model.modules() if isinstance(m, te.Linear))
    print(f"  After Unsloth patching: nn.Linear={nn_before}, te.Linear={te_before}")
    
    # Check if Unsloth changed any module types
    module_types = {}
    for name, m in model.named_modules():
        t = type(m).__name__
        module_types[t] = module_types.get(t, 0) + 1
    print(f"  Module types: {dict(sorted(module_types.items(), key=lambda x: -x[1])[:10])}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    print(f"\n  Running accelerator.prepare()...")
    model, optimizer = accelerator.prepare(model, optimizer)
    
    nn_after = sum(1 for m in model.modules() if type(m).__name__ == 'Linear' and 'torch.nn' in type(m).__module__)
    te_after = sum(1 for m in model.modules() if isinstance(m, te.Linear))
    print(f"  After prepare: nn.Linear={nn_after}, te.Linear={te_after}")
    
    if te_after == 0:
        print("\n  ❌ NO te.Linear layers! Unsloth patching BROKE FP8 conversion!")
        # Check why
        for name, m in model.named_modules():
            if 'Linear' in type(m).__name__:
                shape = getattr(m, 'weight', None)
                if shape is not None:
                    shape = shape.shape
                print(f"    {name}: {type(m).__name__} shape={shape}")
                if len(list(model.named_modules())) > 20:
                    print("    ... (truncated)")
                    break
    else:
        print(f"  ✅ {te_after} layers converted to te.Linear")
    
    # Disable gradient checkpointing if it exists
    for module in model.modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = False
    
    input_ids = torch.randint(0, 32000, (BATCH_SIZE, SEQ_LEN), device=accelerator.device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    # Warmup
    print(f"\n  Warmup...")
    model.train()
    try:
        for _ in range(3):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            accelerator.backward(outputs.loss)
            optimizer.zero_grad()
        torch.cuda.synchronize()
        warmup_ok = True
    except Exception as e:
        print(f"  ❌ Warmup failed: {e}")
        warmup_ok = False
    
    if warmup_ok:
        # Benchmark
        print(f"  Benchmarking...")
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(10):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            accelerator.backward(outputs.loss)
            optimizer.zero_grad()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        mem = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"\nUnsloth + Accelerate FP8 Results:")
        print(f"  Time: {elapsed*1000:.1f} ms")
        print(f"  Memory: {mem:.2f} GB")
        print(f"  te.Linear layers: {te_after}")
        
        # Compare with HF
        if os.path.exists("/tmp/hf_fp8_results.json"):
            with open("/tmp/hf_fp8_results.json") as f:
                hf = json.load(f)
            
            print("\n" + "=" * 80)
            print("COMPARISON: HF vs Unsloth (both with Accelerate FP8)")
            print("=" * 80)
            speedup = hf["time"] / elapsed
            print(f"HF + FP8:      {hf['time']*1000:.1f} ms, {hf['memory']:.2f} GB, {hf['te_layers']} te.Linear")
            print(f"Unsloth + FP8: {elapsed*1000:.1f} ms, {mem:.2f} GB, {te_after} te.Linear")
            print(f"Unsloth vs HF: {speedup:.2f}x")
            
            if te_after == hf["te_layers"]:
                print("\n✅ Same number of te.Linear layers - conversion worked!")
            else:
                print(f"\n⚠️  Different te.Linear count: HF={hf['te_layers']}, Unsloth={te_after}")

else:
    import subprocess
    print("Testing HuggingFace + Accelerate FP8 first...")
    subprocess.run([sys.executable, __file__, "hf"])
    print("\n" + "=" * 80 + "\n")
    print("Testing Unsloth + Accelerate FP8...")
    subprocess.run([sys.executable, __file__, "unsloth"])
