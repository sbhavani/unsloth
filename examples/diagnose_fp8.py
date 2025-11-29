"""
FP8 Diagnostic Script

Check what layers are actually using FP8 and identify bottlenecks.
"""

import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training, check_fp8_training_support
import torch

print("=" * 80)
print("FP8 DIAGNOSTIC")
print("=" * 80)

# Check FP8 support
if not check_fp8_training_support():
    print("❌ FP8 not supported")
    exit(1)

# Enable FP8
print("\n[1/3] Setting up FP8...")
setup_fp8_mixed_precision_training(backend="te")

# Load model
print("\n[2/3] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

# Add LoRA
print("\n[3/3] Adding LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
)

print("\n" + "=" * 80)
print("MODEL LAYER ANALYSIS")
print("=" * 80)

# Check for Transformer Engine layers
te_layers = 0
linear_layers = 0
lora_layers = 0

try:
    from transformer_engine.pytorch import Linear as TELinear
    has_te = True
except ImportError:
    has_te = False
    TELinear = None

for name, module in model.named_modules():
    if hasattr(module, '__class__'):
        class_name = module.__class__.__name__
        
        # Check for TE layers
        if has_te and isinstance(module, TELinear):
            te_layers += 1
            print(f"  ✅ FP8: {name} ({class_name})")
        
        # Check for regular Linear
        elif isinstance(module, torch.nn.Linear):
            linear_layers += 1
            if linear_layers <= 5:  # Show first 5
                print(f"  ⚠️  BF16: {name} ({class_name})")
        
        # Check for LoRA
        elif "lora" in class_name.lower():
            lora_layers += 1
            if lora_layers <= 3:  # Show first 3
                print(f"  📍 LoRA: {name} ({class_name})")

print(f"\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  Transformer Engine (FP8) layers: {te_layers}")
print(f"  Regular Linear (BF16) layers: {linear_layers}")
print(f"  LoRA layers: {lora_layers}")

if te_layers == 0:
    print(f"\n❌ NO FP8 LAYERS FOUND!")
    print(f"   This explains the poor speedup. FP8 is not being applied.")
    print(f"\n   Possible causes:")
    print(f"   1. Accelerate not wrapping layers with Transformer Engine")
    print(f"   2. Model needs to be prepared differently for FP8")
    print(f"   3. FP8 environment variables not being picked up")
elif te_layers < 10:
    print(f"\n⚠️  VERY FEW FP8 LAYERS!")
    print(f"   Only {te_layers} layers using FP8. Most layers still in BF16.")
    print(f"   This explains the minimal speedup.")
else:
    print(f"\n✅ Good FP8 coverage ({te_layers} layers)")

print(f"\n" + "=" * 80)
print("ENVIRONMENT VARIABLES")
print("=" * 80)
for key in ["ACCELERATE_MIXED_PRECISION", "ACCELERATE_FP8_BACKEND", "ACCELERATE_FP8_FORMAT"]:
    print(f"  {key}: {os.environ.get(key, 'NOT SET')}")

print(f"\n" + "=" * 80)
