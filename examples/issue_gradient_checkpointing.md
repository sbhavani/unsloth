# Gradient checkpointing broken for full fine-tuning with transformers 4.57+

## Environment
1. **Did you update?** Yes, `pip install --upgrade unsloth unsloth_zoo`
2. **Platform:** Local / Docker (NVIDIA NGC PyTorch container)
3. **GPUs:** 1x NVIDIA H100 80GB HBM3
4. **Notebook:** N/A - reproducing from script
5. **Versions:**
   - Unsloth: 2025.11.3
   - TRL: 0.22.2
   - Transformers: 4.57.2
   - PyTorch: 2.9.0a0+145a3a7bda.nv25.10
6. **Trainer:** Manual training loop (also affects SFTTrainer)

## Issue

When using `FastLanguageModel.for_training(model, use_gradient_checkpointing=True)` for **full fine-tuning** (not LoRA), training fails with:

```
AttributeError: 'LlamaDecoderLayer' object has no attribute '_gradient_checkpointing_func'
```

This happens because `for_training()` sets `gradient_checkpointing=True` on layers, but doesn't set `_gradient_checkpointing_func` which is required by `transformers.modeling_layers.GradientCheckpointingLayer.__call__()` in transformers 4.57+.

**Note:** This works fine with LoRA via `get_peft_model()` because that function uses `use_gradient_checkpointing="unsloth"` which sets up checkpointing differently.

## Minimal Reproduction

```python
import torch
from unsloth import FastLanguageModel

# Load model for FULL fine-tuning (no LoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# Enable gradient checkpointing for full fine-tuning
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
tokenizer.pad_token = tokenizer.eos_token

# Simple forward pass
model.train()
model = model.cuda()

input_ids = torch.randint(0, 32000, (2, 512), device="cuda")
attention_mask = torch.ones_like(input_ids)

with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )

# This fails with:
# AttributeError: 'LlamaDecoderLayer' object has no attribute '_gradient_checkpointing_func'
outputs.loss.backward()
```

## Expected Behavior

Gradient checkpointing should work for full fine-tuning, same as it does for LoRA training.

## Workaround

Currently, the only workaround is to disable gradient checkpointing:
```python
model = FastLanguageModel.for_training(model, use_gradient_checkpointing=False)
```

But this significantly increases memory usage (77GB vs ~20GB for an 8B model).

## Root Cause Analysis

In `transformers/modeling_layers.py` line 93:
```python
def __call__(self, *args, **kwargs):
    if self.gradient_checkpointing and self.training:
        return self._gradient_checkpointing_func(...)  # <-- This attribute doesn't exist
    return super().__call__(*args, **kwargs)
```

The `for_training()` function sets `gradient_checkpointing=True` but doesn't set `_gradient_checkpointing_func`. 

For LoRA, `get_peft_model()` uses `prepare_model_for_kbit_training()` which properly sets up gradient checkpointing via `model.gradient_checkpointing_enable()`.

## Suggested Fix

In `for_training()`, when `use_gradient_checkpointing=True`, also call:
```python
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

Or ensure `_gradient_checkpointing_func` is set on each `GradientCheckpointingLayer`.
