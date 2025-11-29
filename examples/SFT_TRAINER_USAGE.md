# SFTTrainer API Usage Guide (TRL >= 0.18)

This guide explains the correct usage of `SFTTrainer` with Unsloth for FP8 training.

## Key API Changes in TRL >= 0.18

The TRL library has undergone significant API changes. Here's what changed:

### ❌ REMOVED Parameters
- `tokenizer` → Use `processing_class` instead
- `dataset_text_field` → Removed (use `formatting_func`)
- `max_seq_length` → Removed (set in model loading)
- `dataset_num_proc` → Not a parameter
- `dataset_kwargs` → Not a parameter

### ✅ CORRECT Usage

```python
import os
# CRITICAL: Set before any imports to disable multiprocessing
os.environ["HF_DATASETS_NUM_PROC"] = "1"

# Import unsloth FIRST
from unsloth import FastLanguageModel, setup_fp8_mixed_precision_training

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# 1. Enable FP8
setup_fp8_mixed_precision_training(backend="te")

# 2. Load model with max_seq_length
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,  # Set here, not in SFTTrainer
    dtype=None,
    load_in_4bit=False,
)

# 3. Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
)

# 4. Load dataset (raw, not pre-tokenized)
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

# 5. Define formatting function (BATCHED for Unsloth)
def formatting_func(examples):
    """
    Format examples into text.
    
    IMPORTANT: Unsloth's SFTTrainer expects BATCHED formatting_func
    that returns a LIST of strings, not a single string.
    """
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(text)
    return texts

# 6. Set tokenizer padding
tokenizer.pad_token = tokenizer.eos_token

# 7. Create trainer
trainer = SFTTrainer(
    model=model,                      # PEFT model with LoRA
    processing_class=tokenizer,       # NOT 'tokenizer'!
    train_dataset=dataset,            # Raw dataset
    formatting_func=formatting_func,  # On-the-fly formatting
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=False,                   # Must be False for FP8
        bf16=False,                   # Must be False for FP8
        logging_steps=10,
        output_dir="outputs",
        report_to="none",
        dataloader_num_workers=0,     # CRITICAL: Disable multiprocessing
    ),
)

# 8. Train
trainer.train()
```

## Critical Points

### 1. Environment Variable
```python
os.environ["HF_DATASETS_NUM_PROC"] = "1"
```
- **MUST** be set before any HuggingFace imports
- Prevents multiprocessing pickling errors with Unsloth

### 2. Import Order
```python
# 1. Set env variable
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

# 2. Import unsloth FIRST
from unsloth import ...

# 3. Import other libraries
import torch
from transformers import ...
from trl import ...
```
- Unsloth patches transformers/trl/peft
- Must be imported before those libraries

### 3. SFTTrainer Parameters

**Required:**
- `model`: Your PEFT model
- `processing_class`: Tokenizer (renamed from `tokenizer`)
- `train_dataset`: Raw dataset
- `args`: TrainingArguments instance

**Optional:**
- `formatting_func`: Function to format examples (recommended)
- `eval_dataset`: Evaluation dataset
- `packing`: Whether to pack multiple examples (default: False)

### 4. Formatting Function (BATCHED for Unsloth)

```python
def formatting_func(examples):
    """
    Format BATCHED examples.
    
    IMPORTANT: Unsloth's SFTTrainer expects a BATCHED formatting function
    that processes multiple examples at once.
    
    Args:
        examples: Dict with keys from your dataset, each containing lists
        
    Returns:
        list[str]: List of formatted text strings
    """
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(text)
    return texts
```

**Important:**
- Takes a **dict of lists** (batched examples)
- Returns a **list of strings**
- Processes multiple examples at once
- This is different from standard TRL SFTTrainer!

### 5. Training Arguments for FP8

```python
TrainingArguments(
    per_device_train_batch_size=2,     # Adjust for your GPU
    gradient_accumulation_steps=4,      # Effective batch size = 2 * 4 = 8
    fp16=False,                         # Must disable for FP8
    bf16=False,                         # Must disable for FP8
    dataloader_num_workers=0,           # Must be 0 (no multiprocessing)
    # ... other args
)
```

## Troubleshooting

### Error: `ValueError: Unsloth: The formatting_func should return a list of processed strings`
**Solution:** Your `formatting_func` must be **batched** and return a **list of strings**, not a single string:
```python
# ❌ WRONG (returns single string)
def formatting_func(example):
    return f"Instruction: {example['instruction']}"

# ✅ CORRECT (returns list of strings)
def formatting_func(examples):
    return [f"Instruction: {inst}" for inst in examples['instruction']]
```

### Error: `TypeError: cannot pickle 'ConfigModuleInstance' object`
**Solution:** Set `os.environ["HF_DATASETS_NUM_PROC"] = "1"` before imports

### Error: `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'`
**Solution:** Use `processing_class=tokenizer` instead of `tokenizer=tokenizer`

### Error: `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'max_seq_length'`
**Solution:** Set `max_seq_length` in `FastLanguageModel.from_pretrained()`, not SFTTrainer

### Warning: "Unsloth should be imported before trl, transformers..."
**Solution:** Import unsloth before other libraries (see Import Order above)

## Checking Your TRL Version

Run this to see available parameters:
```bash
python examples/check_sft_trainer_api.py
```

## References

- TRL Documentation: https://huggingface.co/docs/trl
- Unsloth Documentation: https://docs.unsloth.ai
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
