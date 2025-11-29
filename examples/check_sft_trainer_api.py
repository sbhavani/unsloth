"""
Check SFTTrainer API

This script inspects the SFTTrainer API to show what parameters are available
in the installed version of TRL.
"""

import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"

from trl import SFTTrainer
import inspect

print("=" * 80)
print("SFTTrainer API Inspection")
print("=" * 80)

# Get the signature
sig = inspect.signature(SFTTrainer.__init__)
print("\nSFTTrainer.__init__ parameters:")
print("-" * 80)

for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue
    
    default = param.default
    if default == inspect.Parameter.empty:
        default_str = "REQUIRED"
    else:
        default_str = repr(default)
    
    annotation = param.annotation
    if annotation == inspect.Parameter.empty:
        type_str = ""
    else:
        type_str = f": {annotation}"
    
    print(f"  {param_name}{type_str}")
    print(f"    Default: {default_str}")
    print()

print("=" * 80)
print("\nRecommended minimal usage for Unsloth + FP8:")
print("-" * 80)
print("""
from transformers import TrainingArguments
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,                          # Your PEFT model
    processing_class=tokenizer,           # Tokenizer (not 'tokenizer'!)
    train_dataset=dataset,                # Your dataset
    formatting_func=formatting_func,      # Function to format examples
    args=TrainingArguments(               # Training arguments
        per_device_train_batch_size=2,
        max_steps=100,
        fp16=False,
        bf16=False,
        logging_steps=10,
        output_dir="outputs",
        report_to="none",
        dataloader_num_workers=0,         # Important: no multiprocessing
    ),
)

# Where formatting_func is:
def formatting_func(example):
    '''Format a single example (not batched)'''
    instruction = example["instruction"]
    output = example["output"]
    text = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{output}"
    return text
""")
print("=" * 80)
