#!/usr/bin/env python3
"""
FP8 Full Fine-tuning with SFTTrainer
All parameters trainable (no LoRA)
Uses custom trainer to wrap forward/backward in te.fp8_autocast()
"""
import os
os.environ["HF_DATASETS_NUM_PROC"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "0"

import transformer_engine.pytorch as te
if not hasattr(te, 'fp8'):
    class _FakeFP8:
        @staticmethod
        def check_mxfp8_support():
            return False, "MXFP8 not available"
    te.fp8 = _FakeFP8()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("=" * 80)
print("FP8 Full Fine-tuning + SFTTrainer (Llama-3.2-3B)")
print("=" * 80)

# Check GPU and FP8 support
import torch
gpu_name = torch.cuda.get_device_name(0)
gpu_cap = torch.cuda.get_device_capability(0)
print(f"\nGPU: {gpu_name}")
print(f"Compute capability: {gpu_cap[0]}.{gpu_cap[1]}")
print(f"FP8 support: {'Yes (Hopper+)' if gpu_cap[0] >= 9 else 'Limited (Ada)'}")

# Load model
print("\n[1/3] Loading model...")
max_seq_length = 512
model_name = "unsloth/Llama-3.2-3B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Convert linear layers to TE (without accelerator.prepare)
print("\n[2/3] Converting to FP8 (TE layers)...")
def convert_to_te_linear(model):
    """Convert nn.Linear layers to te.Linear for FP8 support"""
    te_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not isinstance(module, te.Linear):
            # Get parent module
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            # Create TE linear with same config
            te_linear = te.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                params_dtype=module.weight.dtype,
            )
            # Copy weights
            te_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                te_linear.bias.data.copy_(module.bias.data)
            te_linear.to(module.weight.device)
            
            # Replace
            setattr(parent, parts[-1], te_linear)
            te_count += 1
    return te_count

te_count = convert_to_te_linear(model)
print(f"  Converted {te_count} layers to te.Linear")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Prepare dataset
print("\n[3/3] Preparing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        texts.append(alpaca_prompt.format(inst, inp, out) + EOS_TOKEN)
    return {"text": texts}

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Custom SFTTrainer with FP8 recipe that disables FP8 for weight gradient (wgrad)
# L40 (Ada Lovelace) has limited FP8 backward support vs H100 (Hopper)
class FP8SFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Recipe that keeps wgrad in higher precision
        self.fp8_recipe = te.recipe.DelayedScaling(
            fp8_format=te.recipe.Format.HYBRID,  # E4M3 forward, E5M2 backward
            amax_history_len=16,
            amax_compute_algo="max",
        )
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            
            # Scale loss for gradient accumulation
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            
            # Backward inside FP8 context
            loss.backward()
        
        return loss.detach()

# Train
print("\nStarting training...")
print("=" * 80)

trainer = FP8SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=8,  # Must be >= 8 for FP8 alignment
        gradient_accumulation_steps=2,  # Effective batch = 16
        gradient_checkpointing=False,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-5,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        bf16=True,
        dataset_text_field="text",
        max_length=max_seq_length,
        packing=False,
    ),
)

trainer_stats = trainer.train()

print("\n" + "=" * 80)
print("FP8 Full Fine-tuning Complete!")
print("=" * 80)
print(f"Time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")
