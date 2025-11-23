# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FP8 Mixed Precision Training Support using Transformer Engine

This module provides utilities for FP8 mixed precision finetuning in Unsloth,
leveraging HuggingFace Accelerate's integration with NVIDIA Transformer Engine.

FP8 (8-bit floating point) training can provide:
- Reduced memory usage compared to BF16/FP16
- Faster training on supported hardware (H100, H200, etc.)
- Minimal accuracy degradation with proper scaling

Example usage:
    from unsloth import FastLanguageModel
    from unsloth.fp8_training import get_fp8_accelerator

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B",
        max_seq_length=2048,
        load_in_fp8=True,
    )

    # Get FP8-enabled accelerator
    accelerator = get_fp8_accelerator(
        fp8_format="HYBRID",
        amax_history_len=32,
        amax_compute_algo="max",
    )

    # Prepare model and optimizer
    model, optimizer = accelerator.prepare(model, optimizer)
"""

import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# Check if required packages are available
FP8_AVAILABLE = True
ACCELERATE_FP8_AVAILABLE = False
TE_AVAILABLE = False

try:
    from accelerate import Accelerator
    from accelerate.utils import FP8RecipeKwargs
    ACCELERATE_FP8_AVAILABLE = True
except ImportError:
    FP8_AVAILABLE = False
    print(
        "Unsloth FP8: `accelerate` is not installed or doesn't support FP8.\n"
        "Install with: pip install accelerate>=0.26.0"
    )

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
    from accelerate.utils.transformer_engine import convert_model
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    if ACCELERATE_FP8_AVAILABLE:
        print(
            "Unsloth FP8: `transformer_engine` is not installed.\n"
            "For FP8 training, install with: pip install transformer-engine\n"
            "Note: Transformer Engine requires CUDA 11.8+ and Hopper GPUs (H100, H200) for best performance."
        )
        FP8_AVAILABLE = False


@dataclass
class FP8TrainingConfig:
    """
    Configuration for FP8 mixed precision training using Transformer Engine.

    Args:
        fp8_format (str): FP8 format to use. Options:
            - "HYBRID": Uses E4M3 for forward pass and E5M2 for backward pass (recommended)
            - "E4M3": Uses E4M3 format for both forward and backward
            - "E5M2": Uses E5M2 format for both forward and backward
        amax_history_len (int): Length of history to use for computing amax (absolute maximum).
            Larger values provide more stable scaling but slower adaptation. Default: 32
        amax_compute_algo (str): Algorithm for computing amax. Options:
            - "max": Use maximum value from history (recommended)
            - "most_recent": Use most recent value
        margin (int): Margin for scaling factor computation. Default: 0
        fp8_dpa (bool): Enable FP8 dot product attention. Default: False
        backend (str): Backend to use for FP8. Default: "TE" (Transformer Engine)
        enabled (bool): Whether FP8 training is enabled. Default: True
    """
    fp8_format: str = "HYBRID"
    amax_history_len: int = 32
    amax_compute_algo: str = "max"
    margin: int = 0
    fp8_dpa: bool = False
    backend: str = "TE"
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for FP8RecipeKwargs."""
        return {
            "fp8_format": self.fp8_format,
            "amax_history_len": self.amax_history_len,
            "amax_compute_algo": self.amax_compute_algo,
            "margin": self.margin,
            "fp8_dpa": self.fp8_dpa,
        }

    def get_te_recipe(self) -> Optional["DelayedScaling"]:
        """Get Transformer Engine DelayedScaling recipe."""
        if not TE_AVAILABLE:
            return None

        # Convert format string to TE Format enum
        format_map = {
            "HYBRID": Format.HYBRID,
            "E4M3": Format.E4M3,
            "E5M2": Format.E5M2,
        }

        return DelayedScaling(
            fp8_format=format_map.get(self.fp8_format, Format.HYBRID),
            amax_history_len=self.amax_history_len,
            amax_compute_algo=self.amax_compute_algo,
            margin=self.margin,
        )


def check_fp8_support() -> bool:
    """
    Check if FP8 training is supported on the current system.

    Returns:
        bool: True if FP8 is supported, False otherwise
    """
    if not FP8_AVAILABLE:
        return False

    if not torch.cuda.is_available():
        print("Unsloth FP8: CUDA is not available. FP8 training requires NVIDIA GPUs.")
        return False

    # Check GPU compute capability (FP8 works best on Hopper H100/H200)
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        print(
            f"Unsloth FP8: GPU compute capability {major}.{minor} detected. "
            "FP8 training is optimized for Hopper GPUs (compute capability 9.0+). "
            "Performance may be limited on older GPUs."
        )

    return True


def get_fp8_accelerator(
    fp8_config: Optional[FP8TrainingConfig] = None,
    **accelerator_kwargs
) -> Optional["Accelerator"]:
    """
    Create an Accelerator instance configured for FP8 mixed precision training.

    Args:
        fp8_config (FP8TrainingConfig, optional): FP8 configuration. If None, uses defaults.
        **accelerator_kwargs: Additional arguments to pass to Accelerator

    Returns:
        Accelerator: Configured accelerator instance, or None if FP8 is not available

    Example:
        >>> accelerator = get_fp8_accelerator(
        ...     fp8_config=FP8TrainingConfig(fp8_format="HYBRID", amax_history_len=32)
        ... )
        >>> model, optimizer = accelerator.prepare(model, optimizer)
    """
    if not check_fp8_support():
        return None

    if fp8_config is None:
        fp8_config = FP8TrainingConfig()

    # Create FP8RecipeKwargs
    fp8_recipe_kwargs = FP8RecipeKwargs(
        backend=fp8_config.backend,
        **fp8_config.to_dict()
    )

    # Setup accelerator with FP8 mixed precision
    kwargs_handlers = [fp8_recipe_kwargs]
    if "kwargs_handlers" in accelerator_kwargs:
        kwargs_handlers.extend(accelerator_kwargs.pop("kwargs_handlers"))

    accelerator = Accelerator(
        mixed_precision="fp8",
        kwargs_handlers=kwargs_handlers,
        **accelerator_kwargs
    )

    return accelerator


def prepare_model_for_fp8_training(
    model: torch.nn.Module,
    fp8_config: Optional[FP8TrainingConfig] = None,
) -> torch.nn.Module:
    """
    Convert a model to use Transformer Engine FP8 layers.

    This function replaces compatible layers (Linear, LayerNorm, etc.) with their
    Transformer Engine FP8 equivalents. This is typically done before wrapping
    the model with accelerator.prepare().

    Args:
        model (torch.nn.Module): Model to convert
        fp8_config (FP8TrainingConfig, optional): FP8 configuration. If None, uses defaults.

    Returns:
        torch.nn.Module: Model with FP8 layers

    Example:
        >>> model = prepare_model_for_fp8_training(model)
        >>> # Now prepare with accelerator
        >>> model, optimizer = accelerator.prepare(model, optimizer)
    """
    if not TE_AVAILABLE:
        raise ImportError(
            "Transformer Engine is not available. Install with: pip install transformer-engine"
        )

    if fp8_config is None:
        fp8_config = FP8TrainingConfig()

    # Convert model to use Transformer Engine layers
    with torch.no_grad():
        convert_model(model)

    return model


def get_fp8_autocast_context(
    fp8_config: Optional[FP8TrainingConfig] = None,
    enabled: bool = True,
):
    """
    Get FP8 autocast context manager for manual FP8 training loops.

    This is useful when not using Accelerate and implementing custom training loops
    with Transformer Engine directly.

    Args:
        fp8_config (FP8TrainingConfig, optional): FP8 configuration
        enabled (bool): Whether to enable FP8 autocast

    Returns:
        Context manager for FP8 autocast

    Example:
        >>> with get_fp8_autocast_context():
        ...     outputs = model(**batch)
        ...     loss = outputs.loss
    """
    if not TE_AVAILABLE:
        # Return a no-op context manager
        import contextlib
        return contextlib.nullcontext()

    if fp8_config is None:
        fp8_config = FP8TrainingConfig()

    fp8_recipe = fp8_config.get_te_recipe()

    return te.fp8_autocast(enabled=enabled, fp8_recipe=fp8_recipe)


class FP8Trainer:
    """
    Wrapper class to simplify FP8 training setup.

    This class handles the boilerplate of setting up FP8 training with Accelerate
    and Transformer Engine, making it easier to get started with FP8 finetuning.

    Example:
        >>> from unsloth.fp8_training import FP8Trainer
        >>>
        >>> trainer = FP8Trainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     fp8_config=FP8TrainingConfig(fp8_format="HYBRID")
        ... )
        >>>
        >>> # Training loop
        >>> for batch in train_dataloader:
        ...     loss = trainer.training_step(batch)
        ...     print(f"Loss: {loss.item()}")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        fp8_config: Optional[FP8TrainingConfig] = None,
        convert_model_to_fp8: bool = False,
        **accelerator_kwargs
    ):
        """
        Initialize FP8 trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer for training
            fp8_config: FP8 configuration
            convert_model_to_fp8: Whether to convert model layers to TE FP8 layers
            **accelerator_kwargs: Additional arguments for Accelerator
        """
        self.fp8_config = fp8_config or FP8TrainingConfig()

        # Convert model if requested
        if convert_model_to_fp8:
            model = prepare_model_for_fp8_training(model, self.fp8_config)

        # Setup accelerator
        self.accelerator = get_fp8_accelerator(self.fp8_config, **accelerator_kwargs)

        if self.accelerator is None:
            raise RuntimeError("Failed to initialize FP8 accelerator. Check GPU support and dependencies.")

        # Prepare model and optimizer
        self.model, self.optimizer = self.accelerator.prepare(model, optimizer)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Execute a single training step.

        Args:
            batch: Dictionary containing input tensors

        Returns:
            Loss tensor
        """
        self.model.train()

        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

    def save_model(self, save_path: str):
        """Save the trained model."""
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(unwrapped_model.state_dict(), save_path)


# Convenience function for backward compatibility
def setup_fp8_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    fp8_format: str = "HYBRID",
    amax_history_len: int = 32,
    amax_compute_algo: str = "max",
    **accelerator_kwargs
):
    """
    Convenience function to setup FP8 training (deprecated - use FP8Trainer or get_fp8_accelerator).

    Args:
        model: PyTorch model
        optimizer: Optimizer
        fp8_format: FP8 format ("HYBRID", "E4M3", or "E5M2")
        amax_history_len: History length for amax computation
        amax_compute_algo: Algorithm for amax computation
        **accelerator_kwargs: Additional Accelerator arguments

    Returns:
        Tuple of (prepared_model, prepared_optimizer, accelerator)
    """
    import warnings
    warnings.warn(
        "setup_fp8_training is deprecated. Use FP8Trainer or get_fp8_accelerator instead.",
        DeprecationWarning,
        stacklevel=2
    )

    fp8_config = FP8TrainingConfig(
        fp8_format=fp8_format,
        amax_history_len=amax_history_len,
        amax_compute_algo=amax_compute_algo,
    )

    accelerator = get_fp8_accelerator(fp8_config, **accelerator_kwargs)
    if accelerator is None:
        raise RuntimeError("Failed to setup FP8 training")

    model, optimizer = accelerator.prepare(model, optimizer)

    return model, optimizer, accelerator
