"""Weight snapshot: saves and restores original weights for repeated quantization.

Keeps CPU copies of the quantizable weights (all nn.Linear except skip_names)
so that the same model can be quantized with different configs without
reloading from disk.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class WeightSnapshot:
    """Saves and restores original float weights for quantization experiments.

    Usage:
        >>> snapshot = WeightSnapshot()
        >>> snap = snapshot.take(model)
        >>> # ... quantize model ...
        >>> snapshot.restore(model)  # weights are back to original

    The snapshot stores CPU copies so GPU memory is not doubled.
    """

    def __init__(self) -> None:
        self._weights: Dict[str, torch.Tensor] = {}
        self._skip_names: Tuple[str, ...] = ()

    def take(
        self,
        model: nn.Module,
        skip_names: Tuple[str, ...] = ("lm_head",),
    ) -> "WeightSnapshot":
        """Save CPU copies of all quantizable Linear weights.

        Args:
            model: The model whose weights to save.
            skip_names: Module name substrings to skip.

        Returns:
            self (for chaining).
        """
        self._weights.clear()
        self._skip_names = skip_names
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(skip in name for skip in skip_names):
                continue
            self._weights[name] = module.weight.data.detach().cpu().clone()
        return self

    def restore(self, model: nn.Module) -> None:
        """Restore saved weights into the model in place.

        Args:
            model: The model to restore. Must have the same architecture
                as when take() was called.

        Raises:
            RuntimeError: If take() was never called.
        """
        if not self._weights:
            raise RuntimeError("No snapshot taken. Call take() first.")

        for name, module in model.named_modules():
            if name in self._weights:
                module.weight.data = self._weights[name].to(
                    module.weight.device, module.weight.dtype
                )

    @property
    def weight_names(self) -> list[str]:
        """List of saved weight names."""
        return list(self._weights.keys())

    def get_original(self, name: str) -> torch.Tensor:
        """Get the original weight tensor for a layer by name.

        Args:
            name: Module name as returned by named_modules().

        Returns:
            CPU float tensor of the original weight.
        """
        return self._weights[name]
