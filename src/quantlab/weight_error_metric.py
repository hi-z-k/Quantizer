"""Weight error metric: measures how much quantization changed the weights.

Computes over all quantizable layers:
- MSE: mean squared error between original and quantized weights.
- SQNR: signal-to-quantization-noise ratio in dB.
    SQNR = 10 * log10( sum(w^2) / sum((w - w_hat)^2) )
- Worst layer: the layer with the lowest SQNR.

Higher SQNR means less distortion (better).
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn

from quantlab.interfaces import Metric
from quantlab.snapshot import WeightSnapshot


class WeightErrorMetric(Metric):
    """Measures weight quantization error against a saved snapshot.

    Args:
        snapshot: A WeightSnapshot taken before quantization.

    Returns dict with: mse, sqnr_db, worst_layer, worst_layer_sqnr_db.
    """

    def __init__(self, snapshot: WeightSnapshot) -> None:
        self._snapshot = snapshot

    def measure(self, model: nn.Module, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Compute MSE and SQNR between original and current weights.

        Args:
            model: The model after quantization.

        Returns:
            Dictionary with keys: mse, sqnr_db, worst_layer, worst_layer_sqnr_db.
        """
        total_signal = 0.0
        total_noise = 0.0
        total_se = 0.0
        total_count = 0
        worst_layer = ""
        worst_sqnr = float("inf")

        for name in self._snapshot.weight_names:
            original = self._snapshot.get_original(name).float()
            current = self._get_current_weight(model, name).float()

            signal = (original ** 2).sum().item()
            noise = ((original - current) ** 2).sum().item()
            count = original.numel()

            total_signal += signal
            total_noise += noise
            total_se += noise
            total_count += count

            # Per-layer SQNR
            if noise > 0:
                layer_sqnr = 10.0 * math.log10(signal / noise)
            else:
                layer_sqnr = float("inf")

            if layer_sqnr < worst_sqnr:
                worst_sqnr = layer_sqnr
                worst_layer = name

        mse = total_se / total_count if total_count > 0 else 0.0

        if total_noise > 0:
            sqnr_db = 10.0 * math.log10(total_signal / total_noise)
        else:
            sqnr_db = float("inf")

        return {
            "mse": mse,
            "sqnr_db": sqnr_db,
            "worst_layer": worst_layer,
            "worst_layer_sqnr_db": worst_sqnr,
        }

    @staticmethod
    def _get_current_weight(model: nn.Module, name: str) -> torch.Tensor:
        """Get the current weight tensor by module name.

        Handles both nn.Linear and QuantizedLinear (via dequantization).
        """
        module = dict(model.named_modules()).get(name)
        if module is None:
            raise KeyError(f"Module '{name}' not found in model")

        if hasattr(module, "weight"):
            return module.weight.data.detach().cpu()

        # QuantizedLinear: dequantize to get the weight
        if hasattr(module, "_dequantize_weight"):
            return module._dequantize_weight(torch.float32).detach().cpu()

        raise TypeError(f"Cannot extract weight from {type(module)}")
