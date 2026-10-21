"""Simulated RTN quantizer: quantize-then-dequantize in place.

Replaces each Linear weight with its quantized-then-dequantized version
(still FP16). Uses the same arithmetic as the packed RTNQuantizer so that
logits match within 1e-3.

This is useful for fast evaluation sweeps because it avoids the overhead
of bit packing and unpacking. The model stays in its original architecture
with standard nn.Linear layers.
"""

from __future__ import annotations

from typing import Set, Tuple

import torch
import torch.nn as nn

from quantlab.config import Granularity, Mode, QuantConfig
from quantlab.grouping import group_weight, ungroup_weight
from quantlab.interfaces import Quantizer
from quantlab.symmetric_params import compute_symmetric_params_batched
from quantlab.asymmetric_params import compute_asymmetric_params_batched


class SimulatedRTNQuantizer(Quantizer):
    """Simulated Round-To-Nearest quantizer.

    Quantizes and immediately dequantizes the weights of every nn.Linear
    whose name is not in skip_names. The weight stays as a float parameter
    but holds only values that are exactly representable at the target bit width.

    Args:
        config: Quantization configuration.
        skip_names: Set of module name substrings to skip (e.g., 'lm_head').
    """

    def __init__(
        self,
        config: QuantConfig,
        skip_names: Tuple[str, ...] = ("lm_head",),
    ) -> None:
        self.config = config
        self.skip_names = skip_names

    def quantize(self, model: nn.Module) -> nn.Module:
        """Quantize-then-dequantize all Linear weights in place.

        Args:
            model: A PyTorch model.

        Returns:
            The same model with weights replaced by their quantized approximations.
        """
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(skip in name for skip in self.skip_names):
                continue
            self._simulate_layer(module)
        return model

    def _simulate_layer(self, linear: nn.Linear) -> None:
        """Quantize-then-dequantize one Linear layer's weight in place."""
        weight = linear.weight.data  # [out_features, in_features]
        original_dtype = weight.dtype
        weight_f32 = weight.float()

        grouped, info = group_weight(
            weight_f32,
            self.config.granularity,
            self.config.group_size,
        )

        if self.config.mode == Mode.SYMMETRIC:
            deq = _sim_symmetric(grouped, self.config.bits)
        else:
            deq = _sim_asymmetric(grouped, self.config.bits)

        result = ungroup_weight(deq, info)
        linear.weight.data = result.to(original_dtype)


def _sim_symmetric(grouped: torch.Tensor, bits: int) -> torch.Tensor:
    """Simulate symmetric quantization on grouped weights.

    Args:
        grouped: [n_groups, group_width] float32 tensor.
        bits: Number of quantization bits.

    Returns:
        [n_groups, group_width] dequantized float32 tensor.
    """
    scales, qmin, qmax = compute_symmetric_params_batched(grouped, bits)
    # scales is float16 [n_groups], expand for broadcasting
    scales_f32 = scales.float().unsqueeze(1)  # [n_groups, 1]

    # Quantize
    codes = torch.round(grouped / scales_f32)
    codes = codes.clamp(qmin, qmax)

    # Dequantize
    return scales_f32 * codes


def _sim_asymmetric(grouped: torch.Tensor, bits: int) -> torch.Tensor:
    """Simulate asymmetric quantization on grouped weights.

    Args:
        grouped: [n_groups, group_width] float32 tensor.
        bits: Number of quantization bits.

    Returns:
        [n_groups, group_width] dequantized float32 tensor.
    """
    scales, zero_points, qmin, qmax = compute_asymmetric_params_batched(grouped, bits)
    scales_f32 = scales.float().unsqueeze(1)  # [n_groups, 1]
    zp_f32 = zero_points.float().unsqueeze(1)  # [n_groups, 1]

    # Quantize
    codes = torch.round(grouped / scales_f32) + zp_f32
    codes = codes.clamp(qmin, qmax)

    # Dequantize
    return scales_f32 * (codes - zp_f32)
