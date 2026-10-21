"""Asymmetric quantization parameter computation.

Asymmetric quantization uses the full unsigned range [0, 2^bits - 1]:
    1. Widen the range so it contains zero:
       w_min = min(min(w), 0)
       w_max = max(max(w), 0)
    2. scale = (w_max - w_min) / (2^bits - 1)
    3. zero_point = clamp(round(-w_min / scale), 0, 2^bits - 1)
    4. q = clamp(round(w / scale) + zero_point, 0, 2^bits - 1)

Example (4-bit per-tensor, weights [-0.8, 0, 0.55, 1.75, 2.2]):
    w_min = -0.8, w_max = 2.2 (both already include zero in between)
    scale = (2.2 - (-0.8)) / 15 = 3.0 / 15 = 0.2
    zero_point = clamp(round(0.8 / 0.2), 0, 15) = clamp(4, 0, 15) = 4
    q(-0.8) = clamp(round(-0.8/0.2) + 4, 0, 15) = clamp(-4+4, 0, 15) = 0   ✓
    q(0)    = clamp(round(0/0.2) + 4, 0, 15)     = clamp(0+4, 0, 15)  = 4   ✓
    q(0.55) = clamp(round(0.55/0.2) + 4, 0, 15)  = clamp(3+4, 0, 15)  = 7
    q(1.75) = clamp(round(1.75/0.2) + 4, 0, 15)  = clamp(9+4, 0, 15)  = 13
    q(2.2)  = clamp(round(2.2/0.2) + 4, 0, 15)   = clamp(11+4, 0, 15) = 15  ✓

Scales are rounded to float16 before they are used. A constant or all-zero
group gets scale = 1.0 to avoid division by zero.
"""

from __future__ import annotations

from typing import Optional

import torch

from quantlab.config import QuantParams
from quantlab.interfaces import QuantParamsStrategy


class AsymmetricParamsStrategy(QuantParamsStrategy):
    """Compute asymmetric quantization parameters for a group of weights."""

    def compute(
        self,
        weight: torch.Tensor,
        bits: int,
        group_size: Optional[int] = None,
    ) -> list[QuantParams]:
        """Compute asymmetric scale and zero_point for each row of the grouped weight.

        Args:
            weight: 2-D tensor [n_groups, group_width] from grouping.
            bits: Number of quantization bits.

        Returns:
            List of QuantParams, one per group (row).
        """
        qmin = 0
        qmax = (1 << bits) - 1

        results: list[QuantParams] = []
        for row in weight:
            w_min = min(row.min().item(), 0.0)
            w_max = max(row.max().item(), 0.0)
            scale = (w_max - w_min) / qmax
            if scale == 0.0:
                scale = 1.0
            # Round scale to float16 precision
            scale = torch.tensor(scale, dtype=torch.float16).item()
            zero_point = int(torch.round(torch.tensor(-w_min / scale)).item())
            zero_point = max(qmin, min(qmax, zero_point))
            results.append(QuantParams(scale=scale, zero_point=zero_point, qmin=qmin, qmax=qmax))
        return results


def compute_asymmetric_params_batched(
    weight: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Vectorised asymmetric parameter computation.

    Args:
        weight: 2-D tensor [n_groups, group_width].
        bits: Number of quantization bits.

    Returns:
        (scales, zero_points, qmin, qmax) where scales is [n_groups] float16
        and zero_points is [n_groups] int.
    """
    qmin = 0
    qmax = (1 << bits) - 1

    w_min = weight.amin(dim=1)  # [n_groups]
    w_max = weight.amax(dim=1)
    # Widen to include zero
    w_min = torch.minimum(w_min, torch.zeros_like(w_min))
    w_max = torch.maximum(w_max, torch.zeros_like(w_max))

    scales = (w_max - w_min) / qmax
    # Replace zero scales with 1.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    scales = scales.to(torch.float16)

    # Compute zero_point using the float16 scales
    zero_points = torch.round(-w_min / scales.float()).to(torch.int64)
    zero_points = zero_points.clamp(qmin, qmax)

    return scales, zero_points, qmin, qmax
