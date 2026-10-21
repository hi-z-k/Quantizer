"""Symmetric quantization parameter computation.

Symmetric quantization centres the quantized range on zero:
    qmax = 2^(bits-1) - 1
    scale = max(|w|) / qmax

The range [-qmax, qmax] is symmetric, so no zero_point is needed (it is 0).

Example (8-bit, per-tensor, weights [0.2, -0.1, 0.3, 8.0, -3.9, 2.0]):
    qmax = 127
    scale = 8.0 / 127 ≈ 0.06299 → rounded to float16 → 0.06299
    q(8.0) = round(8.0 / 0.06299) = round(127.0) = 127  ✓
    q(0.2) = round(0.2 / 0.06299) = round(3.17)  = 3

Scales are rounded to float16 before use. A constant or all-zero group
gets scale = 1.0 to avoid division by zero.
"""

from __future__ import annotations

from typing import Optional

import torch

from quantlab.config import QuantParams
from quantlab.interfaces import QuantParamsStrategy


class SymmetricParamsStrategy(QuantParamsStrategy):
    """Compute symmetric quantization parameters for a group of weights.

    For each group, find the maximum absolute value, compute the scale,
    round it to float16, and return QuantParams with zero_point = 0.
    """

    def compute(
        self,
        weight: torch.Tensor,
        bits: int,
        group_size: Optional[int] = None,
    ) -> list[QuantParams]:
        """Compute symmetric scale for each row of the grouped weight.

        Args:
            weight: 2-D tensor [n_groups, group_width] from grouping.
            bits: Number of quantization bits.

        Returns:
            List of QuantParams, one per group (row).
        """
        qmax = (1 << (bits - 1)) - 1
        qmin = -qmax

        results: list[QuantParams] = []
        for row in weight:
            abs_max = row.abs().max().item()
            if abs_max == 0.0:
                scale = 1.0
            else:
                scale = abs_max / qmax
            # Round scale to float16 precision
            scale = torch.tensor(scale, dtype=torch.float16).item()
            results.append(QuantParams(scale=scale, zero_point=0, qmin=qmin, qmax=qmax))
        return results


def compute_symmetric_params_batched(
    weight: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, int, int]:
    """Vectorised symmetric parameter computation.

    Args:
        weight: 2-D tensor [n_groups, group_width].
        bits: Number of quantization bits.

    Returns:
        (scales, qmin, qmax) where scales is [n_groups] in float16.
    """
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax

    abs_max = weight.abs().amax(dim=1)  # [n_groups]
    scales = abs_max / qmax
    # Replace zero scales with 1.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    # Round to float16
    scales = scales.to(torch.float16)
    return scales, qmin, qmax
