"""Linear dequantizer: recovers approximate float values from quantized codes.

The reconstruction formula is:
    w_hat = scale * (q - zero_point)

For symmetric quantization, zero_point is 0, so:
    w_hat = scale * q

Example (symmetric 8-bit, scale = 8.0/127, code = 3):
    w_hat = (8.0/127) * 3 = 0.189  (original was 0.2, error = 0.011)
"""

from __future__ import annotations

from typing import Optional

import torch

from quantlab.interfaces import Dequantizer


class LinearDequantizer(Dequantizer):
    """Dequantize integer codes back to approximate floating-point values.

    Uses the affine map: w_hat = scale * (q - zero_point).
    """

    def dequantize(
        self,
        codes: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        bits: int,
    ) -> torch.Tensor:
        """Dequantize codes to float values.

        Args:
            codes: Integer codes, same shape as the original weight group.
            scale: Per-group scale factors (broadcastable to codes).
            zero_point: Per-group zero points, or None for symmetric.
            bits: Not used in linear dequantization, kept for interface.

        Returns:
            Float tensor of reconstructed values, same shape as codes.
        """
        codes_float = codes.float()
        if zero_point is not None:
            zp_float = zero_point.float()
            return scale.float() * (codes_float - zp_float)
        return scale.float() * codes_float
