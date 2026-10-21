"""Quantized linear layer that stores weights in packed integer format.

Replaces ``nn.Linear`` with a module that stores:
- packed_weight (uint8): bit-packed quantized weight codes
- scales (float16): per-group scale factors, shape [out_features, n_groups]
- zero_points (float16, asymmetric only): per-group zero points
- bits, group_size, shapes, and the original bias

On forward: unpack → dequantize to the input dtype → ``F.linear``.
The dequantized weight is never cached.

Stored codes are unsigned:
- Symmetric codes for bits < 8: stored as q + 2^(bits-1) (e.g., 4-bit codes 1..15)
- Symmetric 8-bit codes: stored as int8 (signed)
- Asymmetric codes: used as-is (already unsigned)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantlab.bit_packer import BitPacker
from quantlab.config import Granularity, Mode, QuantConfig
from quantlab.linear_dequantizer import LinearDequantizer


class QuantizedLinear(nn.Module):
    """A linear layer with bit-packed quantized weights.

    This module stores quantized weights compactly and dequantizes them
    on every forward pass. It never caches the full-precision weight.

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        bits: Quantization bit width.
        group_size: Number of weights per group.
        n_groups_per_row: Number of groups per output channel.
        mode: SYMMETRIC or ASYMMETRIC.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QuantConfig,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = config.bits
        self.mode = config.mode
        self.granularity = config.granularity

        # Compute group layout
        if config.granularity == Granularity.TENSOR:
            self.group_size = in_features * out_features
            self.n_groups_per_row = 1
        elif config.granularity == Granularity.CHANNEL:
            self.group_size = in_features
            self.n_groups_per_row = 1
        else:  # GROUP
            self.group_size = config.group_size
            self.n_groups_per_row = (in_features + config.group_size - 1) // config.group_size

        # Register buffers (populated by RTNQuantizer)
        self.register_buffer("packed_weight", torch.empty(0, dtype=torch.uint8))
        self.register_buffer(
            "scales",
            torch.empty(out_features, self.n_groups_per_row, dtype=torch.float16),
        )
        if config.mode == Mode.ASYMMETRIC:
            self.register_buffer(
                "zero_points",
                torch.empty(out_features, self.n_groups_per_row, dtype=torch.float16),
            )
        else:
            self.zero_points = None

        if bias:
            self.register_buffer("bias_param", torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias_param = None

        self._packer = BitPacker()
        self._dequantizer = LinearDequantizer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unpack, dequantize, and compute the linear transform.

        Args:
            x: Input tensor of shape [..., in_features].

        Returns:
            Output tensor of shape [..., out_features].
        """
        weight = self._dequantize_weight(x.dtype)
        bias = self.bias_param.to(x.dtype) if self.bias_param is not None else None
        return F.linear(x, weight, bias)

    def _dequantize_weight(self, target_dtype: torch.dtype) -> torch.Tensor:
        """Unpack and dequantize the full weight matrix.

        Returns:
            Float tensor of shape [out_features, in_features].
        """
        total_elements = self.out_features * self.in_features
        n_groups = self.out_features * self.n_groups_per_row

        if self.granularity == Granularity.TENSOR:
            # For tensor granularity, all weights share one scale
            padded_total = self.out_features * self.n_groups_per_row * self.group_size
            codes = self._packer.unpack(self.packed_weight, self.bits, padded_total)
        else:
            # Unpack all codes
            padded_width = self.n_groups_per_row * self.group_size
            padded_total = self.out_features * padded_width
            codes = self._packer.unpack(self.packed_weight, self.bits, padded_total)

        # Convert stored unsigned codes back to signed for symmetric
        codes = codes.to(torch.float32)
        if self.mode == Mode.SYMMETRIC and self.bits < 8:
            codes = codes - (1 << (self.bits - 1))
        elif self.mode == Mode.SYMMETRIC and self.bits == 8:
            # 8-bit symmetric stored as int8, already handled by uint8 view
            codes = codes.to(torch.float32)
            mask = codes > 127
            codes[mask] = codes[mask] - 256

        # Reshape codes into [out_features, n_groups_per_row, group_size]
        if self.granularity == Granularity.TENSOR:
            # Single group for entire matrix
            codes_grouped = codes[:total_elements].reshape(1, -1)
            scales = self.scales.reshape(1)  # single scale
            zp = self.zero_points.reshape(1) if self.zero_points is not None else None
            deq = self._dequantizer.dequantize(codes_grouped, scales.unsqueeze(1), 
                                                zp.unsqueeze(1) if zp is not None else None, 
                                                self.bits)
            weight = deq.reshape(self.out_features, self.in_features)
        else:
            padded_width = self.n_groups_per_row * self.group_size
            codes_2d = codes[:self.out_features * padded_width].reshape(
                self.out_features, self.n_groups_per_row, self.group_size
            )
            # Expand scales for broadcasting: [out, n_groups, 1]
            scales = self.scales.unsqueeze(2)
            zp = self.zero_points.unsqueeze(2) if self.zero_points is not None else None
            deq = self._dequantizer.dequantize(codes_2d, scales, zp, self.bits)
            # Flatten groups and trim to in_features
            weight = deq.reshape(self.out_features, padded_width)[:, :self.in_features]

        return weight.to(target_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, groups={self.n_groups_per_row}, "
            f"mode={self.mode.value}"
        )
