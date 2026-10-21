"""RTN quantizer: replaces nn.Linear with packed QuantizedLinear.

Round-To-Nearest quantization with bit packing. Each nn.Linear (except
those in skip_names) is replaced by a QuantizedLinear that stores the
weights as packed unsigned integers with per-group float16 scales.

Stored codes are unsigned:
- Symmetric, bits < 8: code_stored = q + 2^(bits-1)  (e.g., 4-bit: 1..15)
- Symmetric, bits = 8: stored as signed int8 viewed as uint8
- Asymmetric: codes are already unsigned
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from quantlab.bit_packer import BitPacker
from quantlab.config import Granularity, Mode, QuantConfig
from quantlab.grouping import GroupInfo, group_weight
from quantlab.interfaces import Quantizer
from quantlab.quantized_linear import QuantizedLinear
from quantlab.symmetric_params import compute_symmetric_params_batched
from quantlab.asymmetric_params import compute_asymmetric_params_batched


class RTNQuantizer(Quantizer):
    """Round-To-Nearest quantizer with packed storage.

    Replaces each nn.Linear with a QuantizedLinear that stores bit-packed
    weights, float16 scales, and optional zero points.

    Args:
        config: Quantization configuration.
        skip_names: Module name substrings to skip (e.g., 'lm_head').
    """

    def __init__(
        self,
        config: QuantConfig,
        skip_names: Tuple[str, ...] = ("lm_head",),
    ) -> None:
        self.config = config
        self.skip_names = skip_names
        self._packer = BitPacker()

    def quantize(self, model: nn.Module) -> nn.Module:
        """Replace all Linear layers with packed QuantizedLinear.

        Args:
            model: A PyTorch model.

        Returns:
            The same model with Linear layers replaced.
        """
        replacements: list[tuple[str, nn.Module, str, QuantizedLinear]] = []

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(skip in name for skip in self.skip_names):
                continue
            qlinear = self._quantize_layer(module)
            # Find parent module and attribute name
            parts = name.rsplit(".", 1)
            if len(parts) == 1:
                parent = model
                attr = parts[0]
            else:
                parent = dict(model.named_modules())[parts[0]]
                attr = parts[1]
            replacements.append((name, parent, attr, qlinear))

        for _, parent, attr, qlinear in replacements:
            setattr(parent, attr, qlinear)

        return model

    def _quantize_layer(self, linear: nn.Linear) -> QuantizedLinear:
        """Quantize and pack one Linear layer.

        Args:
            linear: The original nn.Linear module.

        Returns:
            A QuantizedLinear with packed weights.
        """
        weight = linear.weight.data.float()  # [out, in]
        out_features, in_features = weight.shape

        grouped, info = group_weight(
            weight, self.config.granularity, self.config.group_size
        )

        if self.config.mode == Mode.SYMMETRIC:
            codes, scales, zero_points = self._quantize_symmetric(grouped, info)
        else:
            codes, scales, zero_points = self._quantize_asymmetric(grouped, info)

        # Create QuantizedLinear
        qlinear = QuantizedLinear(
            in_features, out_features, self.config,
            bias=linear.bias is not None,
        )

        # Pack and store
        codes_flat = codes.reshape(-1)
        qlinear.packed_weight = self._packer.pack(codes_flat, self.config.bits)
        qlinear.scales = scales
        if zero_points is not None:
            qlinear.zero_points = zero_points
        if linear.bias is not None:
            qlinear.bias_param = linear.bias.data.to(torch.float16)

        return qlinear

    def _quantize_symmetric(
        self, grouped: torch.Tensor, info: GroupInfo
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Symmetric quantization with unsigned code storage.

        Returns:
            (codes_uint, scales, None)
        """
        scales_f16, qmin, qmax = compute_symmetric_params_batched(grouped, self.config.bits)
        scales_f32 = scales_f16.float().unsqueeze(1)

        # Quantize: signed codes
        codes_signed = torch.round(grouped / scales_f32).clamp(qmin, qmax)

        # Convert to unsigned for storage
        if self.config.bits < 8:
            offset = 1 << (self.config.bits - 1)
            codes_uint = (codes_signed + offset).to(torch.uint8)
        else:
            # 8-bit: store as signed viewed through uint8
            codes_uint = codes_signed.to(torch.int8).view(torch.uint8)

        # Reshape scales to [out_features, n_groups_per_row]
        scales_out = scales_f16.reshape(info.out_features, info.n_groups_per_row)
        return codes_uint, scales_out, None

    def _quantize_asymmetric(
        self, grouped: torch.Tensor, info: GroupInfo
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Asymmetric quantization (codes already unsigned).

        Returns:
            (codes_uint, scales, zero_points)
        """
        scales_f16, zp, qmin, qmax = compute_asymmetric_params_batched(
            grouped, self.config.bits
        )
        scales_f32 = scales_f16.float().unsqueeze(1)
        zp_f32 = zp.float().unsqueeze(1)

        codes = torch.round(grouped / scales_f32) + zp_f32
        codes = codes.clamp(qmin, qmax).to(torch.uint8)

        scales_out = scales_f16.reshape(info.out_features, info.n_groups_per_row)
        zp_out = zp.to(torch.float16).reshape(info.out_features, info.n_groups_per_row)
        return codes, scales_out, zp_out
