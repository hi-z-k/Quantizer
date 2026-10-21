"""Model size metric: formula estimates and real tensor byte counts.

Two functions:
- estimate_size: computes bytes from formulas (no model modification needed).
- measure_size: adds up real tensor bytes of a packed model.

Size components:
- packed_weight_bytes = quantized_params × bits / 8
- scale_bytes = 2 bytes (float16) per group
- zero_point_bytes = 2 bytes per group (asymmetric only)
- fp16_rest_bytes = everything not quantized, counted as float16

effective_bits = (packed + scale + zp bytes) × 8 / quantized_params
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from quantlab.config import Granularity, Mode, QuantConfig


@dataclass
class SizeReport:
    """Size breakdown of a quantized model.

    All sizes in bytes unless suffixed with _mb, _pct, or _bits.
    """

    packed_weight_bytes: int
    scale_bytes: int
    zero_point_bytes: int
    fp16_rest_bytes: int
    total_bytes: int
    total_mb: float
    effective_bits: float
    reduction_vs_fp16_pct: float
    fp16_total_bytes: int


def estimate_size(
    model: nn.Module,
    config: QuantConfig,
    skip_names: Tuple[str, ...] = ("lm_head",),
) -> SizeReport:
    """Estimate size from formulas without modifying the model.

    Args:
        model: The original (unquantized) model.
        config: Quantization configuration.
        skip_names: Module name substrings to skip.

    Returns:
        SizeReport with estimated byte counts.
    """
    quant_params = 0
    total_groups = 0
    fp16_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            n_params = module.weight.numel()
            if any(skip in name for skip in skip_names):
                fp16_params += n_params
                if module.bias is not None:
                    fp16_params += module.bias.numel()
            else:
                quant_params += n_params
                n_groups = _count_groups(module, config)
                total_groups += n_groups
                if module.bias is not None:
                    fp16_params += module.bias.numel()
        elif isinstance(module, nn.Embedding):
            fp16_params += module.weight.numel()
        elif hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
            # LayerNorm, RMSNorm, etc.
            if not isinstance(module, nn.Linear):
                fp16_params += module.weight.numel()
                if hasattr(module, "bias") and module.bias is not None:
                    fp16_params += module.bias.numel()

    packed_weight_bytes = quant_params * config.bits // 8
    scale_bytes = total_groups * 2  # float16
    zero_point_bytes = total_groups * 2 if config.mode == Mode.ASYMMETRIC else 0
    fp16_rest_bytes = fp16_params * 2

    total_bytes = packed_weight_bytes + scale_bytes + zero_point_bytes + fp16_rest_bytes
    fp16_total_bytes = (quant_params + fp16_params) * 2

    total_mb = total_bytes / (1024 * 1024)
    reduction = (1 - total_bytes / fp16_total_bytes) * 100 if fp16_total_bytes > 0 else 0.0

    quant_overhead = packed_weight_bytes + scale_bytes + zero_point_bytes
    effective_bits = (quant_overhead * 8 / quant_params) if quant_params > 0 else 0.0

    return SizeReport(
        packed_weight_bytes=packed_weight_bytes,
        scale_bytes=scale_bytes,
        zero_point_bytes=zero_point_bytes,
        fp16_rest_bytes=fp16_rest_bytes,
        total_bytes=total_bytes,
        total_mb=total_mb,
        effective_bits=effective_bits,
        reduction_vs_fp16_pct=reduction,
        fp16_total_bytes=fp16_total_bytes,
    )


def measure_size(model: nn.Module) -> SizeReport:
    """Count real tensor bytes of a (packed) model.

    Args:
        model: A model with QuantizedLinear and/or nn.Linear layers.

    Returns:
        SizeReport with actual byte counts.
    """
    from quantlab.quantized_linear import QuantizedLinear

    packed_bytes = 0
    scale_bytes = 0
    zp_bytes = 0
    fp16_bytes = 0
    quant_params = 0
    fp16_params = 0

    counted_data_ptrs: set[int] = set()

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            pw = module.packed_weight
            if pw.data_ptr() not in counted_data_ptrs:
                packed_bytes += pw.numel() * pw.element_size()
                counted_data_ptrs.add(pw.data_ptr())

            sc = module.scales
            if sc.data_ptr() not in counted_data_ptrs:
                scale_bytes += sc.numel() * sc.element_size()
                counted_data_ptrs.add(sc.data_ptr())

            if module.zero_points is not None:
                zpt = module.zero_points
                if zpt.data_ptr() not in counted_data_ptrs:
                    zp_bytes += zpt.numel() * zpt.element_size()
                    counted_data_ptrs.add(zpt.data_ptr())

            if module.bias_param is not None:
                bp = module.bias_param
                if bp.data_ptr() not in counted_data_ptrs:
                    fp16_bytes += bp.numel() * bp.element_size()
                    counted_data_ptrs.add(bp.data_ptr())

            quant_params += module.out_features * module.in_features
            continue

        # Count all parameters and buffers of non-QuantizedLinear modules
        for pname, param in module.named_parameters(recurse=False):
            if param.data_ptr() not in counted_data_ptrs:
                fp16_bytes += param.numel() * param.element_size()
                fp16_params += param.numel()
                counted_data_ptrs.add(param.data_ptr())
        for bname, buf in module.named_buffers(recurse=False):
            if buf.data_ptr() not in counted_data_ptrs:
                fp16_bytes += buf.numel() * buf.element_size()
                counted_data_ptrs.add(buf.data_ptr())

    total_bytes = packed_bytes + scale_bytes + zp_bytes + fp16_bytes
    fp16_total_bytes = (quant_params + fp16_params) * 2 if (quant_params + fp16_params) > 0 else total_bytes

    total_mb = total_bytes / (1024 * 1024)
    reduction = (1 - total_bytes / fp16_total_bytes) * 100 if fp16_total_bytes > 0 else 0.0

    quant_overhead = packed_bytes + scale_bytes + zp_bytes
    effective_bits = (quant_overhead * 8 / quant_params) if quant_params > 0 else 0.0

    return SizeReport(
        packed_weight_bytes=packed_bytes,
        scale_bytes=scale_bytes,
        zero_point_bytes=zp_bytes,
        fp16_rest_bytes=fp16_bytes,
        total_bytes=total_bytes,
        total_mb=total_mb,
        effective_bits=effective_bits,
        reduction_vs_fp16_pct=reduction,
        fp16_total_bytes=fp16_total_bytes,
    )


def _count_groups(linear: nn.Linear, config: QuantConfig) -> int:
    """Count the number of scale groups for one Linear layer."""
    out_features, in_features = linear.weight.shape

    if config.granularity == Granularity.TENSOR:
        return 1
    elif config.granularity == Granularity.CHANNEL:
        return out_features
    else:  # GROUP
        n_groups_per_row = (in_features + config.group_size - 1) // config.group_size
        return out_features * n_groups_per_row
