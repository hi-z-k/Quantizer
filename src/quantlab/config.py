"""Configuration data classes for quantization.

Defines the three axes of a quantization config:
- Mode: symmetric or asymmetric quantization
- Granularity: per-tensor, per-channel, or per-group scale sharing
- QuantConfig: frozen dataclass combining bits, mode, granularity, and group_size

Example:
    >>> cfg = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)
    >>> cfg.name
    'int4_sym_group32'
    >>> cfg = QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL)
    >>> cfg.name
    'int8_sym_channel'
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class Mode(enum.Enum):
    """Quantization mode.

    SYMMETRIC: the quantized range is centred on zero, [-qmax, qmax].
        No zero_point is needed. Good when weights are roughly symmetric.
    ASYMMETRIC: the quantized range uses the full unsigned range [0, 2^bits - 1].
        A zero_point maps the real zero into this range. Needed when weights
        are one-sided (e.g. after ReLU).
    """

    SYMMETRIC = "sym"
    ASYMMETRIC = "asym"


class Granularity(enum.Enum):
    """Scale granularity: how many weights share one scale value.

    TENSOR: one scale for the entire weight matrix. Cheapest but coarsest.
    CHANNEL: one scale per output channel (per row of the weight matrix).
    GROUP: one scale per block of group_size consecutive weights along the
        input dimension. Finer groups cost more storage but track local
        ranges better.
    """

    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"


@dataclass(frozen=True)
class QuantConfig:
    """Immutable quantization configuration.

    Args:
        bits: Number of bits per weight (e.g. 2, 3, 4, 5, 6, 8).
        mode: Symmetric or asymmetric quantization.
        granularity: Tensor-level, channel-level, or group-level scales.
        group_size: Number of weights per group. Required iff granularity is GROUP.

    The ``.name`` property returns a canonical string for file names and logs::

        'int4_sym_group32', 'int8_sym_channel', 'int4_asym_tensor'
    """

    bits: int
    mode: Mode
    granularity: Granularity
    group_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.granularity == Granularity.GROUP and self.group_size is None:
            raise ValueError("group_size is required when granularity is GROUP")
        if self.granularity != Granularity.GROUP and self.group_size is not None:
            raise ValueError("group_size must be None when granularity is not GROUP")
        if self.bits < 1 or self.bits > 16:
            raise ValueError(f"bits must be between 1 and 16, got {self.bits}")

    @property
    def name(self) -> str:
        """Canonical name like 'int4_sym_group32' or 'int8_asym_channel'."""
        mode_str = self.mode.value
        if self.granularity == Granularity.GROUP:
            gran_str = f"group{self.group_size}"
        else:
            gran_str = self.granularity.value
        return f"int{self.bits}_{mode_str}_{gran_str}"


@dataclass
class QuantParams:
    """Computed quantization parameters for one group of weights.

    These are the numbers that the quantize and dequantize formulas use:
        quantize:   q = clamp(round(w / scale) + zero_point, qmin, qmax)
        dequantize: w_hat = scale * (q - zero_point)

    Attributes:
        scale: The step size in weight space. Rounded to float16 precision.
        zero_point: Integer offset mapping real zero to the code space.
            For symmetric: always 0.
        qmin: Minimum representable code.
        qmax: Maximum representable code.
    """

    scale: float
    zero_point: int
    qmin: int
    qmax: int
