"""Abstract interfaces for quantlab.

All abstract base classes live here. A new technique is a new Quantizer file,
a new format is a new Exporter, new data is a new TextDataset — nothing
existing is edited. Depend on these interfaces, not on concrete classes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class QuantParamsStrategy(ABC):
    """Strategy for computing quantization parameters (scale, zero_point).

    Given a group of weights and a bit width, compute the scale and zero_point
    that minimise the quantization error.
    """

    @abstractmethod
    def compute(
        self, weight: torch.Tensor, bits: int, group_size: Optional[int] = None
    ) -> Any:
        """Compute quantization parameters for a weight group.

        Args:
            weight: A 1-D or 2-D tensor of float weights.
            bits: Number of quantization bits.
            group_size: Optional group size for grouped quantization.

        Returns:
            QuantParams or a batch of QuantParams.
        """
        ...


class Packer(ABC):
    """Packs integer codes into compact byte representations."""

    @abstractmethod
    def pack(self, codes: torch.Tensor, bits: int) -> torch.Tensor:
        """Pack integer codes into uint8 bytes.

        Args:
            codes: 1-D tensor of unsigned integer codes.
            bits: Bits per code (2, 4, or 8).

        Returns:
            1-D uint8 tensor of packed bytes.
        """
        ...

    @abstractmethod
    def unpack(self, packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
        """Unpack codes from packed bytes.

        Args:
            packed: 1-D uint8 tensor of packed bytes.
            bits: Bits per code (2, 4, or 8).
            count: Number of original codes.

        Returns:
            1-D tensor of unsigned integer codes.
        """
        ...


class Dequantizer(ABC):
    """Recovers approximate floating-point values from quantized codes."""

    @abstractmethod
    def dequantize(
        self,
        codes: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        bits: int,
    ) -> torch.Tensor:
        """Dequantize codes back to approximate float values.

        The reconstruction formula is: real = scale * (q - zero_point).
        For symmetric quantization, zero_point is 0.

        Args:
            codes: Integer quantized codes.
            scale: Per-group scale factors.
            zero_point: Per-group zero points (None for symmetric).
            bits: Number of quantization bits.

        Returns:
            Float tensor of reconstructed values.
        """
        ...


class Quantizer(ABC):
    """Quantizes a model's weights in place."""

    @abstractmethod
    def quantize(self, model: nn.Module) -> nn.Module:
        """Quantize the model's Linear layers in place.

        Args:
            model: A PyTorch model with nn.Linear layers.

        Returns:
            The same model with weights quantized.
        """
        ...


class TextDataset(ABC):
    """Tokenized text dataset for evaluation."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor:
        ...


class Metric(ABC):
    """Measures some property of a model."""

    @abstractmethod
    def measure(self, model: nn.Module, *args: Any, **kwargs: Any) -> Any:
        ...


class Exporter(ABC):
    """Exports a quantized model to a specific format."""

    @abstractmethod
    def export(
        self, model: nn.Module, config: Any, out_dir: Any, **kwargs: Any
    ) -> Any:
        ...


class TemplateExporter(ABC):
    """Exports using a template file (e.g. GGUF)."""

    @abstractmethod
    def export(self, config: Any, out_path: Any) -> Any:
        ...
