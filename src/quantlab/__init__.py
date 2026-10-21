"""quantlab: Post-training weight quantization for language models.

A post-training quantization library written from scratch in raw PyTorch.
Architecture:
- Core Quantize/Dequantize Logic & Interfaces: directly inside `quantlab/`
- Model Serialization & I/O: inside `quantlab.io`
- Evaluation, Datasets & Metrics: inside `quantlab.testing`
"""

import sys

# 1. Core Interfaces & Quantize/Dequantize Logic (Directly inside package)
from quantlab.interfaces import (
    Quantizer,
    QuantParamsStrategy,
    Packer,
    Dequantizer,
    TextDataset,
    Metric,
    Exporter,
    TemplateExporter,
)
from quantlab.config import Mode, Granularity, QuantConfig, QuantParams
from quantlab.grouping import group_weight, ungroup_weight
from quantlab.symmetric_params import SymmetricParamsStrategy, compute_symmetric_params_batched
from quantlab.asymmetric_params import AsymmetricParamsStrategy, compute_asymmetric_params_batched
from quantlab.bit_packer import BitPacker
from quantlab.linear_dequantizer import LinearDequantizer
from quantlab.quantized_linear import QuantizedLinear
from quantlab.simulated_quantizer import SimulatedRTNQuantizer
from quantlab.rtn_quantizer import RTNQuantizer
from quantlab.snapshot import WeightSnapshot

# 2. I/O Subpackage (Exporters & Loaders)
from quantlab import io
from quantlab.io import NativeExporter, load_native, GGUFExporter
from quantlab.io import native_exporter, native_loader, gguf_exporter

# 3. Testing & Evaluation Subpackage (Datasets & Metrics)
from quantlab import testing
from quantlab.testing import (
    WikiText2Dataset,
    PerplexityMetric,
    WeightErrorMetric,
    estimate_size,
    measure_size,
    SizeReport,
)
from quantlab.testing import (
    wikitext2_dataset,
    perplexity_metric,
    weight_error_metric,
    model_size_metric,
)

# Convenience alias: eval -> testing
eval = testing
sys.modules["quantlab.eval"] = testing

# Backward-compatible module paths for contract consumers
sys.modules["quantlab.wikitext2_dataset"] = wikitext2_dataset
sys.modules["quantlab.perplexity_metric"] = perplexity_metric
sys.modules["quantlab.weight_error_metric"] = weight_error_metric
sys.modules["quantlab.model_size_metric"] = model_size_metric
sys.modules["quantlab.native_exporter"] = native_exporter
sys.modules["quantlab.native_loader"] = native_loader
sys.modules["quantlab.gguf_exporter"] = gguf_exporter

__all__ = [
    # Interfaces
    "Quantizer",
    "QuantParamsStrategy",
    "Packer",
    "Dequantizer",
    "TextDataset",
    "Metric",
    "Exporter",
    "TemplateExporter",
    # Core Quantize/Dequantize Logic
    "Mode",
    "Granularity",
    "QuantConfig",
    "QuantParams",
    "group_weight",
    "ungroup_weight",
    "SymmetricParamsStrategy",
    "compute_symmetric_params_batched",
    "AsymmetricParamsStrategy",
    "compute_asymmetric_params_batched",
    "BitPacker",
    "LinearDequantizer",
    "QuantizedLinear",
    "SimulatedRTNQuantizer",
    "RTNQuantizer",
    "WeightSnapshot",
    # Subpackages
    "io",
    "testing",
    "eval",
    # I/O
    "NativeExporter",
    "load_native",
    "GGUFExporter",
    # Testing & Metrics
    "WikiText2Dataset",
    "PerplexityMetric",
    "WeightErrorMetric",
    "estimate_size",
    "measure_size",
    "SizeReport",
]
