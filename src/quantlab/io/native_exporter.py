"""Native exporter: saves quantized models in safetensors format.

Writes:
- model.safetensors: packed tensors as <layer>.qweight, <layer>.scales,
  <layer>.zeros; every other tensor as FP16.
- quant_config.json: config and per-layer shapes for reconstruction.

If source_dir is given, copies config.json and tokenizer files so the
exported folder is self-contained.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import save_file

from quantlab.config import Mode, QuantConfig
from quantlab.interfaces import Exporter
from quantlab.quantized_linear import QuantizedLinear


class NativeExporter(Exporter):
    """Export a quantized model to safetensors with metadata.

    Usage:
        >>> exporter = NativeExporter()
        >>> path = exporter.export(model, config, 'output/', source_dir='model_dir/')
    """

    def export(
        self,
        model: torch.nn.Module,
        config: QuantConfig,
        out_dir: str | Path,
        source_dir: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> Path:
        """Export the quantized model.

        Args:
            model: Quantized model with QuantizedLinear layers.
            config: The QuantConfig used for quantization.
            out_dir: Output directory path.
            source_dir: Original model directory to copy tokenizer files from.

        Returns:
            Path to the output directory.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        tensors: Dict[str, torch.Tensor] = {}
        layer_info: Dict[str, Dict[str, Any]] = {}

        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinear):
                self._save_quantized_layer(name, module, tensors, layer_info)
            elif hasattr(module, "weight") and isinstance(
                module.weight, (torch.nn.Parameter, torch.Tensor)
            ):
                if not isinstance(module, QuantizedLinear):
                    key = f"{name}.weight" if name else "weight"
                    tensors[key] = module.weight.data.to(torch.float16).contiguous()
                    if hasattr(module, "bias") and module.bias is not None:
                        tensors[f"{name}.bias"] = module.bias.data.to(torch.float16).contiguous()

        # Save tensors
        save_file(tensors, out_path / "model.safetensors")

        # Save config
        quant_meta = {
            "bits": config.bits,
            "mode": config.mode.value,
            "granularity": config.granularity.value,
            "group_size": config.group_size,
            "config_name": config.name,
            "layers": layer_info,
        }
        with open(out_path / "quant_config.json", "w") as f:
            json.dump(quant_meta, f, indent=2)

        # Save HuggingFace model config if available
        if hasattr(model, "config") and hasattr(model.config, "save_pretrained"):
            try:
                model.config.save_pretrained(out_path)
            except Exception:
                pass

        # Save tokenizer if provided
        tokenizer = kwargs.get("tokenizer")
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            try:
                tokenizer.save_pretrained(out_path)
            except Exception:
                pass

        # Copy tokenizer and config files if source_dir is provided
        if source_dir is not None:
            self._copy_source_files(Path(source_dir), out_path)

        return out_path

    @staticmethod
    def _save_quantized_layer(
        name: str,
        module: QuantizedLinear,
        tensors: Dict[str, torch.Tensor],
        layer_info: Dict[str, Dict[str, Any]],
    ) -> None:
        """Add one QuantizedLinear's tensors to the output dict."""
        prefix = name
        tensors[f"{prefix}.qweight"] = module.packed_weight.contiguous()
        tensors[f"{prefix}.scales"] = module.scales.contiguous()
        if module.zero_points is not None:
            tensors[f"{prefix}.zeros"] = module.zero_points.contiguous()
        if module.bias_param is not None:
            tensors[f"{prefix}.bias"] = module.bias_param.contiguous()

        layer_info[name] = {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bits": module.bits,
            "group_size": module.group_size,
            "n_groups_per_row": module.n_groups_per_row,
            "mode": module.mode.value,
            "has_bias": module.bias_param is not None,
        }

    @staticmethod
    def _copy_source_files(source_dir: Path, out_path: Path) -> None:
        """Copy config.json and tokenizer files from source directory."""
        files_to_copy = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "generation_config.json",
        ]
        for fname in files_to_copy:
            src = source_dir / fname
            if src.exists():
                shutil.copy2(src, out_path / fname)
