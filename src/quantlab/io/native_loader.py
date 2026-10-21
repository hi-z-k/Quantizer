"""Native loader: loads a quantized model from safetensors + quant_config.json.

Builds an empty model from the saved config on the meta device, swaps
nn.Linear layers for QuantizedLinear, loads the tensors, and returns
(model, tokenizer). The full-precision model is never loaded.

Accepts a local folder or a HuggingFace repo ID with an optional subfolder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file

from quantlab.config import Granularity, Mode, QuantConfig
from quantlab.quantized_linear import QuantizedLinear


def load_native(
    path_or_repo_id: str,
    subfolder: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[nn.Module, object]:
    """Load a natively exported quantized model.

    Args:
        path_or_repo_id: Local directory path or HuggingFace repo ID.
        subfolder: Optional subfolder within the repo.
        device: Device to load tensors onto (default 'cpu').

    Returns:
        (model, tokenizer) tuple.
    """
    model_dir = _resolve_path(path_or_repo_id, subfolder)

    # Load quant config
    with open(model_dir / "quant_config.json") as f:
        quant_meta = json.load(f)

    # Build QuantConfig
    granularity = Granularity(quant_meta["granularity"])
    mode = Mode(quant_meta["mode"])
    config = QuantConfig(
        bits=quant_meta["bits"],
        mode=mode,
        granularity=granularity,
        group_size=quant_meta.get("group_size"),
    )

    # Load the model architecture using transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_config = AutoConfig.from_pretrained(model_dir)

    # Create model on meta device (no memory allocated)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)

    # Replace Linear layers with QuantizedLinear based on saved info
    layer_info = quant_meta["layers"]
    _replace_with_quantized(model, layer_info, config)

    # Materialize model parameters/buffers (e.g. RoPE inv_freq) on target device
    # Note: Linear layers are already swapped with QuantizedLinear so full FP16 weights are never allocated
    model.to_empty(device=device)

    # Load tensors from safetensors
    tensors = load_file(model_dir / "model.safetensors", device=device)
    _load_tensors(model, tensors, layer_info, device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.eval()
    return model, tokenizer


def _resolve_path(path_or_repo_id: str, subfolder: Optional[str]) -> Path:
    """Resolve a local path or download from HuggingFace."""
    local = Path(path_or_repo_id)
    if local.exists():
        if subfolder:
            return local / subfolder
        return local

    # Try HuggingFace
    from huggingface_hub import snapshot_download

    downloaded = snapshot_download(
        repo_id=path_or_repo_id,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
    )
    path = Path(downloaded)
    if subfolder:
        path = path / subfolder
    return path


def _replace_with_quantized(
    model: nn.Module,
    layer_info: dict,
    config: QuantConfig,
) -> None:
    """Replace nn.Linear layers that were quantized with QuantizedLinear."""
    replacements = []
    for name, module in model.named_modules():
        if name in layer_info:
            info = layer_info[name]
            # Build a layer-specific config for the right granularity
            layer_config = QuantConfig(
                bits=info["bits"],
                mode=Mode(info["mode"]),
                granularity=config.granularity,
                group_size=info.get("group_size", config.group_size),
            )
            qlinear = QuantizedLinear(
                in_features=info["in_features"],
                out_features=info["out_features"],
                config=layer_config,
                bias=info.get("has_bias", False),
            )
            parts = name.rsplit(".", 1)
            if len(parts) == 1:
                replacements.append((model, parts[0], qlinear))
            else:
                parent = dict(model.named_modules())[parts[0]]
                replacements.append((parent, parts[1], qlinear))

    for parent, attr, qlinear in replacements:
        setattr(parent, attr, qlinear)


def _load_tensors(
    model: nn.Module,
    tensors: dict,
    layer_info: dict,
    device: str,
) -> None:
    """Load saved tensors into the model."""
    for name, module in model.named_modules():
        if name in layer_info:
            # QuantizedLinear
            prefix = name
            if f"{prefix}.qweight" in tensors:
                module.packed_weight = tensors[f"{prefix}.qweight"].to(device)
            if f"{prefix}.scales" in tensors:
                module.scales = tensors[f"{prefix}.scales"].to(device)
            if f"{prefix}.zeros" in tensors:
                module.zero_points = tensors[f"{prefix}.zeros"].to(device)
            if f"{prefix}.bias" in tensors:
                module.bias_param = tensors[f"{prefix}.bias"].to(device)
        else:
            # Regular module: load weight and bias
            key_w = f"{name}.weight"
            key_b = f"{name}.bias"
            if key_w in tensors and hasattr(module, "weight"):
                module.weight = nn.Parameter(tensors[key_w].to(device))
            if key_b in tensors and hasattr(module, "bias"):
                module.bias = nn.Parameter(tensors[key_b].to(device))
