"""Tests for native exporter and loader."""

import tempfile
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from quantlab.config import Granularity, Mode, QuantConfig
from quantlab.native_exporter import NativeExporter
from quantlab.native_loader import load_native
from quantlab.rtn_quantizer import RTNQuantizer


def test_native_export_and_load():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config).half().eval()

    quant_cfg = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)
    RTNQuantizer(quant_cfg).quantize(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir)
        # Also save config.json in tmpdir so AutoConfig can read it
        config.save_pretrained(out_path)

        exporter = NativeExporter()
        exported_dir = exporter.export(model, quant_cfg, out_path)
        assert (exported_dir / "model.safetensors").exists()
        assert (exported_dir / "quant_config.json").exists()

        # Reload with load_native (mock tokenizer not needed if config.json present)
        # We test loading tensors and running forward pass
        from quantlab.native_loader import _resolve_path, _replace_with_quantized, _load_tensors
        from safetensors.torch import load_file
        import json

        with open(exported_dir / "quant_config.json") as f:
            meta = json.load(f)

        with torch.device("meta"):
            reloaded_model = LlamaForCausalLM(config)

        _replace_with_quantized(reloaded_model, meta["layers"], quant_cfg)
        reloaded_model.to_empty(device="cpu")
        tensors = load_file(exported_dir / "model.safetensors")
        _load_tensors(reloaded_model, tensors, meta["layers"], device="cpu")

        # Forward pass on reloaded model
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            out = reloaded_model(x).logits
            orig_out = model(x).logits

        assert torch.allclose(out.float(), orig_out.float(), atol=1e-2)
