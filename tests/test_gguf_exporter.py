"""Tests for GGUF export and verification (Case 8)."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import gguf
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

from quantlab.config import Granularity, Mode, QuantConfig
from quantlab.gguf_exporter import (
    GGUFExporter,
    dequantize_block_q8_0,
    dequantize_block_q4_0,
    quantize_block_q8_0,
    quantize_block_q4_0,
)


def create_synthetic_f16_gguf(path: Path) -> None:
    """Build a tiny synthetic F16 GGUF template."""
    writer = GGUFWriter(str(path), arch="llama")
    writer.add_name("tiny-llama-test")

    # Linear weight that will be quantized: shape [32, 64] = 2048 weights
    np.random.seed(42)
    w_linear = np.random.randn(32, 64).astype(np.float16)
    writer.add_tensor("blk.0.attn_q.weight", w_linear)

    # Embedding weight that should stay F16: shape [10, 64]
    w_emb = np.random.randn(10, 64).astype(np.float16)
    writer.add_tensor("token_embd.weight", w_emb)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_hand_calc_case8_gguf_export_and_dequantize():
    """Case 8: Build synthetic F16 GGUF, export Q8_0 and Q4_0, reopen, dequantize by hand and compare."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        template_file = tmp_path / "template.gguf"
        create_synthetic_f16_gguf(template_file)

        exporter = GGUFExporter(template_file)

        # 1. Test Q8_0 Export
        cfg_q8 = QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)
        out_q8 = tmp_path / "model_q8.gguf"
        exporter.export(cfg_q8, out_q8)
        assert out_q8.exists()

        reader_q8 = GGUFReader(str(out_q8))
        assert len(reader_q8.tensors) == 2
        tensor_dict_q8 = {t.name: t for t in reader_q8.tensors}
        assert tensor_dict_q8["blk.0.attn_q.weight"].tensor_type == GGMLQuantizationType.Q8_0
        assert tensor_dict_q8["token_embd.weight"].tensor_type == GGMLQuantizationType.F16

        # Hand-dequantize Q8_0 block 0
        raw_bytes_q8 = tensor_dict_q8["blk.0.attn_q.weight"].data.tobytes()
        scale_hand_q8 = np.frombuffer(raw_bytes_q8[:2], dtype="<f2")[0]
        qs_hand_q8 = np.frombuffer(raw_bytes_q8[2:34], dtype=np.int8)
        w_hand_block0_q8 = float(scale_hand_q8) * qs_hand_q8.astype(np.float32)

        # Compare with library dequantizer
        w_lib_q8 = dequantize_block_q8_0(raw_bytes_q8, count=32 * 64)
        assert np.allclose(w_hand_block0_q8, w_lib_q8[:32])

        # 2. Test Q4_0 Export
        cfg_q4 = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)
        out_q4 = tmp_path / "model_q4.gguf"
        exporter.export(cfg_q4, out_q4)
        assert out_q4.exists()

        reader_q4 = GGUFReader(str(out_q4))
        tensor_dict_q4 = {t.name: t for t in reader_q4.tensors}
        assert tensor_dict_q4["blk.0.attn_q.weight"].tensor_type == GGMLQuantizationType.Q4_0
        assert tensor_dict_q4["token_embd.weight"].tensor_type == GGMLQuantizationType.F16

        # Hand-dequantize Q4_0 block 0 (18 bytes)
        raw_bytes_q4 = tensor_dict_q4["blk.0.attn_q.weight"].data.tobytes()
        scale_hand_q4 = np.frombuffer(raw_bytes_q4[:2], dtype="<f2")[0]
        packed_qs = np.frombuffer(raw_bytes_q4[2:18], dtype=np.uint8)
        low_nibbles = (packed_qs & 0x0F).astype(np.float32) - 8.0
        high_nibbles = ((packed_qs >> 4) & 0x0F).astype(np.float32) - 8.0
        w_hand_block0_q4 = float(scale_hand_q4) * np.concatenate([low_nibbles, high_nibbles])

        # Compare with library dequantizer
        w_lib_q4 = dequantize_block_q4_0(raw_bytes_q4, count=32 * 64)
        assert np.allclose(w_hand_block0_q4, w_lib_q4[:32])


def test_gguf_exporter_invalid_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        template_file = Path(tmpdir) / "template.gguf"
        create_synthetic_f16_gguf(template_file)
        exporter = GGUFExporter(template_file)

        # Channel granularity unsupported for GGUF
        bad_cfg = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL)
        with pytest.raises(ValueError):
            exporter.export(bad_cfg, Path(tmpdir) / "out.gguf")
