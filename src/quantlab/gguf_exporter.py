"""GGUF exporter for quantlab models.

Exports models to GGUF format using a full-precision (F16) template file.
Reads the template with gguf.GGUFReader, quantizes the target linear weights
using quantlab's own block quantization, and writes the output with gguf.GGUFWriter.

Supports:
- int8 symmetric group 32 (Q8_0 layout: 2 bytes fp16 scale + 32 int8 weights = 34 bytes)
- int4 symmetric group 32 (Q4_0 layout: 2 bytes fp16 scale + 16 packed uint8 bytes = 18 bytes)

Only linear projection weights (attn_q, attn_k, attn_v, attn_output, ffn_gate,
ffn_up, ffn_down) are quantized. Token embeddings and output heads remain F16.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Set

import numpy as np
import gguf
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

from quantlab.config import Granularity, Mode, QuantConfig
from quantlab.interfaces import TemplateExporter


# Projection weight names to quantize inside transformer blocks
QUANT_LAYER_NAMES: Set[str] = {
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
}


def quantize_block_q8_0(weights: np.ndarray) -> np.ndarray:
    """Quantize 1D float weights into GGML Q8_0 block format.

    Per 32 weights:
    - 1 float16 scale d = max(abs(w)) / 127
    - 32 int8 codes q = clamp(round(w / d), -127, 127)
    Total: 34 bytes per block.

    Args:
        weights: 1D float numpy array, size must be a multiple of 32.

    Returns:
        Structured numpy array of dtype [('d', '<f2'), ('qs', 'i1', (32,))]
    """
    assert weights.size % 32 == 0, f"Weight size {weights.size} must be multiple of 32"
    blocks = weights.reshape(-1, 32).astype(np.float32)
    abs_max = np.max(np.abs(blocks), axis=1)
    scales = np.where(abs_max == 0, 1.0, abs_max / 127.0).astype(np.float16)

    scales_f32 = scales.astype(np.float32)[:, None]
    codes = np.clip(np.round(blocks / scales_f32), -127, 127).astype(np.int8)

    block_dtype = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
    packed = np.empty(len(blocks), dtype=block_dtype)
    packed["d"] = scales
    packed["qs"] = codes
    return packed


def quantize_block_q4_0(weights: np.ndarray) -> np.ndarray:
    """Quantize 1D float weights into GGML Q4_0 block format.

    Per 32 weights:
    - 1 float16 scale d = max(abs(w)) / 7
    - 16 bytes where byte j holds code[j] in low nibble and code[j+16] in high nibble
      where codes = q + 8 (range 1 to 15, or 0 if -8)
    Total: 18 bytes per block.

    Args:
        weights: 1D float numpy array, size must be a multiple of 32.

    Returns:
        Structured numpy array of dtype [('d', '<f2'), ('qs', 'u1', (16,))]
    """
    assert weights.size % 32 == 0, f"Weight size {weights.size} must be multiple of 32"
    blocks = weights.reshape(-1, 32).astype(np.float32)
    abs_max = np.max(np.abs(blocks), axis=1)
    scales = np.where(abs_max == 0, 1.0, abs_max / 7.0).astype(np.float16)

    scales_f32 = scales.astype(np.float32)[:, None]
    q = np.clip(np.round(blocks / scales_f32), -7, 7).astype(np.int32)
    codes = (q + 8).astype(np.uint8)

    low_nibbles = codes[:, :16] & 0x0F
    high_nibbles = (codes[:, 16:] & 0x0F) << 4
    packed_bytes = (low_nibbles | high_nibbles).astype(np.uint8)

    block_dtype = np.dtype([("d", "<f2"), ("qs", "u1", (16,))])
    packed = np.empty(len(blocks), dtype=block_dtype)
    packed["d"] = scales
    packed["qs"] = packed_bytes
    return packed


def dequantize_block_q8_0(packed_bytes: bytes, count: int) -> np.ndarray:
    """Dequantize Q8_0 blocks back to float32 values."""
    assert count % 32 == 0
    n_blocks = count // 32
    block_dtype = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
    blocks = np.frombuffer(packed_bytes, dtype=block_dtype, count=n_blocks)
    d = blocks["d"].astype(np.float32)[:, None]
    qs = blocks["qs"].astype(np.float32)
    return (d * qs).reshape(-1)


def dequantize_block_q4_0(packed_bytes: bytes, count: int) -> np.ndarray:
    """Dequantize Q4_0 blocks back to float32 values."""
    assert count % 32 == 0
    n_blocks = count // 32
    block_dtype = np.dtype([("d", "<f2"), ("qs", "u1", (16,))])
    blocks = np.frombuffer(packed_bytes, dtype=block_dtype, count=n_blocks)
    d = blocks["d"].astype(np.float32)[:, None]
    qs = blocks["qs"]

    low = (qs & 0x0F).astype(np.float32) - 8.0
    high = ((qs >> 4) & 0x0F).astype(np.float32) - 8.0
    codes = np.concatenate([low, high], axis=1)
    return (d * codes).reshape(-1)


class GGUFExporter(TemplateExporter):
    """Exports model weights to GGUF format using an F16 template.

    Args:
        template_gguf: Path to the unquantized F16 GGUF template file.
    """

    def __init__(self, template_gguf: str | Path) -> None:
        self.template_path = Path(template_gguf)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template GGUF not found: {self.template_path}")

    def export(self, config: QuantConfig, out_path: str | Path) -> Path:
        """Export quantized model to GGUF.

        Args:
            config: Quantization config. Must be Q8_0 or Q4_0.
            out_path: Destination path for the new .gguf file.

        Returns:
            Path to the written GGUF file.

        Raises:
            ValueError: If config is not Q8_0 or Q4_0.
        """
        is_q8_0 = (
            config.bits == 8
            and config.mode == Mode.SYMMETRIC
            and config.granularity == Granularity.GROUP
            and config.group_size == 32
        )
        is_q4_0 = (
            config.bits == 4
            and config.mode == Mode.SYMMETRIC
            and config.granularity == Granularity.GROUP
            and config.group_size == 32
        )

        if not (is_q8_0 or is_q4_0):
            raise ValueError(
                f"Unsupported GGUF configuration: {config.name}. "
                "Only int8 symmetric group 32 (Q8_0) and int4 symmetric group 32 (Q4_0) "
                "are supported by GGUFExporter."
            )

        target_ggml_type = GGMLQuantizationType.Q8_0 if is_q8_0 else GGMLQuantizationType.Q4_0
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        reader = GGUFReader(str(self.template_path))

        # Determine architecture
        arch = "llama"
        for field in reader.fields.values():
            if field.name == "general.architecture":
                val = field.parts[field.data[0]]
                if isinstance(val, (bytes, bytearray)):
                    arch = val.decode("utf-8")
                elif isinstance(val, str):
                    arch = val
                break

        writer = GGUFWriter(str(out_path), arch=arch)

        # Copy metadata
        for field in reader.fields.values():
            if field.name.startswith("GGUF.") or field.name == "general.architecture":
                continue
            val_type = field.types[0]
            if val_type == gguf.GGUFValueType.ARRAY:
                sub_type = field.types[1]
                if sub_type == gguf.GGUFValueType.STRING:
                    val = [bytes(field.parts[idx]).decode("utf-8") for idx in field.data]
                else:
                    val = [field.parts[idx].tolist()[0] for idx in field.data]
                writer.add_key_value(field.name, val, val_type, sub_type)
            elif val_type == gguf.GGUFValueType.STRING:
                val = bytes(field.parts[field.data[0]]).decode("utf-8")
                writer.add_key_value(field.name, val, val_type)
            else:
                val = field.parts[field.data[0]].tolist()[0]
                writer.add_key_value(field.name, val, val_type)

        # Copy or quantize tensors
        for tensor in reader.tensors:
            name = tensor.name
            should_quantize = self._should_quantize_tensor(name)

            if should_quantize and tensor.data.size % 32 == 0:
                weights = tensor.data.astype(np.float32)
                type_size = 34 if is_q8_0 else 18

                if is_q8_0:
                    packed = quantize_block_q8_0(weights)
                else:
                    packed = quantize_block_q4_0(weights)

                # Reshape uint8 buffer to match row-based byte layout expected by GGUFWriter
                # For [rows, cols], byte shape is [rows, (cols // 32) * type_size]
                if tensor.data.ndim >= 2:
                    rows = int(np.prod(tensor.data.shape[:-1]))
                    bytes_per_row = (tensor.data.shape[-1] // 32) * type_size
                    packed_uint8 = packed.view(np.uint8).reshape(rows, bytes_per_row)
                else:
                    packed_uint8 = packed.view(np.uint8).reshape(-1)

                # Add tensor with raw bytes and GGML quantization type
                writer.add_tensor(
                    name=name,
                    tensor=packed_uint8,
                    raw_dtype=target_ggml_type,
                )
            else:
                # Copy unchanged (F16/F32, embeddings, head, norms)
                writer.add_tensor(
                    name=name,
                    tensor=tensor.data,
                    raw_dtype=tensor.tensor_type,
                )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        # Verification step: reopen with GGUFReader and verify counts and types
        verify_reader = GGUFReader(str(out_path))
        if len(verify_reader.tensors) != len(reader.tensors):
            raise RuntimeError(
                f"Export verification failed: expected {len(reader.tensors)} tensors, "
                f"got {len(verify_reader.tensors)}"
            )

        for vt in verify_reader.tensors:
            if self._should_quantize_tensor(vt.name) and vt.data.size % 32 == 0:
                if vt.tensor_type != target_ggml_type:
                    raise RuntimeError(
                        f"Tensor {vt.name} expected type {target_ggml_type}, got {vt.tensor_type}"
                    )

        return out_path

    @staticmethod
    def _should_quantize_tensor(name: str) -> bool:
        """Check if tensor should be quantized to block format."""
        # Embeddings and lm_head remain F16
        if "token_embd" in name or "output.weight" in name or "output_norm" in name:
            return False
        if not name.endswith(".weight"):
            return False
        # Must be in blk.N
        if not name.startswith("blk."):
            return False
        # Match one of the projection names
        return any(qname in name for qname in QUANT_LAYER_NAMES)
