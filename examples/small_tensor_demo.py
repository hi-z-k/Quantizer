"""Small tensor demo for quantlab.

Demonstrates quantizing and dequantizing a small 3x3 weight tensor
from scratch, printing the original weights, scales, quantized codes,
reconstructed weights, and quantization error.
"""

import torch
from quantlab.config import Granularity, Mode, QuantConfig
from quantlab.grouping import group_weight, ungroup_weight
from quantlab.symmetric_params import compute_symmetric_params_batched
from quantlab.linear_dequantizer import LinearDequantizer
from quantlab.bit_packer import BitPacker


def main() -> None:
    print("=" * 60)
    print("quantlab: 3x3 Small Tensor Quantization Demo")
    print("=" * 60)

    # 1. Create a representative 3x3 weight matrix
    torch.manual_seed(42)
    w = torch.tensor(
        [
            [0.2, -0.1, 0.3],
            [8.0, -3.9, 2.0],
            [-0.5, 1.2, -0.8],
        ],
        dtype=torch.float32,
    )
    print(f"\nOriginal 3x3 Weight Tensor:\n{w}")

    # 2. Config: 4-bit symmetric per-channel
    config = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL)
    print(f"\nQuantization Config: {config.name} ({config.bits} bits, mode={config.mode.value})")

    # 3. Group weights (CHANNEL: 1 group per row)
    grouped, info = group_weight(w, config.granularity, config.group_size)

    # 4. Compute symmetric scales
    # qmax = 2^(4-1) - 1 = 7
    scales, qmin, qmax = compute_symmetric_params_batched(grouped, config.bits)
    print(f"Per-channel scales (float16): {scales.tolist()}")
    print(f"Code range: [{qmin}, {qmax}]")

    # 5. Quantize: q = clamp(round(w / scale), qmin, qmax)
    scales_f32 = scales.float().unsqueeze(1)
    codes_signed = torch.round(grouped / scales_f32).clamp(qmin, qmax)
    print(f"\nQuantized Signed Codes:\n{codes_signed}")

    # 6. Dequantize: w_hat = scale * q
    dequantizer = LinearDequantizer()
    w_hat = dequantizer.dequantize(codes_signed, scales_f32, None, config.bits)
    w_hat = ungroup_weight(w_hat, info)
    print(f"\nReconstructed (Dequantized) Weight:\n{w_hat}")

    # 7. Compute errors
    error = (w - w_hat).abs()
    mse = ((w - w_hat) ** 2).mean().item()
    max_err = error.max().item()
    print(f"\nAbsolute Error Matrix:\n{error}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Maximum Absolute Error:   {max_err:.6f}")
    print(f"Theoretical Max Error (scale/2 per row): {(scales.float() / 2).tolist()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
