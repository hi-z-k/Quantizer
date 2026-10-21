"""Tests for quantization parameter computation and math invariants."""

import math
import torch
import pytest
from quantlab.asymmetric_params import (
    AsymmetricParamsStrategy,
    compute_asymmetric_params_batched,
)
from quantlab.symmetric_params import (
    SymmetricParamsStrategy,
    compute_symmetric_params_batched,
)


def test_hand_calc_case1_asymmetric_4bit():
    """Case 1: Asymmetric 4-bit per-tensor on [-0.8, 0, 0.55, 1.75, 2.2]:

    scale 0.2, zero_point 4, codes [0, 4, 7, 13, 15], read-back [-0.8, 0, 0.6, 1.8, 2.2].
    """
    w = torch.tensor([[-0.8, 0.0, 0.55, 1.75, 2.2]], dtype=torch.float32)
    scales, zero_points, qmin, qmax = compute_asymmetric_params_batched(w, bits=4)

    # Scale should be 0.2
    assert abs(scales.item() - 0.2) < 1e-4
    # Zero point should be 4
    assert zero_points.item() == 4
    assert qmin == 0
    assert qmax == 15

    # Codes
    scale_f32 = scales.float()
    codes = torch.round(w / scale_f32) + zero_points.float()
    codes = codes.clamp(qmin, qmax)
    expected_codes = torch.tensor([[0, 4, 7, 13, 15]], dtype=torch.float32)
    assert torch.equal(codes, expected_codes)

    # Read-back: float16 representation of 0.2 is 0.199951, so read-back is accurate to ~5e-4
    read_back = scale_f32 * (codes - zero_points.float())
    expected_read_back = torch.tensor([[-0.8, 0.0, 0.6, 1.8, 2.2]], dtype=torch.float32)
    assert torch.allclose(read_back, expected_read_back, atol=1e-3)


def test_hand_calc_case2_symmetric_8bit():
    """Case 2: Symmetric 8-bit on [[0.2, -0.1, 0.3], [8.0, -3.9, 2.0]]:

    per-tensor scale 8/127, row 0 codes 3, -2, 5;
    per-channel row 0 scale 0.3/127, codes 85, -42, 127.
    """
    w = torch.tensor([[0.2, -0.1, 0.3], [8.0, -3.9, 2.0]], dtype=torch.float32)

    # 1. Per-tensor: shape (1, 6)
    w_flat = w.reshape(1, -1)
    scales_tensor, qmin, qmax = compute_symmetric_params_batched(w_flat, bits=8)
    expected_scale_tensor = torch.tensor(8.0 / 127.0, dtype=torch.float16)
    assert scales_tensor.item() == expected_scale_tensor.item()

    row0_codes_tensor = torch.round(w[0] / scales_tensor.float()).clamp(qmin, qmax)
    assert row0_codes_tensor.tolist() == [3, -2, 5]

    # 2. Per-channel: shape (2, 3)
    scales_channel, qmin, qmax = compute_symmetric_params_batched(w, bits=8)
    expected_scale_row0 = torch.tensor(0.3 / 127.0, dtype=torch.float16)
    assert scales_channel[0].item() == expected_scale_row0.item()

    row0_codes_channel = torch.round(w[0] / scales_channel[0].float()).clamp(qmin, qmax)
    assert row0_codes_channel.tolist() == [85, -42, 127]


def test_hand_calc_case4_error_bound():
    """Case 4: Error bound: for in-range values, abs(w - w_hat) is at most scale / 2 (symmetric and asymmetric)."""
    torch.manual_seed(42)
    # Test symmetric
    w = torch.randn(10, 64) * 2.0
    scales_sym, qmin_s, qmax_s = compute_symmetric_params_batched(w, bits=4)
    scales_sym_f32 = scales_sym.float().unsqueeze(1)
    q_sym = torch.round(w / scales_sym_f32).clamp(qmin_s, qmax_s)
    w_hat_sym = scales_sym_f32 * q_sym

    # For in-range values (not clamped), error <= scale / 2
    in_range_sym = (q_sym > qmin_s) & (q_sym < qmax_s)
    error_sym = (w[in_range_sym] - w_hat_sym[in_range_sym]).abs()
    bound_sym = (scales_sym_f32.expand_as(w)[in_range_sym] / 2.0) + 1e-4
    assert torch.all(error_sym <= bound_sym)

    # Test asymmetric
    scales_asym, zp, qmin_a, qmax_a = compute_asymmetric_params_batched(w, bits=4)
    scales_asym_f32 = scales_asym.float().unsqueeze(1)
    zp_f32 = zp.float().unsqueeze(1)
    q_asym = (torch.round(w / scales_asym_f32) + zp_f32).clamp(qmin_a, qmax_a)
    w_hat_asym = scales_asym_f32 * (q_asym - zp_f32)

    in_range_asym = (q_asym > qmin_a) & (q_asym < qmax_a)
    error_asym = (w[in_range_asym] - w_hat_asym[in_range_asym]).abs()
    bound_asym = (scales_asym_f32.expand_as(w)[in_range_asym] / 2.0) + 1e-4
    assert torch.all(error_asym <= bound_asym)


def test_hand_calc_case5_edge_cases():
    """Case 5: All-positive group with asymmetric keeps zero exactly representable.

    Constant and all-zero groups give no NaN.
    """
    # 1. All-positive group
    w_pos = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    scales, zp, qmin, qmax = compute_asymmetric_params_batched(w_pos, bits=4)
    # Check that real 0.0 is exactly representable: w_hat = scale * (zp - zp) = 0.0
    zero_code = zp.item()
    assert qmin <= zero_code <= qmax
    w_hat_at_zero = scales.item() * (zero_code - zp.item())
    assert w_hat_at_zero == 0.0

    # 2. All-zero group
    w_zero = torch.zeros(2, 8, dtype=torch.float32)
    scales_sym_z, _, _ = compute_symmetric_params_batched(w_zero, bits=4)
    assert not torch.isnan(scales_sym_z).any()
    assert scales_sym_z[0].item() == 1.0  # fallback scale is 1.0

    scales_asym_z, zp_z, _, _ = compute_asymmetric_params_batched(w_zero, bits=4)
    assert not torch.isnan(scales_asym_z).any()
    assert not torch.isnan(zp_z.float()).any()

    # 3. Constant group
    w_const = torch.full((2, 8), 5.0, dtype=torch.float32)
    scales_c, zp_c, _, _ = compute_asymmetric_params_batched(w_const, bits=4)
    assert not torch.isnan(scales_c).any()
    assert not torch.isnan(zp_c.float()).any()
