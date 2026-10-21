"""Tests for linear dequantizer."""

import torch
from quantlab.linear_dequantizer import LinearDequantizer


def test_symmetric_dequantize():
    dequantizer = LinearDequantizer()
    codes = torch.tensor([3, -2, 5], dtype=torch.float32)
    scale = torch.tensor([0.1], dtype=torch.float32)

    w_hat = dequantizer.dequantize(codes, scale, zero_point=None, bits=8)
    expected = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float32)
    assert torch.allclose(w_hat, expected, atol=1e-5)


def test_asymmetric_dequantize():
    dequantizer = LinearDequantizer()
    codes = torch.tensor([0, 4, 7, 13, 15], dtype=torch.float32)
    scale = torch.tensor([0.2], dtype=torch.float32)
    zp = torch.tensor([4.0], dtype=torch.float32)

    w_hat = dequantizer.dequantize(codes, scale, zero_point=zp, bits=4)
    expected = torch.tensor([-0.8, 0.0, 0.6, 1.8, 2.2], dtype=torch.float32)
    assert torch.allclose(w_hat, expected, atol=1e-5)
