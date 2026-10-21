"""Tests for QuantizedLinear layer and equivalence with SimulatedRTNQuantizer."""

import torch
import torch.nn as nn
from quantlab.config import Granularity, Mode, QuantConfig
from quantlab.rtn_quantizer import RTNQuantizer
from quantlab.simulated_quantizer import SimulatedRTNQuantizer


def test_quantized_linear_forward():
    torch.manual_seed(42)
    linear = nn.Linear(64, 32, bias=True)
    x = torch.randn(2, 64)

    config = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)
    quantizer = RTNQuantizer(config)
    qlinear = quantizer._quantize_layer(linear)

    y = qlinear(x)
    assert y.shape == (2, 32)
    assert torch.isfinite(y).all()


def test_non_divisible_row_width_case6():
    """Case 6: Row width not divisible by group_size (e.g. 70 with group 32)."""
    torch.manual_seed(42)
    linear = nn.Linear(70, 24, bias=True)
    x = torch.randn(3, 70)

    config = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)
    quantizer = RTNQuantizer(config)
    qlinear = quantizer._quantize_layer(linear)

    y = qlinear(x)
    assert y.shape == (3, 24)
    assert torch.isfinite(y).all()


def test_quantized_linear_matches_simulated_case7():
    """Case 7: QuantizedLinear output equals SimulatedRTNQuantizer output within 1e-3 on small random Linear."""
    torch.manual_seed(42)

    configs = [
        QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32),
        QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32),
        QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL),
        QuantConfig(bits=4, mode=Mode.ASYMMETRIC, granularity=Granularity.GROUP, group_size=32),
    ]

    for cfg in configs:
        linear_orig = nn.Linear(64, 32, bias=True)
        # Deep copy for simulated
        linear_sim = nn.Linear(64, 32, bias=True)
        linear_sim.weight.data.copy_(linear_orig.weight.data)
        linear_sim.bias.data.copy_(linear_orig.bias.data)

        # Packed
        rtn = RTNQuantizer(cfg)
        qlinear = rtn._quantize_layer(linear_orig)

        # Simulated
        sim = SimulatedRTNQuantizer(cfg)
        sim._simulate_layer(linear_sim)

        x = torch.randn(4, 64)
        out_packed = qlinear(x)
        out_sim = linear_sim(x)

        max_diff = (out_packed - out_sim).abs().max().item()
        assert max_diff < 1e-3, f"Mismatch for {cfg.name}: max_diff={max_diff}"
