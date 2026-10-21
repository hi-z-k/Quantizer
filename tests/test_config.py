"""Tests for config module."""

import pytest
from quantlab.config import Granularity, Mode, QuantConfig, QuantParams


def test_quant_config_names():
    cfg1 = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)
    assert cfg1.name == "int4_sym_group32"

    cfg2 = QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL)
    assert cfg2.name == "int8_sym_channel"

    cfg3 = QuantConfig(bits=4, mode=Mode.ASYMMETRIC, granularity=Granularity.TENSOR)
    assert cfg3.name == "int4_asym_tensor"


def test_quant_config_validation():
    # group_size required for GROUP
    with pytest.raises(ValueError, match="group_size is required"):
        QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=None)

    # group_size must be None for non-GROUP
    with pytest.raises(ValueError, match="group_size must be None"):
        QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL, group_size=32)

    # bits out of range
    with pytest.raises(ValueError, match="bits must be between 1 and 16"):
        QuantConfig(bits=0, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL)

    with pytest.raises(ValueError, match="bits must be between 1 and 16"):
        QuantConfig(bits=17, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL)


def test_quant_config_immutability():
    cfg = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL)
    with pytest.raises(Exception):
        cfg.bits = 8  # FrozenInstanceError


def test_quant_params():
    qp = QuantParams(scale=0.05, zero_point=0, qmin=-7, qmax=7)
    assert qp.scale == 0.05
    assert qp.zero_point == 0
    assert qp.qmin == -7
    assert qp.qmax == 7
