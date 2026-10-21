"""Tests for model size estimation and measurement."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from quantlab.config import Granularity, Mode, QuantConfig
from quantlab.model_size_metric import estimate_size, measure_size
from quantlab.rtn_quantizer import RTNQuantizer


def test_estimate_and_measure_size():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config).eval()

    quant_cfg = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)

    # 1. Estimate
    report_est = estimate_size(model, quant_cfg)
    assert report_est.packed_weight_bytes > 0
    assert report_est.scale_bytes > 0
    assert report_est.zero_point_bytes == 0
    assert report_est.fp16_rest_bytes > 0
    assert report_est.total_bytes > 0
    assert 0 < report_est.reduction_vs_fp16_pct < 100
    assert report_est.effective_bits > 4.0  # slightly above 4 bits due to scales

    # 2. Measure on packed model
    RTNQuantizer(quant_cfg).quantize(model)
    report_meas = measure_size(model)
    assert report_meas.total_bytes > 0
    assert report_meas.packed_weight_bytes == report_est.packed_weight_bytes
    assert report_meas.scale_bytes == report_est.scale_bytes
