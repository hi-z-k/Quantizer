"""Tests for perplexity, weight error, and dataset metrics."""

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from quantlab.config import Granularity, Mode, QuantConfig
from quantlab.interfaces import TextDataset
from quantlab.perplexity_metric import PerplexityMetric
from quantlab.simulated_quantizer import SimulatedRTNQuantizer
from quantlab.snapshot import WeightSnapshot
from quantlab.weight_error_metric import WeightErrorMetric


class DummyDataset(TextDataset):
    """Dummy dataset providing synthetic token windows for testing."""

    def __init__(self, vocab_size: int = 100, n_windows: int = 4, window: int = 16):
        torch.manual_seed(42)
        self.tokens = torch.randint(0, vocab_size, (n_windows, window))

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tokens[idx]


def test_weight_error_metric():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config).eval()
    snapshot = WeightSnapshot().take(model)

    quant_cfg = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)
    SimulatedRTNQuantizer(quant_cfg).quantize(model)

    metric = WeightErrorMetric(snapshot)
    report = metric.measure(model)

    assert "mse" in report
    assert "sqnr_db" in report
    assert "worst_layer" in report
    assert "worst_layer_sqnr_db" in report
    assert report["mse"] > 0
    assert report["sqnr_db"] > 0
    assert len(report["worst_layer"]) > 0


def test_perplexity_metric():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config).eval()
    dataset = DummyDataset(vocab_size=100, n_windows=4, window=16)

    metric = PerplexityMetric(batch_size=2)
    ppl = metric.measure(model, dataset)

    assert ppl > 0
    assert not torch.isnan(torch.tensor(ppl))
