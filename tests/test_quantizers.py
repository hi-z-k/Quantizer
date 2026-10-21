"""Tests for RTNQuantizer, SimulatedRTNQuantizer, and WeightSnapshot."""

import copy
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from quantlab.config import Granularity, Mode, QuantConfig
from quantlab.rtn_quantizer import RTNQuantizer
from quantlab.simulated_quantizer import SimulatedRTNQuantizer
from quantlab.snapshot import WeightSnapshot


def test_weight_snapshot():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config)

    # Take snapshot
    snapshot = WeightSnapshot().take(model)
    assert len(snapshot.weight_names) > 0

    # Modify weights
    for name, module in model.named_modules():
        if hasattr(module, "weight") and name in snapshot.weight_names:
            module.weight.data.zero_()

    # Restore
    snapshot.restore(model)

    # Check restored
    for name, module in model.named_modules():
        if hasattr(module, "weight") and name in snapshot.weight_names:
            orig = snapshot.get_original(name)
            assert torch.equal(module.weight.data.cpu(), orig)


def test_tiny_llama_equivalence_case7():
    """Case 7: QuantizedLinear output equals SimulatedRTNQuantizer output on a tiny LlamaConfig model within 1e-3."""
    torch.manual_seed(42)
    config_llama = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model_orig = LlamaForCausalLM(config_llama).eval()

    model_sim = copy.deepcopy(model_orig)
    model_rtn = copy.deepcopy(model_orig)

    quant_cfg = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)

    # Simulate
    sim = SimulatedRTNQuantizer(quant_cfg)
    sim.quantize(model_sim)

    # Packed
    rtn = RTNQuantizer(quant_cfg)
    rtn.quantize(model_rtn)

    # Forward pass comparison
    input_ids = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out_sim = model_sim(input_ids).logits
        out_rtn = model_rtn(input_ids).logits

    diff = (out_sim - out_rtn).abs().max().item()
    assert diff < 1e-3, f"Max difference was {diff}, expected < 1e-3"
