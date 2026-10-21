"""Tests for grouping module."""

import torch
from quantlab.config import Granularity
from quantlab.grouping import group_weight, ungroup_weight


def test_tensor_grouping():
    w = torch.randn(4, 8)
    grouped, info = group_weight(w, Granularity.TENSOR)
    assert grouped.shape == (1, 32)
    restored = ungroup_weight(grouped, info)
    assert torch.equal(w, restored)


def test_channel_grouping():
    w = torch.randn(5, 10)
    grouped, info = group_weight(w, Granularity.CHANNEL)
    assert grouped.shape == (5, 10)
    restored = ungroup_weight(grouped, info)
    assert torch.equal(w, restored)


def test_group_blocking_divisible():
    w = torch.randn(4, 64)
    grouped, info = group_weight(w, Granularity.GROUP, group_size=32)
    assert grouped.shape == (8, 32)
    assert not info.padded
    restored = ungroup_weight(grouped, info)
    assert torch.equal(w, restored)


def test_group_blocking_non_divisible_case6():
    # Exact case 6: row width not divisible by group_size (e.g. 70 with group 32)
    w = torch.randn(4, 70)
    grouped, info = group_weight(w, Granularity.GROUP, group_size=32)
    # 70 // 32 = 2 groups of 32, plus 1 group of 6 -> 3 groups per row * 4 rows = 12 groups of 32
    assert grouped.shape == (12, 32)
    assert info.padded
    restored = ungroup_weight(grouped, info)
    assert restored.shape == (4, 70)
    assert torch.equal(w, restored)
