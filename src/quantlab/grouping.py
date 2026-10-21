"""Weight grouping utility.

Reshapes a 2-D weight matrix [out_features, in_features] into groups
along the input dimension so that each group gets its own scale.

Three granularities:
- TENSOR: flatten the entire matrix into one group.
- CHANNEL: each row is one group (one scale per output neuron).
- GROUP: each row is split into blocks of ``group_size``. If in_features
  is not divisible by group_size, the last block is shorter. We pad by
  repeating the row's last value so min/max see full groups, then cut
  the padding after rounding.

Example (GROUP, group_size=32, in_features=70):
    Row of 70 values → 3 groups: [32, 32, 6].
    The last group is padded to 32 for scale computation, then trimmed back to 6.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from quantlab.config import Granularity


class GroupInfo(NamedTuple):
    """Bookkeeping for reversing the grouping.

    Attributes:
        out_features: Number of output channels (rows).
        in_features: Original input dimension width.
        n_groups_per_row: How many groups each row was split into.
        group_size: Width of each full group.
        padded: Whether the last group was padded.
        original_in_features: Same as in_features (kept for clarity).
    """

    out_features: int
    in_features: int
    n_groups_per_row: int
    group_size: int
    padded: bool
    original_in_features: int


def group_weight(
    weight: torch.Tensor,
    granularity: Granularity,
    group_size: Optional[int] = None,
) -> tuple[torch.Tensor, GroupInfo]:
    """Reshape a weight matrix into groups along the input dimension.

    Args:
        weight: 2-D tensor of shape [out_features, in_features].
        granularity: TENSOR, CHANNEL, or GROUP.
        group_size: Required when granularity is GROUP.

    Returns:
        (grouped, info) where grouped has shape [n_total_groups, group_width]
        and info allows reversing the operation.
    """
    assert weight.ndim == 2, f"Expected 2-D weight, got {weight.ndim}-D"
    out_features, in_features = weight.shape

    if granularity == Granularity.TENSOR:
        return _group_tensor(weight, out_features, in_features)
    elif granularity == Granularity.CHANNEL:
        return _group_channel(weight, out_features, in_features)
    elif granularity == Granularity.GROUP:
        assert group_size is not None
        return _group_blocked(weight, out_features, in_features, group_size)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")


def _group_tensor(
    weight: torch.Tensor, out_features: int, in_features: int
) -> tuple[torch.Tensor, GroupInfo]:
    """One group for the whole matrix."""
    grouped = weight.reshape(1, -1)
    info = GroupInfo(out_features, in_features, 1, out_features * in_features, False, in_features)
    return grouped, info


def _group_channel(
    weight: torch.Tensor, out_features: int, in_features: int
) -> tuple[torch.Tensor, GroupInfo]:
    """One group per row (output channel)."""
    grouped = weight  # already [out_features, in_features]
    info = GroupInfo(out_features, in_features, 1, in_features, False, in_features)
    return grouped, info


def _group_blocked(
    weight: torch.Tensor,
    out_features: int,
    in_features: int,
    group_size: int,
) -> tuple[torch.Tensor, GroupInfo]:
    """Blocks of group_size along each row, with padding if needed."""
    n_groups = (in_features + group_size - 1) // group_size
    padded_width = n_groups * group_size
    needs_pad = padded_width > in_features

    if needs_pad:
        # Pad by repeating the last value in each row
        pad_count = padded_width - in_features
        last_col = weight[:, -1:].expand(-1, pad_count)
        weight_padded = torch.cat([weight, last_col], dim=1)
    else:
        weight_padded = weight

    # Reshape: [out, padded_width] → [out * n_groups, group_size]
    grouped = weight_padded.reshape(out_features * n_groups, group_size)
    info = GroupInfo(out_features, in_features, n_groups, group_size, needs_pad, in_features)
    return grouped, info


def ungroup_weight(
    grouped: torch.Tensor,
    info: GroupInfo,
) -> torch.Tensor:
    """Reverse the grouping, restoring the original [out_features, in_features] shape.

    Args:
        grouped: Tensor from group_weight, shape [n_total_groups, group_width].
        info: The GroupInfo returned by group_weight.

    Returns:
        2-D tensor of shape [out_features, in_features].
    """
    if info.n_groups_per_row == 1 and info.group_size == info.out_features * info.in_features:
        # TENSOR granularity
        return grouped.reshape(info.out_features, info.in_features)

    if info.n_groups_per_row == 1 and not info.padded:
        # CHANNEL granularity
        return grouped

    # GROUP granularity: reshape back and trim padding
    padded_width = info.n_groups_per_row * info.group_size
    full = grouped.reshape(info.out_features, padded_width)
    return full[:, : info.in_features]
