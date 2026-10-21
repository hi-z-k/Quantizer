"""Perplexity metric for language models.

Perplexity = exp(mean negative log-likelihood) over all tokens.

Uses non-overlapping windows from a TextDataset. Each window is fed to the
model, and the cross-entropy loss is computed over all positions except the
first (which has no previous context).

Computation is done in float32 for numerical stability.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from quantlab.interfaces import Metric, TextDataset


class PerplexityMetric(Metric):
    """Measures perplexity of a language model on a text dataset.

    Args:
        batch_size: Number of windows to process at once (default 8).

    Example:
        >>> metric = PerplexityMetric(batch_size=4)
        >>> ppl = metric.measure(model, dataset)
        >>> print(f"Perplexity: {ppl:.2f}")
    """

    def __init__(self, batch_size: int = 8) -> None:
        self.batch_size = batch_size

    @torch.no_grad()
    def measure(self, model: torch.nn.Module, dataset: TextDataset) -> float:  # type: ignore[override]
        """Compute perplexity over the dataset.

        Args:
            model: A causal language model.
            dataset: A TextDataset providing token windows.

        Returns:
            Perplexity as a float. Returns float('nan') if the loss is not finite.
        """
        model.eval()
        device = next(model.parameters()).device

        total_loss = 0.0
        total_tokens = 0

        for start in range(0, len(dataset), self.batch_size):
            end = min(start + self.batch_size, len(dataset))
            batch = torch.stack([dataset[i] for i in range(start, end)]).to(device)

            # Forward pass
            outputs = model(batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Shift: predict token t+1 from position t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            # Cross-entropy in float32
            loss = F.cross_entropy(
                shift_logits.float().reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="sum",
            )

            n_tokens = shift_labels.numel()
            total_loss += loss.item()
            total_tokens += n_tokens

        if total_tokens == 0:
            return float("nan")

        avg_loss = total_loss / total_tokens
        if not math.isfinite(avg_loss):
            return float("nan")

        return math.exp(avg_loss)
