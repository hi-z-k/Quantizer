"""WikiText-2 dataset for language model evaluation.

Loads the WikiText-2 test split, tokenizes it, and provides non-overlapping
windows of a fixed length for perplexity measurement.

Uses only the first ``max_tokens`` tokens of the test split (default 100,000)
to keep evaluation fast and consistent across experiments.
"""

from __future__ import annotations

import torch
from datasets import load_dataset

from quantlab.interfaces import TextDataset


class WikiText2Dataset(TextDataset):
    """Tokenized WikiText-2 test set in non-overlapping windows.

    Args:
        tokenizer: A HuggingFace tokenizer.
        split: Dataset split (default 'test').
        max_tokens: Maximum tokens to use (default 100,000).
        window: Window length in tokens (default 1024).

    Example:
        >>> from transformers import AutoTokenizer
        >>> tok = AutoTokenizer.from_pretrained('gpt2')
        >>> ds = WikiText2Dataset(tok, max_tokens=10000, window=512)
        >>> len(ds)  # number of windows
        19
        >>> ds[0].shape
        torch.Size([512])
    """

    def __init__(
        self,
        tokenizer: object,
        split: str = "test",
        max_tokens: int = 100_000,
        window: int = 1024,
    ) -> None:
        self.window = window

        # Load and concatenate all text (try canonical Salesforce/wikitext then fallback to wikitext)
        try:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        except Exception:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        full_text = "\n\n".join(dataset["text"])

        # Tokenize
        encoded = tokenizer(full_text, return_tensors="pt")  # type: ignore
        all_tokens = encoded["input_ids"].squeeze(0)  # [total_tokens]

        # Truncate to max_tokens
        if all_tokens.numel() > max_tokens:
            all_tokens = all_tokens[:max_tokens]

        # Split into non-overlapping windows
        n_windows = all_tokens.numel() // window
        usable = n_windows * window
        self._tokens = all_tokens[:usable].reshape(n_windows, window)

    def __len__(self) -> int:
        """Number of non-overlapping windows."""
        return self._tokens.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get one window of token IDs.

        Args:
            idx: Window index.

        Returns:
            1-D tensor of shape [window].
        """
        return self._tokens[idx]
