"""Data splitting helpers for cross-validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class KFold:
    """K-fold splitter yielding train and validation indices."""

    n_splits: int = 5
    shuffle: bool = False
    random_state: int | None = None

    def split(self, X: Sequence[Any], y: Sequence[Any] | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield index pairs for successive training and validation splits."""

        del y  # kept for signature parity with scikit-learn-like API

        n_samples = _num_samples(X)
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if n_samples < self.n_splits:
            raise ValueError("n_splits cannot exceed the number of samples")

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        fold_sizes = _fold_sizes(n_samples, self.n_splits)
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate((indices[:start], indices[stop:]))
            yield train_indices, test_indices
            current = stop


def _num_samples(data: Sequence[Any]) -> int:
    if hasattr(data, "shape") and len(getattr(data, "shape")) > 0:
        return int(data.shape[0])
    if isinstance(data, Sequence):
        return len(data)
    raise TypeError("Unable to determine number of samples for provided dataset.")


def _fold_sizes(n_samples: int, n_splits: int) -> Iterable[int]:
    base = n_samples // n_splits
    remainder = n_samples % n_splits
    for fold in range(n_splits):
        if fold < remainder:
            yield base + 1
        else:
            yield base


__all__ = ["KFold"]
