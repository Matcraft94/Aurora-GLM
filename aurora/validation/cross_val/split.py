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


@dataclass(frozen=True)
class StratifiedKFold:
    """Stratified K-fold splitter preserving label proportions."""

    n_splits: int = 5
    shuffle: bool = False
    random_state: int | None = None

    def split(
        self,
        X: Sequence[Any],
        y: Sequence[Any],
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield stratified train and validation index pairs."""

        if y is None:
            raise ValueError("StratifiedKFold requires target labels.")

        n_samples = _num_samples(X)
        y_arr = _to_numpy_labels(y)

        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if n_samples < self.n_splits:
            raise ValueError("n_splits cannot exceed the number of samples")

        unique, counts = np.unique(y_arr, return_counts=True)
        if np.any(counts < self.n_splits):
            raise ValueError("Each class must have at least n_splits members.")

        rng = np.random.default_rng(self.random_state) if self.shuffle else None
        fold_indices: list[list[int]] = [[] for _ in range(self.n_splits)]

        for cls in unique:
            cls_indices = np.where(y_arr == cls)[0]
            if rng is not None:
                rng.shuffle(cls_indices)

            sizes = list(_fold_sizes(cls_indices.shape[0], self.n_splits))
            start = 0
            for fold_id, fold_size in enumerate(sizes):
                stop = start + fold_size
                fold_indices[fold_id].extend(cls_indices[start:stop])
                start = stop

        indices = np.arange(n_samples)
        for fold_id in range(self.n_splits):
            test_idx = np.array(sorted(fold_indices[fold_id]), dtype=int)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            train_idx = indices[train_mask]
            yield train_idx, test_idx


def _to_numpy_labels(labels: Sequence[Any]) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("Labels must be a one-dimensional sequence.")
    return arr


__all__ = ["KFold", "StratifiedKFold"]
