"""Classification-oriented evaluation metrics."""
from __future__ import annotations

from typing import Any

import numpy as np


def accuracy_score(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Any | None = None,
    normalize: bool = True,
) -> float:
    """Compute classification accuracy, optionally weighted."""

    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    _validate_shape(true, pred)

    hits = (true == pred).astype(np.float64)
    total = _weighted_sum(hits, sample_weight)
    if normalize:
        weight_sum = _weighted_sum(np.ones_like(hits), sample_weight)
        if weight_sum <= 0.0:
            raise ValueError("sample weights must have positive sum")
        return float(total / weight_sum)
    return float(total)


def log_loss(
    y_true: Any,
    y_prob: Any,
    *,
    eps: float = 1e-15,
    sample_weight: Any | None = None,
) -> float:
    """Compute the negative log-likelihood for probabilistic predictions."""

    true = _to_numpy(y_true)
    prob = _to_numpy(y_prob)

    if true.ndim != 1:
        raise ValueError("y_true must be a one-dimensional array of labels")
    if prob.ndim == 1:
        if not _is_binary_labels(true):
            raise ValueError("Binary probabilities provided but labels are not binary")
        p = np.clip(prob, eps, 1.0 - eps)
        losses = -(true * np.log(p) + (1.0 - true) * np.log(1.0 - p))
    elif prob.ndim == 2:
        n_samples, n_classes = prob.shape
        if true.shape[0] != n_samples:
            raise ValueError("y_true and y_prob must share the same number of samples")
        prob = np.clip(prob, eps, 1.0 - eps)
        row_sums = prob.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Each probability row must sum to 1.0")
        labels = true.astype(int)
        if np.any(labels < 0) or np.any(labels >= n_classes):
            raise ValueError("Labels out of range for provided probability matrix")
        losses = -np.log(prob[np.arange(n_samples), labels])
    else:  # pragma: no cover - defensive branch
        raise ValueError("y_prob must be one- or two-dimensional")

    return float(_weighted_mean(losses, sample_weight))


def brier_score_loss(
    y_true: Any,
    y_prob: Any,
    *,
    sample_weight: Any | None = None,
) -> float:
    """Compute the Brier score for probabilistic binary predictions."""

    true = _to_numpy(y_true)
    prob = _to_numpy(y_prob)

    if prob.ndim != 1:
        raise ValueError("Brier score currently supports 1D binary probabilities only")
    if not _is_binary_labels(true):
        raise ValueError("Brier score requires binary labels")

    diff = true - prob
    losses = diff * diff
    return float(_weighted_mean(losses, sample_weight))


def _is_binary_labels(labels: np.ndarray) -> bool:
    unique = np.unique(labels)
    return np.array_equal(unique, [0]) or np.array_equal(unique, [1]) or np.array_equal(unique, [0, 1])


def _weighted_sum(values: np.ndarray, sample_weight: Any | None) -> float:
    if sample_weight is None:
        return float(np.sum(values))
    weights = _to_numpy(sample_weight)
    if weights.shape != values.shape:
        weights = np.broadcast_to(weights, values.shape)
    return float(np.sum(weights * values))


def _weighted_mean(values: np.ndarray, sample_weight: Any | None) -> float:
    total = _weighted_sum(values, sample_weight)
    if sample_weight is None:
        return total / values.size
    weights = _to_numpy(sample_weight)
    if weights.shape != values.shape:
        weights = np.broadcast_to(weights, values.shape)
    weight_sum = np.sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("sample weights must have positive sum")
    return total / weight_sum


def _validate_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError("Inputs must have identical shapes")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


__all__ = ["accuracy_score", "log_loss", "brier_score_loss"]
