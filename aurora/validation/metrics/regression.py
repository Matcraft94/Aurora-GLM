"""Regression metric utilities built on top of NumPy."""
from __future__ import annotations

from typing import Any

import numpy as np


def mean_squared_error(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Any | None = None,
    squared: bool = True,
) -> float:
    """Return the (weighted) mean squared error between ``y_true`` and ``y_pred``."""

    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    _validate_shape(y_true_np, y_pred_np)

    diff = y_true_np - y_pred_np
    squared_error = diff * diff
    mean = _weighted_mean(squared_error, sample_weight)
    if squared:
        return float(mean)
    return float(np.sqrt(mean))


def mean_absolute_error(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Any | None = None,
) -> float:
    """Return the (weighted) mean absolute error."""

    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    _validate_shape(y_true_np, y_pred_np)

    absolute_error = np.abs(y_true_np - y_pred_np)
    mean = _weighted_mean(absolute_error, sample_weight)
    return float(mean)


def root_mean_squared_error(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Any | None = None,
) -> float:
    """Convenience wrapper computing the square root of the mean squared error."""

    return mean_squared_error(y_true, y_pred, sample_weight=sample_weight, squared=False)


def _weighted_mean(values: np.ndarray, sample_weight: Any | None) -> float:
    if sample_weight is None:
        return float(np.mean(values))

    weights = _to_numpy(sample_weight)
    if weights.shape != values.shape:
        weights = np.broadcast_to(weights, values.shape)
    total_weight = np.sum(weights)
    if total_weight <= 0:
        raise ValueError("sample weights must have positive sum")
    return float(np.sum(weights * values) / total_weight)


def _validate_shape(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


__all__ = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
]
