"""Tests for validation metric utilities."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models.glm import fit_glm
from aurora.validation.metrics import (
    mean_absolute_error,
    mean_squared_error,
    pseudo_r2,
    root_mean_squared_error,
)


def test_mean_squared_error_matches_manual_computation():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    mse = mean_squared_error(y_true, y_pred)
    manual = np.mean((y_true - y_pred) ** 2)
    assert mse == pytest.approx(manual)


def test_mean_squared_error_with_sample_weights():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.8, 2.5])
    weights = np.array([1.0, 2.0, 3.0])
    mse = mean_squared_error(y_true, y_pred, sample_weight=weights)
    manual = np.average((y_true - y_pred) ** 2, weights=weights)
    assert mse == pytest.approx(manual)


def test_mean_absolute_error_and_rmse():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.5, 1.5, 1.0])
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    assert mae == pytest.approx(np.mean(np.abs(y_true - y_pred)))
    assert rmse == pytest.approx(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _gaussian_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 2))
    coef = np.array([0.7, -0.4])
    intercept = 1.5
    noise = rng.normal(scale=0.6, size=X.shape[0])
    y = intercept + np.sum(X * coef, axis=1) + noise
    return X, y


def test_pseudo_r2_mcfadden():
    X, y = _gaussian_dataset()
    result = fit_glm(X, y, family="gaussian", link=None, max_iter=50, tol=1e-9)

    score = pseudo_r2(result, method="mcfadden")
    assert 0.0 <= score <= 1.0
    assert score > 0.0

    same_score = pseudo_r2(result, method="deviance")
    assert same_score == pytest.approx(score)

    with pytest.raises(NotImplementedError):
        pseudo_r2(result, method="cox-snell")
