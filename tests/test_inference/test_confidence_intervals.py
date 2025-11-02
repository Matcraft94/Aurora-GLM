"""Tests for confidence interval computation utilities."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.inference import confidence_intervals
from aurora.models.glm import fit_glm


def _generate_gaussian_data(seed: int = 123) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(600, 3))
    coef = np.array([0.4, -0.6, 0.2])
    intercept = 1.2
    noise = rng.normal(scale=0.5, size=X.shape[0])
    y = intercept + np.sum(X * coef, axis=1) + noise
    return X, y, coef, intercept


def test_confidence_intervals_contains_true_parameters():
    X, y, coef, intercept = _generate_gaussian_data()
    result = fit_glm(X, y, family="gaussian", link=None, max_iter=50, tol=1e-9)

    ci = confidence_intervals(result, level=0.95)

    assert ci.lower.shape == coef.shape
    assert ci.upper.shape == coef.shape
    assert ci.intercept is not None

    for lower, upper, true_value in zip(ci.lower, ci.upper, coef):
        assert lower < true_value < upper

    intercept_lower, intercept_upper = ci.intercept
    assert intercept_lower < intercept < intercept_upper


def test_confidence_interval_parameter_validation():
    X, y, _, _ = _generate_gaussian_data()
    result = fit_glm(X, y, family="gaussian", link=None)

    with pytest.raises(ValueError):
        confidence_intervals(result, level=1.0)

    with pytest.raises(NotImplementedError):
        confidence_intervals(result, method="profile")

    ci = confidence_intervals(result, include_intercept=False)
    assert ci.intercept is None
