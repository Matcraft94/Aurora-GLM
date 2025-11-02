"""Tests for Wald hypothesis testing utilities."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.inference import wald_test
from aurora.models.glm import fit_glm


def _dataset(seed: int = 321) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(500, 3))
    coef = np.array([0.5, -0.7, 0.25])
    intercept = 0.9
    noise = rng.normal(scale=0.4, size=X.shape[0])
    y = intercept + np.sum(X * coef, axis=1) + noise
    return X, y, coef


def test_wald_test_detects_nonzero_coefficient():
    X, y, coef = _dataset()
    result = fit_glm(X, y, family="gaussian", link=None, max_iter=50, tol=1e-9)

    test = wald_test(result, [0.0, 1.0, 0.0, 0.0])

    assert test["df"] == pytest.approx(1.0)
    assert test["statistic"] > 0.0
    assert test["p_value"] < 1e-6


def test_wald_test_validates_input_lengths():
    X, y, _ = _dataset(456)
    result = fit_glm(X, y, family="gaussian", link=None)

    with pytest.raises(ValueError):
        wald_test(result, [1.0, 0.0])

    with pytest.raises(ValueError):
        wald_test(result, [[1.0, 0.0], [0.0, 1.0]])


def test_wald_without_intercept():
    X, y, coef = _dataset(654)
    result = fit_glm(X, y, family="gaussian", link=None, fit_intercept=False)

    test = wald_test(result, coef, include_intercept=False)
    assert test["p_value"] < 1e-6


def test_wald_test_supports_multiple_constraints():
    X, y, _ = _dataset(987)
    result = fit_glm(X, y, family="gaussian", link=None, max_iter=60, tol=1e-9)

    contrast = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    outcome = wald_test(result, contrast, value=[0.0, 0.0])

    assert outcome["df"] == pytest.approx(2.0)
    assert outcome["statistic"] >= 0.0
    assert 0.0 <= outcome["p_value"] <= 1.0
