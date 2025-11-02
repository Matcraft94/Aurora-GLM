"""Tests for GLM diagnostic utilities."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.inference import glm_diagnostics
from aurora.models.glm import fit_glm


def _gaussian_dataset(
    seed: int = 123,
    n_samples: int = 600,
    n_features: int = 3,
    intercept: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    coef = rng.normal(scale=0.5, size=n_features)
    noise = rng.normal(scale=0.5, size=n_samples)
    y = intercept + np.sum(X * coef, axis=1) + noise
    return X, y


def test_glm_diagnostics_gaussian_identity():
    X, y = _gaussian_dataset()
    result = fit_glm(X, y, family="gaussian", link=None, max_iter=50, tol=1e-9)

    diagnostics = glm_diagnostics(result)

    mu = np.asarray(result.mu_, dtype=float)
    response_expected = y - mu
    np.testing.assert_allclose(diagnostics.response_residuals, response_expected, atol=1e-6)

    variance = np.asarray(result.family.variance(result.mu_), dtype=float)
    sqrt_var = np.sqrt(np.clip(variance, 1e-12, None))
    pearson_expected = response_expected / sqrt_var
    np.testing.assert_allclose(diagnostics.pearson_residuals, pearson_expected, rtol=1e-6, atol=1e-6)

    deriv = np.asarray(result.link.derivative(result.mu_), dtype=float)
    working_expected = response_expected * deriv
    np.testing.assert_allclose(diagnostics.working_residuals, working_expected, atol=1e-6)

    deviance_expected = np.sign(response_expected) * np.sqrt(np.clip((response_expected**2) / np.clip(variance, 1e-12, None), 0.0, None))
    np.testing.assert_allclose(diagnostics.deviance_residuals, deviance_expected, atol=1e-6)

    leverage_sum = np.sum(diagnostics.leverage)
    p = result.coef_.shape[0] + (1 if result.intercept_ is not None else 0)
    assert leverage_sum == pytest.approx(p, rel=1e-5)
    assert np.all(diagnostics.leverage >= -1e-9)
    assert np.all(diagnostics.leverage <= 1.0 + 1e-6)
    assert np.all(diagnostics.cooks_distance >= -1e-9)


def test_glm_diagnostics_without_intercept():
    X, y = _gaussian_dataset(intercept=0.0)
    result = fit_glm(X, y, family="gaussian", link=None, fit_intercept=False, max_iter=50, tol=1e-9)

    diagnostics = glm_diagnostics(result)

    p = result.coef_.shape[0]
    leverage_sum = np.sum(diagnostics.leverage)
    assert leverage_sum == pytest.approx(p, rel=1e-5)
    assert diagnostics.cooks_distance.shape == y.shape


def test_glmresult_diagnostics_property_caches_result():
    X, y = _gaussian_dataset(seed=999)
    result = fit_glm(X, y, family="gaussian", link=None, max_iter=50, tol=1e-9)

    diagnostics_first = result.diagnostics_
    diagnostics_second = result.diagnostics_

    assert diagnostics_first is diagnostics_second
    assert diagnostics_first.response_residuals.shape == y.shape
