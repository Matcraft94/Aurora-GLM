"""Tests for the GLM IRLS fitting routine."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models.glm import fit_glm, predict_glm


def test_gaussian_identity_fit_converges_and_matches_least_squares():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 3))
    true_coef = np.array([0.5, -1.2, 0.3])
    intercept = 0.7
    noise = rng.normal(scale=0.1, size=200)
    y = intercept + X @ true_coef + noise

    result = fit_glm(X, y, family="gaussian", link=None, max_iter=50, tol=1e-9)

    assert result.converged_
    np.testing.assert_allclose(result.coef_, true_coef, atol=1e-2)
    assert result.intercept_ == pytest.approx(intercept, abs=1e-2)
    np.testing.assert_allclose(result.predict(X, type="response"), result.mu_)
    np.testing.assert_allclose(result.predict(X, type="link"), result.eta_)
    assert result.n_iter_ <= 50

    direct = result.predict(X[:10])
    wrapped = predict_glm(result, X[:10])
    np.testing.assert_allclose(direct, wrapped)


def test_poisson_log_fit_reduces_deviance_and_supports_prediction_modes():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(150, 2))
    coef = np.array([0.8, -0.4])
    intercept = -0.2
    eta = intercept + X @ coef
    mu = np.exp(eta)
    y = rng.poisson(mu)

    result = fit_glm(X, y, family="poisson", link="log", max_iter=60, tol=1e-9)

    assert result.converged_
    assert result.deviance_ < result.null_deviance_
    assert result.aic_ > 0.0
    assert result.bic_ > 0.0
    link_pred = result.predict(X[:5], type="link")
    resp_pred = result.predict(X[:5], type="response")
    np.testing.assert_allclose(link_pred, result.eta_[:5], atol=1e-6)
    np.testing.assert_allclose(resp_pred, result.mu_[:5], atol=1e-6)

    with pytest.raises(ValueError):
        result.predict(X[:5], type="unsupported")

    resp_pred_ci = result.predict(X[:5], interval="confidence")
    assert isinstance(resp_pred_ci, tuple) and len(resp_pred_ci) == 3
    resp_mean, resp_lower, resp_upper = resp_pred_ci
    np.testing.assert_allclose(resp_mean, resp_pred)
    assert np.all(resp_lower <= resp_mean)
    assert np.all(resp_upper >= resp_mean)

    link_mean, link_lower, link_upper = result.predict(X[:5], type="link", interval="confidence")
    np.testing.assert_allclose(link_mean, link_pred)
    assert np.all(link_lower <= link_mean)
    assert np.all(link_upper >= link_mean)

    with pytest.raises(NotImplementedError):
        result.predict(X[:5], interval="prediction")

    std_errors = result.std_errors_
    assert std_errors.shape == coef.shape
    assert np.all(np.isfinite(std_errors))

    if result.intercept_ is not None:
        assert result.intercept_std_error_ is not None
        assert np.isfinite(result.intercept_std_error_)
        assert result.intercept_p_value_ is not None

    p_values = result.p_values_
    assert p_values.shape == coef.shape
    assert np.all((p_values >= 0.0) & (p_values <= 1.0))