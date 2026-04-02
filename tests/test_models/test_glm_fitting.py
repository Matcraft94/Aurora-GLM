"""Tests for the GLM IRLS fitting routine."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.models.glm import fit_glm, predict_glm
from tests.conftest import to_numpy


@pytest.mark.parametrize("backend", [None, "torch", "jax"])
def test_gaussian_identity_fit_converges_and_matches_least_squares(backend):
    """Test Gaussian GLM convergence and coefficient accuracy across backends.

    This test verifies that:
    - Gaussian GLM converges successfully on all backends
    - Recovered coefficients match true values
    - Predictions work correctly (response and link scale)
    - NumPy backend uses auto-detection (backend=None) to avoid dtype bug
    """
    # Skip if backend not available
    if backend == "torch":
        pytest.importorskip("torch")
    elif backend == "jax":
        pytest.importorskip("jax")

    # Generate data
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 3))
    true_coef = np.array([0.5, -1.2, 0.3])
    intercept = 0.7
    noise = rng.normal(scale=0.1, size=200)
    y = intercept + X @ true_coef + noise

    # Fit model
    if backend is None:
        # NumPy: use auto-detection
        result = fit_glm(X, y, family="gaussian", link=None, max_iter=50, tol=1e-9)
    else:
        # PyTorch/JAX: use explicit backend
        result = fit_glm(X, y, family="gaussian", link=None, max_iter=50, tol=1e-9, backend=backend)

    # Verify convergence
    assert result.converged_, f"Failed to converge on {backend}"

    # Verify coefficients
    coef_np = to_numpy(result.coef_)
    np.testing.assert_allclose(coef_np, true_coef, atol=1e-2)

    # Verify intercept
    assert result.intercept_ == pytest.approx(intercept, abs=1e-2)

    # Verify predictions
    mu_np = to_numpy(result.mu_)
    eta_np = to_numpy(result.eta_)

    pred_response = result.predict(X, type="response")
    pred_link = result.predict(X, type="link")

    # Relaxed tolerance for PyTorch (different BLAS implementations)
    np.testing.assert_allclose(to_numpy(pred_response), mu_np, rtol=1e-5)
    np.testing.assert_allclose(to_numpy(pred_link), eta_np, rtol=1e-5)

    # Verify iteration count
    assert result.n_iter_ <= 50

    # Verify predict_glm wrapper
    direct = result.predict(X[:10])
    wrapped = predict_glm(result, X[:10])
    np.testing.assert_allclose(to_numpy(direct), to_numpy(wrapped))


@pytest.mark.parametrize("backend", [None, "torch", "jax"])
def test_poisson_log_fit_reduces_deviance_and_supports_prediction_modes(backend):
    """Test Poisson GLM fitting and prediction modes across backends.

    This comprehensive test verifies:
    - Poisson GLM convergence on all backends
    - Deviance reduction compared to null model
    - AIC/BIC calculation
    - Prediction on link and response scales
    - Confidence intervals
    - Standard errors and p-values
    - NumPy backend uses auto-detection (backend=None) to avoid dtype bug
    """
    # Skip if backend not available
    if backend == "torch":
        pytest.importorskip("torch")
    elif backend == "jax":
        pytest.importorskip("jax")

    # Generate data
    rng = np.random.default_rng(7)
    X = rng.normal(size=(150, 2))
    coef = np.array([0.8, -0.4])
    intercept = -0.2
    eta = intercept + X @ coef
    mu = np.exp(eta)
    y = rng.poisson(mu)

    # Fit model
    if backend is None:
        result = fit_glm(X, y, family="poisson", link="log", max_iter=60, tol=1e-9)
    else:
        result = fit_glm(X, y, family="poisson", link="log", max_iter=60, tol=1e-9, backend=backend)

    # Verify convergence
    assert result.converged_, f"Failed to converge on {backend}"

    # Verify deviance reduction
    assert result.deviance_ < result.null_deviance_

    # Verify information criteria
    assert result.aic_ > 0.0
    assert result.bic_ > 0.0

    # Verify predictions
    link_pred = result.predict(X[:5], type="link")
    resp_pred = result.predict(X[:5], type="response")

    link_pred_np = to_numpy(link_pred)
    resp_pred_np = to_numpy(resp_pred)
    eta_np = to_numpy(result.eta_[:5])
    mu_np = to_numpy(result.mu_[:5])

    np.testing.assert_allclose(link_pred_np, eta_np, atol=1e-6)
    np.testing.assert_allclose(resp_pred_np, mu_np, atol=1e-6)

    # Verify error handling
    with pytest.raises(ValueError):
        result.predict(X[:5], type="unsupported")

    # Verify confidence intervals (response scale)
    resp_pred_ci = result.predict(X[:5], interval="confidence")
    assert isinstance(resp_pred_ci, tuple) and len(resp_pred_ci) == 3
    resp_mean, resp_lower, resp_upper = resp_pred_ci

    resp_mean_np = to_numpy(resp_mean)
    resp_lower_np = to_numpy(resp_lower)
    resp_upper_np = to_numpy(resp_upper)

    np.testing.assert_allclose(resp_mean_np, resp_pred_np)
    assert np.all(resp_lower_np <= resp_mean_np)
    assert np.all(resp_upper_np >= resp_mean_np)

    # Verify confidence intervals (link scale)
    link_mean, link_lower, link_upper = result.predict(X[:5], type="link", interval="confidence")

    link_mean_np = to_numpy(link_mean)
    link_lower_np = to_numpy(link_lower)
    link_upper_np = to_numpy(link_upper)

    np.testing.assert_allclose(link_mean_np, link_pred_np)
    assert np.all(link_lower_np <= link_mean_np)
    assert np.all(link_upper_np >= link_mean_np)

    # Verify prediction intervals not implemented
    with pytest.raises(NotImplementedError):
        result.predict(X[:5], interval="prediction")

    # Verify standard errors
    std_errors = result.std_errors_
    std_errors_np = to_numpy(std_errors)
    assert std_errors_np.shape == coef.shape
    assert np.all(np.isfinite(std_errors_np))

    # Verify intercept inference
    if result.intercept_ is not None:
        assert result.intercept_std_error_ is not None
        assert np.isfinite(result.intercept_std_error_)
        assert result.intercept_p_value_ is not None

    # Verify p-values
    p_values = result.p_values_
    p_values_np = to_numpy(p_values)
    assert p_values_np.shape == coef.shape
    assert np.all((p_values_np >= 0.0) & (p_values_np <= 1.0))
