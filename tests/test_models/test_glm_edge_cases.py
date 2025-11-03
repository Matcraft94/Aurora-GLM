"""Edge case tests for GLM fitting to improve coverage."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models.glm import fit_glm
from aurora.utils.exceptions import ConfigurationError


def test_fit_glm_with_invalid_family_raises_error():
    """fit_glm should raise error for unknown family."""
    X = np.random.randn(50, 2)
    y = np.random.randn(50)

    with pytest.raises((KeyError, ValueError)):
        fit_glm(X, y, family="invalid_family")


def test_fit_glm_with_invalid_link_raises_error():
    """fit_glm should raise ValueError for invalid link."""
    X = np.random.randn(50, 2)
    y = np.random.randn(50)

    with pytest.raises(ValueError, match="Unknown link"):
        fit_glm(X, y, family="gaussian", link="invalid_link")


def test_fit_glm_with_singular_design_matrix():
    """fit_glm should handle singular/rank-deficient design matrices."""
    # Create rank-deficient X (duplicate columns)
    X_base = np.random.randn(100, 2)
    X = np.column_stack([X_base, X_base[:, 0]])  # Third column = first column
    y = np.random.randn(100)

    # Should still run and return a result
    result = fit_glm(X, y, family="gaussian")
    # IRLS may still converge even with rank deficiency
    assert result is not None


def test_fit_glm_with_zero_max_iter():
    """fit_glm with max_iter=0 should return initial guess without iterating."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 2))
    y = rng.normal(size=100)

    result = fit_glm(X, y, family="gaussian", max_iter=0)

    # Should not converge with zero iterations
    assert not result.converged_
    assert result.n_iter_ == 0


def test_fit_glm_poisson_with_integer_counts():
    """Poisson GLM should work well with integer count data."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 2))
    # Generate count data
    y = rng.poisson(np.exp(X[:, 0] * 0.3 + 0.5))

    result = fit_glm(X, y, family="poisson", link="log")

    assert result.converged_
    assert result.coef_.shape == (2,)


def test_fit_glm_binomial_with_binary_response():
    """Binomial GLM should work with binary 0/1 response."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(150, 2))
    # Generate binary data
    probs = 1 / (1 + np.exp(-(X @ np.array([0.5, -0.7]) + 0.2)))
    y = rng.binomial(1, probs)

    result = fit_glm(X, y, family="binomial", link="logit")

    assert result.converged_
    assert result.coef_.shape == (2,)


def test_fit_glm_gamma_with_positive_response():
    """Gamma GLM should work with positive continuous response."""
    rng = np.random.default_rng(456)
    X = rng.normal(size=(100, 2))
    # Generate gamma-distributed data
    shape = 2.0
    mu = np.exp(X @ np.array([0.3, -0.2]) + 1.0)
    scale = mu / shape
    y = rng.gamma(shape, scale)

    result = fit_glm(X, y, family="gamma", link="log")

    assert result.converged_
    assert result.coef_.shape == (2,)


def test_fit_glm_with_1d_input_reshapes_to_2d():
    """fit_glm should handle 1D X input by reshaping to column vector."""
    X_1d = np.random.randn(50)
    y = np.random.randn(50)

    result = fit_glm(X_1d, y, family="gaussian")

    assert result.coef_.shape == (1,)
    assert result.intercept_ is not None


def test_fit_glm_without_intercept():
    """fit_glm with fit_intercept=False should not include intercept."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(100, 3))
    y = rng.normal(size=100)

    result = fit_glm(X, y, family="gaussian", fit_intercept=False)

    assert result.intercept_ is None
    assert result.coef_.shape == (3,)


def test_fit_glm_with_extremely_tight_tolerance():
    """fit_glm with very small tolerance should iterate until max_iter."""
    rng = np.random.default_rng(456)
    X = rng.normal(size=(50, 2))
    y = rng.normal(size=50)

    result = fit_glm(X, y, family="gaussian", tol=1e-20, max_iter=5)

    # Likely won't converge with such tight tolerance
    assert result.n_iter_ <= 5


def test_fit_glm_with_default_optimizer():
    """fit_glm should work with default IRLS optimizer."""
    rng = np.random.default_rng(789)
    X = rng.normal(size=(100, 2))
    y_gauss = rng.normal(X @ np.array([1.0, -0.5]) + 0.5, scale=0.5)

    # Test default optimizer
    result = fit_glm(X, y_gauss, family="gaussian")
    assert result.converged_
    assert result.coef_.shape == (2,)


def test_fit_glm_with_single_sample():
    """fit_glm should handle single-sample edge case."""
    X = np.array([[1.0, 2.0]])
    y = np.array([3.0])

    # With single sample, may not converge or may fit trivially
    result = fit_glm(X, y, family="gaussian", fit_intercept=False)
    assert result is not None


def test_fit_glm_with_perfect_separation_binomial():
    """Binomial GLM with perfect separation should still run (may not converge)."""
    # Create perfectly separable data
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    # Perfect separation can cause convergence issues
    result = fit_glm(X, y, family="binomial", max_iter=100)

    # Should return a result (may or may not converge)
    assert result is not None


def test_fit_glm_caches_diagnostics():
    """GLMResult should cache diagnostics on first access."""
    rng = np.random.default_rng(999)
    X = rng.normal(size=(100, 2))
    y = rng.poisson(np.exp(X[:, 0] * 0.5))

    result = fit_glm(X, y, family="poisson")

    # First access should compute
    diag1 = result.diagnostics_
    assert result._diagnostics_cache is not None

    # Second access should use cache
    diag2 = result.diagnostics_
    assert diag1 is diag2  # Same object


def test_fit_glm_gaussian_equivalent_to_ols():
    """Gaussian GLM with identity link should give OLS-like results."""
    rng = np.random.default_rng(111)
    X = rng.normal(size=(200, 3))
    true_coef = np.array([1.5, -0.8, 0.6])
    true_intercept = 2.0
    y = true_intercept + X @ true_coef + rng.normal(scale=0.5, size=200)

    result = fit_glm(X, y, family="gaussian", link="identity")

    # Coefficients should be close to true values
    np.testing.assert_allclose(result.coef_, true_coef, atol=0.2)
    np.testing.assert_allclose(result.intercept_, true_intercept, atol=0.2)
