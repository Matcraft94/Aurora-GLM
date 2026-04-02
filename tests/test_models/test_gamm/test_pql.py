"""Tests for Penalized Quasi-Likelihood (PQL) estimation."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.models.gamm.pql import fit_pql


def test_pql_gaussian_identity():
    """PQL with Gaussian family should match LMM."""
    np.random.seed(42)

    # Generate data
    n_groups, n_per_group = 5, 20
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    # Random effects design (indicator matrix)
    Z = np.zeros((n, n_groups))
    for i in range(n):
        Z[i, groups[i]] = 1

    # Generate data with random intercepts
    b_true = np.random.randn(n_groups) * 0.5
    y = 2.0 + 0.5 * x + b_true[groups] + np.random.randn(n) * 0.3

    # Fit with PQL
    result = fit_pql(X, Z, y, family="gaussian", maxiter_outer=10)

    # Should converge
    assert result.converged

    # Fixed effects should be reasonable
    assert 1.5 < result.beta[0] < 2.5  # Intercept
    assert 0.2 < result.beta[1] < 0.8  # Slope

    # Variance should be positive
    assert result.psi[0, 0] > 0
    assert result.sigma2 > 0

    # Deviance should be reasonable
    assert result.deviance > 0


def test_pql_poisson_log():
    """PQL should fit Poisson GLMM with log link."""
    np.random.seed(42)

    # Generate data
    n_groups, n_per_group = 8, 25
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    # Random effects design
    Z = np.zeros((n, n_groups))
    for i in range(n):
        Z[i, groups[i]] = 1

    # Generate Poisson data with random intercepts
    b_true = np.random.randn(n_groups) * 0.4
    eta = 1.0 + 0.3 * x + b_true[groups]
    mu_true = np.exp(eta)
    y = np.random.poisson(mu_true)

    # Fit with PQL
    result = fit_pql(X, Z, y, family="poisson", maxiter_outer=15, maxiter_inner=10)

    # Should converge (may take a few iterations)
    assert result.converged

    # Fixed effects should recover true values approximately
    assert 0.5 < result.beta[0] < 1.5  # Intercept around 1.0
    assert 0.0 < result.beta[1] < 0.6  # Slope around 0.3

    # Variance should be positive
    assert result.psi[0, 0] > 0

    # Fitted values should be in reasonable range
    assert np.all(result.fitted_values > 0)
    assert np.all(np.isfinite(result.fitted_values))


def test_pql_binomial_logit():
    """PQL should fit Binomial GLMM with logit link."""
    np.random.seed(123)

    # Generate data
    n_groups, n_per_group = 6, 30
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    # Random effects design
    Z = np.zeros((n, n_groups))
    for i in range(n):
        Z[i, groups[i]] = 1

    # Generate Binomial data with random intercepts
    b_true = np.random.randn(n_groups) * 0.6
    eta = 0.5 + 0.4 * x + b_true[groups]
    p_true = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p_true)

    # Fit with PQL
    result = fit_pql(X, Z, y, family="binomial", maxiter_outer=20, maxiter_inner=10)

    # Should converge
    assert result.converged

    # Fixed effects should be reasonable
    assert -0.5 < result.beta[0] < 1.5  # Intercept around 0.5
    assert 0.0 < result.beta[1] < 0.8  # Slope around 0.4

    # Variance should be positive
    assert result.psi[0, 0] > 0

    # Fitted values should be probabilities
    assert np.all(result.fitted_values >= 0)
    assert np.all(result.fitted_values <= 1)


def test_pql_with_penalty():
    """PQL should work with smoothing penalty."""
    np.random.seed(42)

    # Generate data
    n_groups, n_per_group = 5, 20
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    # Random effects design
    Z = np.zeros((n, n_groups))
    for i in range(n):
        Z[i, groups[i]] = 1

    # Generate Poisson data
    b_true = np.random.randn(n_groups) * 0.3
    eta = 1.0 + 0.2 * x + b_true[groups]
    y = np.random.poisson(np.exp(eta))

    # Penalty on slope coefficient
    S = np.diag([0, 1])
    lambda_ = 0.1

    # Fit with penalty
    result = fit_pql(X, Z, y, family="poisson", S=S, lambda_=lambda_, maxiter_outer=15)

    assert result.converged

    # Penalty should shrink slope toward zero
    assert np.abs(result.beta[1]) < 0.5


def test_pql_no_variance_update():
    """PQL should work with fixed variance components."""
    np.random.seed(42)

    # Generate data
    n_groups, n_per_group = 4, 15
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    # Random effects design
    Z = np.zeros((n, n_groups))
    for i in range(n):
        Z[i, groups[i]] = 1

    b_true = np.random.randn(n_groups) * 0.5
    y = 2.0 + 0.5 * x + b_true[groups] + np.random.randn(n) * 0.3

    # Fix variance at specific value
    psi_fixed = np.array([[0.25]])

    result = fit_pql(
        X,
        Z,
        y,
        family="gaussian",
        psi_init=psi_fixed,
        update_psi=False,  # Don't update variance
        maxiter_outer=5,
    )

    # Variance should remain at initial value
    assert np.abs(result.psi[0, 0] - 0.25) < 1e-10

    # Coefficients should still be estimated
    assert 1.5 < result.beta[0] < 2.5


def test_pql_convergence_tracking():
    """PQL should track iteration counts."""
    np.random.seed(42)

    # Generate data
    n_groups, n_per_group = 3, 10
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    X = np.column_stack([np.ones(n), np.random.randn(n)])
    Z = np.zeros((n, n_groups))
    for i in range(n):
        Z[i, groups[i]] = 1

    y = np.random.randn(n)

    result = fit_pql(X, Z, y, family="gaussian", maxiter_outer=10, maxiter_inner=5)

    # Should have iteration counts
    assert result.n_iter_outer > 0
    assert result.n_iter_inner > 0
    assert result.n_iter_outer <= 10
    assert result.n_iter_inner <= 10 * 5  # Max possible


def test_pql_diagnostics():
    """PQL result should include diagnostic statistics."""
    np.random.seed(42)

    # Generate data
    n_groups, n_per_group = 4, 20
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])
    Z = np.zeros((n, n_groups))
    for i in range(n):
        Z[i, groups[i]] = 1

    b_true = np.random.randn(n_groups) * 0.4
    eta = 1.0 + 0.3 * x + b_true[groups]
    y = np.random.poisson(np.exp(eta))

    result = fit_pql(X, Z, y, family="poisson", maxiter_outer=15)

    # Should have all diagnostic fields
    assert hasattr(result, "deviance")
    assert hasattr(result, "log_likelihood")
    assert hasattr(result, "fitted_values")
    assert hasattr(result, "linear_predictor")

    # Values should be finite
    assert np.isfinite(result.deviance)
    assert np.isfinite(result.log_likelihood)
    assert np.all(np.isfinite(result.fitted_values))
    assert np.all(np.isfinite(result.linear_predictor))

    # Deviance should be positive
    assert result.deviance > 0


def test_pql_invalid_family_raises():
    """PQL should raise error for invalid family."""
    X = np.random.randn(50, 2)
    Z = np.random.randn(50, 5)
    y = np.random.randn(50)

    with pytest.raises(ValueError, match="Unknown family"):
        fit_pql(X, Z, y, family="invalid_family")


def test_pql_small_groups():
    """PQL should handle small group sizes."""
    np.random.seed(42)

    # Very small groups
    n_groups, n_per_group = 10, 3
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    X = np.column_stack([np.ones(n), np.random.randn(n)])
    Z = np.zeros((n, n_groups))
    for i in range(n):
        Z[i, groups[i]] = 1

    b_true = np.random.randn(n_groups) * 0.3
    y = 2.0 + b_true[groups] + np.random.randn(n) * 0.5

    result = fit_pql(X, Z, y, family="gaussian", maxiter_outer=15)

    # Should still converge (though estimates may be less accurate)
    assert result.converged or result.n_iter_outer == 15


def test_pql_large_variance():
    """PQL should handle large random effect variance robustly."""
    np.random.seed(42)

    # Generate data with large random effects
    n_groups, n_per_group = 5, 25
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    X = np.column_stack([np.ones(n), np.random.randn(n)])
    Z = np.zeros((n, n_groups))
    for i in range(n):
        Z[i, groups[i]] = 1

    # Large random effects
    b_true = np.random.randn(n_groups) * 2.0
    eta = 1.0 + b_true[groups]
    y = np.random.poisson(np.exp(eta))

    result = fit_pql(X, Z, y, family="poisson", maxiter_outer=25)

    # Should complete without error (may or may not converge with extreme data)
    assert result.n_iter_outer > 0

    # Variance should be positive definite
    assert result.psi[0, 0] >= 0.0

    # Should have valid fixed effects
    assert len(result.beta) == 2
    assert np.all(np.isfinite(result.beta))


def test_pql_zero_counts():
    """PQL should handle data with many zeros (Poisson)."""
    np.random.seed(42)

    # Generate sparse count data
    n_groups, n_per_group = 6, 20
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])
    Z = np.zeros((n, n_groups))
    for i in range(n):
        Z[i, groups[i]] = 1

    # Low mean -> many zeros
    b_true = np.random.randn(n_groups) * 0.2
    eta = -1.0 + 0.2 * x + b_true[groups]
    y = np.random.poisson(np.exp(eta))

    result = fit_pql(X, Z, y, family="poisson", maxiter_outer=20)

    # Should handle zeros gracefully
    assert result.converged or result.n_iter_outer == 20
    assert np.all(np.isfinite(result.fitted_values))
    assert np.all(result.fitted_values >= 0)


def test_pql_working_response():
    """PQL should compute reasonable working response."""
    np.random.seed(42)

    # Simple case to verify algorithm
    n_groups, n_per_group = 3, 10
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    X = np.ones((n, 1))
    Z = np.zeros((n, n_groups))
    for i in range(n):
        Z[i, groups[i]] = 1

    # Constant mean
    y = np.random.poisson(5, size=n)

    result = fit_pql(X, Z, y, family="poisson", maxiter_outer=10)

    # Working response should be finite
    assert np.all(np.isfinite(result.linear_predictor))

    # Fitted mean should be close to data mean
    assert 3 < np.mean(result.fitted_values) < 7
