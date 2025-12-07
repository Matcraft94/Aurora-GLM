"""Tests for REML variance component estimation."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg

from aurora.models.gamm import construct_Z_matrix, RandomEffect
from aurora.models.gamm.estimation import (
    REMLResult,
    compute_P_matrix,
    compute_V_matrix,
    estimate_fixed_effects,
    estimate_random_effects,
    estimate_variance_components,
    reml_log_likelihood,
    reml_objective,
)


# Helper functions for generating test data

def generate_random_intercept_data(
    n_groups: int = 10,
    n_per_group: int = 5,
    beta: float = 2.0,
    psi: float = 1.0,
    sigma2: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray]:
    """Generate data from random intercept model."""
    np.random.seed(seed)

    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    # Design matrices
    X = np.ones((n, 1))
    Z = np.zeros((n, n_groups))
    Z[np.arange(n), groups] = 1

    # Generate random effects
    b = np.random.randn(n_groups) * np.sqrt(psi)

    # Generate response
    y = X @ np.array([beta]) + Z @ b + np.random.randn(n) * np.sqrt(sigma2)

    Z_info = [{
        'grouping': 'subject',
        'n_effects': 1,
        'n_groups': n_groups,
        'groups': np.arange(n_groups),
        'start_col': 0,
        'end_col': n_groups,
    }]

    return y, X, Z, Z_info, groups


def generate_random_slope_data(
    n_groups: int = 10,
    n_per_group: int = 5,
    beta: np.ndarray = None,
    psi: np.ndarray = None,
    sigma2: float = 0.5,
    seed: int = 42,
    covariance: str = 'unstructured',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Generate data from random intercept + slope model."""
    if beta is None:
        beta = np.array([2.0, 0.5])
    if psi is None:
        psi = np.array([[1.0, 0.3], [0.3, 0.5]])

    np.random.seed(seed)

    n = n_groups * n_per_group

    # Time variable
    time = np.tile(np.arange(n_per_group), n_groups)
    groups = np.repeat(np.arange(n_groups), n_per_group)

    # Fixed effects design
    X = np.column_stack([np.ones(n), time])

    # Random effects design
    groups_data = {'subject': groups}
    re = RandomEffect(grouping='subject', variables=(1,), covariance=covariance)
    Z, Z_info = construct_Z_matrix(X, [re], groups_data)

    # Generate random effects (intercept + slope per group)
    L = linalg.cholesky(psi, lower=True)
    b_raw = np.random.randn(n_groups, 2)
    b = (L @ b_raw.T).T.flatten()  # Shape: (n_groups * 2,)

    # Generate response
    y = X @ beta + Z @ b + np.random.randn(n) * np.sqrt(sigma2)

    return y, X, Z, Z_info


# Tests for helper functions

def test_compute_V_matrix():
    """compute_V_matrix should compute V = ZΨZ' + σ²I."""
    n, q = 6, 2
    Z = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
    ], dtype=float)
    psi = np.array([[2.0, 0.5], [0.5, 1.0]])
    sigma2 = 0.5

    V = compute_V_matrix(Z, psi, sigma2)

    # Expected: ZΨZ' + σ²I
    # ZΨZ' has blocks with correlations between all obs
    expected = np.array([
        [2.5, 2.0, 2.0, 0.5, 0.5, 0.5],
        [2.0, 2.5, 2.0, 0.5, 0.5, 0.5],
        [2.0, 2.0, 2.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 1.5, 1.0, 1.0],
        [0.5, 0.5, 0.5, 1.0, 1.5, 1.0],
        [0.5, 0.5, 0.5, 1.0, 1.0, 1.5],
    ])

    assert V.shape == (6, 6)
    np.testing.assert_allclose(V, expected, atol=1e-10)


def test_compute_P_matrix():
    """compute_P_matrix should compute P = V⁻¹ - V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹."""
    n = 4
    V = np.eye(n) * 2.0  # Simple diagonal
    X = np.ones((n, 1))

    P = compute_P_matrix(V, X)

    # For V = 2I and X = 1, we have:
    # V⁻¹ = 0.5I
    # X'V⁻¹X = 1' * 0.5I * 1 = 2
    # (X'V⁻¹X)⁻¹ = 0.5
    # V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹ = 0.5 * 1 * 0.5 * 1' * 0.5 = 0.125 * 11'

    V_inv = 0.5 * np.eye(n)
    projection = 0.125 * np.ones((n, n))
    expected = V_inv - projection

    assert P.shape == (4, 4)
    np.testing.assert_allclose(P, expected, atol=1e-10)


def test_reml_log_likelihood_known():
    """reml_log_likelihood should compute correct value for simple case."""
    n = 4
    y = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.ones((n, 1))
    V = 2.0 * np.eye(n)

    P = compute_P_matrix(V, X)
    log_lik = reml_log_likelihood(y, X, V, P)

    # Manual calculation:
    # log|V| = log(2^4) = 4*log(2)
    # V⁻¹ = 0.5I
    # X'V⁻¹X = 2, log|X'V⁻¹X| = log(2)
    # y'Py = computed below
    # l = -0.5 * [log|V| + log|X'V⁻¹X| + y'Py] - 0.5(n-p)log(2π)

    yPy = y @ P @ y
    log_det_V = 4 * np.log(2)
    log_det_XtV_invX = np.log(2)
    expected = -0.5 * (log_det_V + log_det_XtV_invX + yPy) - 0.5 * 3 * np.log(2 * np.pi)

    assert np.isclose(log_lik, expected, atol=1e-10)


def test_reml_objective():
    """reml_objective should return negative log-likelihood."""
    y, X, Z, Z_info, _ = generate_random_intercept_data(
        n_groups=5, n_per_group=4, seed=123
    )

    from aurora.models.gamm.covariance import IdentityCovariance

    cov_structure = IdentityCovariance()
    n_effects = 1

    # theta = [log(psi), log(sigma2)]
    theta = np.array([0.0, -0.5])  # psi=1.0, sigma2≈0.6

    neg_log_lik = reml_objective(theta, y, X, Z, cov_structure, n_effects)

    assert isinstance(neg_log_lik, float)
    assert neg_log_lik > 0  # Should be positive (negative of log-likelihood)


def test_reml_objective_invalid_params():
    """reml_objective should return large value for invalid parameters."""
    y, X, Z, Z_info, _ = generate_random_intercept_data(n_groups=5, n_per_group=4)

    from aurora.models.gamm.covariance import IdentityCovariance

    cov_structure = IdentityCovariance()
    n_effects = 1

    # Invalid: very large negative log(sigma2) causes numerical issues
    theta = np.array([0.0, -100.0])

    neg_log_lik = reml_objective(theta, y, X, Z, cov_structure, n_effects)

    assert neg_log_lik >= 1e10


# Tests for variance component estimation

def test_estimate_variance_components_random_intercept():
    """estimate_variance_components should work for random intercept model."""
    y, X, Z, Z_info, _ = generate_random_intercept_data(
        n_groups=10,
        n_per_group=5,
        beta=2.0,
        psi=1.0,
        sigma2=0.5,
        seed=42,
    )

    result = estimate_variance_components(
        y, X, Z, Z_info, covariance='identity', maxiter=500
    )

    assert isinstance(result, REMLResult)
    assert result.psi.shape == (1, 1)
    assert result.sigma2 > 0
    assert result.converged

    # Check estimates are reasonable (within 2x of true values)
    assert 0.3 < result.psi[0, 0] < 3.0  # True: 1.0
    assert 0.1 < result.sigma2 < 2.0  # True: 0.5


def test_estimate_variance_components_random_slope():
    """estimate_variance_components should work for random slope model."""
    y, X, Z, Z_info = generate_random_slope_data(
        n_groups=10,
        n_per_group=5,
        beta=np.array([2.0, 0.5]),
        psi=np.array([[1.0, 0.0], [0.0, 0.5]]),
        sigma2=0.5,
        seed=42,
        covariance='diagonal',  # Use diagonal covariance in RandomEffect
    )

    result = estimate_variance_components(
        y, X, Z, Z_info, covariance='diagonal', maxiter=500
    )

    assert result.psi.shape == (2, 2)
    assert result.converged

    # Check diagonal structure (should have zero off-diagonals since diagonal covariance is used)
    off_diag = result.psi - np.diag(np.diag(result.psi))
    np.testing.assert_allclose(off_diag, 0, atol=1e-10)

    # Check estimates are positive
    assert result.psi[0, 0] > 0
    assert result.psi[1, 1] > 0
    assert result.sigma2 > 0


def test_estimate_variance_components_unstructured():
    """estimate_variance_components should handle unstructured covariance."""
    y, X, Z, Z_info = generate_random_slope_data(
        n_groups=10,
        n_per_group=5,
        psi=np.array([[1.5, 0.4], [0.4, 0.8]]),
        seed=42,
    )

    result = estimate_variance_components(
        y, X, Z, Z_info, covariance='unstructured', maxiter=500
    )

    assert result.psi.shape == (2, 2)
    assert result.converged

    # Check positive definite
    eigvals = np.linalg.eigvalsh(result.psi)
    assert np.all(eigvals > 0)

    # Check correlation is reasonable
    corr = result.psi[0, 1] / np.sqrt(result.psi[0, 0] * result.psi[1, 1])
    assert -1 < corr < 1


def test_estimate_variance_components_initial_values():
    """estimate_variance_components should accept initial values."""
    y, X, Z, Z_info, _ = generate_random_intercept_data(
        n_groups=5, n_per_group=4, seed=123
    )

    initial_psi = np.array([[2.0]])
    initial_sigma2 = 1.0

    result = estimate_variance_components(
        y,
        X,
        Z,
        Z_info,
        covariance='identity',
        initial_psi=initial_psi,
        initial_sigma2=initial_sigma2,
        maxiter=500,
    )

    assert result.converged
    # Should converge to reasonable values regardless of initial guess
    assert 0.1 < result.psi[0, 0] < 10.0
    assert 0.1 < result.sigma2 < 10.0


def test_estimate_variance_components_store_matrices():
    """estimate_variance_components should store V and P if requested."""
    y, X, Z, Z_info, _ = generate_random_intercept_data(
        n_groups=5, n_per_group=4, seed=123
    )

    result = estimate_variance_components(
        y, X, Z, Z_info, covariance='identity', store_matrices=True, maxiter=500
    )

    assert result.V is not None
    assert result.P is not None
    assert result.V.shape == (20, 20)
    assert result.P.shape == (20, 20)

    # Check V is positive definite
    eigvals = np.linalg.eigvalsh(result.V)
    assert np.all(eigvals > 0)


def test_estimate_variance_components_multiple_terms_raises():
    """estimate_variance_components should handle multiple random effect terms."""
    y, X, Z, Z_info, _ = generate_random_intercept_data(n_groups=5, n_per_group=4)

    # Create Z_info with multiple terms
    # Note: Z needs to have enough columns for both terms
    n_per_term = 5
    Z_combined = np.column_stack([Z, Z])  # Duplicate for two terms

    Z_info_multiple = [
        {'n_effects': 1, 'n_groups': 5, 'grouping': 'term1', 'start_col': 0, 'end_col': 5},
        {'n_effects': 1, 'n_groups': 5, 'grouping': 'term2', 'start_col': 5, 'end_col': 10},
    ]

    # Should work with multiple terms (no longer raises)
    result = estimate_variance_components(y, X, Z_combined, Z_info_multiple, covariance='identity')

    # Result should have variance structure for each term
    assert result.converged
    assert result.psi.shape[0] >= 2  # At least 2x2 for two terms


def test_estimate_variance_components_convergence():
    """estimate_variance_components should indicate convergence status."""
    y, X, Z, Z_info, _ = generate_random_intercept_data(n_groups=5, n_per_group=4)

    # Use very low maxiter to test non-convergence
    result = estimate_variance_components(
        y, X, Z, Z_info, covariance='identity', maxiter=1
    )

    # Check convergence flag exists (may or may not converge in 1 iter)
    assert isinstance(result.converged, bool)
    assert result.n_iterations >= 0


# Tests for fixed and random effects estimation

def test_estimate_fixed_effects():
    """estimate_fixed_effects should compute GLS estimates."""
    n = 4
    y = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.ones((n, 1))
    V = 2.0 * np.eye(n)

    beta, cov_beta = estimate_fixed_effects(y, X, V)

    # For V = 2I and X = 1:
    # β = (X'V⁻¹X)⁻¹X'V⁻¹y
    # V⁻¹ = 0.5I
    # X'V⁻¹X = 1' * 0.5I * 1 = 0.5 * 4 = 2
    # (X'V⁻¹X)⁻¹ = 0.5
    # X'V⁻¹y = 1' * 0.5I * y = 0.5 * sum(y) = 5.0
    # β = 0.5 * 5.0 = 2.5 = mean(y)
    expected_beta = np.mean(y)

    assert beta.shape == (1,)
    np.testing.assert_allclose(beta, expected_beta, atol=1e-10)

    # Cov(β) = (X'V⁻¹X)⁻¹ = 0.5
    expected_cov = np.array([[0.5]])
    np.testing.assert_allclose(cov_beta, expected_cov, atol=1e-10)


def test_estimate_fixed_effects_multiple_predictors():
    """estimate_fixed_effects should work with multiple predictors."""
    n = 10
    np.random.seed(42)
    X = np.column_stack([np.ones(n), np.arange(n)])
    beta_true = np.array([1.0, 0.5])
    V = np.eye(n) * 2.0
    y = X @ beta_true + np.random.randn(n) * 0.1

    beta, cov_beta = estimate_fixed_effects(y, X, V)

    assert beta.shape == (2,)
    assert cov_beta.shape == (2, 2)

    # Should be close to true values
    np.testing.assert_allclose(beta, beta_true, atol=0.5)

    # Covariance should be positive definite
    eigvals = np.linalg.eigvalsh(cov_beta)
    assert np.all(eigvals > 0)


def test_estimate_random_effects():
    """estimate_random_effects should compute BLUPs."""
    y, X, Z, Z_info, groups = generate_random_intercept_data(
        n_groups=5,
        n_per_group=4,
        beta=2.0,
        psi=1.0,
        sigma2=0.5,
        seed=42,
    )

    # Use true parameter values
    psi = np.array([[1.0]])
    sigma2 = 0.5
    beta = np.array([2.0])

    b = estimate_random_effects(y, X, Z, beta, psi, sigma2, Z_info=Z_info)

    assert b.shape == (5,)  # 5 groups

    # BLUPs should shrink towards zero
    # Check that they're not all zero and have reasonable magnitude
    assert not np.allclose(b, 0)
    assert np.abs(b).max() < 5.0  # Reasonable shrinkage


def test_estimate_random_effects_high_correlation():
    """estimate_random_effects should handle correlated random effects."""
    y, X, Z, Z_info = generate_random_slope_data(
        n_groups=5,
        n_per_group=4,
        psi=np.array([[1.0, 0.5], [0.5, 0.8]]),
        seed=42,
    )

    psi = np.array([[1.0, 0.5], [0.5, 0.8]])
    sigma2 = 0.5
    beta = np.array([2.0, 0.5])

    b = estimate_random_effects(y, X, Z, beta, psi, sigma2, Z_info=Z_info)

    # Should have 5 groups * 2 effects = 10 coefficients
    assert b.shape == (10,)

    # Should not be all zero
    assert not np.allclose(b, 0)


# Integration tests

def test_full_estimation_pipeline():
    """Full pipeline: estimate variances, then fixed/random effects."""
    y, X, Z, Z_info, groups = generate_random_intercept_data(
        n_groups=10,
        n_per_group=5,
        beta=3.0,
        psi=1.5,
        sigma2=0.8,
        seed=123,
    )

    # Step 1: Estimate variance components
    reml_result = estimate_variance_components(
        y, X, Z, Z_info, covariance='identity', store_matrices=True, maxiter=500
    )

    assert reml_result.converged

    # Step 2: Estimate fixed effects
    beta, cov_beta = estimate_fixed_effects(y, X, reml_result.V)

    assert beta.shape == (1,)
    # Should be reasonably close to true value 3.0
    assert 2.0 < beta[0] < 4.0

    # Step 3: Estimate random effects
    b = estimate_random_effects(
        y, X, Z, beta, reml_result.psi, reml_result.sigma2, Z_info=Z_info
    )

    assert b.shape == (10,)  # 10 groups
    # Should have some variation
    assert np.std(b) > 0.1


def test_estimation_balanced_vs_unbalanced():
    """Estimation should work for both balanced and unbalanced designs."""
    # Balanced design
    y_bal, X_bal, Z_bal, Z_info_bal, _ = generate_random_intercept_data(
        n_groups=5, n_per_group=4, seed=42
    )

    result_bal = estimate_variance_components(
        y_bal, X_bal, Z_bal, Z_info_bal, covariance='identity', maxiter=500
    )

    assert result_bal.converged

    # Unbalanced design (manual construction)
    np.random.seed(42)
    n_per_group = [3, 5, 4, 6, 2]  # Unbalanced
    n_groups = 5
    n = sum(n_per_group)
    groups = np.concatenate([np.full(n_i, i) for i, n_i in enumerate(n_per_group)])

    X_unbal = np.ones((n, 1))
    Z_unbal = np.zeros((n, n_groups))
    Z_unbal[np.arange(n), groups] = 1

    # Generate data
    b_true = np.random.randn(n_groups) * np.sqrt(1.0)
    y_unbal = X_unbal @ np.array([2.0]) + Z_unbal @ b_true + np.random.randn(n) * np.sqrt(0.5)

    Z_info_unbal = [{
        'grouping': 'subject',
        'n_effects': 1,
        'n_groups': n_groups,
        'groups': np.arange(n_groups),
        'start_col': 0,
        'end_col': n_groups,
    }]

    result_unbal = estimate_variance_components(
        y_unbal, X_unbal, Z_unbal, Z_info_unbal, covariance='identity', maxiter=500
    )

    assert result_unbal.converged
    # Both should give reasonable estimates
    assert result_bal.sigma2 > 0
    assert result_unbal.sigma2 > 0


def test_reml_result_attributes():
    """REMLResult should have all expected attributes."""
    y, X, Z, Z_info, _ = generate_random_intercept_data(n_groups=5, n_per_group=4)

    result = estimate_variance_components(
        y, X, Z, Z_info, covariance='identity', store_matrices=True, maxiter=500
    )

    # Check all attributes exist
    assert hasattr(result, 'psi')
    assert hasattr(result, 'sigma2')
    assert hasattr(result, 'log_likelihood')
    assert hasattr(result, 'theta')
    assert hasattr(result, 'n_iterations')
    assert hasattr(result, 'converged')
    assert hasattr(result, 'V')
    assert hasattr(result, 'P')

    # Check types
    assert isinstance(result.psi, np.ndarray)
    assert isinstance(result.sigma2, (float, np.floating))
    assert isinstance(result.log_likelihood, (float, np.floating))
    assert isinstance(result.theta, np.ndarray)
    assert isinstance(result.n_iterations, (int, np.integer))
    assert isinstance(result.converged, bool)
