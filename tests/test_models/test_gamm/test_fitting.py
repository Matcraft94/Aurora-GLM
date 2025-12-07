"""Tests for GAMM fitting (Gaussian family)."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models.gamm import RandomEffect, construct_Z_matrix
from aurora.models.gamm.fitting import (
    GAMMResult,
    compute_edf,
    fit_gamm_gaussian,
    predict_gamm,
    solve_mixed_model_equations,
)


# Helper functions for test data generation

def generate_lmm_data(
    n_groups: int = 10,
    n_per_group: int = 10,
    beta_true: np.ndarray = None,
    psi_true: float = 1.0,
    sigma2_true: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray]:
    """Generate data from simple LMM with random intercept."""
    if beta_true is None:
        beta_true = np.array([2.0, 0.5])

    np.random.seed(seed)
    n = n_groups * n_per_group

    # Fixed effects: intercept + covariate
    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    # Random effects
    groups = np.repeat(np.arange(n_groups), n_per_group)
    Z = np.zeros((n, n_groups))
    Z[np.arange(n), groups] = 1

    # Generate random intercepts
    b = np.random.randn(n_groups) * np.sqrt(psi_true)

    # Generate response
    y = X @ beta_true + Z @ b + np.random.randn(n) * np.sqrt(sigma2_true)

    Z_info = [{
        'grouping': 'subject',
        'n_effects': 1,
        'n_groups': n_groups,
        'groups': np.arange(n_groups),
        'start_col': 0,
        'end_col': n_groups,
    }]

    return y, X, Z, Z_info, groups


def generate_gamm_data_with_smooth(
    n_groups: int = 10,
    n_per_group: int = 10,
    seed: int = 42,
) -> tuple:
    """Generate data from GAMM with smooth term + random intercept."""
    np.random.seed(seed)
    n = n_groups * n_per_group

    # Smooth covariate
    x = np.linspace(0, 2 * np.pi, n)
    f_true = np.sin(x)  # True smooth function

    # Parametric covariate
    z = np.random.randn(n)

    # Fixed effects design
    X_para = np.column_stack([np.ones(n), z])
    beta_para_true = np.array([2.0, 0.5])

    # Simple basis for smooth (cubic)
    X_smooth = np.column_stack([x, x**2, x**3])
    # True coefficients chosen to approximate sin(x)
    beta_smooth_true = np.array([1.0, -0.3, 0.05])

    # Random effects
    groups = np.repeat(np.arange(n_groups), n_per_group)
    Z = np.zeros((n, n_groups))
    Z[np.arange(n), groups] = 1
    b_true = np.random.randn(n_groups) * 0.5

    # Generate response
    y = (X_para @ beta_para_true +
         X_smooth @ beta_smooth_true +
         Z @ b_true +
         np.random.randn(n) * 0.3)

    Z_info = [{
        'grouping': 'subject',
        'n_effects': 1,
        'n_groups': n_groups,
        'groups': np.arange(n_groups),
        'start_col': 0,
        'end_col': n_groups,
    }]

    # Penalty matrix for smooth (second derivative)
    S_smooth = np.array([
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 6],
    ])

    return y, X_para, {'s1': X_smooth}, Z, Z_info, {'s1': S_smooth}


# Tests for solve_mixed_model_equations

def test_solve_mixed_model_equations_basic():
    """solve_mixed_model_equations should solve simple system."""
    n = 20
    np.random.seed(42)

    X = np.random.randn(n, 2)
    Z = np.random.randn(n, 3)
    y = np.random.randn(n)

    psi_inv = np.eye(3) * 0.5  # Ψ⁻¹ = 0.5I

    beta, b = solve_mixed_model_equations(X, Z, y, psi_inv=psi_inv)

    assert beta.shape == (2,)
    assert b.shape == (3,)
    assert not np.any(np.isnan(beta))
    assert not np.any(np.isnan(b))


def test_solve_mixed_model_equations_with_penalty():
    """solve_mixed_model_equations should handle smoothing penalty."""
    n = 20
    p = 3
    np.random.seed(42)

    X = np.random.randn(n, p)
    Z = np.random.randn(n, 2)
    y = np.random.randn(n)

    # Smoothing penalty
    S_smooth = np.eye(p)
    lambda_smooth = 1.0

    psi_inv = np.eye(2)

    beta, b = solve_mixed_model_equations(
        X, Z, y,
        S_smooth=S_smooth,
        psi_inv=psi_inv,
        lambda_smooth=lambda_smooth
    )

    assert beta.shape == (p,)
    assert b.shape == (2,)


def test_solve_mixed_model_equations_no_random():
    """solve_mixed_model_equations should work with no random effects."""
    n = 20
    p = 3
    np.random.seed(42)

    X = np.random.randn(n, p)
    Z = np.zeros((n, 0))  # No random effects
    y = np.random.randn(n)

    beta, b = solve_mixed_model_equations(X, Z, y)

    assert beta.shape == (p,)
    assert b.shape == (0,)


# Tests for compute_edf

def test_compute_edf_no_penalty():
    """compute_edf should return nominal DF with no penalty."""
    n, p, q = 20, 3, 2
    X = np.random.randn(n, p)
    Z = np.random.randn(n, q)

    edf_fixed, edf_random = compute_edf(X, Z, None, None, 0.0)

    # Without penalty, EDF should be close to p and q
    assert edf_fixed > 0
    assert edf_random > 0
    assert edf_fixed <= p
    assert edf_random <= q


def test_compute_edf_with_penalty():
    """compute_edf should decrease with stronger penalty."""
    n, p, q = 20, 3, 2
    np.random.seed(42)
    X = np.random.randn(n, p)
    Z = np.random.randn(n, q)

    S_smooth = np.eye(p)
    psi_inv = np.eye(q)

    # Weak penalty
    edf_fixed_weak, _ = compute_edf(X, Z, S_smooth, psi_inv, lambda_smooth=0.1)

    # Strong penalty
    edf_fixed_strong, _ = compute_edf(X, Z, S_smooth, psi_inv, lambda_smooth=10.0)

    # EDF should decrease with stronger penalty
    assert edf_fixed_strong < edf_fixed_weak


# Tests for fit_gamm_gaussian

def test_fit_gamm_gaussian_random_intercept():
    """fit_gamm_gaussian should fit random intercept model."""
    y, X, Z, Z_info, groups = generate_lmm_data(
        n_groups=5, n_per_group=10, seed=42
    )

    result = fit_gamm_gaussian(
        X_parametric=X,
        X_smooth=None,
        Z=Z,
        Z_info=Z_info,
        y=y,
        covariance='identity',
    )

    assert isinstance(result, GAMMResult)
    assert result.beta_parametric.shape == (2,)
    assert result.converged
    assert result.n_groups == 5
    assert result.n_obs == 50

    # Check variance components
    assert result.variance_components[0].shape == (1, 1)
    assert result.residual_variance > 0

    # Check fitted values
    assert result.fitted_values.shape == (50,)
    assert result.residuals.shape == (50,)

    # Fixed effects should be reasonable
    assert 1.0 < result.beta_parametric[0] < 3.0  # Intercept near 2.0
    assert 0.0 < result.beta_parametric[1] < 1.0  # Slope near 0.5


def test_fit_gamm_gaussian_with_smooth_term():
    """fit_gamm_gaussian should fit model with smooth term."""
    y, X_para, X_smooth, Z, Z_info, S_smooth = generate_gamm_data_with_smooth(
        n_groups=8, n_per_group=10, seed=42
    )

    lambda_smooth = {'s1': 0.1}

    result = fit_gamm_gaussian(
        X_parametric=X_para,
        X_smooth=X_smooth,
        Z=Z,
        Z_info=Z_info,
        y=y,
        S_smooth=S_smooth,
        lambda_smooth=lambda_smooth,
        covariance='identity',
    )

    assert result.converged
    assert result.beta_parametric.shape == (2,)
    assert 's1' in result.beta_smooth
    assert result.beta_smooth['s1'].shape == (3,)

    # Check EDF
    assert result.edf_parametric == 2.0
    assert 's1' in result.edf_smooth
    assert result.edf_smooth['s1'] > 0

    # Check AIC/BIC
    assert result.aic > 0
    assert result.bic > 0
    assert result.bic > result.aic  # BIC penalizes more


def test_fit_gamm_gaussian_random_effects_extracted():
    """fit_gamm_gaussian should extract random effects by group."""
    y, X, Z, Z_info, groups = generate_lmm_data(
        n_groups=5, n_per_group=8, seed=123
    )

    result = fit_gamm_gaussian(
        X_parametric=X,
        X_smooth=None,
        Z=Z,
        Z_info=Z_info,
        y=y,
        covariance='identity',
    )

    # Check random effects structure
    # extract_random_effects returns: {'grouping_var': {group_id: array, ...}}
    assert isinstance(result.random_effects, dict)
    assert 'subject' in result.random_effects
    assert len(result.random_effects['subject']) == 5  # 5 groups

    # Each group should have 1 random effect (intercept)
    for group_id in range(5):
        assert group_id in result.random_effects['subject']
        assert result.random_effects['subject'][group_id].shape == (1,)


def test_fit_gamm_gaussian_multiple_smooth_terms():
    """fit_gamm_gaussian should handle multiple smooth terms."""
    np.random.seed(42)
    n = 50

    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)

    X_para = np.ones((n, 1))
    X_smooth = {
        's1': np.column_stack([x1, x1**2]),
        's2': np.column_stack([x2, x2**2]),
    }

    # Random effects
    groups = np.repeat(np.arange(5), 10)
    Z = np.zeros((n, 5))
    Z[np.arange(n), groups] = 1

    # Generate response
    y = (2.0 +
         (X_smooth['s1'] @ [1.0, -0.5]) +
         (X_smooth['s2'] @ [0.5, 0.3]) +
         Z @ np.random.randn(5) * 0.3 +
         np.random.randn(n) * 0.2)

    Z_info = [{
        'grouping': 'subject',
        'n_effects': 1,
        'n_groups': 5,
        'groups': np.arange(5),
        'start_col': 0,
        'end_col': 5,
    }]

    S_smooth = {
        's1': np.eye(2),
        's2': np.eye(2),
    }

    lambda_smooth = {'s1': 0.1, 's2': 0.1}

    result = fit_gamm_gaussian(
        X_parametric=X_para,
        X_smooth=X_smooth,
        Z=Z,
        Z_info=Z_info,
        y=y,
        S_smooth=S_smooth,
        lambda_smooth=lambda_smooth,
    )

    assert result.converged
    assert 's1' in result.beta_smooth
    assert 's2' in result.beta_smooth
    assert 's1' in result.edf_smooth
    assert 's2' in result.edf_smooth


def test_fit_gamm_gaussian_edf_calculation():
    """fit_gamm_gaussian should compute reasonable EDF."""
    y, X, Z, Z_info, _ = generate_lmm_data(
        n_groups=10, n_per_group=10, seed=42
    )

    result = fit_gamm_gaussian(
        X_parametric=X,
        X_smooth=None,
        Z=Z,
        Z_info=Z_info,
        y=y,
        covariance='identity',
    )

    # EDF should be positive and reasonable
    assert result.edf_total > 0
    assert result.edf_total < result.n_obs  # Must be less than n

    # Parametric EDF should be nominal (no penalty on parametric terms)
    assert result.edf_parametric == 2.0


def test_fit_gamm_gaussian_unstructured_covariance():
    """fit_gamm_gaussian should handle unstructured covariance."""
    # Generate data with random slope
    np.random.seed(42)
    n_groups = 5
    n_per_group = 10
    n = n_groups * n_per_group

    time = np.tile(np.arange(n_per_group), n_groups)
    groups = np.repeat(np.arange(n_groups), n_per_group)

    X = np.column_stack([np.ones(n), time])

    # Random effects: intercept + slope
    groups_data = {'subject': groups}
    re = RandomEffect(grouping='subject', variables=(1,))
    Z, Z_info = construct_Z_matrix(X, [re], groups_data)

    # Generate response
    psi_true = np.array([[1.0, 0.3], [0.3, 0.5]])
    L = np.linalg.cholesky(psi_true)
    b_raw = np.random.randn(n_groups, 2)
    b = (L @ b_raw.T).T.flatten()

    y = X @ [2.0, 0.5] + Z @ b + np.random.randn(n) * 0.3

    result = fit_gamm_gaussian(
        X_parametric=X,
        X_smooth=None,
        Z=Z,
        Z_info=Z_info,
        y=y,
        covariance='unstructured',
    )

    assert result.converged
    assert result.variance_components[0].shape == (2, 2)

    # Check positive definite
    eigvals = np.linalg.eigvalsh(result.variance_components[0])
    assert np.all(eigvals > 0)


# Tests for predict_gamm

def test_predict_gamm_without_random():
    """predict_gamm should make population-level predictions."""
    y, X, Z, Z_info, _ = generate_lmm_data(
        n_groups=5, n_per_group=10, seed=42
    )

    result = fit_gamm_gaussian(
        X_parametric=X,
        X_smooth=None,
        Z=Z,
        Z_info=Z_info,
        y=y,
        covariance='identity',
    )

    # New data (same structure)
    n_new = 20
    X_new = np.column_stack([np.ones(n_new), np.random.randn(n_new)])

    pred = predict_gamm(result, X_new, include_random=False)

    assert pred.shape == (n_new,)
    assert not np.any(np.isnan(pred))


def test_predict_gamm_with_random():
    """predict_gamm should include random effects when requested."""
    y, X, Z, Z_info, _ = generate_lmm_data(
        n_groups=5, n_per_group=10, seed=42
    )

    result = fit_gamm_gaussian(
        X_parametric=X,
        X_smooth=None,
        Z=Z,
        Z_info=Z_info,
        y=y,
        covariance='identity',
    )

    # New data with random effects
    n_new = 20
    X_new = np.column_stack([np.ones(n_new), np.random.randn(n_new)])
    groups_new = np.repeat(np.arange(5), 4)  # Use existing groups
    Z_new = np.zeros((n_new, 5))
    Z_new[np.arange(n_new), groups_new] = 1

    pred_with_random = predict_gamm(
        result, X_new, Z_new=Z_new, include_random=True
    )
    pred_without_random = predict_gamm(
        result, X_new, include_random=False
    )

    assert pred_with_random.shape == (n_new,)
    assert pred_without_random.shape == (n_new,)

    # Predictions should differ
    assert not np.allclose(pred_with_random, pred_without_random)


def test_predict_gamm_with_smooth():
    """predict_gamm should work with smooth terms."""
    y, X_para, X_smooth, Z, Z_info, S_smooth = generate_gamm_data_with_smooth(
        n_groups=5, n_per_group=10, seed=42
    )

    result = fit_gamm_gaussian(
        X_parametric=X_para,
        X_smooth=X_smooth,
        Z=Z,
        Z_info=Z_info,
        y=y,
        S_smooth=S_smooth,
        lambda_smooth={'s1': 0.1},
    )

    # New data
    n_new = 15
    x_new = np.linspace(0, 2 * np.pi, n_new)
    X_para_new = np.column_stack([np.ones(n_new), np.random.randn(n_new)])
    X_smooth_new = {'s1': np.column_stack([x_new, x_new**2, x_new**3])}

    pred = predict_gamm(
        result,
        X_para_new,
        X_smooth_new=X_smooth_new,
        include_random=False
    )

    assert pred.shape == (n_new,)
    assert not np.any(np.isnan(pred))


# Integration tests

def test_gamm_result_attributes():
    """GAMMResult should have all expected attributes."""
    y, X, Z, Z_info, _ = generate_lmm_data(
        n_groups=5, n_per_group=10, seed=42
    )

    result = fit_gamm_gaussian(
        X_parametric=X,
        X_smooth=None,
        Z=Z,
        Z_info=Z_info,
        y=y,
    )

    # Check all attributes exist
    assert hasattr(result, 'coefficients')
    assert hasattr(result, 'beta_parametric')
    assert hasattr(result, 'beta_smooth')
    assert hasattr(result, 'random_effects')
    assert hasattr(result, 'variance_components')
    assert hasattr(result, 'residual_variance')
    assert hasattr(result, 'smoothing_parameters')
    assert hasattr(result, 'edf_total')
    assert hasattr(result, 'edf_parametric')
    assert hasattr(result, 'edf_smooth')
    assert hasattr(result, 'fitted_values')
    assert hasattr(result, 'residuals')
    assert hasattr(result, 'log_likelihood')
    assert hasattr(result, 'aic')
    assert hasattr(result, 'bic')
    assert hasattr(result, 'converged')
    assert hasattr(result, 'n_iterations')
    assert hasattr(result, 'n_obs')
    assert hasattr(result, 'n_groups')
    assert hasattr(result, 'family')

    # Check types
    assert isinstance(result.beta_parametric, np.ndarray)
    assert isinstance(result.beta_smooth, dict)
    assert isinstance(result.random_effects, dict)
    assert isinstance(result.variance_components, list)
    assert len(result.variance_components) > 0
    assert isinstance(result.variance_components[0], np.ndarray)
    assert isinstance(result.residual_variance, (float, np.floating))
    assert isinstance(result.fitted_values, np.ndarray)
    assert isinstance(result.residuals, np.ndarray)
    assert isinstance(result.converged, bool)
    assert result.family == 'gaussian'


def test_gamm_fit_quality():
    """GAMM fit should have reasonable quality metrics."""
    y, X, Z, Z_info, _ = generate_lmm_data(
        n_groups=10, n_per_group=15,
        beta_true=np.array([3.0, 0.8]),
        psi_true=1.2,
        sigma2_true=0.6,
        seed=42
    )

    result = fit_gamm_gaussian(
        X_parametric=X,
        X_smooth=None,
        Z=Z,
        Z_info=Z_info,
        y=y,
        covariance='identity',
    )

    # Fixed effects should be close to true values
    assert 2.5 < result.beta_parametric[0] < 3.5  # True: 3.0
    assert 0.5 < result.beta_parametric[1] < 1.1  # True: 0.8

    # Variance components should be reasonable
    assert 0.5 < result.variance_components[0][0, 0] < 2.5  # True: 1.2
    assert 0.3 < result.residual_variance < 1.2  # True: 0.6

    # Residuals should have reasonable properties
    assert np.abs(np.mean(result.residuals)) < 0.2  # Close to 0
    assert 0.5 < np.std(result.residuals) < 1.5  # Reasonable spread
