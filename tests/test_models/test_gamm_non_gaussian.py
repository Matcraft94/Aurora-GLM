"""Integration tests for non-Gaussian GAMM using fit_gamm() interface."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurora.models.gamm import RandomEffect, fit_gamm


def test_fit_gamm_poisson_random_intercept():
    """Test fit_gamm() with Poisson family and random intercepts."""
    np.random.seed(42)

    # Generate Poisson data with random intercepts
    n_groups, n_per_group = 8, 25
    n = n_groups * n_per_group

    groups = np.repeat(np.arange(n_groups), n_per_group)
    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    # True parameters
    b_true = np.random.randn(n_groups) * 0.3
    eta = 1.0 + 0.4 * x + b_true[groups]
    mu_true = np.exp(eta)
    y = np.random.poisson(mu_true)

    # Fit model
    re = RandomEffect(grouping='group')
    result = fit_gamm(
        y=y,
        X=X,
        random_effects=[re],
        groups_data={'group': groups},
        family='poisson',
        covariance='identity'
    )

    # Verify result
    assert result.converged
    assert result.family == 'poisson'
    assert result.n_groups == n_groups

    # Fixed effects should be reasonable
    assert 0.5 < result.beta_parametric[0] < 1.5  # Intercept near 1.0
    assert 0.1 < result.beta_parametric[1] < 0.7  # Slope near 0.4

    # Variance components should be positive
    assert result.variance_components[0][0, 0] > 0

    # Should have fitted values
    assert len(result.fitted_values) == n
    assert np.all(result.fitted_values > 0)


def test_fit_gamm_poisson_formula():
    """Test fit_gamm() with Poisson family using formula interface."""
    np.random.seed(123)

    # Generate data
    n_groups, n_per_group = 6, 20
    n = n_groups * n_per_group

    df = pd.DataFrame({
        'count': np.random.poisson(5, n),
        'x': np.random.randn(n),
        'subject': np.repeat(np.arange(n_groups), n_per_group)
    })

    # Add group effect to counts
    for i in range(n_groups):
        mask = df['subject'] == i
        df.loc[mask, 'count'] = np.random.poisson(
            np.exp(1.0 + 0.3 * df.loc[mask, 'x'] + np.random.randn() * 0.4),
            size=mask.sum()
        )

    # Fit with formula
    result = fit_gamm(
        formula='count ~ x + (1 | subject)',
        data=df,
        family='poisson',
        covariance='identity'
    )

    assert result.converged
    assert result.family == 'poisson'
    assert result.n_groups == n_groups
    assert len(result.beta_parametric) == 2  # Intercept + x


def test_fit_gamm_binomial_random_intercept():
    """Test fit_gamm() with Binomial family and random intercepts."""
    np.random.seed(42)

    # Generate binomial data
    n_groups, n_per_group = 6, 30
    n = n_groups * n_per_group

    groups = np.repeat(np.arange(n_groups), n_per_group)
    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    # True parameters
    b_true = np.random.randn(n_groups) * 0.5
    eta = 0.2 + 0.5 * x + b_true[groups]
    p_true = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p_true)

    # Fit model
    re = RandomEffect(grouping='group')
    result = fit_gamm(
        y=y,
        X=X,
        random_effects=[re],
        groups_data={'group': groups},
        family='binomial',
        covariance='identity'
    )

    # Verify result
    assert result.converged
    assert result.family == 'binomial'
    assert result.n_groups == n_groups

    # Fixed effects should be reasonable
    assert -0.5 < result.beta_parametric[0] < 1.0  # Intercept near 0.2
    assert 0.0 < result.beta_parametric[1] < 1.0  # Slope near 0.5

    # Fitted values should be probabilities
    assert np.all(result.fitted_values >= 0)
    assert np.all(result.fitted_values <= 1)


def test_fit_gamm_binomial_formula():
    """Test fit_gamm() with Binomial family using formula interface."""
    np.random.seed(456)

    # Generate data
    n_groups, n_per_group = 5, 25
    n = n_groups * n_per_group

    df = pd.DataFrame({
        'success': np.random.binomial(1, 0.5, n),
        'predictor': np.random.randn(n),
        'group_id': np.repeat(np.arange(n_groups), n_per_group)
    })

    # Fit with formula
    result = fit_gamm(
        formula='success ~ predictor + (1 | group_id)',
        data=df,
        family='binomial',
        covariance='identity'
    )

    assert result.converged
    assert result.family == 'binomial'
    assert len(result.fitted_values) == n


def test_fit_gamm_gamma_random_intercept():
    """Test fit_gamm() with Gamma family and random intercepts."""
    np.random.seed(789)

    # Generate gamma data
    n_groups, n_per_group = 5, 20
    n = n_groups * n_per_group

    groups = np.repeat(np.arange(n_groups), n_per_group)
    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    # True parameters
    b_true = np.random.randn(n_groups) * 0.2
    eta = 1.0 + 0.3 * x + b_true[groups]
    mu_true = np.exp(eta)  # log link

    # Generate gamma with fixed shape parameter
    shape = 2.0
    scale = mu_true / shape
    y = np.random.gamma(shape, scale)

    # Fit model
    re = RandomEffect(grouping='group')
    result = fit_gamm(
        y=y,
        X=X,
        random_effects=[re],
        groups_data={'group': groups},
        family='gamma',
        covariance='identity'
    )

    # Verify result
    assert result.converged
    assert result.family == 'gamma'
    assert result.n_groups == n_groups

    # Fitted values should be positive
    assert np.all(result.fitted_values > 0)


def test_fit_gamm_poisson_random_slope():
    """Test fit_gamm() with Poisson and random slopes."""
    np.random.seed(42)

    # Generate data with random slopes
    n_groups, n_per_group = 5, 25
    n = n_groups * n_per_group

    groups = np.repeat(np.arange(n_groups), n_per_group)
    time = np.tile(np.arange(n_per_group), n_groups)
    X = np.column_stack([np.ones(n), time])

    # Random intercepts and slopes
    b_intercept = np.random.randn(n_groups) * 0.3
    b_slope = np.random.randn(n_groups) * 0.05

    eta = 1.0 + 0.1 * time + b_intercept[groups] + b_slope[groups] * time
    mu_true = np.exp(eta)
    y = np.random.poisson(mu_true)

    # Fit with random slope
    re = RandomEffect(
        grouping='group',
        variables=(1,),  # Random slope on variable 1 (time)
        include_intercept=True
    )

    result = fit_gamm(
        y=y,
        X=X,
        random_effects=[re],
        groups_data={'group': groups},
        family='poisson',
        covariance='unstructured'  # Allow correlation
    )

    assert result.converged
    assert result.variance_components[0].shape == (2, 2)  # 2x2 for intercept + slope


def test_fit_gamm_convergence_info():
    """Test that non-Gaussian GAMM returns proper convergence info."""
    np.random.seed(42)

    n_groups, n_per_group = 4, 15
    n = n_groups * n_per_group

    groups = np.repeat(np.arange(n_groups), n_per_group)
    X = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.poisson(5, n)

    re = RandomEffect(grouping='group')
    result = fit_gamm(
        y=y,
        X=X,
        random_effects=[re],
        groups_data={'group': groups},
        family='poisson',
        maxiter=15
    )

    # Should have convergence info
    assert hasattr(result, 'converged')
    assert hasattr(result, 'n_iterations')
    assert result.n_iterations > 0
    assert result.n_iterations <= 15


def test_fit_gamm_invalid_family():
    """Test that invalid family raises appropriate error."""
    n = 50
    X = np.random.randn(n, 2)
    y = np.random.randn(n)
    groups = np.random.randint(0, 5, n)

    re = RandomEffect(grouping='group')

    with pytest.raises(ValueError, match="not supported"):
        fit_gamm(
            y=y,
            X=X,
            random_effects=[re],
            groups_data={'group': groups},
            family='invalid_family'
        )


def test_fit_gamm_diagnostics():
    """Test that non-Gaussian GAMM returns proper diagnostics."""
    np.random.seed(42)

    n_groups, n_per_group = 5, 20
    n = n_groups * n_per_group

    groups = np.repeat(np.arange(n_groups), n_per_group)
    X = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.poisson(3, n)

    re = RandomEffect(grouping='group')
    result = fit_gamm(
        y=y,
        X=X,
        random_effects=[re],
        groups_data={'group': groups},
        family='poisson'
    )

    # Should have all diagnostic fields
    assert hasattr(result, 'aic')
    assert hasattr(result, 'bic')
    assert hasattr(result, 'log_likelihood')
    assert hasattr(result, 'residuals')
    assert hasattr(result, 'fitted_values')

    # Values should be finite
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)
    assert np.isfinite(result.log_likelihood)
    assert np.all(np.isfinite(result.residuals))
    assert np.all(np.isfinite(result.fitted_values))
