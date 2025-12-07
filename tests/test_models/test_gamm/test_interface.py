"""Integration tests for high-level GAMM interface."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurora.models.gamm import (
    RandomEffect,
    fit_gamm,
    fit_gamm_with_smooth,
    predict_from_gamm,
)


def test_fit_gamm_basic():
    """fit_gamm should fit simple random intercept model."""
    np.random.seed(42)

    # Generate data
    n_groups, n_per_group = 5, 20
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)
    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    # Random intercepts
    b_true = np.random.randn(n_groups) * 0.8
    y = 2.0 + 0.5 * x + b_true[groups] + np.random.randn(n) * 0.4

    # Fit model
    re = RandomEffect(grouping="subject")
    result = fit_gamm(
        y=y, X=X, random_effects=[re], groups_data={"subject": groups}, covariance="identity"
    )

    # Check results
    assert result.converged
    assert result.n_groups == 5
    assert result.n_obs == 100
    assert result.family == "gaussian"

    # Fixed effects should be reasonable
    assert 1.5 < result.beta_parametric[0] < 2.5  # Intercept
    assert 0.2 < result.beta_parametric[1] < 0.8  # Slope


def test_fit_gamm_with_pandas():
    """fit_gamm should accept pandas inputs."""
    np.random.seed(42)

    # Generate data
    n_groups, n_per_group = 5, 10
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)
    x = np.random.randn(n)

    # Create DataFrame
    df = pd.DataFrame({"y": 2.0 + 0.5 * x + np.random.randn(n) * 0.5, "x": x, "subject": groups})

    # Fit model
    re = RandomEffect(grouping="subject")
    result = fit_gamm(
        y=df["y"], X=df[["x"]], random_effects=[re], groups_data=df[["subject"]], covariance="identity"
    )

    assert result.converged
    assert result.n_groups == 5


def test_fit_gamm_intercept_only():
    """fit_gamm should fit intercept-only model when X is None."""
    np.random.seed(42)

    # Generate data
    n_groups, n_per_group = 5, 10
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    b_true = np.random.randn(n_groups) * 0.8
    y = 2.0 + b_true[groups] + np.random.randn(n) * 0.5

    # Fit model
    re = RandomEffect(grouping="subject")
    result = fit_gamm(y=y, X=None, random_effects=[re], groups_data={"subject": groups}, covariance="identity")

    assert result.converged
    assert result.beta_parametric.shape == (1,)  # Intercept only
    assert 1.5 < result.beta_parametric[0] < 2.5


def test_fit_gamm_random_slope():
    """fit_gamm should fit random intercept + slope model."""
    np.random.seed(42)

    # Generate data
    n_groups, n_per_group = 8, 15
    n = n_groups * n_per_group
    time = np.tile(np.arange(n_per_group), n_groups)
    groups = np.repeat(np.arange(n_groups), n_per_group)

    X = np.column_stack([np.ones(n), time])

    # Random intercepts and slopes
    b_int = np.random.randn(n_groups) * 0.5
    b_slope = np.random.randn(n_groups) * 0.1

    y = 2.0 + 0.5 * time + b_int[groups] + b_slope[groups] * time + np.random.randn(n) * 0.3

    # Fit model with random slope
    re = RandomEffect(grouping="subject", variables=(1,))
    result = fit_gamm(
        y=y, X=X, random_effects=[re], groups_data={"subject": groups}, covariance="unstructured"
    )

    assert result.converged
    assert result.variance_components[0].shape == (2, 2)

    # Check positive definite
    eigvals = np.linalg.eigvalsh(result.variance_components[0])
    assert np.all(eigvals > 0)


def test_fit_gamm_no_random_raises():
    """fit_gamm should raise if no random effects provided."""
    np.random.seed(42)

    n = 50
    X = np.random.randn(n, 2)
    y = np.random.randn(n)

    with pytest.raises(ValueError, match="At least one random effect required"):
        fit_gamm(y=y, X=X, random_effects=None, groups_data=None)


def test_fit_gamm_non_gaussian_supported():
    """fit_gamm now supports non-Gaussian families via PQL."""
    np.random.seed(42)

    n = 50
    groups = np.repeat(np.arange(5), 10)
    X = np.ones((n, 1))
    # Generate count data for Poisson
    y = np.random.poisson(3, n)

    re = RandomEffect(grouping="subject")

    # Should now work with non-Gaussian families (Phase 5 implementation)
    result = fit_gamm(y=y, X=X, random_effects=[re], groups_data={"subject": groups}, family="poisson")
    
    # Verify result structure (uses beta_parametric and random_effects attributes)
    assert hasattr(result, 'beta_parametric')
    assert hasattr(result, 'random_effects')


def test_fit_gamm_missing_groups_raises():
    """fit_gamm should raise if groups_data not provided with random effects."""
    np.random.seed(42)

    n = 50
    X = np.random.randn(n, 2)
    y = np.random.randn(n)

    re = RandomEffect(grouping="subject")

    with pytest.raises(ValueError, match="groups_data must be provided"):
        fit_gamm(y=y, X=X, random_effects=[re], groups_data=None)


def test_fit_gamm_mismatched_dimensions_raises():
    """fit_gamm should raise if y and X dimensions don't match."""
    np.random.seed(42)

    y = np.random.randn(50)
    X = np.random.randn(40, 2)  # Wrong size
    groups = np.repeat(np.arange(5), 10)

    re = RandomEffect(grouping="subject")

    with pytest.raises(ValueError, match="Length of y.*must match"):
        fit_gamm(y=y, X=X, random_effects=[re], groups_data={"subject": groups})


def test_fit_gamm_with_smooth_basic():
    """fit_gamm_with_smooth should fit model with smooth term."""
    np.random.seed(42)

    # Generate data
    n_groups, n_per_group = 5, 20
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    x_smooth = np.linspace(0, 2 * np.pi, n)
    f_true = np.sin(x_smooth)

    X_para = np.ones((n, 1))
    X_smooth = {"s1": np.column_stack([x_smooth, x_smooth**2, x_smooth**3])}

    b_true = np.random.randn(n_groups) * 0.3
    y = 2.0 + f_true + b_true[groups] + np.random.randn(n) * 0.3

    # Penalty matrix
    S_smooth = {"s1": np.diag([0, 1, 2])}
    lambda_smooth = {"s1": 0.1}

    # Fit model
    re = RandomEffect(grouping="subject")
    result = fit_gamm_with_smooth(
        y=y,
        X_parametric=X_para,
        X_smooth=X_smooth,
        S_smooth=S_smooth,
        random_effects=[re],
        groups_data={"subject": groups},
        lambda_smooth=lambda_smooth,
        covariance="identity",
    )

    assert result.converged
    assert "s1" in result.beta_smooth
    assert result.beta_smooth["s1"].shape == (3,)
    assert "s1" in result.edf_smooth


def test_predict_from_gamm_population():
    """predict_from_gamm should make population-level predictions."""
    np.random.seed(42)

    # Generate and fit model
    n_groups, n_per_group = 5, 20
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)
    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    b_true = np.random.randn(n_groups) * 0.8
    y = 2.0 + 0.5 * x + b_true[groups] + np.random.randn(n) * 0.4

    re = RandomEffect(grouping="subject")
    result = fit_gamm(
        y=y, X=X, random_effects=[re], groups_data={"subject": groups}, covariance="identity"
    )

    # New data
    n_new = 30
    X_new = np.column_stack([np.ones(n_new), np.random.randn(n_new)])

    pred = predict_from_gamm(result, X_new, include_random=False)

    assert pred.shape == (n_new,)
    assert not np.any(np.isnan(pred))


def test_predict_from_gamm_conditional():
    """predict_from_gamm should make conditional predictions."""
    np.random.seed(42)

    # Generate and fit model
    n_groups, n_per_group = 5, 20
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)
    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    b_true = np.random.randn(n_groups) * 0.8
    y = 2.0 + 0.5 * x + b_true[groups] + np.random.randn(n) * 0.4

    re = RandomEffect(grouping="subject")
    result = fit_gamm(
        y=y, X=X, random_effects=[re], groups_data={"subject": groups}, covariance="identity"
    )

    # New data with groups
    n_new = 20
    X_new = np.column_stack([np.ones(n_new), np.random.randn(n_new)])
    groups_new = np.repeat(np.arange(5), 4)  # Use existing groups

    pred_pop = predict_from_gamm(result, X_new, include_random=False)
    pred_cond = predict_from_gamm(result, X_new, groups_new=groups_new, include_random=True)

    assert pred_pop.shape == (n_new,)
    assert pred_cond.shape == (n_new,)

    # Predictions should differ
    assert not np.allclose(pred_pop, pred_cond)


def test_predict_from_gamm_with_pandas():
    """predict_from_gamm should accept pandas inputs."""
    np.random.seed(42)

    # Generate and fit model
    n_groups, n_per_group = 5, 10
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)
    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    y = 2.0 + 0.5 * x + np.random.randn(n_groups)[groups] * 0.5 + np.random.randn(n) * 0.3

    re = RandomEffect(grouping="subject")
    result = fit_gamm(
        y=y, X=X, random_effects=[re], groups_data={"subject": groups}, covariance="identity"
    )

    # New data as DataFrame
    df_new = pd.DataFrame({"intercept": np.ones(15), "x": np.random.randn(15)})

    pred = predict_from_gamm(result, df_new)

    assert pred.shape == (15,)


def test_predict_from_gamm_missing_groups_raises():
    """predict_from_gamm should raise if groups_new missing with include_random=True."""
    np.random.seed(42)

    # Fit model
    n_groups, n_per_group = 5, 10
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)
    X = np.random.randn(n, 2)
    y = np.random.randn(n)

    re = RandomEffect(grouping="subject")
    result = fit_gamm(
        y=y, X=X, random_effects=[re], groups_data={"subject": groups}, covariance="identity"
    )

    # Try to predict without groups
    X_new = np.random.randn(10, 2)

    with pytest.raises(ValueError, match="groups_new required"):
        predict_from_gamm(result, X_new, include_random=True)


def test_end_to_end_workflow():
    """Complete workflow: fit, predict, extract results."""
    np.random.seed(123)

    # Generate data
    n_groups, n_per_group = 10, 15
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    X = np.column_stack([np.ones(n), x1, x2])

    # True model: y = 3 + 0.5*x1 - 0.3*x2 + b_i + ε
    b_true = np.random.randn(n_groups) * 1.0
    y = 3.0 + 0.5 * x1 - 0.3 * x2 + b_true[groups] + np.random.randn(n) * 0.6

    # Fit GAMM
    re = RandomEffect(grouping="subject")
    result = fit_gamm(
        y=y, X=X, random_effects=[re], groups_data={"subject": groups}, covariance="identity"
    )

    # Check convergence
    assert result.converged

    # Check fixed effects
    assert 2.5 < result.beta_parametric[0] < 3.5  # Intercept
    assert 0.3 < result.beta_parametric[1] < 0.7  # x1 slope
    assert -0.5 < result.beta_parametric[2] < -0.1  # x2 slope

    # Check variance components
    assert 0.5 < result.variance_components[0][0, 0] < 2.0  # True: 1.0
    assert 0.3 < result.residual_variance < 1.0  # True: 0.6

    # Check random effects extracted
    assert "subject" in result.random_effects
    assert len(result.random_effects["subject"]) == 10

    # Make predictions
    n_new = 20
    X_new = np.column_stack([np.ones(n_new), np.random.randn(n_new), np.random.randn(n_new)])

    pred_pop = predict_from_gamm(result, X_new)
    assert pred_pop.shape == (n_new,)

    # Conditional predictions for existing groups
    groups_new = np.random.choice(n_groups, n_new)
    pred_cond = predict_from_gamm(result, X_new, groups_new=groups_new, include_random=True)
    assert pred_cond.shape == (n_new,)

    # Check diagnostics
    assert result.aic > 0
    assert result.bic > result.aic
    assert result.edf_total > 0
    assert len(result.residuals) == n
    assert len(result.fitted_values) == n
