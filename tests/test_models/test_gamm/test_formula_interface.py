"""Integration tests for formula-based GAMM interface."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurora.models.gamm import fit_gamm
from aurora.models.gamm.random_effects import RandomEffect


def test_formula_random_intercept():
    """Test formula mode with random intercept."""
    np.random.seed(42)

    # Generate data
    n_groups, n_per_group = 5, 20
    n = n_groups * n_per_group

    data = pd.DataFrame({
        'y': np.random.randn(n),
        'x1': np.random.randn(n),
        'subject': np.repeat(np.arange(n_groups), n_per_group),
    })

    # Add random effects to y
    b_true = np.random.randn(n_groups) * 0.5
    data['y'] = 2.0 + 0.5 * data['x1'] + b_true[data['subject']] + np.random.randn(n) * 0.3

    # Fit with formula
    result = fit_gamm(
        formula="y ~ x1 + (1 | subject)",
        data=data,
        covariance='identity',
    )

    # Check results
    assert result.converged
    assert result.beta_parametric.shape == (2,)  # Intercept + x1
    assert 1.5 < result.beta_parametric[0] < 2.5  # Intercept around 2.0
    assert 0.2 < result.beta_parametric[1] < 0.8  # Slope around 0.5
    assert len(result.variance_components) > 0


def test_formula_random_intercept_and_slope():
    """Test formula mode with random intercept + slope."""
    np.random.seed(123)

    # Generate data
    n_groups, n_per_group = 8, 15
    n = n_groups * n_per_group

    data = pd.DataFrame({
        'y': np.zeros(n),
        'time': np.tile(np.arange(n_per_group), n_groups),
        'subject': np.repeat(np.arange(n_groups), n_per_group),
    })

    # Generate with random intercepts and slopes
    b0_true = np.random.randn(n_groups) * 0.3
    b1_true = np.random.randn(n_groups) * 0.1
    for i in range(n):
        subj = data.loc[i, 'subject']
        data.loc[i, 'y'] = (
            3.0 + b0_true[subj] + (0.2 + b1_true[subj]) * data.loc[i, 'time']
            + np.random.randn() * 0.2
        )

    # Fit with formula
    result = fit_gamm(
        formula="y ~ time + (1 + time | subject)",
        data=data,
        covariance='unstructured',
    )

    # Check results
    assert result.converged
    assert result.beta_parametric.shape == (2,)
    assert 2.5 < result.beta_parametric[0] < 3.5  # Intercept
    assert 0.0 < result.beta_parametric[1] < 0.4  # Slope

    # Should have 2x2 covariance matrix for random effects
    assert result.variance_components[0].shape == (2, 2)


def test_formula_crossed_random_effects():
    """Test formula with crossed random effects."""
    np.random.seed(42)

    # Generate data with crossed structure
    n_subjects = 6
    n_items = 5
    n = n_subjects * n_items

    data = pd.DataFrame({
        'y': np.zeros(n),
        'x1': np.random.randn(n),
        'subject': np.repeat(np.arange(n_subjects), n_items),
        'item': np.tile(np.arange(n_items), n_subjects),
    })

    # Generate with crossed random effects
    b_subj = np.random.randn(n_subjects) * 0.4
    b_item = np.random.randn(n_items) * 0.3

    for i in range(n):
        subj = data.loc[i, 'subject']
        item = data.loc[i, 'item']
        data.loc[i, 'y'] = (
            2.0 + 0.5 * data.loc[i, 'x1']
            + b_subj[subj] + b_item[item]
            + np.random.randn() * 0.2
        )

    # Fit with crossed random effects
    result = fit_gamm(
        formula="y ~ x1 + (1 | subject) + (1 | item)",
        data=data,
        covariance='identity',
    )

    assert result.converged
    assert result.beta_parametric.shape == (2,)
    # Should have 2 variance components (one per grouping variable)
    assert len(result.variance_components) == 2


def test_formula_nested_random_effects():
    """Test formula with nested random effects."""
    np.random.seed(42)

    # Generate data with nested structure
    n_clinics = 3
    n_subjects_per_clinic = 4
    n_obs_per_subject = 5
    n = n_clinics * n_subjects_per_clinic * n_obs_per_subject

    data = pd.DataFrame({
        'y': np.zeros(n),
        'x1': np.random.randn(n),
        'clinic': np.repeat(np.arange(n_clinics), n_subjects_per_clinic * n_obs_per_subject),
        'subject': np.repeat(
            np.arange(n_clinics * n_subjects_per_clinic),
            n_obs_per_subject
        ),
    })

    # Generate with nested structure
    b_clinic = np.random.randn(n_clinics) * 0.5
    b_subject = np.random.randn(n_clinics * n_subjects_per_clinic) * 0.3

    for i in range(n):
        clinic = data.loc[i, 'clinic']
        subject = data.loc[i, 'subject']
        data.loc[i, 'y'] = (
            2.0 + 0.3 * data.loc[i, 'x1']
            + b_clinic[clinic] + b_subject[subject]
            + np.random.randn() * 0.2
        )

    # Fit with nested random effects
    result = fit_gamm(
        formula="y ~ x1 + (1 | clinic/subject)",
        data=data,
        covariance='identity',
    )

    assert result.converged
    assert result.beta_parametric.shape == (2,)
    # Nested creates 2 random effect terms
    assert len(result.variance_components) == 2


def test_formula_without_fixed_effects():
    """Test formula with only intercept and random effects."""
    np.random.seed(42)

    n_groups, n_per_group = 5, 10
    n = n_groups * n_per_group

    data = pd.DataFrame({
        'y': np.random.randn(n),
        'subject': np.repeat(np.arange(n_groups), n_per_group),
    })

    # Generate with random intercepts only
    b_true = np.random.randn(n_groups)
    data['y'] = 3.0 + b_true[data['subject']] + np.random.randn(n) * 0.5

    # Fit intercept-only model
    result = fit_gamm(
        formula="y ~ (1 | subject)",
        data=data,
        covariance='identity',
    )

    assert result.converged
    assert result.beta_parametric.shape == (1,)  # Intercept only
    assert 2.0 < result.beta_parametric[0] < 4.0


def test_formula_multiple_fixed_effects():
    """Test formula with multiple fixed effects."""
    np.random.seed(42)

    n_groups, n_per_group = 6, 15
    n = n_groups * n_per_group

    data = pd.DataFrame({
        'y': np.random.randn(n),
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'subject': np.repeat(np.arange(n_groups), n_per_group),
    })

    # Generate
    b_true = np.random.randn(n_groups) * 0.4
    data['y'] = (
        2.0 + 0.5*data['x1'] + 0.3*data['x2'] - 0.4*data['x3']
        + b_true[data['subject']] + np.random.randn(n) * 0.3
    )

    # Fit with multiple predictors
    result = fit_gamm(
        formula="y ~ x1 + x2 + x3 + (1 | subject)",
        data=data,
        covariance='identity',
    )

    assert result.converged
    assert result.beta_parametric.shape == (4,)  # Intercept + 3 predictors


def test_formula_mode_validation():
    """Test validation errors in formula mode."""
    data = pd.DataFrame({
        'y': np.random.randn(50),
        'x1': np.random.randn(50),
        'subject': np.repeat(np.arange(5), 10),
    })

    # Missing data
    with pytest.raises(ValueError, match="data must be provided"):
        fit_gamm(formula="y ~ x1 + (1 | subject)")

    # Cannot mix modes
    with pytest.raises(ValueError, match="Cannot mix"):
        fit_gamm(
            formula="y ~ x1 + (1 | subject)",
            data=data,
            y=data['y'].values,
        )

    # Response not in data
    with pytest.raises(ValueError, match="not found in data"):
        fit_gamm(formula="z ~ x1 + (1 | subject)", data=data)

    # Variable not in data
    with pytest.raises(ValueError, match="not found in data"):
        fit_gamm(formula="y ~ x999 + (1 | subject)", data=data)

    # Grouping variable not in data
    with pytest.raises(ValueError, match="not found in data"):
        fit_gamm(formula="y ~ x1 + (1 | clinic)", data=data)


def test_matrix_mode_still_works():
    """Ensure matrix mode (original interface) still works."""
    np.random.seed(42)

    n_groups, n_per_group = 5, 20
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)
    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])

    b_true = np.random.randn(n_groups) * 0.5
    y = 2.0 + 0.5*x + b_true[groups] + np.random.randn(n)*0.3

    # Fit with matrix mode
    re = RandomEffect(grouping='subject')
    result = fit_gamm(
        y=y,
        X=X,
        random_effects=[re],
        groups_data={'subject': groups},
        covariance='identity',
    )

    assert result.converged
    assert result.beta_parametric.shape == (2,)


def test_formula_with_dict_data():
    """Test formula mode with dict instead of DataFrame."""
    np.random.seed(42)

    n_groups, n_per_group = 4, 10
    n = n_groups * n_per_group

    data = {
        'y': np.random.randn(n),
        'x1': np.random.randn(n),
        'subject': np.repeat(np.arange(n_groups), n_per_group),
    }

    # Generate
    b_true = np.random.randn(n_groups) * 0.5
    data['y'] = 2.0 + 0.5*data['x1'] + b_true[data['subject']] + np.random.randn(n)*0.3

    # Fit with dict
    result = fit_gamm(
        formula="y ~ x1 + (1 | subject)",
        data=data,
        covariance='identity',
    )

    assert result.converged
    assert result.beta_parametric.shape == (2,)


def test_formula_random_slope_adds_variable_to_X():
    """Test that variables in random slopes are automatically added to X."""
    np.random.seed(42)

    n_groups, n_per_group = 5, 15
    n = n_groups * n_per_group

    data = pd.DataFrame({
        'y': np.random.randn(n),
        'time': np.tile(np.arange(n_per_group), n_groups),
        'subject': np.repeat(np.arange(n_groups), n_per_group),
    })

    # Generate
    b0 = np.random.randn(n_groups) * 0.3
    b1 = np.random.randn(n_groups) * 0.1
    for i in range(n):
        subj = data.loc[i, 'subject']
        data.loc[i, 'y'] = (
            2.0 + b0[subj] + (0.5 + b1[subj]) * data.loc[i, 'time']
            + np.random.randn() * 0.2
        )

    # Formula doesn't include 'time' as fixed effect,
    # but it's in random slope
    result = fit_gamm(
        formula="y ~ (1 + time | subject)",
        data=data,
        covariance='unstructured',
    )

    assert result.converged
    # Should have intercept + time in X (time added automatically)
    assert result.beta_parametric.shape == (2,)


def test_formula_diagonal_covariance():
    """Test formula mode with diagonal covariance."""
    np.random.seed(42)

    n_groups, n_per_group = 5, 15
    n = n_groups * n_per_group

    data = pd.DataFrame({
        'y': np.random.randn(n),
        'time': np.tile(np.arange(n_per_group), n_groups),
        'subject': np.repeat(np.arange(n_groups), n_per_group),
    })

    b0 = np.random.randn(n_groups) * 0.3
    b1 = np.random.randn(n_groups) * 0.1
    for i in range(n):
        subj = data.loc[i, 'subject']
        data.loc[i, 'y'] = (
            2.0 + b0[subj] + (0.5 + b1[subj]) * data.loc[i, 'time']
            + np.random.randn() * 0.2
        )

    # Diagonal covariance (independent random effects)
    result = fit_gamm(
        formula="y ~ time + (1 + time | subject)",
        data=data,
        covariance='diagonal',
    )

    assert result.converged
    # Diagonal: off-diagonal elements should be zero or very small
    cov = result.variance_components[0]
    assert cov.shape == (2, 2)
    assert abs(cov[0, 1]) < 0.1  # Off-diagonal near zero
    assert abs(cov[1, 0]) < 0.1


def test_formula_comparison_with_matrix_mode():
    """Compare formula mode results with matrix mode."""
    np.random.seed(42)

    n_groups, n_per_group = 5, 20
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)
    x = np.random.randn(n)

    b_true = np.random.randn(n_groups) * 0.5
    y = 2.0 + 0.5*x + b_true[groups] + np.random.randn(n)*0.3

    # Formula mode
    data = pd.DataFrame({'y': y, 'x1': x, 'subject': groups})
    result_formula = fit_gamm(
        formula="y ~ x1 + (1 | subject)",
        data=data,
        covariance='identity',
    )

    # Matrix mode
    X = np.column_stack([np.ones(n), x])
    re = RandomEffect(grouping='subject')
    result_matrix = fit_gamm(
        y=y,
        X=X,
        random_effects=[re],
        groups_data={'subject': groups},
        covariance='identity',
    )

    # Should give same results
    np.testing.assert_allclose(
        result_formula.beta_parametric,
        result_matrix.beta_parametric,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result_formula.residual_variance,
        result_matrix.residual_variance,
        rtol=1e-6,
    )
