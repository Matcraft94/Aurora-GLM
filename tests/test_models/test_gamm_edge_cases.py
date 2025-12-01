"""Edge case tests for GAMM module."""
import numpy as np
import pytest
from aurora.models.gamm import fit_gamm, RandomEffect


def test_gamm_single_group():
    """GAMM with only one group (degenerate case)."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = 1 + 0.5 * X[:, 0] + np.random.randn(50) * 0.2
    groups = np.zeros(50, dtype=int)  # All same group

    # Should handle gracefully (variance may be singular)
    try:
        result = fit_gamm(
            y=y, X=X,
            random_effects=[RandomEffect(grouping='group')],
            groups_data={'group': groups},
            family='gaussian'
        )
        assert result is not None
    except (np.linalg.LinAlgError, ValueError):
        # Acceptable to fail with singular variance
        pass


def test_gamm_unbalanced_groups():
    """Groups with very different sample sizes."""
    np.random.seed(42)
    # Group 0: 5 observations, Group 1: 95 observations
    groups = np.concatenate([np.zeros(5), np.ones(95)]).astype(int)
    X = np.random.randn(100, 2)
    y = 1 + 0.5 * X[:, 0] + np.random.randn(100) * 0.2

    result = fit_gamm(
        y=y, X=X,
        random_effects=[RandomEffect(grouping='group')],
        groups_data={'group': groups},
        family='gaussian'
    )
    assert result.converged


def test_gamm_random_intercept_only():
    """Random intercept without slope."""
    np.random.seed(42)
    n_groups = 10
    n_per_group = 20
    groups = np.repeat(np.arange(n_groups), n_per_group)

    X = np.random.randn(n_groups * n_per_group, 2)
    random_intercepts = np.random.randn(n_groups) * 0.5
    y = 1 + 0.5 * X[:, 0] + random_intercepts[groups] + np.random.randn(len(groups)) * 0.1

    result = fit_gamm(
        y=y, X=X,
        random_effects=[RandomEffect(variables=(), grouping='group', include_intercept=True)],
        groups_data={'group': groups},
        family='gaussian'
    )
    assert result.converged
    assert 'group' in result.random_effects


def test_gamm_random_slope():
    """Random intercept + slope."""
    np.random.seed(42)
    n_groups = 8
    n_per_group = 15
    groups = np.repeat(np.arange(n_groups), n_per_group)

    X = np.random.randn(n_groups * n_per_group, 2)
    random_intercepts = np.random.randn(n_groups) * 0.5
    random_slopes = np.random.randn(n_groups) * 0.2

    y = (1 + random_intercepts[groups] +
         (0.5 + random_slopes[groups]) * X[:, 0] +
         np.random.randn(len(groups)) * 0.1)

    result = fit_gamm(
        y=y, X=X,
        random_effects=[RandomEffect(variables=(0,), grouping='group', include_intercept=True)],
        groups_data={'group': groups},
        family='gaussian'
    )
    assert result.converged


def test_gamm_predict_new_group():
    """Prediction for group not seen in training (should use population effect)."""
    np.random.seed(42)
    n_groups = 5
    n_per_group = 20
    groups_train = np.repeat(np.arange(n_groups), n_per_group)

    X_train = np.random.randn(n_groups * n_per_group, 2)
    y_train = 1 + 0.5 * X_train[:, 0] + np.random.randn(len(groups_train)) * 0.2

    result = fit_gamm(
        y=y_train, X=X_train,
        random_effects=[RandomEffect(grouping='group', include_intercept=True)],
        groups_data={'group': groups_train},
        family='gaussian'
    )

    # Predict for new group (not in training)
    from aurora.models import predict_from_gamm
    X_new = np.random.randn(10, 2)
    groups_new = np.full(10, 999)  # New group ID

    y_pred = predict_from_gamm(result, X_new, groups_new={'group': groups_new}, include_random=False)
    assert y_pred.shape[0] == 10


def test_gamm_predict_without_random_effects():
    """Population-level prediction (include_random=False)."""
    np.random.seed(42)
    n_groups = 5
    n_per_group = 20
    groups = np.repeat(np.arange(n_groups), n_per_group)

    X = np.random.randn(n_groups * n_per_group, 2)
    y = 1 + 0.5 * X[:, 0] + np.random.randn(len(groups)) * 0.2

    result = fit_gamm(
        y=y, X=X,
        random_effects=[RandomEffect(grouping='group', include_intercept=True)],
        groups_data={'group': groups},
        family='gaussian'
    )

    from aurora.models import predict_from_gamm
    X_new = np.random.randn(10, 2)

    y_pred = predict_from_gamm(result, X_new, include_random=False)
    # Should return population-level predictions
    assert y_pred.shape[0] == 10


def test_gamm_covariance_diagonal():
    """Diagonal covariance structure."""
    np.random.seed(42)
    n_groups = 6
    n_per_group = 15
    groups = np.repeat(np.arange(n_groups), n_per_group)

    X = np.random.randn(n_groups * n_per_group, 2)
    y = 1 + 0.5 * X[:, 0] + np.random.randn(len(groups)) * 0.2

    result = fit_gamm(
        y=y, X=X,
        random_effects=[
            RandomEffect(variables=(0,), grouping='group',
                        include_intercept=True, covariance='diagonal')
        ],
        groups_data={'group': groups},
        family='gaussian',
        covariance='diagonal'
    )
    assert result.converged


def test_gamm_variance_components_positive():
    """Variance components should be non-negative."""
    np.random.seed(42)
    n_groups = 10
    n_per_group = 20
    groups = np.repeat(np.arange(n_groups), n_per_group)

    X = np.random.randn(n_groups * n_per_group, 2)
    y = 1 + 0.5 * X[:, 0] + np.random.randn(len(groups)) * 0.2

    result = fit_gamm(
        y=y, X=X,
        random_effects=[RandomEffect(grouping='group', include_intercept=True)],
        groups_data={'group': groups},
        family='gaussian'
    )

    # Check residual variance is positive
    assert result.residual_variance > 0

    # variance_components is a list of covariance matrices (one per random effect term)
    assert isinstance(result.variance_components, list)
    assert len(result.variance_components) == 1  # Single random effect term

    # Check that variance components (diagonal elements) are positive
    psi_arr = np.atleast_2d(result.variance_components[0])
    # Check diagonal (variances) are positive
    assert np.all(np.diag(psi_arr) >= -1e-10)  # Allow small numerical errors
    # Check matrix is positive semi-definite
    eigenvalues = np.linalg.eigvalsh(psi_arr)
    assert np.all(eigenvalues >= -1e-10)


def test_gamm_nested_structure():
    """Nested random effects: clinic/subject."""
    np.random.seed(42)
    n_clinics = 3
    n_subjects_per_clinic = 4
    n_obs_per_subject = 10

    clinics = np.repeat(np.arange(n_clinics), n_subjects_per_clinic * n_obs_per_subject)
    subjects = np.repeat(
        np.arange(n_clinics * n_subjects_per_clinic),
        n_obs_per_subject
    )

    X = np.random.randn(len(subjects), 2)
    y = 1 + 0.5 * X[:, 0] + np.random.randn(len(subjects)) * 0.2

    # Nested: subject within clinic
    result = fit_gamm(
        y=y, X=X,
        random_effects=[
            RandomEffect(grouping='clinic', include_intercept=True),
            RandomEffect(grouping='subject', include_intercept=True)
        ],
        groups_data={'clinic': clinics, 'subject': subjects},
        family='gaussian'
    )
    assert result.converged


def test_gamm_multiple_variance_components():
    """Multiple random effects with separate variance components."""
    np.random.seed(42)
    n_groups_a = 10
    n_groups_b = 8
    n_per_group = 5

    # Fix: Crossed design - ensure both arrays have same length
    # Total observations = n_groups_a * n_per_group = 50
    n = n_groups_a * n_per_group
    group_a = np.tile(np.arange(n_groups_a), n_per_group)
    # For group_b, tile to match length n
    group_b = np.tile(np.arange(n_groups_b), n // n_groups_b + 1)[:n]

    X = np.random.randn(n, 2)
    y = 1 + 0.5 * X[:, 0] + np.random.randn(n) * 0.2

    result = fit_gamm(
        y=y, X=X,
        random_effects=[
            RandomEffect(grouping='group_a', include_intercept=True),
            RandomEffect(grouping='group_b', include_intercept=True)
        ],
        groups_data={'group_a': group_a, 'group_b': group_b},
        family='gaussian'
    )

    # variance_components is a list of covariance matrices (one per random effect term)
    assert result.variance_components is not None
    assert isinstance(result.variance_components, list)
    assert len(result.variance_components) == 2  # Two random effect terms
    for psi in result.variance_components:
        assert isinstance(psi, np.ndarray)
    assert result.converged
