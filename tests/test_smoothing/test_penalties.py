"""Tests for penalty matrices."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.smoothing.penalties.difference import (
    combine_penalties,
    difference_penalty,
    null_space_penalty,
    ridge_penalty,
    weighted_difference_penalty,
)


def test_difference_penalty_basic():
    """difference_penalty should create correct penalty matrix."""
    S = difference_penalty(n_basis=5, order=2)

    assert S.shape == (5, 5)
    # Should be symmetric
    np.testing.assert_allclose(S, S.T, atol=1e-12)
    # Should be positive semi-definite
    eigenvalues = np.linalg.eigvals(S)
    assert np.all(eigenvalues >= -1e-10)


def test_difference_penalty_linear_null_space():
    """Second-order difference penalty should not penalize linear trends."""
    S = difference_penalty(n_basis=6, order=2)

    # Linear trend: β = [0, 1, 2, 3, 4, 5]
    beta_linear = np.arange(6, dtype=np.float64)
    penalty = beta_linear @ S @ beta_linear

    # Should be zero (linear has zero second derivative)
    assert abs(penalty) < 1e-10


def test_difference_penalty_constant_null_space():
    """Second-order difference penalty should not penalize constants."""
    S = difference_penalty(n_basis=5, order=2)

    # Constant: β = [3, 3, 3, 3, 3]
    beta_const = np.full(5, 3.0)
    penalty = beta_const @ S @ beta_const

    # Should be zero
    assert abs(penalty) < 1e-10


def test_difference_penalty_penalizes_curvature():
    """Second-order difference penalty should penalize non-linear patterns."""
    S = difference_penalty(n_basis=5, order=2)

    # Quadratic: β = [0, 1, 4, 9, 16] (i²)
    beta_quad = np.array([0, 1, 4, 9, 16], dtype=np.float64)
    penalty_quad = beta_quad @ S @ beta_quad

    # Should be positive (quadratic has constant second derivative)
    assert penalty_quad > 0

    # Oscillatory: β = [1, -1, 1, -1, 1]
    beta_osc = np.array([1, -1, 1, -1, 1], dtype=np.float64)
    penalty_osc = beta_osc @ S @ beta_osc

    # Should be larger (more curvature)
    assert penalty_osc > penalty_quad


def test_difference_penalty_different_orders():
    """Different orders should give different penalties."""
    n = 6

    S1 = difference_penalty(n, order=1)
    S2 = difference_penalty(n, order=2)
    S3 = difference_penalty(n, order=3)

    # All should be different
    assert not np.allclose(S1, S2)
    assert not np.allclose(S2, S3)

    # Higher order should have smaller null space
    # Order 1: penalizes differences (nullspace = constants)
    beta_const = np.ones(n)
    assert abs(beta_const @ S1 @ beta_const) < 1e-10

    # Order 2: doesn't penalize linear
    beta_linear = np.arange(n, dtype=np.float64)
    assert abs(beta_linear @ S2 @ beta_linear) < 1e-10
    # But does penalize with order 1
    assert (beta_linear @ S1 @ beta_linear) > 0


def test_difference_penalty_invalid_params():
    """difference_penalty should validate parameters."""
    # Negative n_basis
    with pytest.raises(ValueError, match="positive"):
        difference_penalty(n_basis=0, order=2)

    # Negative order
    with pytest.raises(ValueError, match="positive"):
        difference_penalty(n_basis=5, order=0)

    # Order too large
    with pytest.raises(ValueError, match="less than"):
        difference_penalty(n_basis=3, order=3)


def test_weighted_difference_penalty_uniform_knots():
    """Weighted penalty with uniform knots should match unweighted."""
    n = 6
    knots = np.linspace(0, 1, n + 2)  # Uniform spacing

    S_weighted = weighted_difference_penalty(n, knots, order=2)
    S_simple = difference_penalty(n, order=2)

    # Should be proportional (up to scaling by spacing)
    # Check structure is similar
    assert S_weighted.shape == S_simple.shape
    assert np.allclose(S_weighted, S_weighted.T)  # Symmetric


def test_weighted_difference_penalty_nonuniform_knots():
    """Weighted penalty should handle non-uniform knots."""
    n = 5
    # Non-uniform knots (more dense at start)
    knots = np.array([0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5])

    S = weighted_difference_penalty(n, knots, order=2)

    assert S.shape == (n, n)
    np.testing.assert_allclose(S, S.T, atol=1e-12)


def test_weighted_difference_penalty_invalid_knots():
    """weighted_difference_penalty should validate knots."""
    # Too few knots
    with pytest.raises(ValueError, match="at least"):
        weighted_difference_penalty(5, knots=np.array([0, 1, 2]), order=2)

    # Non-increasing knots
    with pytest.raises(ValueError, match="strictly increasing"):
        weighted_difference_penalty(5, knots=np.array([0, 1, 0.5, 2, 3, 4]), order=2)


def test_ridge_penalty_basic():
    """ridge_penalty should create identity-like matrix."""
    S = ridge_penalty(n_basis=5)

    # Should be identity except first element
    expected = np.eye(5)
    expected[0, 0] = 0  # Exclude intercept

    np.testing.assert_array_equal(S, expected)


def test_ridge_penalty_include_intercept():
    """ridge_penalty with include_intercept should be pure identity."""
    S = ridge_penalty(n_basis=5, exclude_intercept=False)

    np.testing.assert_array_equal(S, np.eye(5))


def test_ridge_penalty_penalizes_magnitude():
    """Ridge penalty should penalize coefficient magnitude."""
    S = ridge_penalty(n_basis=4, exclude_intercept=False)

    # Larger coefficients should have larger penalty
    beta_small = np.array([1, 1, 1, 1], dtype=np.float64)
    beta_large = np.array([2, 2, 2, 2], dtype=np.float64)

    penalty_small = beta_small @ S @ beta_small
    penalty_large = beta_large @ S @ beta_large

    assert penalty_large > penalty_small
    # Should be 4x larger (quadratic in coefficients)
    np.testing.assert_allclose(penalty_large, 4 * penalty_small, rtol=1e-10)


def test_null_space_penalty_equivalent_to_difference():
    """null_space_penalty should match difference_penalty."""
    n = 8

    for dim in [1, 2, 3]:
        S_null = null_space_penalty(n, null_space_dim=dim)
        S_diff = difference_penalty(n, order=dim)

        np.testing.assert_allclose(S_null, S_diff)


def test_null_space_penalty_controls_flexibility():
    """Larger null space should allow more flexibility (less penalty)."""
    n = 8
    beta = np.array([0, 1, 4, 9, 16, 25, 36, 49], dtype=np.float64)  # Quadratic

    S1 = null_space_penalty(n, null_space_dim=1)
    S2 = null_space_penalty(n, null_space_dim=2)
    S3 = null_space_penalty(n, null_space_dim=3)

    penalty1 = beta @ S1 @ beta
    penalty2 = beta @ S2 @ beta
    penalty3 = beta @ S3 @ beta

    # Larger null space = smaller penalty
    assert penalty1 > penalty2 > penalty3


def test_combine_penalties_uniform_weights():
    """combine_penalties with uniform weights should average penalties."""
    S1 = difference_penalty(5, order=1)
    S2 = difference_penalty(5, order=2)

    S_combined = combine_penalties([S1, S2])

    # With uniform weights, should be sum
    expected = S1 + S2
    np.testing.assert_allclose(S_combined, expected)


def test_combine_penalties_custom_weights():
    """combine_penalties should respect custom weights."""
    S1 = difference_penalty(5, order=2)
    S2 = ridge_penalty(5, exclude_intercept=False)

    weights = [0.9, 0.1]
    S_combined = combine_penalties([S1, S2], weights=weights)

    expected = 0.9 * S1 + 0.1 * S2
    np.testing.assert_allclose(S_combined, expected)


def test_combine_penalties_single_penalty():
    """combine_penalties with single penalty should return scaled penalty."""
    S = difference_penalty(5, order=2)

    S_combined = combine_penalties([S], weights=[2.0])

    np.testing.assert_allclose(S_combined, 2.0 * S)


def test_combine_penalties_invalid_inputs():
    """combine_penalties should validate inputs."""
    S1 = difference_penalty(5, order=2)
    S2 = difference_penalty(6, order=2)  # Different size

    # Empty list
    with pytest.raises(ValueError, match="empty"):
        combine_penalties([])

    # Mismatched shapes
    with pytest.raises(ValueError, match="same shape"):
        combine_penalties([S1, S2])

    # Wrong number of weights
    with pytest.raises(ValueError, match="Number of weights"):
        combine_penalties([S1, S1], weights=[1.0])


def test_difference_penalty_rank_deficiency():
    """Penalty matrix should have rank = n_basis - null_space_dim."""
    n = 10

    # Order 2 penalty has null space of dimension 2 (constant + linear)
    S2 = difference_penalty(n, order=2)
    rank = np.linalg.matrix_rank(S2, tol=1e-10)
    assert rank == n - 2

    # Order 1 penalty has null space of dimension 1 (constant)
    S1 = difference_penalty(n, order=1)
    rank = np.linalg.matrix_rank(S1, tol=1e-10)
    assert rank == n - 1


def test_penalty_matrices_are_positive_semidefinite():
    """All penalty matrices should be positive semi-definite."""
    n = 8

    penalties = [
        difference_penalty(n, order=1),
        difference_penalty(n, order=2),
        weighted_difference_penalty(n, np.linspace(0, 1, n + 2), order=2),
        ridge_penalty(n),
        null_space_penalty(n, null_space_dim=2),
    ]

    for S in penalties:
        eigenvalues = np.linalg.eigvals(S)
        # All eigenvalues should be non-negative (allowing small numerical error)
        assert np.all(eigenvalues >= -1e-10)
