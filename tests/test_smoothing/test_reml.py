"""Tests for REML smoothing parameter selection."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.smoothing.selection.reml import (
    reml_score,
    select_multiple_smoothing_parameters_reml,
    select_smoothing_parameter_reml,
)
from aurora.smoothing.splines.bspline import BSplineBasis
from aurora.smoothing.splines.cubic import CubicSplineBasis


def test_reml_score_basic():
    """reml_score should compute REML criterion."""
    rng = np.random.default_rng(42)
    n = 100

    # Generate data
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    # Create basis
    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Compute REML score
    reml = reml_score(y, X, S, lambda_=0.1)

    # Should be finite
    assert np.isfinite(reml)

    # REML can be negative (it's -2*log-likelihood, but without constants)
    # Just check it's reasonable
    assert -1000 < reml < 1000


def test_reml_score_varies_with_lambda():
    """REML score should vary with lambda."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Compute REML for different lambdas
    lambdas = [1e-6, 1e-2, 1e0, 1e4]
    scores = [reml_score(y, X, S, lam) for lam in lambdas]

    # All should be finite
    assert all(np.isfinite(s) for s in scores)

    # Scores should be different (not all the same)
    assert len(set(scores)) > 1

    # Should have variation
    assert np.std(scores) > 0.1


def test_reml_score_negative_lambda():
    """reml_score should return inf for negative lambda."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 1, n)
    y = rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    score = reml_score(y, X, S, lambda_=-1.0)
    assert score == np.inf


def test_reml_score_with_weights():
    """reml_score should handle observation weights."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)
    weights = rng.uniform(0.5, 1.5, size=n)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # With weights
    reml_weighted = reml_score(y, X, S, lambda_=0.1, weights=weights)

    # Without weights
    reml_unweighted = reml_score(y, X, S, lambda_=0.1)

    # Should be different
    assert reml_weighted != reml_unweighted

    # Both should be finite
    assert np.isfinite(reml_weighted)
    assert np.isfinite(reml_unweighted)


def test_select_smoothing_parameter_reml_basic():
    """select_smoothing_parameter_reml should find optimal lambda."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Select lambda via REML
    result = select_smoothing_parameter_reml(y, X, S)

    # Check result structure
    assert "lambda_opt" in result
    assert "reml_score" in result
    assert "coefficients" in result
    assert "fitted_values" in result
    assert "edf" in result

    # Lambda should be positive
    assert result["lambda_opt"] > 0

    # EDF should be reasonable
    assert 0 < result["edf"] < X.shape[1]

    # Coefficients should match basis size
    assert len(result["coefficients"]) == X.shape[1]

    # Fitted values should match sample size
    assert len(result["fitted_values"]) == n


def test_select_smoothing_parameter_reml_fit_quality():
    """REML should produce good fit to data."""
    rng = np.random.default_rng(42)
    n = 200

    x = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.2 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=15, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    result = select_smoothing_parameter_reml(y, X, S)

    # Compute MSE
    mse = np.mean((result["fitted_values"] - y_true) ** 2)

    # MSE should be close to noise level (0.2^2 = 0.04)
    # Allow some slack since we're estimating
    assert mse < 0.15


def test_select_smoothing_parameter_reml_vs_gcv():
    """REML and GCV should give similar (but not identical) results."""
    from aurora.smoothing.selection.gcv import select_smoothing_parameter

    rng = np.random.default_rng(42)
    n = 150

    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=12, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Select via REML
    result_reml = select_smoothing_parameter_reml(y, X, S)

    # Select via GCV
    result_gcv = select_smoothing_parameter(y, X, S)

    # Both should find reasonable solutions
    assert result_reml["lambda_opt"] > 0
    assert result_gcv["lambda_opt"] > 0

    # EDFs should both be reasonable
    assert 0 < result_reml["edf"] < X.shape[1]
    assert 0 < result_gcv["edf"] < X.shape[1]

    # Fitted values should be reasonably correlated
    # (REML and GCV can give quite different results)
    corr = np.corrcoef(result_reml["fitted_values"], result_gcv["fitted_values"])[0, 1]
    assert corr > 0.7


def test_select_smoothing_parameter_reml_with_weights():
    """REML should handle weighted observations."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)
    weights = rng.uniform(0.5, 1.5, size=n)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    result = select_smoothing_parameter_reml(y, X, S, weights=weights)

    assert "lambda_opt" in result
    assert result["lambda_opt"] > 0
    assert 0 < result["edf"] < X.shape[1]


def test_select_smoothing_parameter_reml_cubic_basis():
    """REML should work with cubic spline basis."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots_interior = CubicSplineBasis.create_knots(x, n_knots=8)
    basis = CubicSplineBasis(knots_interior)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix()

    result = select_smoothing_parameter_reml(y, X, S)

    assert result["lambda_opt"] > 0
    assert len(result["coefficients"]) == X.shape[1]


def test_select_smoothing_parameter_reml_default_method():
    """REML should work with default method."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Default method (brent -> bounded)
    result = select_smoothing_parameter_reml(y, X, S)

    # Should find reasonable solution
    assert result["lambda_opt"] > 0
    assert 0 < result["edf"] < X.shape[1]


def test_select_smoothing_parameter_reml_invalid_inputs():
    """REML should validate inputs."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 1, n)
    y = rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Mismatched X and y
    with pytest.raises(ValueError, match="same number of rows"):
        select_smoothing_parameter_reml(y[:-10], X, S)

    # Wrong S shape
    with pytest.raises(ValueError, match="must have shape"):
        select_smoothing_parameter_reml(y, X, S[:-1, :-1])

    # Invalid lambda bounds
    with pytest.raises(ValueError, match="lambda_min < lambda_max"):
        select_smoothing_parameter_reml(y, X, S, lambda_min=10, lambda_max=1)


def test_select_multiple_smoothing_parameters_reml_basic():
    """Multiple REML should optimize multiple lambdas."""
    rng = np.random.default_rng(42)
    n = 200

    # Generate data with two smooth terms
    X = rng.normal(size=(n, 2))
    y = np.sin(2 * X[:, 0]) + np.cos(X[:, 1]) + 0.1 * rng.normal(size=n)

    # Create bases for each term
    knots1 = BSplineBasis.create_knots(X[:, 0], n_basis=10, degree=3)
    basis1 = BSplineBasis(knots1, degree=3)
    X1 = basis1.basis_matrix(X[:, 0])
    S1 = basis1.penalty_matrix(order=2)

    knots2 = BSplineBasis.create_knots(X[:, 1], n_basis=10, degree=3)
    basis2 = BSplineBasis(knots2, degree=3)
    X2 = basis2.basis_matrix(X[:, 1])
    S2 = basis2.penalty_matrix(order=2)

    # Select multiple lambdas
    result = select_multiple_smoothing_parameters_reml(y, [X1, X2], [S1, S2], max_iter=10)

    # Check result structure
    assert "lambda_opt" in result
    assert "reml_score" in result
    assert "coefficients" in result
    assert "fitted_values" in result
    assert "edf_values" in result
    assert "converged" in result
    assert "n_iter" in result

    # Should have two lambdas
    assert len(result["lambda_opt"]) == 2

    # Both should be positive
    assert all(lam > 0 for lam in result["lambda_opt"])

    # Should have two EDFs
    assert len(result["edf_values"]) == 2

    # Each EDF should exist and be finite
    assert all(np.isfinite(edf) for edf in result["edf_values"])


def test_select_multiple_smoothing_parameters_reml_convergence():
    """Multiple REML should converge."""
    rng = np.random.default_rng(42)
    n = 150

    X = rng.normal(size=(n, 2))
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * rng.normal(size=n)

    knots1 = BSplineBasis.create_knots(X[:, 0], n_basis=10, degree=3)
    basis1 = BSplineBasis(knots1, degree=3)
    X1 = basis1.basis_matrix(X[:, 0])
    S1 = basis1.penalty_matrix(order=2)

    knots2 = BSplineBasis.create_knots(X[:, 1], n_basis=10, degree=3)
    basis2 = BSplineBasis(knots2, degree=3)
    X2 = basis2.basis_matrix(X[:, 1])
    S2 = basis2.penalty_matrix(order=2)

    result = select_multiple_smoothing_parameters_reml(y, [X1, X2], [S1, S2], max_iter=20, tol=1e-3)

    # Should converge within 20 iterations
    assert result["converged"]
    assert result["n_iter"] <= 20


def test_select_multiple_smoothing_parameters_reml_three_terms():
    """Multiple REML should handle three smooth terms."""
    rng = np.random.default_rng(42)
    n = 200

    X = rng.normal(size=(n, 3))
    y = np.sin(2 * X[:, 0]) + np.cos(X[:, 1]) + X[:, 2] ** 2 + 0.1 * rng.normal(size=n)

    # Create three bases
    X_list = []
    S_list = []

    for j in range(3):
        knots = BSplineBasis.create_knots(X[:, j], n_basis=10, degree=3)
        basis = BSplineBasis(knots, degree=3)
        X_j = basis.basis_matrix(X[:, j])
        S_j = basis.penalty_matrix(order=2)
        X_list.append(X_j)
        S_list.append(S_j)

    result = select_multiple_smoothing_parameters_reml(y, X_list, S_list, max_iter=15)

    # Should have three lambdas
    assert len(result["lambda_opt"]) == 3

    # Should have three EDFs
    assert len(result["edf_values"]) == 3


def test_select_multiple_smoothing_parameters_reml_with_init():
    """Multiple REML should accept initial lambdas."""
    rng = np.random.default_rng(42)
    n = 150

    X = rng.normal(size=(n, 2))
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=n)

    knots1 = BSplineBasis.create_knots(X[:, 0], n_basis=10, degree=3)
    basis1 = BSplineBasis(knots1, degree=3)
    X1 = basis1.basis_matrix(X[:, 0])
    S1 = basis1.penalty_matrix(order=2)

    knots2 = BSplineBasis.create_knots(X[:, 1], n_basis=10, degree=3)
    basis2 = BSplineBasis(knots2, degree=3)
    X2 = basis2.basis_matrix(X[:, 1])
    S2 = basis2.penalty_matrix(order=2)

    # With custom initialization
    result = select_multiple_smoothing_parameters_reml(
        y, [X1, X2], [S1, S2], lambda_init=[0.1, 0.5], max_iter=10
    )

    assert len(result["lambda_opt"]) == 2


def test_select_multiple_smoothing_parameters_reml_invalid_inputs():
    """Multiple REML should validate inputs."""
    rng = np.random.default_rng(42)
    n = 100

    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)

    knots = BSplineBasis.create_knots(X[:, 0], n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X1 = basis.basis_matrix(X[:, 0])
    S1 = basis.penalty_matrix(order=2)

    # Mismatched X_list and S_list lengths
    with pytest.raises(ValueError, match="same length"):
        select_multiple_smoothing_parameters_reml(y, [X1], [S1, S1])

    # Wrong lambda_init length
    with pytest.raises(ValueError, match="lambda_init must have length"):
        select_multiple_smoothing_parameters_reml(y, [X1, X1], [S1, S1], lambda_init=[0.1])


def test_reml_stability_extreme_lambda():
    """REML should handle extreme lambda values gracefully."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 1, n)
    y = rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Very small lambda
    reml_small = reml_score(y, X, S, lambda_=1e-10)
    assert np.isfinite(reml_small)

    # Very large lambda
    reml_large = reml_score(y, X, S, lambda_=1e10)
    assert np.isfinite(reml_large)
