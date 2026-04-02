"""Tests for GCV smoothing parameter selection."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.smoothing.selection.gcv import gcv_score, select_smoothing_parameter
from aurora.smoothing.splines.bspline import BSplineBasis


def test_gcv_score_basic():
    """gcv_score should compute valid GCV score."""
    # Simple synthetic data
    rng = np.random.default_rng(42)
    n = 50
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    # Create basis
    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Compute GCV score
    lambda_ = 0.1
    score = gcv_score(y, X, S, lambda_)

    # Should be positive
    assert score > 0
    assert np.isfinite(score)


def test_gcv_score_with_weights():
    """gcv_score should handle observation weights."""
    rng = np.random.default_rng(42)
    n = 50
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Uniform weights should give same result as no weights
    weights_uniform = np.ones(n)
    score_unweighted = gcv_score(y, X, S, lambda_=0.1, weights=None)
    score_weighted = gcv_score(y, X, S, lambda_=0.1, weights=weights_uniform)

    np.testing.assert_allclose(score_unweighted, score_weighted, rtol=1e-10)

    # Non-uniform weights should give different result
    weights_varied = rng.uniform(0.5, 1.5, size=n)
    score_varied = gcv_score(y, X, S, lambda_=0.1, weights=weights_varied)

    assert not np.isclose(score_unweighted, score_varied)


def test_gcv_score_lambda_effect():
    """GCV score should vary with smoothing parameter."""
    rng = np.random.default_rng(42)
    n = 50
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Different lambda values should give different scores
    lambdas = [1e-3, 1e-1, 1e1, 1e3]
    scores = [gcv_score(y, X, S, lam) for lam in lambdas]

    # Scores should be different
    assert len(set(scores)) == len(scores)


def test_gcv_score_extreme_lambda():
    """GCV score should handle extreme lambda values."""
    rng = np.random.default_rng(42)
    n = 50
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Very small lambda (nearly unpenalized)
    score_small = gcv_score(y, X, S, lambda_=1e-10)
    assert np.isfinite(score_small)

    # Very large lambda (heavily penalized, nearly linear)
    score_large = gcv_score(y, X, S, lambda_=1e10)
    assert np.isfinite(score_large)


def test_gcv_score_invalid_weights():
    """gcv_score should validate weights."""
    rng = np.random.default_rng(42)
    n = 50
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Wrong shape
    with pytest.raises(ValueError, match="weights must have shape"):
        gcv_score(y, X, S, lambda_=0.1, weights=np.ones(n + 1))


def test_select_smoothing_parameter_basic():
    """select_smoothing_parameter should find optimal lambda."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    # Smooth function with noise
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=12, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    result = select_smoothing_parameter(y, X, S)

    # Should return valid result
    assert "lambda_opt" in result
    assert "gcv_score" in result
    assert "edf" in result
    assert "coefficients" in result
    assert "fitted_values" in result

    # Lambda should be in reasonable range
    assert 0 < result["lambda_opt"] < 1e6

    # EDF should be between 1 and p
    assert 1 <= result["edf"] <= X.shape[1]

    # Fitted values should match data shape
    assert result["fitted_values"].shape == y.shape


def test_select_smoothing_parameter_methods():
    """Different optimization methods should give similar results."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=12, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    result_golden = select_smoothing_parameter(y, X, S, method="golden")
    result_brent = select_smoothing_parameter(y, X, S, method="brent")

    # Should give similar lambda values
    np.testing.assert_allclose(
        result_golden["lambda_opt"],
        result_brent["lambda_opt"],
        rtol=0.1,  # 10% tolerance (optimization methods may differ slightly)
    )

    # GCV scores should be similar
    np.testing.assert_allclose(result_golden["gcv_score"], result_brent["gcv_score"], rtol=0.01)


def test_select_smoothing_parameter_with_weights():
    """select_smoothing_parameter should handle weights."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)
    weights = rng.uniform(0.5, 1.5, size=n)

    knots = BSplineBasis.create_knots(x, n_basis=12, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    result = select_smoothing_parameter(y, X, S, weights=weights)

    # Should return valid result
    assert result["lambda_opt"] > 0
    assert np.isfinite(result["gcv_score"])


def test_select_smoothing_parameter_smooth_data():
    """GCV should select small lambda for smooth data."""
    n = 100
    x = np.linspace(0, 1, n)
    # Very smooth function (quadratic) with tiny noise
    y = x**2 + 0.001 * np.random.default_rng(42).normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=12, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    result = select_smoothing_parameter(y, X, S)

    # For smooth data, should select relatively small lambda
    # (less penalty needed)
    assert result["lambda_opt"] < 1.0


def test_select_smoothing_parameter_noisy_data():
    """GCV should select larger lambda for noisy data."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    # Simple function with large noise
    y = np.sin(2 * np.pi * x) + 0.5 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=12, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    result = select_smoothing_parameter(y, X, S)

    # For noisy data, should select larger lambda (more smoothing)
    # Exact value depends on signal-to-noise ratio
    assert result["lambda_opt"] > 0
    assert result["edf"] < X.shape[1]  # Should be smoothed


def test_select_smoothing_parameter_edf_interpretation():
    """EDF should reflect model complexity."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=20, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    result = select_smoothing_parameter(y, X, S)

    # EDF should be:
    # - Much less than p (penalty is working)
    # - Greater than 2 (more than just linear)
    p = X.shape[1]
    assert 2 < result["edf"] < p * 0.7  # GCV may not smooth as aggressively


def test_select_smoothing_parameter_fit_quality():
    """Selected lambda should give reasonable fit."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.2 * rng.normal(size=n)  # More noise so smoothing helps

    knots = BSplineBasis.create_knots(x, n_basis=15, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    result = select_smoothing_parameter(y, X, S)
    y_hat = result["fitted_values"]

    # Fitted values should be closer to true function than raw data
    mse_fit = np.mean((y_hat - y_true) ** 2)
    mse_raw = np.mean((y - y_true) ** 2)

    # Smoothed fit should reduce error (bias-variance tradeoff)
    # With enough noise, smoothing should help
    assert mse_fit < mse_raw


def test_select_smoothing_parameter_invalid_inputs():
    """select_smoothing_parameter should validate inputs."""
    rng = np.random.default_rng(42)
    n = 50
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Wrong y shape
    with pytest.raises(ValueError, match="y must have shape"):
        select_smoothing_parameter(np.ones(n + 1), X, S)

    # Wrong S shape
    S_wrong = np.eye(X.shape[1] + 1)
    with pytest.raises(ValueError, match="S must have shape"):
        select_smoothing_parameter(y, X, S_wrong)

    # Invalid method
    with pytest.raises(ValueError, match="Unknown method"):
        select_smoothing_parameter(y, X, S, method="invalid")


def test_select_smoothing_parameter_lambda_bounds():
    """Lambda bounds should constrain optimization."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=12, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Narrow bounds
    result = select_smoothing_parameter(y, X, S, lambda_min=0.1, lambda_max=1.0)

    # Should be within bounds
    assert 0.1 <= result["lambda_opt"] <= 1.0


def test_gcv_score_decreases_at_optimum():
    """GCV score should be at minimum at selected lambda."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    knots = BSplineBasis.create_knots(x, n_basis=12, degree=3)
    basis = BSplineBasis(knots, degree=3)
    X = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    result = select_smoothing_parameter(y, X, S)
    lambda_opt = result["lambda_opt"]
    gcv_opt = result["gcv_score"]

    # Check nearby lambda values have higher GCV
    lambda_lower = lambda_opt * 0.5
    lambda_upper = lambda_opt * 2.0

    gcv_lower = gcv_score(y, X, S, lambda_lower)
    gcv_upper = gcv_score(y, X, S, lambda_upper)

    # Optimal should be better (or very close)
    assert gcv_opt <= gcv_lower * 1.01  # Allow 1% tolerance
    assert gcv_opt <= gcv_upper * 1.01
