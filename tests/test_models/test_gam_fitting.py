"""Tests for GAM fitting functionality."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models.gam import GAMResult, fit_gam


def test_fit_gam_basic():
    """fit_gam should fit a simple GAM."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=12)

    # Should return GAMResult
    assert isinstance(result, GAMResult)

    # Basic attributes
    assert result.coefficients.shape[0] == 12
    assert result.fitted_values.shape == (n,)
    assert result.residuals.shape == (n,)
    assert result.lambda_ > 0
    assert result.edf > 0


def test_fit_gam_automatic_lambda():
    """fit_gam should automatically select lambda via GCV."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=12, lambda_=None)

    # Should have selected lambda
    assert result.lambda_ > 0
    # GCV score should be available
    assert result.gcv_score is not None
    assert result.gcv_score > 0


def test_fit_gam_fixed_lambda():
    """fit_gam should accept fixed lambda."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    lambda_fixed = 0.1
    result = fit_gam(x, y, n_basis=12, lambda_=lambda_fixed)

    # Should use provided lambda
    assert result.lambda_ == lambda_fixed
    # GCV score should not be available
    assert result.gcv_score is None


def test_fit_gam_bspline_basis():
    """fit_gam should work with B-spline basis."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=12, basis_type="bspline")

    assert result.coefficients.shape[0] == 12
    assert "BSplineBasis" in type(result.basis).__name__


def test_fit_gam_cubic_basis():
    """fit_gam should work with cubic spline basis."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=12, basis_type="cubic")

    assert result.coefficients.shape[0] == 12
    assert "CubicSplineBasis" in type(result.basis).__name__


def test_fit_gam_with_weights():
    """fit_gam should handle observation weights."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)
    weights = rng.uniform(0.5, 1.5, size=n)

    result = fit_gam(x, y, n_basis=12, weights=weights)

    assert result.weights is not None
    np.testing.assert_array_equal(result.weights, weights)


def test_fit_gam_different_degrees():
    """fit_gam should work with different polynomial degrees."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    # Linear splines
    result_linear = fit_gam(x, y, n_basis=12, degree=1)
    assert result_linear.coefficients.shape[0] == 12

    # Cubic splines
    result_cubic = fit_gam(x, y, n_basis=12, degree=3)
    assert result_cubic.coefficients.shape[0] == 12

    # Should give different fits
    assert not np.allclose(
        result_linear.fitted_values, result_cubic.fitted_values
    )


def test_fit_gam_knot_methods():
    """fit_gam should support different knot placement methods."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    result_quantile = fit_gam(x, y, n_basis=12, knot_method="quantile")
    result_uniform = fit_gam(x, y, n_basis=12, knot_method="uniform")

    # Both should succeed
    assert result_quantile.coefficients.shape[0] == 12
    assert result_uniform.coefficients.shape[0] == 12


def test_fit_gam_prediction():
    """GAMResult.predict should make predictions at new points."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=12)

    # Predict at new points
    x_new = np.linspace(0, 1, 200)
    y_pred = result.predict(x_new)

    assert y_pred.shape == (200,)
    assert np.all(np.isfinite(y_pred))


def test_fit_gam_prediction_scalar():
    """GAMResult.predict should handle scalar input."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=12)

    # Predict at single point
    y_pred = result.predict(0.5)

    assert y_pred.shape == (1,)
    assert np.isfinite(y_pred[0])


def test_fit_gam_fit_quality():
    """GAM should provide good fit to smooth data."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=15)

    # MSE should be small (close to noise level)
    mse = np.mean((result.fitted_values - y_true) ** 2)
    noise_var = 0.1**2

    # Should be within 2x of noise variance
    assert mse < 2 * noise_var


def test_fit_gam_residuals():
    """Residuals should be computed correctly."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=12)

    # Residuals = y - fitted_values
    expected_residuals = y - result.fitted_values
    np.testing.assert_allclose(result.residuals, expected_residuals)


def test_fit_gam_edf_interpretation():
    """Effective degrees of freedom should reflect smoothing."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    # Small lambda = less smoothing = higher EDF
    result_small = fit_gam(x, y, n_basis=15, lambda_=0.001)

    # Large lambda = more smoothing = lower EDF
    result_large = fit_gam(x, y, n_basis=15, lambda_=100.0)

    assert result_small.edf > result_large.edf


def test_fit_gam_summary():
    """GAMResult.summary should generate formatted summary."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=12)
    summary = result.summary()

    # Summary should contain key information
    assert "GAM" in summary or "Generalized Additive Model" in summary
    # Lambda can be in scientific notation, so just check it's mentioned
    assert "Lambda" in summary or "lambda" in summary
    # EDF should be mentioned
    assert "DoF" in summary or "Effective" in summary
    assert str(n) in summary  # Number of observations


def test_fit_gam_repr():
    """GAMResult.__repr__ should return informative string."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=12)
    repr_str = repr(result)

    assert "GAMResult" in repr_str
    assert "n_obs" in repr_str
    assert "edf" in repr_str


def test_fit_gam_invalid_inputs():
    """fit_gam should validate inputs."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    # Multidimensional x
    with pytest.raises(ValueError, match="1-dimensional"):
        fit_gam(np.ones((n, 2)), y)

    # Multidimensional y
    with pytest.raises(ValueError, match="1-dimensional"):
        fit_gam(x, np.ones((n, 2)))

    # Mismatched lengths
    with pytest.raises(ValueError, match="same length"):
        fit_gam(x, y[:-10])

    # Invalid weights shape
    with pytest.raises(ValueError, match="weights must have shape"):
        fit_gam(x, y, weights=np.ones(n + 1))

    # Too few basis functions
    with pytest.raises(ValueError, match="n_basis must be at least"):
        fit_gam(x, y, n_basis=2)

    # Invalid basis type
    with pytest.raises(ValueError, match="Unknown basis_type"):
        fit_gam(x, y, basis_type="invalid")


def test_fit_gam_extrapolation():
    """GAM should handle extrapolation gracefully."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=12)

    # Extrapolate beyond training range
    x_extrap = np.array([-0.5, 1.5])
    y_extrap = result.predict(x_extrap)

    # Should not crash and should return finite values
    assert y_extrap.shape == (2,)
    assert np.all(np.isfinite(y_extrap))


def test_fit_gam_penalty_orders():
    """fit_gam should work with different penalty orders."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    # First-order penalty (penalizes slope)
    result_order1 = fit_gam(x, y, n_basis=12, penalty_order=1)

    # Second-order penalty (penalizes curvature)
    result_order2 = fit_gam(x, y, n_basis=12, penalty_order=2)

    # Both should succeed
    assert result_order1.coefficients.shape[0] == 12
    assert result_order2.coefficients.shape[0] == 12

    # Should give different fits
    assert not np.allclose(
        result_order1.fitted_values, result_order2.fitted_values
    )


def test_fit_gam_linear_function():
    """GAM should exactly fit linear functions with order-2 penalty."""
    n = 100
    x = np.linspace(0, 1, n)
    y = 2 * x + 1  # Perfect linear function

    # With order-2 penalty, linear functions are in null space
    result = fit_gam(x, y, n_basis=10, penalty_order=2)

    # Should fit very closely (within numerical precision)
    np.testing.assert_allclose(result.fitted_values, y, atol=1e-6)


def test_fit_gam_constant_function():
    """GAM should exactly fit constant functions."""
    n = 100
    x = np.linspace(0, 1, n)
    y = np.full(n, 5.0)  # Constant

    result = fit_gam(x, y, n_basis=10, penalty_order=2)

    # Should fit exactly
    np.testing.assert_allclose(result.fitted_values, y, atol=1e-6)
