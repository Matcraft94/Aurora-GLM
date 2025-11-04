"""Tests for additive GAM with multiple smooth terms."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models.gam import (
    AdditiveGAMResult,
    ParametricTerm,
    SmoothTerm,
    fit_additive_gam,
)


def test_smooth_term_creation():
    """SmoothTerm should create with valid parameters."""
    term = SmoothTerm(variable=0)

    assert term.variable == 0
    assert term.basis_type == "bspline"
    assert term.n_basis == 10
    assert term.penalty_order == 2


def test_smooth_term_validation():
    """SmoothTerm should validate parameters."""
    # Too few basis functions
    with pytest.raises(ValueError, match="n_basis must be at least"):
        SmoothTerm(variable=0, n_basis=2)

    # Invalid penalty order
    with pytest.raises(ValueError, match="penalty_order must be positive"):
        SmoothTerm(variable=0, penalty_order=0)

    # Unknown basis type
    with pytest.raises(ValueError, match="Unknown basis_type"):
        SmoothTerm(variable=0, basis_type="invalid")  # type: ignore

    # Negative lambda
    with pytest.raises(ValueError, match="lambda_ must be non-negative"):
        SmoothTerm(variable=0, lambda_=-1.0)


def test_fit_additive_gam_two_smooths():
    """fit_additive_gam should fit model with two smooth terms."""
    rng = np.random.default_rng(42)
    n = 200

    # Generate data: y = sin(2πx₁) + cos(x₂) + ε
    X = np.random.randn(n, 2)
    y_true = np.sin(2 * np.pi * X[:, 0]) + np.cos(X[:, 1])
    y = y_true + 0.1 * rng.normal(size=n)

    # Fit additive GAM
    result = fit_additive_gam(
        X, y,
        smooth_terms=[
            SmoothTerm(variable=0, n_basis=12),
            SmoothTerm(variable=1, n_basis=12)
        ]
    )

    # Check result attributes
    assert isinstance(result, AdditiveGAMResult)
    assert result.n_smooth_terms_ == 2
    assert result.n_parametric_terms_ == 0  # No parametric terms
    assert result.n_obs_ == n

    # Check smooth terms exist
    assert "s(0)" in result.smooth_coef
    assert "s(1)" in result.smooth_coef
    assert "s(0)" in result.smooth_bases
    assert "s(1)" in result.smooth_bases

    # Check fit quality (should capture nonlinear relationships)
    r_squared = 1 - np.sum(result.residuals**2) / np.sum((y - np.mean(y))**2)
    assert r_squared > 0.6  # Should explain reasonable variance


def test_fit_additive_gam_with_parametric():
    """fit_additive_gam should handle mixed smooth and parametric terms."""
    rng = np.random.default_rng(42)
    n = 200

    # Generate data: y = sin(x₁) + 2*x₂ + ε
    X = np.random.randn(n, 2)
    y = np.sin(2 * X[:, 0]) + 2 * X[:, 1] + 0.1 * rng.normal(size=n)

    # Fit with one smooth and one parametric
    result = fit_additive_gam(
        X, y,
        smooth_terms=[SmoothTerm(variable=0, n_basis=10)],
        parametric_terms=[ParametricTerm(variable=1)]
    )

    assert result.n_smooth_terms_ == 1
    assert result.n_parametric_terms_ == 1

    # Parametric coefficient should be close to 2
    assert len(result.parametric_coef) == 2  # Intercept + x2
    assert abs(result.parametric_coef[1] - 2.0) < 0.5  # Rough check


def test_fit_additive_gam_predictions():
    """AdditiveGAMResult.predict should work on new data."""
    rng = np.random.default_rng(42)
    n = 100

    X = np.random.randn(n, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(
        X, y,
        smooth_terms=[
            SmoothTerm(variable=0, n_basis=10),
            SmoothTerm(variable=1, n_basis=10)
        ]
    )

    # Predict on new data
    X_new = np.random.randn(50, 2)
    y_pred = result.predict(X_new)

    assert y_pred.shape == (50,)
    assert np.all(np.isfinite(y_pred))


def test_fit_additive_gam_single_smooth():
    """fit_additive_gam should work with single smooth term."""
    rng = np.random.default_rng(42)
    n = 100

    X = np.random.randn(n, 1)
    y = np.sin(2 * np.pi * X[:, 0]) + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(
        X, y,
        smooth_terms=[SmoothTerm(variable=0, n_basis=12)]
    )

    assert result.n_smooth_terms_ == 1
    assert result.fitted_values.shape == (n,)


def test_fit_additive_gam_with_weights():
    """fit_additive_gam should handle observation weights."""
    rng = np.random.default_rng(42)
    n = 100

    X = np.random.randn(n, 2)
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=n)
    weights = rng.uniform(0.5, 1.5, size=n)

    result = fit_additive_gam(
        X, y,
        smooth_terms=[SmoothTerm(variable=0, n_basis=10)],
        weights=weights
    )

    assert result.weights is not None
    np.testing.assert_array_equal(result.weights, weights)


def test_fit_additive_gam_different_bases():
    """fit_additive_gam should support different basis types per term."""
    rng = np.random.default_rng(42)
    n = 150

    X = np.random.randn(n, 2)
    y = np.sin(X[:, 0]) + X[:, 1]**2 + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(
        X, y,
        smooth_terms=[
            SmoothTerm(variable=0, basis_type="bspline", n_basis=10),
            SmoothTerm(variable=1, basis_type="cubic", n_basis=10)
        ]
    )

    assert result.n_smooth_terms_ == 2
    # Check both bases were created
    assert "s(0)" in result.smooth_bases
    assert "s(1)" in result.smooth_bases


def test_additive_gam_summary():
    """AdditiveGAMResult.summary should generate formatted output."""
    rng = np.random.default_rng(42)
    n = 100

    X = np.random.randn(n, 2)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(
        X, y,
        smooth_terms=[SmoothTerm(variable=0, n_basis=10)],
        parametric_terms=[ParametricTerm(variable=1)]
    )

    summary = result.summary()

    # Check summary contains key information
    assert "Additive" in summary or "GAM" in summary
    assert "s(0)" in summary
    assert "Lambda" in summary or "lambda" in summary
    assert "EDF" in summary or "DoF" in summary
    assert str(n) in summary


def test_additive_gam_repr():
    """AdditiveGAMResult.__repr__ should return informative string."""
    rng = np.random.default_rng(42)
    n = 100

    X = np.random.randn(n, 2)
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(
        X, y,
        smooth_terms=[SmoothTerm(variable=0, n_basis=10)]
    )

    repr_str = repr(result)

    assert "AdditiveGAMResult" in repr_str
    assert "n_obs" in repr_str
    assert "n_smooth" in repr_str


def test_fit_additive_gam_invalid_inputs():
    """fit_additive_gam should validate inputs."""
    rng = np.random.default_rng(42)
    n = 100

    X = np.random.randn(n, 2)
    y = np.random.randn(n)

    # 1D X
    with pytest.raises(ValueError, match="X must be 2-dimensional"):
        fit_additive_gam(
            np.random.randn(n),
            y,
            smooth_terms=[SmoothTerm(variable=0)]
        )

    # 2D y
    with pytest.raises(ValueError, match="y must be 1-dimensional"):
        fit_additive_gam(
            X,
            np.random.randn(n, 2),
            smooth_terms=[SmoothTerm(variable=0)]
        )

    # Mismatched lengths
    with pytest.raises(ValueError, match="same number of rows"):
        fit_additive_gam(
            X,
            np.random.randn(n + 10),
            smooth_terms=[SmoothTerm(variable=0)]
        )

    # No smooth terms
    with pytest.raises(ValueError, match="at least one smooth term"):
        fit_additive_gam(X, y, smooth_terms=[])

    # Variable index out of range
    with pytest.raises(ValueError, match="out of range"):
        fit_additive_gam(
            X,
            y,
            smooth_terms=[SmoothTerm(variable=5)]
        )

    # Invalid weights shape
    with pytest.raises(ValueError, match="weights must have shape"):
        fit_additive_gam(
            X,
            y,
            smooth_terms=[SmoothTerm(variable=0)],
            weights=np.ones(n + 1)
        )


def test_fit_additive_gam_three_smooths():
    """fit_additive_gam should handle three smooth terms."""
    rng = np.random.default_rng(42)
    n = 200

    X = np.random.randn(n, 3)
    y = (np.sin(2 * X[:, 0]) +
         np.cos(X[:, 1]) +
         X[:, 2]**2 +
         0.1 * rng.normal(size=n))

    result = fit_additive_gam(
        X, y,
        smooth_terms=[
            SmoothTerm(variable=0, n_basis=10),
            SmoothTerm(variable=1, n_basis=10),
            SmoothTerm(variable=2, n_basis=10)
        ]
    )

    assert result.n_smooth_terms_ == 3
    assert len(result.smooth_coef) == 3
    assert "s(0)" in result.smooth_coef
    assert "s(1)" in result.smooth_coef
    assert "s(2)" in result.smooth_coef


def test_fit_additive_gam_edf_tracking():
    """AdditiveGAMResult should track EDF for each smooth."""
    rng = np.random.default_rng(42)
    n = 150

    X = np.random.randn(n, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(
        X, y,
        smooth_terms=[
            SmoothTerm(variable=0, n_basis=12),
            SmoothTerm(variable=1, n_basis=12)
        ]
    )

    # Each smooth should have EDF tracked
    assert "s(0)" in result.edf_values
    assert "s(1)" in result.edf_values

    # EDF should be non-negative and finite
    assert result.edf_values["s(0)"] >= 0
    assert result.edf_values["s(1)"] >= 0
    assert np.isfinite(result.edf_values["s(0)"])
    assert np.isfinite(result.edf_values["s(1)"])

    # Total EDF should be non-negative
    assert result.total_edf_ >= 0


def test_fit_additive_gam_residuals():
    """Residuals should be computed correctly."""
    rng = np.random.default_rng(42)
    n = 100

    X = np.random.randn(n, 2)
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(
        X, y,
        smooth_terms=[SmoothTerm(variable=0, n_basis=10)]
    )

    # Residuals = y - fitted_values
    expected_residuals = y - result.fitted_values
    np.testing.assert_allclose(result.residuals, expected_residuals)


def test_fit_additive_gam_fit_quality():
    """Additive GAM should provide good fit to data."""
    rng = np.random.default_rng(42)
    n = 200

    # Known functional form
    X = np.random.randn(n, 2)
    y_true = 2 * np.sin(np.pi * X[:, 0]) + 3 * np.cos(2 * X[:, 1])
    y = y_true + 0.2 * rng.normal(size=n)

    result = fit_additive_gam(
        X, y,
        smooth_terms=[
            SmoothTerm(variable=0, n_basis=15),
            SmoothTerm(variable=1, n_basis=15)
        ]
    )

    # Should capture the nonlinear patterns well
    mse = np.mean((result.fitted_values - y_true) ** 2)
    noise_var = 0.2**2

    # MSE should be close to noise level
    assert mse < 3 * noise_var
