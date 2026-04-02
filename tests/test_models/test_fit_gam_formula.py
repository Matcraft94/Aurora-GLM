"""Tests for formula-based GAM fitting."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.models.gam import AdditiveGAMResult, fit_gam_formula


def test_fit_gam_formula_dict_basic():
    """fit_gam_formula should work with dict data."""
    rng = np.random.default_rng(42)
    n = 150

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = np.sin(2 * x1) + np.cos(x2) + 0.1 * rng.normal(size=n)

    data = {"y": y, "x1": x1, "x2": x2}

    result = fit_gam_formula("y ~ s(x1) + s(x2)", data)

    assert isinstance(result, AdditiveGAMResult)
    assert result.n_smooth_terms_ == 2
    assert result.n_parametric_terms_ == 0


def test_fit_gam_formula_dict_mixed():
    """fit_gam_formula should handle mixed smooth and parametric terms."""
    rng = np.random.default_rng(42)
    n = 150

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    y = np.sin(2 * x1) + 2 * x2 + 0.5 * x3 + 0.1 * rng.normal(size=n)

    data = {"response": y, "temp": x1, "pressure": x2, "humidity": x3}

    result = fit_gam_formula("response ~ s(temp) + pressure + humidity", data)

    assert result.n_smooth_terms_ == 1
    assert result.n_parametric_terms_ == 2


def test_fit_gam_formula_array_basic():
    """fit_gam_formula should work with array data and column indices."""
    rng = np.random.default_rng(42)
    n = 150

    X = np.random.randn(n, 2)
    y = np.sin(2 * X[:, 0]) + np.cos(X[:, 1]) + 0.1 * rng.normal(size=n)

    # Stack y as first column
    data = np.column_stack([y, X])

    result = fit_gam_formula("0 ~ s(1) + s(2)", data)

    assert isinstance(result, AdditiveGAMResult)
    assert result.n_smooth_terms_ == 2
    assert result.n_parametric_terms_ == 0


def test_fit_gam_formula_array_mixed():
    """fit_gam_formula should handle mixed terms with array data."""
    rng = np.random.default_rng(42)
    n = 150

    X = np.random.randn(n, 3)
    y = np.sin(2 * X[:, 0]) + 2 * X[:, 1] + 0.5 * X[:, 2] + 0.1 * rng.normal(size=n)

    data = np.column_stack([y, X])

    result = fit_gam_formula("0 ~ s(1) + 2 + 3", data)

    assert result.n_smooth_terms_ == 1
    assert result.n_parametric_terms_ == 2


def test_fit_gam_formula_with_options():
    """fit_gam_formula should parse smooth term options."""
    rng = np.random.default_rng(42)
    n = 150

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = np.sin(2 * x1) + np.cos(x2) + 0.1 * rng.normal(size=n)

    data = {"y": y, "x1": x1, "x2": x2}

    result = fit_gam_formula("y ~ s(x1, n_basis=15) + s(x2, basis='cubic')", data)

    assert result.n_smooth_terms_ == 2
    # Options should have been applied (though we can't directly inspect them from result)


def test_fit_gam_formula_with_weights():
    """fit_gam_formula should handle observation weights."""
    rng = np.random.default_rng(42)
    n = 150

    x1 = np.random.randn(n)
    y = np.sin(2 * x1) + 0.1 * rng.normal(size=n)
    weights = rng.uniform(0.5, 1.5, size=n)

    data = {"y": y, "x": x1}

    result = fit_gam_formula("y ~ s(x)", data, weights=weights)

    assert result.weights is not None
    np.testing.assert_array_equal(result.weights, weights)


def test_fit_gam_formula_with_method():
    """fit_gam_formula should support REML method."""
    rng = np.random.default_rng(42)
    n = 150

    x1 = np.random.randn(n)
    y = np.sin(2 * x1) + 0.1 * rng.normal(size=n)

    data = {"y": y, "x": x1}

    result = fit_gam_formula("y ~ s(x)", data, method="REML")

    assert isinstance(result, AdditiveGAMResult)


def test_fit_gam_formula_predictions():
    """fit_gam_formula result should support predictions."""
    rng = np.random.default_rng(42)
    n = 100

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = np.sin(x1) + np.cos(x2) + 0.1 * rng.normal(size=n)

    data = {"y": y, "x1": x1, "x2": x2}

    result = fit_gam_formula("y ~ s(x1) + s(x2)", data)

    # Predict on new data
    x1_new = np.random.randn(50)
    x2_new = np.random.randn(50)
    X_new = np.column_stack([x1_new, x2_new])

    y_pred = result.predict(X_new)

    assert y_pred.shape == (50,)
    assert np.all(np.isfinite(y_pred))


def test_fit_gam_formula_dict_invalid_var():
    """fit_gam_formula should raise error for missing variable in dict."""
    data = {"y": np.random.randn(100), "x1": np.random.randn(100)}

    with pytest.raises(KeyError):
        fit_gam_formula("y ~ s(x1) + s(x2)", data)  # x2 not in data


def test_fit_gam_formula_array_invalid_response():
    """fit_gam_formula should validate response for array data."""
    data = np.random.randn(100, 3)

    with pytest.raises(ValueError, match="response must be a column index"):
        fit_gam_formula("y ~ s(1) + s(2)", data)  # Response is string, not index


def test_fit_gam_formula_array_invalid_predictor():
    """fit_gam_formula should validate predictors for array data."""
    data = np.random.randn(100, 3)

    with pytest.raises(ValueError, match="all variables must be column indices"):
        fit_gam_formula("0 ~ s(x1) + s(2)", data)  # x1 is string, not index


def test_fit_gam_formula_array_not_2d():
    """fit_gam_formula should validate array dimensionality."""
    data = np.random.randn(100)  # 1D array

    with pytest.raises(ValueError, match="must be 2-dimensional"):
        fit_gam_formula("0 ~ s(1)", data)


def test_fit_gam_formula_fit_quality():
    """fit_gam_formula should provide good fit."""
    rng = np.random.default_rng(42)
    n = 200

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y_true = 2 * np.sin(np.pi * x1) + 3 * np.cos(2 * x2)
    y = y_true + 0.2 * rng.normal(size=n)

    data = {"y": y, "x1": x1, "x2": x2}

    result = fit_gam_formula("y ~ s(x1, n_basis=15) + s(x2, n_basis=15)", data)

    # Should provide good fit
    r_squared = 1 - np.sum(result.residuals**2) / np.sum((y - np.mean(y)) ** 2)
    assert r_squared > 0.6
