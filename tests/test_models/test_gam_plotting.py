"""Tests for GAM plotting functions."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.models.gam import (
    ParametricTerm,
    SmoothTerm,
    fit_additive_gam,
    fit_gam,
    plot_all_smooths,
    plot_smooth,
)

# Check if matplotlib is available
pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt


def test_plot_smooth_univariate():
    """plot_smooth should work with univariate GAMResult."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 10, n)
    y = np.sin(x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=15)

    # Should create plot without error
    fig = plot_smooth(result)

    assert fig is not None
    assert len(fig.axes) == 1

    ax = fig.axes[0]
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "f(x)"

    plt.close(fig)


def test_plot_smooth_univariate_custom_labels():
    """plot_smooth should accept custom labels."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 10, n)
    y = np.sin(x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=15)

    fig = plot_smooth(result, title="Custom Title", xlabel="Custom X", ylabel="Custom Y")

    ax = fig.axes[0]
    assert ax.get_title() == "Custom Title"
    assert ax.get_xlabel() == "Custom X"
    assert ax.get_ylabel() == "Custom Y"

    plt.close(fig)


def test_plot_smooth_univariate_no_partial_residuals():
    """plot_smooth should work without partial residuals."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 10, n)
    y = np.sin(x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=15)

    fig = plot_smooth(result, partial_residuals=False)

    assert fig is not None
    plt.close(fig)


def test_plot_smooth_univariate_confidence_level():
    """plot_smooth should respect confidence level."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 10, n)
    y = np.sin(x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=15)

    fig = plot_smooth(result, confidence_level=0.90)

    assert fig is not None
    ax = fig.axes[0]

    # Check legend contains confidence level
    legend_text = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("90" in t for t in legend_text)

    plt.close(fig)


def test_plot_smooth_additive_single_term():
    """plot_smooth should work with AdditiveGAMResult for single term."""
    rng = np.random.default_rng(42)
    n = 150

    X = np.random.randn(n, 2)
    y = np.sin(2 * X[:, 0]) + np.cos(X[:, 1]) + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(
        X, y, smooth_terms=[SmoothTerm(variable=0, n_basis=12), SmoothTerm(variable=1, n_basis=12)]
    )

    # Plot first term using term name
    fig1 = plot_smooth(result, term="s(0)")
    assert fig1 is not None
    plt.close(fig1)

    # Plot second term using variable index
    fig2 = plot_smooth(result, term=1)
    assert fig2 is not None
    plt.close(fig2)


def test_plot_smooth_additive_requires_term():
    """plot_smooth should require term for AdditiveGAMResult."""
    rng = np.random.default_rng(42)
    n = 100

    X = np.random.randn(n, 2)
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(X, y, smooth_terms=[SmoothTerm(variable=0)])

    # Should raise error without term
    with pytest.raises(ValueError, match="must specify which term"):
        plot_smooth(result, term=None)


def test_plot_smooth_additive_invalid_term():
    """plot_smooth should reject invalid term name."""
    rng = np.random.default_rng(42)
    n = 100

    X = np.random.randn(n, 2)
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(X, y, smooth_terms=[SmoothTerm(variable=0)])

    with pytest.raises(ValueError, match="not found"):
        plot_smooth(result, term="s(5)")


def test_plot_smooth_univariate_rejects_term():
    """plot_smooth should reject term parameter for univariate GAM."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 10, n)
    y = np.sin(x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y)

    with pytest.raises(ValueError, match="term should be None"):
        plot_smooth(result, term="s(0)")


def test_plot_smooth_with_custom_ax():
    """plot_smooth should work with provided axes."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 10, n)
    y = np.sin(x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y)

    # Create custom figure/axes
    fig_custom, ax_custom = plt.subplots()
    fig_returned = plot_smooth(result, ax=ax_custom)

    assert fig_returned is fig_custom
    assert ax_custom.has_data()

    plt.close(fig_custom)


def test_plot_all_smooths_basic():
    """plot_all_smooths should create grid of all smooth terms."""
    rng = np.random.default_rng(42)
    n = 150

    X = np.random.randn(n, 3)
    y = np.sin(2 * X[:, 0]) + np.cos(X[:, 1]) + X[:, 2] ** 2 + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(
        X,
        y,
        smooth_terms=[
            SmoothTerm(variable=0, n_basis=10),
            SmoothTerm(variable=1, n_basis=10),
            SmoothTerm(variable=2, n_basis=10),
        ],
    )

    fig = plot_all_smooths(result, ncols=2)

    assert fig is not None
    # Should have at least 3 axes (3 terms)
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == 3

    plt.close(fig)


def test_plot_all_smooths_single_term():
    """plot_all_smooths should work with single smooth term."""
    rng = np.random.default_rng(42)
    n = 100

    X = np.random.randn(n, 1)
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(X, y, smooth_terms=[SmoothTerm(variable=0)])

    fig = plot_all_smooths(result)

    assert fig is not None
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == 1

    plt.close(fig)


def test_plot_all_smooths_ncols():
    """plot_all_smooths should respect ncols parameter."""
    rng = np.random.default_rng(42)
    n = 150

    X = np.random.randn(n, 4)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2] ** 2 + X[:, 3] + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(
        X,
        y,
        smooth_terms=[
            SmoothTerm(variable=0),
            SmoothTerm(variable=1),
            SmoothTerm(variable=2),
            SmoothTerm(variable=3),
        ],
    )

    # 4 terms with ncols=2 should give 2 rows
    fig = plot_all_smooths(result, ncols=2)

    assert fig is not None
    # Total axes should be ncols * nrows = 2 * 2 = 4
    assert len(fig.axes) >= 4

    plt.close(fig)


def test_plot_all_smooths_no_partial_residuals():
    """plot_all_smooths should work without partial residuals."""
    rng = np.random.default_rng(42)
    n = 100

    X = np.random.randn(n, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(X, y, smooth_terms=[SmoothTerm(variable=0), SmoothTerm(variable=1)])

    fig = plot_all_smooths(result, partial_residuals=False)

    assert fig is not None
    plt.close(fig)


def test_plot_all_smooths_requires_additive():
    """plot_all_smooths should require AdditiveGAMResult."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 10, n)
    y = np.sin(x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y)

    with pytest.raises(TypeError, match="requires AdditiveGAMResult"):
        plot_all_smooths(result)


def test_plot_smooth_different_n_points():
    """plot_smooth should respect n_points parameter."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 10, n)
    y = np.sin(x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y)

    fig = plot_smooth(result, n_points=50)

    assert fig is not None
    ax = fig.axes[0]

    # Check that smooth line exists
    lines = ax.get_lines()
    # Should have at least one line (the smooth curve)
    assert len(lines) > 0

    plt.close(fig)


def test_plot_smooth_cubic_basis():
    """plot_smooth should work with cubic spline basis."""
    rng = np.random.default_rng(42)
    n = 100

    x = np.linspace(0, 10, n)
    y = np.sin(x) + 0.1 * rng.normal(size=n)

    result = fit_gam(x, y, basis_type="cubic", n_basis=12)

    fig = plot_smooth(result)

    assert fig is not None
    plt.close(fig)


def test_plot_smooth_additive_with_parametric():
    """plot_smooth should work for additive GAM with parametric terms."""
    rng = np.random.default_rng(42)
    n = 150

    X = np.random.randn(n, 2)
    y = np.sin(X[:, 0]) + 2 * X[:, 1] + 0.1 * rng.normal(size=n)

    result = fit_additive_gam(
        X, y, smooth_terms=[SmoothTerm(variable=0)], parametric_terms=[ParametricTerm(variable=1)]
    )

    # Plot the smooth term
    fig = plot_smooth(result, term=0)

    assert fig is not None
    plt.close(fig)


def test_plot_smooth_handles_singular_precision():
    """plot_smooth should handle singular precision matrix gracefully."""
    rng = np.random.default_rng(42)
    n = 50

    # Use very few points to potentially create singular matrix
    x = np.linspace(0, 1, n)
    y = x + 0.01 * rng.normal(size=n)

    result = fit_gam(x, y, n_basis=30)  # Many basis functions for few points

    # Should not raise error even if precision is singular
    fig = plot_smooth(result)

    assert fig is not None
    plt.close(fig)
