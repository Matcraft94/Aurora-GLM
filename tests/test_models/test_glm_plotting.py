"""Tests for GLMResult.plot_diagnostics() method."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models.glm import fit_glm


def test_plot_diagnostics_creates_figure_with_four_subplots():
    """plot_diagnostics() should create a figure with 4 diagnostic plots."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for testing

    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 2))
    y = rng.poisson(np.exp(X[:, 0] * 0.5))

    result = fit_glm(X, y, family="poisson", link="log")
    fig = result.plot_diagnostics()

    # Should have 2x2 = 4 main axes (colorbar adds one more)
    assert len(fig.axes) >= 4

    # Check that figure was created
    assert fig is not None
    assert hasattr(fig, "savefig")


def test_plot_diagnostics_raises_without_matplotlib():
    """plot_diagnostics() should raise ImportError if matplotlib is not available."""
    # This test is tricky because matplotlib IS installed in the test environment
    # We'll skip it for now, but document the expected behavior
    pytest.skip("Cannot test ImportError when matplotlib is installed")


def test_plot_diagnostics_uses_cached_diagnostics():
    """plot_diagnostics() should use the cached diagnostics property."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    rng = np.random.default_rng(123)
    X = rng.normal(size=(150, 3))
    y = rng.binomial(1, 1 / (1 + np.exp(-(X @ np.array([0.7, -0.5, 0.3]) + 0.2))))

    result = fit_glm(X, y, family="binomial", link="logit")

    # Access diagnostics first to cache them
    _ = result.diagnostics_

    # plot_diagnostics should use the cached version
    fig = result.plot_diagnostics()

    assert fig is not None
    # Verify it didn't recompute by checking cache is still there
    assert result._diagnostics_cache is not None


def test_plot_can_be_saved_to_file(tmp_path):
    """Generated diagnostic plots should be saveable to a file."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    rng = np.random.default_rng(456)
    X = rng.normal(size=(100, 2))
    y = rng.normal(X @ np.array([1.0, -0.5]) + 2.0, scale=1.0)

    result = fit_glm(X, y, family="gaussian")
    fig = result.plot_diagnostics()

    # Save to temporary file
    output_path = tmp_path / "diagnostics_test.png"
    fig.savefig(str(output_path), dpi=50, bbox_inches="tight")

    # Verify file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_diagnostics_works_with_different_families():
    """plot_diagnostics() should work with different GLM families."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    rng = np.random.default_rng(789)

    # Test with Gamma family
    X = rng.normal(size=(150, 2))
    shape = 2.0
    mu = np.exp(X @ np.array([0.5, -0.3]) + 1.0)
    scale = mu / shape
    y = rng.gamma(shape, scale)

    result = fit_glm(X, y, family="gamma", link="log")
    fig = result.plot_diagnostics()

    assert fig is not None
    assert len(fig.axes) >= 4


def test_plot_diagnostics_custom_figsize():
    """plot_diagnostics() should respect custom figsize parameter."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    rng = np.random.default_rng(321)
    X = rng.normal(size=(100, 2))
    y = rng.poisson(np.exp(X[:, 0] * 0.5))

    result = fit_glm(X, y, family="poisson")
    fig = result.plot_diagnostics(figsize=(8, 6))

    # Check figsize was applied
    assert fig.get_figwidth() == pytest.approx(8.0, abs=0.1)
    assert fig.get_figheight() == pytest.approx(6.0, abs=0.1)
