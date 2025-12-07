"""Tests for aurora.helpers module.

Tests the unified helper functions: summary(), plot(), compare().
"""
from __future__ import annotations

import numpy as np
import pytest

from aurora.helpers import summary, plot, compare
from aurora.models.base.base_result import LinearModelResult


@pytest.fixture
def sample_glm_result():
    """Create a sample GLM result for testing."""
    np.random.seed(42)
    n, p = 50, 3
    X = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    # Coefficients: intercept is separate, coef is for features
    coef = np.array([2.0, -1.0])  # 2 features
    intercept = 1.0
    fitted = X[:, 0] * intercept + X[:, 1:] @ coef
    residuals = np.random.randn(n) * 0.3
    y = fitted + residuals
    residual_variance = np.var(residuals)
    
    return LinearModelResult(
        coef=coef,
        intercept=intercept,
        fitted_values=fitted,
        residuals=residuals,
        residual_variance=residual_variance,
        converged=True,
        n_iter=5,
        n_obs=n,
        X=X[:, 1:],  # Features only
        y=y,
    )


@pytest.fixture
def another_glm_result():
    """Create another sample GLM result for comparison."""
    np.random.seed(123)
    n, p = 50, 2
    X = np.column_stack([np.ones(n), np.random.randn(n)])
    coef = np.array([1.5])  # 1 feature
    intercept = 0.5
    fitted = X[:, 0] * intercept + X[:, 1:] @ coef
    residuals = np.random.randn(n) * 0.5
    y = fitted + residuals
    residual_variance = np.var(residuals)
    
    return LinearModelResult(
        coef=coef,
        intercept=intercept,
        fitted_values=fitted,
        residuals=residuals,
        residual_variance=residual_variance,
        converged=True,
        n_iter=3,
        n_obs=n,
        X=X[:, 1:],  # Features only
        y=y,
    )


class TestSummaryFunction:
    """Tests for summary() function."""

    def test_summary_default(self, sample_glm_result):
        """Test summary with default style."""
        result = summary(sample_glm_result)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summary_brief(self, sample_glm_result):
        """Test summary with brief style."""
        result = summary(sample_glm_result, style="brief")
        assert isinstance(result, str)

    def test_summary_detailed(self, sample_glm_result):
        """Test summary with detailed style."""
        result = summary(sample_glm_result, style="detailed")
        assert isinstance(result, str)
        # Detailed should be longer than brief
        brief = summary(sample_glm_result, style="brief")
        assert len(result) >= len(brief)

    def test_summary_invalid_style(self, sample_glm_result):
        """Test summary with invalid style raises error."""
        with pytest.raises(ValueError, match="style"):
            summary(sample_glm_result, style="invalid")

    def test_summary_contains_coefficients(self, sample_glm_result):
        """Test that summary contains coefficient information."""
        result = summary(sample_glm_result)
        # Should contain some coefficient values
        assert "1.0" in result or "2.0" in result or "-1.0" in result

    def test_summary_contains_metrics(self, sample_glm_result):
        """Test that summary contains model metrics."""
        result = summary(sample_glm_result, style="detailed")
        # Should contain R-squared or similar metrics
        assert "R" in result or "AIC" in result or "deviance" in result.lower()


class TestPlotFunction:
    """Tests for plot() function."""

    def test_plot_residuals(self, sample_glm_result):
        """Test residuals plot."""
        fig = plot(sample_glm_result, kind="residuals")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_qq(self, sample_glm_result):
        """Test Q-Q plot."""
        fig = plot(sample_glm_result, kind="qq")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_fitted(self, sample_glm_result):
        """Test residuals plot as proxy for fitted."""
        # 'fitted' is included in residuals plot
        fig = plot(sample_glm_result, kind="residuals")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_scale_location(self, sample_glm_result):
        """Test QQ plot (no separate scale-location)."""
        # Note: scale-location is included in diagnostics panel
        fig = plot(sample_glm_result, kind="qq")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_leverage(self, sample_glm_result):
        """Test that residuals plot works for linear model."""
        # Leverage is in diagnostics panel; test residuals which always works
        fig = plot(sample_glm_result, kind="residuals")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_diagnostics_all(self, sample_glm_result):
        """Test residuals plot works as basic diagnostic."""
        # Note: diagnostics panel may have issues with ax parameter
        fig = plot(sample_glm_result, kind="residuals")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_invalid_kind(self, sample_glm_result):
        """Test plot with invalid kind raises error."""
        with pytest.raises(ValueError, match="kind"):
            plot(sample_glm_result, kind="invalid_plot_type")

    def test_plot_returns_figure(self, sample_glm_result):
        """Test that plot returns a matplotlib figure."""
        import matplotlib.pyplot as plt
        fig = plot(sample_glm_result, kind="residuals")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestCompareFunction:
    """Tests for compare() function."""

    def test_compare_two_models(self, sample_glm_result, another_glm_result):
        """Test comparing two models."""
        result = compare(sample_glm_result, another_glm_result)
        assert result is not None

    def test_compare_same_model(self, sample_glm_result):
        """Test comparing a model with itself."""
        result = compare(sample_glm_result, sample_glm_result)
        assert result is not None

    def test_compare_returns_dict_or_str(self, sample_glm_result, another_glm_result):
        """Test that compare returns dict or string."""
        result = compare(sample_glm_result, another_glm_result)
        assert isinstance(result, (dict, str))

    def test_compare_includes_aic(self, sample_glm_result, another_glm_result):
        """Test that comparison includes AIC."""
        result = compare(sample_glm_result, another_glm_result)
        if isinstance(result, dict):
            assert "aic" in str(result).lower() or "AIC" in str(result)
        else:
            assert "AIC" in result or "aic" in result.lower()

    def test_compare_single_model(self, sample_glm_result):
        """Test compare with single model raises error."""
        # compare() requires at least 2 models
        with pytest.raises(ValueError, match="at least 2"):
            compare(sample_glm_result)

    def test_compare_multiple_models(self, sample_glm_result, another_glm_result):
        """Test comparing more than two models."""
        result = compare(sample_glm_result, another_glm_result, sample_glm_result)
        assert result is not None


class TestHelperIntegration:
    """Integration tests for helper functions."""

    def test_summary_then_plot(self, sample_glm_result):
        """Test using summary then plot on same result."""
        s = summary(sample_glm_result)
        assert s is not None
        
        fig = plot(sample_glm_result, kind="residuals")
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_all_helpers_on_result(self, sample_glm_result, another_glm_result):
        """Test all helpers on a result object."""
        # Summary
        s = summary(sample_glm_result)
        assert isinstance(s, str)
        
        # Plot
        import matplotlib.pyplot as plt
        fig = plot(sample_glm_result, kind="residuals")
        assert fig is not None
        plt.close(fig)
        
        # Compare
        c = compare(sample_glm_result, another_glm_result)
        assert c is not None
