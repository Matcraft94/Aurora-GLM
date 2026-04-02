"""Tests for LOESS smoother."""

import numpy as np
import pytest

from aurora.smoothing.local import LOESSResult, LOESSSmoother, loess


class TestLOESSSmoother:
    """Tests for LOESSSmoother class."""

    def test_init_valid(self):
        """Test valid initialization."""
        smoother = LOESSSmoother(span=0.5, degree=1, robust=False)
        assert smoother.span == 0.5
        assert smoother.degree == 1
        assert smoother.robust is False

    def test_init_invalid_span(self):
        """Test invalid span raises error."""
        with pytest.raises(ValueError, match="span must be in"):
            LOESSSmoother(span=0.0)
        with pytest.raises(ValueError, match="span must be in"):
            LOESSSmoother(span=1.5)

    def test_init_invalid_degree(self):
        """Test invalid degree raises error."""
        with pytest.raises(ValueError, match="degree must be 1 or 2"):
            LOESSSmoother(degree=3)


class TestLOESSFitting:
    """Tests for LOESS fitting."""

    @pytest.fixture
    def sine_data(self):
        """Generate noisy sine data."""
        np.random.seed(42)
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x) + np.random.normal(0, 0.2, 100)
        return x, y

    def test_fit_returns_result(self, sine_data):
        """Test that fit returns LOESSResult."""
        x, y = sine_data
        smoother = LOESSSmoother(span=0.3)
        result = smoother.fit(x, y)

        assert isinstance(result, LOESSResult)

    def test_fit_fitted_values_shape(self, sine_data):
        """Test fitted values have correct shape."""
        x, y = sine_data
        result = loess(x, y, span=0.3)

        assert result.fitted_values_.shape == x.shape

    def test_fit_reduces_noise(self, sine_data):
        """Test that smoothing reduces noise."""
        x, y = sine_data
        result = loess(x, y, span=0.3)

        # Smoothed values should be closer to true sine
        true_y = np.sin(x)
        error_raw = np.mean((y - true_y) ** 2)
        error_smooth = np.mean((result.fitted_values_ - true_y) ** 2)

        assert error_smooth < error_raw

    def test_span_controls_smoothness(self, sine_data):
        """Test that larger span produces smoother fit."""
        x, y = sine_data

        result_smooth = loess(x, y, span=0.8)
        result_wiggly = loess(x, y, span=0.2)

        # Smoother fit should have lower variance in second differences
        diff2_smooth = np.diff(result_smooth.fitted_values_, n=2)
        diff2_wiggly = np.diff(result_wiggly.fitted_values_, n=2)

        assert np.var(diff2_smooth) < np.var(diff2_wiggly)

    def test_degree_1_vs_2(self, sine_data):
        """Test local linear vs quadratic."""
        x, y = sine_data

        result_linear = loess(x, y, span=0.3, degree=1)
        result_quad = loess(x, y, span=0.3, degree=2)

        # Both should produce reasonable fits
        true_y = np.sin(x)
        r2_linear = 1 - np.sum((result_linear.fitted_values_ - true_y) ** 2) / np.sum(
            (true_y - true_y.mean()) ** 2
        )
        r2_quad = 1 - np.sum((result_quad.fitted_values_ - true_y) ** 2) / np.sum(
            (true_y - true_y.mean()) ** 2
        )

        assert r2_linear > 0.7
        assert r2_quad > 0.7


class TestLOESSRobust:
    """Tests for robust LOESS."""

    def test_robust_handles_outliers(self):
        """Test that robust LOESS handles outliers."""
        np.random.seed(42)
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)

        # Add outliers
        outlier_idx = [25, 50, 75]
        y[outlier_idx] = [5, -5, 5]

        result_standard = loess(x, y, span=0.3, robust=False)
        result_robust = loess(x, y, span=0.3, robust=True)

        # Robust should be closer to true values at outlier points
        true_y = np.sin(x)
        error_standard = np.abs(result_standard.fitted_values_[outlier_idx] - true_y[outlier_idx])
        error_robust = np.abs(result_robust.fitted_values_[outlier_idx] - true_y[outlier_idx])

        assert np.mean(error_robust) < np.mean(error_standard)


class TestLOESSPrediction:
    """Tests for LOESS prediction."""

    @pytest.fixture
    def fitted_model(self):
        """Fit a LOESS model."""
        np.random.seed(42)
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)
        return loess(x, y, span=0.3)

    def test_predict_at_training_points(self, fitted_model):
        """Test prediction at training points."""
        result = fitted_model
        y_pred = result.predict(result.x_)

        # Should be close to fitted values (interpolation)
        np.testing.assert_allclose(y_pred, result.fitted_values_, atol=1e-10)

    def test_predict_at_new_points(self, fitted_model):
        """Test prediction at new points."""
        result = fitted_model
        x_new = np.linspace(0, 2 * np.pi, 200)
        y_pred = result.predict(x_new)

        assert y_pred.shape == x_new.shape

    def test_predict_interpolation(self, fitted_model):
        """Test that interpolation is reasonable."""
        result = fitted_model

        # Predict at midpoints
        x_mid = (result.x_[:-1] + result.x_[1:]) / 2
        y_pred = result.predict(x_mid)

        # Should be between neighboring fitted values (approximately)
        y_lo = np.minimum(result.fitted_values_[:-1], result.fitted_values_[1:])
        y_hi = np.maximum(result.fitted_values_[:-1], result.fitted_values_[1:])

        # Allow some margin
        margin = 0.1
        assert np.all(y_pred >= y_lo - margin)
        assert np.all(y_pred <= y_hi + margin)

    def test_residuals(self, fitted_model):
        """Test residual computation."""
        result = fitted_model
        resid = result.residuals()

        assert resid.shape == result.y_.shape
        np.testing.assert_allclose(resid, result.y_ - result.fitted_values_)


class TestLOESSSummary:
    """Tests for LOESS summary."""

    def test_summary_contents(self):
        """Test summary contains expected fields."""
        np.random.seed(42)
        x = np.linspace(0, 1, 50)
        y = x**2 + np.random.normal(0, 0.1, 50)

        result = loess(x, y, span=0.5)
        summary = result.summary()

        assert "n_obs" in summary
        assert "span" in summary
        assert "degree" in summary
        assert "r_squared" in summary
        assert "rss" in summary

    def test_summary_r_squared(self):
        """Test R-squared is reasonable."""
        np.random.seed(42)
        x = np.linspace(0, 1, 100)
        y = x**2 + np.random.normal(0, 0.1, 100)

        result = loess(x, y, span=0.3)
        summary = result.summary()

        assert 0 <= summary["r_squared"] <= 1
        assert summary["r_squared"] > 0.8  # Should fit quadratic well


class TestLOESSEdgeCases:
    """Test edge cases."""

    def test_minimum_data_points(self):
        """Test with minimum data points."""
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])

        result = loess(x, y, span=1.0)  # Use all points
        assert result.fitted_values_.shape == (3,)

    def test_constant_response(self):
        """Test with constant response."""
        x = np.linspace(0, 1, 50)
        y = np.ones(50) * 3.0

        result = loess(x, y, span=0.5)

        np.testing.assert_allclose(result.fitted_values_, 3.0, atol=1e-10)

    def test_linear_response(self):
        """Test with linear response (no noise)."""
        x = np.linspace(0, 1, 50)
        y = 2 * x + 1

        result = loess(x, y, span=0.5, degree=1)

        np.testing.assert_allclose(result.fitted_values_, y, atol=1e-10)

    def test_unsorted_data(self):
        """Test with unsorted x values."""
        np.random.seed(42)
        x = np.random.uniform(0, 2 * np.pi, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)

        result = loess(x, y, span=0.3)

        # Should still work
        assert result.fitted_values_.shape == (100,)

        # Check that fitted values are reasonable
        true_y = np.sin(x)
        correlation = np.corrcoef(result.fitted_values_, true_y)[0, 1]
        assert correlation > 0.9
