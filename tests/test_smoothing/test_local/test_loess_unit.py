"""Unit tests for LOESS components.

These tests verify individual components of the LOESS implementation
in isolation.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from aurora.smoothing.local import LOESSSmoother, loess, LOESSResult


class TestLOESSSmootherConstruction:
    """Unit tests for LOESSSmoother construction."""

    def test_default_parameters(self):
        """Test default parameter values."""
        smoother = LOESSSmoother()
        assert smoother.span == 0.75
        assert smoother.degree == 1
        assert smoother.robust is False
        assert smoother.n_robust_iter == 3

    def test_custom_parameters(self):
        """Test custom parameter values."""
        smoother = LOESSSmoother(span=0.5, degree=2, robust=True, n_robust_iter=5)
        assert smoother.span == 0.5
        assert smoother.degree == 2
        assert smoother.robust is True
        assert smoother.n_robust_iter == 5

    def test_invalid_span_low_raises(self):
        """Test that span <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="span must be in"):
            LOESSSmoother(span=0.0)

    def test_invalid_span_high_raises(self):
        """Test that span > 1 raises ValueError."""
        with pytest.raises(ValueError, match="span must be in"):
            LOESSSmoother(span=1.5)

    def test_invalid_degree_raises(self):
        """Test that degree not in {1, 2} raises ValueError."""
        with pytest.raises(ValueError, match="degree must be 1 or 2"):
            LOESSSmoother(degree=3)


class TestTricubeWeights:
    """Unit tests for tricube weight function."""

    def test_tricube_at_zero(self):
        """Test tricube weight at u=0."""
        smoother = LOESSSmoother()
        # Access private method for testing
        w = smoother._tricube_weights(np.array([0.0]))
        assert_allclose(w, [1.0])

    def test_tricube_at_boundary(self):
        """Test tricube weight at u=1."""
        smoother = LOESSSmoother()
        w = smoother._tricube_weights(np.array([1.0]))
        assert_allclose(w, [0.0], atol=1e-10)

    def test_tricube_outside_boundary(self):
        """Test tricube weight for |u| > 1."""
        smoother = LOESSSmoother()
        w = smoother._tricube_weights(np.array([1.5, 2.0, -1.5]))
        assert_allclose(w, [0.0, 0.0, 0.0])

    def test_tricube_symmetric(self):
        """Test tricube weight is symmetric."""
        smoother = LOESSSmoother()
        u = np.array([0.3, 0.5, 0.7])
        w_pos = smoother._tricube_weights(u)
        w_neg = smoother._tricube_weights(-u)
        assert_allclose(w_pos, w_neg)

    def test_tricube_decreasing(self):
        """Test tricube weight decreases as |u| increases."""
        smoother = LOESSSmoother()
        u = np.array([0.0, 0.3, 0.5, 0.7, 0.9])
        w = smoother._tricube_weights(u)
        assert np.all(np.diff(w) <= 0)


class TestBisquareWeights:
    """Unit tests for bisquare weight function (robust fitting)."""

    def test_bisquare_at_zero(self):
        """Test bisquare weight at u=0."""
        smoother = LOESSSmoother()
        w = smoother._bisquare_weights(np.array([0.0]))
        assert_allclose(w, [1.0])

    def test_bisquare_at_boundary(self):
        """Test bisquare weight at u=1."""
        smoother = LOESSSmoother()
        w = smoother._bisquare_weights(np.array([1.0]))
        assert_allclose(w, [0.0], atol=1e-10)

    def test_bisquare_outside_boundary(self):
        """Test bisquare weight for |u| > 1."""
        smoother = LOESSSmoother()
        w = smoother._bisquare_weights(np.array([1.5, 2.0]))
        assert_allclose(w, [0.0, 0.0])


class TestLocalRegression:
    """Unit tests for local regression computation."""

    @pytest.fixture
    def linear_data(self):
        """Linear data for testing."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + np.random.normal(0, 0.5, 100)
        return x, y

    def test_local_linear_on_linear_data(self, linear_data):
        """Test that local linear fits linear data well."""
        x, y = linear_data
        result = loess(x, y, span=0.5, degree=1)

        # Should fit linear trend well
        expected = 2 * x + 1
        corr = np.corrcoef(result.fitted_values_, expected)[0, 1]
        assert corr > 0.99

    def test_local_quadratic_on_quadratic_data(self):
        """Test that local quadratic fits quadratic data well."""
        np.random.seed(42)
        x = np.linspace(-2, 2, 100)
        y = x**2 + np.random.normal(0, 0.1, 100)

        result = loess(x, y, span=0.5, degree=2)

        expected = x**2
        corr = np.corrcoef(result.fitted_values_, expected)[0, 1]
        assert corr > 0.95


class TestSpanEffect:
    """Unit tests for span parameter effect."""

    @pytest.fixture
    def noisy_sine(self):
        np.random.seed(42)
        x = np.linspace(0, 4 * np.pi, 200)
        y = np.sin(x) + np.random.normal(0, 0.3, 200)
        return x, y

    def test_small_span_more_wiggly(self, noisy_sine):
        """Test that small span produces more variable fit."""
        x, y = noisy_sine
        result_small = loess(x, y, span=0.1)
        result_large = loess(x, y, span=0.9)

        var_small = np.var(result_small.fitted_values_)
        var_large = np.var(result_large.fitted_values_)

        assert var_small > var_large

    def test_large_span_smoother(self, noisy_sine):
        """Test that large span produces smoother fit."""
        x, y = noisy_sine
        result_small = loess(x, y, span=0.2)
        result_large = loess(x, y, span=0.8)

        # Compute roughness as sum of squared second differences
        def roughness(f):
            return np.sum(np.diff(f, n=2) ** 2)

        assert roughness(result_large.fitted_values_) < roughness(result_small.fitted_values_)


class TestRobustFitting:
    """Unit tests for robust LOESS fitting."""

    def test_robust_handles_outliers(self):
        """Test that robust fitting handles outliers."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1

        # Add outliers
        y_outliers = y.copy()
        y_outliers[20] = 100
        y_outliers[50] = -50
        y_outliers[80] = 80

        result_standard = loess(x, y_outliers, span=0.3, robust=False)
        result_robust = loess(x, y_outliers, span=0.3, robust=True)

        # Robust fit should be closer to true line
        true_y = 2 * x + 1
        mse_standard = np.mean((result_standard.fitted_values_ - true_y) ** 2)
        mse_robust = np.mean((result_robust.fitted_values_ - true_y) ** 2)

        assert mse_robust < mse_standard

    def test_robust_iterations_improve_fit(self):
        """Test that more robust iterations can improve fit."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        y[10] = 10  # Outlier

        result_1iter = LOESSSmoother(span=0.5, robust=True, n_robust_iter=1).fit(x, y)
        result_3iter = LOESSSmoother(span=0.5, robust=True, n_robust_iter=3).fit(x, y)

        # More iterations should give fit closer to true sine
        true_y = np.sin(x)
        mse_1 = np.mean((result_1iter.fitted_values_ - true_y) ** 2)
        mse_3 = np.mean((result_3iter.fitted_values_ - true_y) ** 2)

        assert mse_3 <= mse_1 * 1.1  # Allow small tolerance


class TestLOESSResultMethods:
    """Unit tests for LOESSResult methods."""

    @pytest.fixture
    def fitted_result(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.2, 100)
        return loess(x, y, span=0.3)

    def test_predict_at_training_points(self, fitted_result):
        """Test prediction at training points matches fitted values."""
        y_pred = fitted_result.predict(fitted_result.x_)
        assert_allclose(y_pred, fitted_result.fitted_values_, rtol=1e-5)

    def test_predict_at_new_points(self, fitted_result):
        """Test prediction at new points."""
        x_new = np.linspace(0, 10, 200)
        y_pred = fitted_result.predict(x_new)
        assert y_pred.shape == (200,)
        assert np.all(np.isfinite(y_pred))

    def test_residuals_shape(self, fitted_result):
        """Test residuals have correct shape."""
        residuals = fitted_result.y_ - fitted_result.fitted_values_
        assert residuals.shape == fitted_result.y_.shape

    def test_summary_contains_required_keys(self, fitted_result):
        """Test that summary contains required keys."""
        summary = fitted_result.summary()
        required_keys = ["n_obs", "span", "degree", "r_squared"]
        for key in required_keys:
            assert key in summary

    def test_r_squared_bounded(self, fitted_result):
        """Test that R-squared is in [0, 1]."""
        summary = fitted_result.summary()
        assert 0 <= summary["r_squared"] <= 1


class TestEdgeCases:
    """Unit tests for edge cases."""

    def test_minimum_data_points(self):
        """Test with minimum number of data points."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        result = loess(x, y, span=0.8)
        assert result.fitted_values_.shape == (5,)

    def test_constant_response(self):
        """Test with constant response."""
        x = np.linspace(0, 1, 50)
        y = np.ones(50) * 5.0

        result = loess(x, y, span=0.5)
        assert_allclose(result.fitted_values_, 5.0, atol=1e-10)

    def test_unsorted_data(self):
        """Test that unsorted data is handled correctly."""
        np.random.seed(42)
        x = np.random.rand(50) * 10
        y = np.sin(x)

        result = loess(x, y, span=0.5)
        assert result.fitted_values_.shape == (50,)
        assert np.all(np.isfinite(result.fitted_values_))

    def test_duplicate_x_values(self):
        """Test handling of duplicate x values."""
        x = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=float)
        y = np.array([1.1, 0.9, 2.1, 1.9, 3.1, 2.9, 4.1, 3.9, 5.1, 4.9])

        result = loess(x, y, span=0.5)
        assert result.fitted_values_.shape == (10,)
