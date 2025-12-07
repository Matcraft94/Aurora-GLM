"""Tests for P-Spline smoothing."""

import numpy as np
import pytest

from aurora.smoothing.splines import PSplineBasis, PSplineResult, fit_pspline


class TestPSplineBasis:
    """Tests for PSplineBasis class."""

    def test_init_valid(self):
        """Test valid initialization."""
        basis = PSplineBasis(n_basis=20, degree=3, penalty_order=2)
        assert basis.n_basis == 20
        assert basis.degree == 3
        assert basis.penalty_order == 2

    def test_init_invalid_n_basis(self):
        """Test that n_basis < 4 raises error."""
        with pytest.raises(ValueError, match="n_basis must be at least 4"):
            PSplineBasis(n_basis=3)

    def test_init_invalid_penalty_order(self):
        """Test that penalty_order >= n_basis raises error."""
        with pytest.raises(ValueError, match="penalty_order.*must be less than n_basis"):
            PSplineBasis(n_basis=10, penalty_order=10)

    def test_basis_matrix_shape(self):
        """Test basis matrix has correct shape."""
        basis = PSplineBasis(n_basis=15, degree=3)
        x = np.linspace(0, 1, 100)
        B = basis.basis_matrix(x)

        assert B.shape == (100, 15)

    def test_basis_matrix_partition_of_unity(self):
        """Test that B-splines sum to 1 (partition of unity)."""
        basis = PSplineBasis(n_basis=20, degree=3)
        x = np.linspace(0, 1, 50)
        B = basis.basis_matrix(x)

        # Interior points should sum to 1
        row_sums = B[5:-5].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_penalty_matrix_shape(self):
        """Test penalty matrix has correct shape."""
        basis = PSplineBasis(n_basis=15, penalty_order=2)
        x = np.linspace(0, 1, 50)
        basis.basis_matrix(x)  # Setup
        S = basis.penalty_matrix()

        assert S.shape == (15, 15)

    def test_penalty_matrix_symmetric(self):
        """Test penalty matrix is symmetric."""
        basis = PSplineBasis(n_basis=15)
        x = np.linspace(0, 1, 50)
        basis.basis_matrix(x)
        S = basis.penalty_matrix()

        np.testing.assert_allclose(S, S.T, atol=1e-12)

    def test_penalty_matrix_positive_semidefinite(self):
        """Test penalty matrix is positive semi-definite."""
        basis = PSplineBasis(n_basis=15, penalty_order=2)
        x = np.linspace(0, 1, 50)
        basis.basis_matrix(x)
        S = basis.penalty_matrix()

        eigenvalues = np.linalg.eigvalsh(S)
        assert np.all(eigenvalues >= -1e-10)

    def test_penalty_null_space(self):
        """Test that linear functions are in null space of 2nd order penalty."""
        basis = PSplineBasis(n_basis=15, penalty_order=2)
        x = np.linspace(0, 1, 50)
        basis.basis_matrix(x)
        S = basis.penalty_matrix()

        # Linear coefficients
        beta_linear = np.arange(15, dtype=float)
        penalty = beta_linear @ S @ beta_linear

        # Should be near zero
        assert penalty < 1e-10


class TestPSplineFitting:
    """Tests for P-spline fitting."""

    @pytest.fixture
    def sine_data(self):
        """Generate noisy sine data."""
        np.random.seed(42)
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x) + np.random.normal(0, 0.2, 100)
        return x, y

    def test_fit_returns_result(self, sine_data):
        """Test that fit returns PSplineResult."""
        x, y = sine_data
        basis = PSplineBasis(n_basis=20)
        result = basis.fit(x, y, lambda_=10.0)

        assert isinstance(result, PSplineResult)

    def test_fit_coefficients_shape(self, sine_data):
        """Test fitted coefficients have correct shape."""
        x, y = sine_data
        result = fit_pspline(x, y, n_basis=20)

        assert result.coef_.shape == (20,)

    def test_fit_fitted_values_shape(self, sine_data):
        """Test fitted values have correct shape."""
        x, y = sine_data
        result = fit_pspline(x, y, n_basis=20)

        assert result.fitted_values_.shape == x.shape

    def test_fit_gcv_selection(self, sine_data):
        """Test GCV smoothing selection."""
        x, y = sine_data
        result = fit_pspline(x, y, n_basis=25, lambda_="gcv")

        assert result.lambda_ > 0
        assert result.edf_ > 0
        assert result.edf_ < 25  # Should be regularized

    def test_fit_aic_selection(self, sine_data):
        """Test AIC smoothing selection."""
        x, y = sine_data
        result = fit_pspline(x, y, n_basis=25, lambda_="aic")

        assert result.lambda_ > 0

    def test_fit_reml_selection(self, sine_data):
        """Test REML smoothing selection."""
        x, y = sine_data
        result = fit_pspline(x, y, n_basis=25, lambda_="reml")

        assert result.lambda_ > 0

    def test_fit_high_lambda_smooths(self, sine_data):
        """Test that high lambda produces smooth fit."""
        x, y = sine_data
        result_smooth = fit_pspline(x, y, n_basis=25, lambda_=1e6)
        result_wiggly = fit_pspline(x, y, n_basis=25, lambda_=0.01)

        # Smooth fit should have lower variance
        var_smooth = np.var(result_smooth.fitted_values_)
        var_wiggly = np.var(result_wiggly.fitted_values_)

        assert var_smooth < var_wiggly

    def test_fit_with_weights(self, sine_data):
        """Test fitting with observation weights."""
        x, y = sine_data
        weights = np.ones(len(y))
        weights[:10] = 0.1  # Downweight first 10 points

        result = fit_pspline(x, y, n_basis=20, weights=weights)

        assert result.fitted_values_.shape == y.shape

    def test_predict_at_new_points(self, sine_data):
        """Test prediction at new points."""
        x, y = sine_data
        result = fit_pspline(x, y, n_basis=20, lambda_=10.0)

        x_new = np.linspace(0, 2 * np.pi, 200)
        y_pred = result.predict(x_new)

        assert y_pred.shape == x_new.shape

    def test_derivative(self, sine_data):
        """Test derivative computation."""
        x, y = sine_data
        result = fit_pspline(x, y, n_basis=25, lambda_=10.0)

        deriv = result.derivative(x, order=1)

        assert deriv.shape == x.shape
        # Derivative of sine should be approximately cosine
        expected_deriv = np.cos(x)
        # Check correlation (not exact due to noise)
        corr = np.corrcoef(deriv, expected_deriv)[0, 1]
        assert corr > 0.8

    def test_confidence_band(self, sine_data):
        """Test confidence band computation."""
        x, y = sine_data
        result = fit_pspline(x, y, n_basis=20, lambda_=10.0)

        lower, upper = result.confidence_band(level=0.95)

        assert lower.shape == x.shape
        assert upper.shape == x.shape
        assert np.all(lower <= result.fitted_values_)
        assert np.all(upper >= result.fitted_values_)

    def test_summary(self, sine_data):
        """Test summary statistics."""
        x, y = sine_data
        result = fit_pspline(x, y, n_basis=20)

        summary = result.summary()

        assert "n_obs" in summary
        assert "edf" in summary
        assert "lambda" in summary
        assert "r_squared" in summary
        assert 0 <= summary["r_squared"] <= 1


class TestPSplineEdgeCases:
    """Test edge cases and special scenarios."""

    def test_small_dataset(self):
        """Test with small dataset."""
        x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        y = np.array([0.0, 0.5, 1.0, 0.5, 0.0])

        result = fit_pspline(x, y, n_basis=4, lambda_=1.0)

        assert result.fitted_values_.shape == (5,)

    def test_constant_response(self):
        """Test with constant response."""
        x = np.linspace(0, 1, 50)
        y = np.ones(50) * 5.0

        result = fit_pspline(x, y, n_basis=10, lambda_=1.0)

        # Fitted values should be close to constant
        np.testing.assert_allclose(result.fitted_values_, 5.0, atol=0.1)

    def test_linear_response(self):
        """Test with linear response."""
        x = np.linspace(0, 1, 50)
        y = 2 * x + 1

        result = fit_pspline(x, y, n_basis=10, penalty_order=2, lambda_=100.0)

        # Should fit linear trend well (allowing some boundary effects)
        # Check interior points (excluding boundary effects)
        np.testing.assert_allclose(result.fitted_values_[5:-5], y[5:-5], atol=0.15)

    def test_different_penalty_orders(self):
        """Test different penalty orders."""
        np.random.seed(42)
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x) + np.random.normal(0, 0.2, 100)

        result_order1 = fit_pspline(x, y, n_basis=20, penalty_order=1, lambda_=10.0)
        result_order2 = fit_pspline(x, y, n_basis=20, penalty_order=2, lambda_=10.0)
        result_order3 = fit_pspline(x, y, n_basis=20, penalty_order=3, lambda_=10.0)

        # All should produce reasonable fits
        for result in [result_order1, result_order2, result_order3]:
            assert result.summary()["r_squared"] > 0.5
