"""Tests for sparse matrix support in GAM fitting.

This module tests that GAM fitting with sparse B-spline matrices produces
identical or nearly identical results to dense matrix fitting, while offering
performance benefits for large problems.
"""

import numpy as np
import pytest

try:
    from scipy.sparse import issparse  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from aurora.models.gam import fit_gam


class TestSparseGAMFitting:
    """Tests for sparse GAM fitting."""

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_vs_dense_identical(self):
        """Test sparse and dense GAM fitting produce identical results."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y_true = np.sin(x) + 0.5 * np.cos(2 * x)
        y = y_true + 0.1 * np.random.randn(100)

        # Fit with dense matrices
        result_dense = fit_gam(x, y, n_basis=15, lambda_=0.1, use_sparse=False)

        # Fit with sparse matrices
        result_sparse = fit_gam(x, y, n_basis=15, lambda_=0.1, use_sparse=True)

        # Coefficients should be nearly identical
        np.testing.assert_allclose(
            result_dense.coefficients, result_sparse.coefficients, rtol=1e-10
        )

        # Fitted values should be nearly identical
        np.testing.assert_allclose(
            result_dense.fitted_values, result_sparse.fitted_values, rtol=1e-10
        )

        # Lambda should be the same
        assert result_dense.lambda_ == result_sparse.lambda_

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_with_weights(self):
        """Test sparse GAM with weighted least squares."""
        np.random.seed(123)
        x = np.linspace(0, 10, 80)
        y = 2 * x + 0.5 * np.random.randn(80)

        # Non-uniform weights
        weights = np.exp(-((x - 5) ** 2) / 10)

        # Fit with sparse
        result_sparse = fit_gam(x, y, n_basis=12, lambda_=1.0, weights=weights, use_sparse=True)

        # Fit with dense for comparison
        result_dense = fit_gam(x, y, n_basis=12, lambda_=1.0, weights=weights, use_sparse=False)

        # Should produce same results
        np.testing.assert_allclose(
            result_sparse.fitted_values, result_dense.fitted_values, rtol=1e-10
        )

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_large_problem(self):
        """Test sparse GAM on larger problem."""
        np.random.seed(456)
        x = np.linspace(0, 20, 500)  # Larger problem
        y_true = np.sin(x) * np.exp(-x / 10)
        y = y_true + 0.05 * np.random.randn(500)

        # Fit with sparse (should be faster)
        result = fit_gam(x, y, n_basis=40, lambda_=0.01, use_sparse=True)

        # Check basic properties
        assert result.coefficients.shape == (40,)
        assert result.fitted_values.shape == (500,)
        assert result.lambda_ == 0.01

        # Check fit quality
        rmse = np.sqrt(np.mean(result.residuals**2))
        assert rmse < 0.2  # Should fit reasonably well

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_prediction(self):
        """Test predictions work with sparse-fitted GAM."""
        np.random.seed(789)
        x_train = np.linspace(0, 10, 100)
        y_train = np.cos(x_train) + 0.1 * np.random.randn(100)

        # Fit with sparse
        result = fit_gam(x_train, y_train, n_basis=15, lambda_=0.5, use_sparse=True)

        # Make predictions on new data
        x_new = np.linspace(0, 10, 200)
        y_pred = result.predict(x_new)

        # Predictions should have correct shape
        assert y_pred.shape == (200,)

        # Predictions should be reasonable (close to cos(x) on average)
        expected = np.cos(x_new)
        mae = np.mean(np.abs(y_pred - expected))
        assert mae < 0.5  # Reasonable fit with noise

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_only_bspline(self):
        """Test that sparse only works with bspline basis."""
        np.random.seed(111)
        x = np.linspace(0, 10, 80)
        y = x**2 + 0.5 * np.random.randn(80)

        # Should work with bspline
        result_bspline = fit_gam(
            x, y, n_basis=12, basis_type="bspline", lambda_=0.1, use_sparse=True
        )
        assert result_bspline.fitted_values.shape == (80,)

        # Should raise error with cubic
        with pytest.raises(
            ValueError, match="use_sparse=True only supported for basis_type='bspline'"
        ):
            fit_gam(x, y, n_basis=12, basis_type="cubic", lambda_=0.1, use_sparse=True)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_different_lambda(self):
        """Test sparse GAM with different smoothing parameters."""
        np.random.seed(222)
        x = np.linspace(0, 10, 100)
        y_true = np.sin(x)
        y = y_true + 0.2 * np.random.randn(100)

        # Small lambda (less smoothing)
        result_small = fit_gam(x, y, n_basis=15, lambda_=0.001, use_sparse=True)

        # Large lambda (more smoothing)
        result_large = fit_gam(x, y, n_basis=15, lambda_=10.0, use_sparse=True)

        # Large lambda should produce smoother fit
        # Measure smoothness via second differences
        smooth_small = np.mean(np.diff(result_small.fitted_values, n=2) ** 2)
        smooth_large = np.mean(np.diff(result_large.fitted_values, n=2) ** 2)

        assert smooth_large < smooth_small

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_vs_dense_different_degrees(self):
        """Test sparse vs dense for different B-spline degrees."""
        np.random.seed(333)
        x = np.linspace(0, 5, 80)
        y = x**2 + 0.3 * np.random.randn(80)

        for degree in [1, 2, 3, 5]:
            # Dense
            result_dense = fit_gam(x, y, n_basis=12, degree=degree, lambda_=0.1, use_sparse=False)

            # Sparse
            result_sparse = fit_gam(x, y, n_basis=12, degree=degree, lambda_=0.1, use_sparse=True)

            # Should match
            np.testing.assert_allclose(
                result_dense.fitted_values,
                result_sparse.fitted_values,
                rtol=1e-9,
                err_msg=f"Mismatch at degree={degree}",
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_knot_placement(self):
        """Test sparse GAM with different knot placement methods."""
        np.random.seed(444)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * np.random.randn(100)

        # Quantile knots (default)
        result_quantile = fit_gam(
            x, y, n_basis=15, knot_method="quantile", lambda_=0.1, use_sparse=True
        )

        # Uniform knots
        result_uniform = fit_gam(
            x, y, n_basis=15, knot_method="uniform", lambda_=0.1, use_sparse=True
        )

        # Both should produce reasonable fits
        assert result_quantile.fitted_values.shape == (100,)
        assert result_uniform.fitted_values.shape == (100,)

        # Results might differ slightly but both should be reasonable
        rmse_quantile = np.sqrt(np.mean(result_quantile.residuals**2))
        rmse_uniform = np.sqrt(np.mean(result_uniform.residuals**2))

        assert rmse_quantile < 0.5
        assert rmse_uniform < 0.5

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_residuals(self):
        """Test that residuals are computed correctly with sparse."""
        np.random.seed(555)
        x = np.linspace(0, 10, 100)
        y = x + 0.2 * np.random.randn(100)

        result = fit_gam(x, y, n_basis=12, lambda_=1.0, use_sparse=True)

        # Residuals should equal y - fitted
        expected_residuals = y - result.fitted_values
        np.testing.assert_allclose(result.residuals, expected_residuals, rtol=1e-12)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_boundary_behavior(self):
        """Test sparse GAM behavior at boundaries."""
        np.random.seed(666)
        x = np.linspace(0, 10, 100)
        y = np.where(x < 5, x, 10 - x) + 0.1 * np.random.randn(100)

        result = fit_gam(x, y, n_basis=15, lambda_=0.5, use_sparse=True)

        # Check boundary predictions are reasonable
        # (B-splines should extrapolate smoothly)
        assert np.isfinite(result.fitted_values[0])
        assert np.isfinite(result.fitted_values[-1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
