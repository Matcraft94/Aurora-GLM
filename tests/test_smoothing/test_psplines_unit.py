"""Unit tests for P-Spline components.

These tests verify individual components of the P-spline implementation
in isolation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aurora.smoothing.splines import PSplineBasis, fit_pspline


class TestPSplineBasisConstruction:
    """Unit tests for PSplineBasis construction."""

    def test_default_parameters(self):
        """Test default parameter values."""
        basis = PSplineBasis()
        assert basis.n_basis == 20
        assert basis.degree == 3
        assert basis.penalty_order == 2

    def test_custom_parameters(self):
        """Test custom parameter values."""
        basis = PSplineBasis(n_basis=30, degree=2, penalty_order=1)
        assert basis.n_basis == 30
        assert basis.degree == 2
        assert basis.penalty_order == 1

    def test_invalid_n_basis_raises(self):
        """Test that n_basis < 4 raises ValueError."""
        with pytest.raises(ValueError, match="n_basis must be at least 4"):
            PSplineBasis(n_basis=3)

    def test_invalid_penalty_order_raises(self):
        """Test that penalty_order >= n_basis raises ValueError."""
        with pytest.raises(ValueError, match="penalty_order.*must be less than n_basis"):
            PSplineBasis(n_basis=10, penalty_order=10)

    def test_penalty_order_equal_n_basis_raises(self):
        """Test that penalty_order == n_basis raises ValueError."""
        with pytest.raises(ValueError):
            PSplineBasis(n_basis=5, penalty_order=5)


class TestBasisMatrix:
    """Unit tests for basis matrix computation."""

    @pytest.fixture
    def basis(self):
        return PSplineBasis(n_basis=15, degree=3)

    @pytest.fixture
    def x(self):
        return np.linspace(0, 1, 100)

    def test_basis_matrix_shape(self, basis, x):
        """Test basis matrix dimensions."""
        B = basis.basis_matrix(x)
        assert B.shape == (100, 15)

    def test_basis_matrix_non_negative(self, basis, x):
        """Test that B-spline basis values are non-negative."""
        B = basis.basis_matrix(x)
        assert np.all(B >= -1e-10)  # Allow small numerical errors

    def test_basis_matrix_partition_of_unity(self, basis, x):
        """Test that B-splines sum to 1 (partition of unity) for interior points."""
        B = basis.basis_matrix(x)
        # Interior points should sum to approximately 1
        row_sums = B[10:-10].sum(axis=1)
        assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_basis_matrix_local_support(self, basis, x):
        """Test that each B-spline has local support."""
        B = basis.basis_matrix(x)
        # Each column should have many zeros (local support)
        sparsity = (B < 1e-10).sum() / B.size
        assert sparsity > 0.5  # More than half should be zero

    def test_basis_matrix_reproducible(self, basis, x):
        """Test that basis matrix is reproducible."""
        B1 = basis.basis_matrix(x)
        B2 = basis.basis_matrix(x)
        assert_allclose(B1, B2)

    def test_basis_matrix_different_domains(self):
        """Test basis matrix on different domains."""
        basis = PSplineBasis(n_basis=10)

        x1 = np.linspace(0, 1, 50)
        x2 = np.linspace(-5, 5, 50)
        x3 = np.linspace(100, 200, 50)

        for x in [x1, x2, x3]:
            B = basis.basis_matrix(x)
            assert B.shape == (50, 10)
            assert np.all(np.isfinite(B))


class TestPenaltyMatrix:
    """Unit tests for penalty matrix computation."""

    @pytest.fixture
    def basis(self):
        basis = PSplineBasis(n_basis=15, penalty_order=2)
        x = np.linspace(0, 1, 50)
        basis.basis_matrix(x)  # Setup required
        return basis

    def test_penalty_matrix_shape(self, basis):
        """Test penalty matrix dimensions."""
        S = basis.penalty_matrix()
        assert S.shape == (15, 15)

    def test_penalty_matrix_symmetric(self, basis):
        """Test penalty matrix is symmetric."""
        S = basis.penalty_matrix()
        assert_allclose(S, S.T, atol=1e-12)

    def test_penalty_matrix_positive_semidefinite(self, basis):
        """Test penalty matrix is positive semi-definite."""
        S = basis.penalty_matrix()
        eigenvalues = np.linalg.eigvalsh(S)
        assert np.all(eigenvalues >= -1e-10)

    def test_penalty_matrix_null_space_order2(self):
        """Test that linear functions are in null space of 2nd order penalty."""
        basis = PSplineBasis(n_basis=15, penalty_order=2)
        x = np.linspace(0, 1, 50)
        basis.basis_matrix(x)
        S = basis.penalty_matrix()

        # Linear coefficients should give zero penalty
        beta_linear = np.arange(15, dtype=float)
        penalty = beta_linear @ S @ beta_linear
        assert penalty < 1e-10

    def test_penalty_matrix_null_space_order1(self):
        """Test that constants are in null space of 1st order penalty."""
        basis = PSplineBasis(n_basis=15, penalty_order=1)
        x = np.linspace(0, 1, 50)
        basis.basis_matrix(x)
        S = basis.penalty_matrix()

        # Constant coefficients should give zero penalty
        beta_const = np.ones(15)
        penalty = beta_const @ S @ beta_const
        assert penalty < 1e-10


class TestSmoothingParameterSelection:
    """Unit tests for smoothing parameter selection."""

    @pytest.fixture
    def noisy_data(self):
        np.random.seed(42)
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x) + np.random.normal(0, 0.2, 100)
        return x, y

    def test_gcv_returns_positive_lambda(self, noisy_data):
        """Test that GCV returns positive lambda."""
        x, y = noisy_data
        result = fit_pspline(x, y, n_basis=20, lambda_="gcv")
        assert result.lambda_ > 0

    def test_aic_returns_positive_lambda(self, noisy_data):
        """Test that AIC returns positive lambda."""
        x, y = noisy_data
        result = fit_pspline(x, y, n_basis=20, lambda_="aic")
        assert result.lambda_ > 0

    def test_reml_returns_positive_lambda(self, noisy_data):
        """Test that REML returns positive lambda."""
        x, y = noisy_data
        result = fit_pspline(x, y, n_basis=20, lambda_="reml")
        assert result.lambda_ > 0

    def test_fixed_lambda_used(self, noisy_data):
        """Test that fixed lambda is used when specified."""
        x, y = noisy_data
        result = fit_pspline(x, y, n_basis=20, lambda_=100.0)
        assert result.lambda_ == 100.0


class TestEffectiveDegreesOfFreedom:
    """Unit tests for effective degrees of freedom."""

    @pytest.fixture
    def noisy_data(self):
        np.random.seed(42)
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x) + np.random.normal(0, 0.2, 100)
        return x, y

    def test_edf_bounded(self, noisy_data):
        """Test that EDF is bounded by n_basis."""
        x, y = noisy_data
        result = fit_pspline(x, y, n_basis=25, lambda_=1.0)
        assert 0 < result.edf_ < 25

    def test_high_lambda_low_edf(self, noisy_data):
        """Test that high lambda gives low EDF."""
        x, y = noisy_data
        result_high = fit_pspline(x, y, n_basis=25, lambda_=1e6)
        result_low = fit_pspline(x, y, n_basis=25, lambda_=0.01)
        assert result_high.edf_ < result_low.edf_

    def test_edf_positive(self, noisy_data):
        """Test that EDF is positive."""
        x, y = noisy_data
        result = fit_pspline(x, y, n_basis=20, lambda_=10.0)
        assert result.edf_ > 0


class TestPSplineResultMethods:
    """Unit tests for PSplineResult methods."""

    @pytest.fixture
    def fitted_result(self):
        np.random.seed(42)
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x) + np.random.normal(0, 0.2, 100)
        return fit_pspline(x, y, n_basis=20, lambda_=10.0)

    def test_predict_at_training_points(self, fitted_result):
        """Test prediction at training points."""
        x = np.linspace(0, 2 * np.pi, 100)
        y_pred = fitted_result.predict(x)
        assert y_pred.shape == (100,)

    def test_predict_at_new_points(self, fitted_result):
        """Test prediction at new points."""
        x_new = np.linspace(0, 2 * np.pi, 200)
        y_pred = fitted_result.predict(x_new)
        assert y_pred.shape == (200,)

    def test_derivative_shape(self, fitted_result):
        """Test derivative computation shape."""
        x = np.linspace(0, 2 * np.pi, 100)
        deriv = fitted_result.derivative(x, order=1)
        assert deriv.shape == (100,)

    def test_derivative_of_sine_approximates_cosine(self, fitted_result):
        """Test that derivative of fitted sine approximates cosine."""
        x = np.linspace(0.5, 2 * np.pi - 0.5, 80)  # Avoid boundaries
        deriv = fitted_result.derivative(x, order=1)
        expected = np.cos(x)
        corr = np.corrcoef(deriv, expected)[0, 1]
        assert corr > 0.8

    def test_confidence_band_contains_fit(self, fitted_result):
        """Test that confidence band contains fitted values."""
        lower, upper = fitted_result.confidence_band(level=0.95)
        assert np.all(lower <= fitted_result.fitted_values_)
        assert np.all(upper >= fitted_result.fitted_values_)

    def test_confidence_band_ordering(self, fitted_result):
        """Test that lower bound <= upper bound."""
        lower, upper = fitted_result.confidence_band(level=0.95)
        assert np.all(lower <= upper)

    def test_summary_contains_required_keys(self, fitted_result):
        """Test that summary contains required keys."""
        summary = fitted_result.summary()
        required_keys = ["n_obs", "edf", "lambda", "r_squared"]
        for key in required_keys:
            assert key in summary

    def test_r_squared_bounded(self, fitted_result):
        """Test that R-squared is in [0, 1]."""
        summary = fitted_result.summary()
        assert 0 <= summary["r_squared"] <= 1
