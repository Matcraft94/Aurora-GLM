"""Tests for data-parallel IRLS."""

import numpy as np
import pytest

from aurora.models.distributed import (
    DataParallelIRLS,
    ParallelResult,
    fit_glm_parallel,
)


@pytest.fixture
def gaussian_data():
    """Generate Gaussian regression data."""
    np.random.seed(42)
    n = 500
    p = 3

    X = np.random.randn(n, p)
    beta_true = np.array([1.0, -0.5, 0.3])
    sigma = 0.5

    y = X @ beta_true + np.random.normal(0, sigma, n)

    return X, y, beta_true


@pytest.fixture
def poisson_data():
    """Generate Poisson regression data."""
    np.random.seed(42)
    n = 500
    p = 2

    X = np.random.randn(n, p)
    beta_true = np.array([0.5, -0.3])

    mu = np.exp(X @ beta_true)
    y = np.random.poisson(mu)

    return X, y, beta_true


class TestFitGLMParallelGaussian:
    """Tests for data-parallel Gaussian regression."""

    def test_returns_result(self, gaussian_data):
        """Test that fit returns ParallelResult."""
        X, y, _ = gaussian_data
        X_chunks = np.array_split(X, 5)
        y_chunks = np.array_split(y, 5)

        result = fit_glm_parallel(X_chunks, y_chunks, family="gaussian")

        assert isinstance(result, ParallelResult)

    def test_coef_shape(self, gaussian_data):
        """Test coefficient shape."""
        X, y, _ = gaussian_data
        X_chunks = np.array_split(X, 5)
        y_chunks = np.array_split(y, 5)

        result = fit_glm_parallel(X_chunks, y_chunks, family="gaussian")

        assert result.coef_.shape == (X.shape[1],)

    def test_recovers_parameters(self, gaussian_data):
        """Test that parallel IRLS recovers true parameters."""
        X, y, beta_true = gaussian_data
        X_chunks = np.array_split(X, 10)
        y_chunks = np.array_split(y, 10)

        result = fit_glm_parallel(X_chunks, y_chunks, family="gaussian")

        # Should be very close to true parameters (IRLS is exact for Gaussian)
        np.testing.assert_allclose(result.coef_, beta_true, atol=0.1)

    def test_converges(self, gaussian_data):
        """Test that IRLS converges."""
        X, y, _ = gaussian_data
        X_chunks = np.array_split(X, 5)
        y_chunks = np.array_split(y, 5)

        result = fit_glm_parallel(X_chunks, y_chunks, family="gaussian")

        assert result.converged_

    def test_matches_single_machine(self, gaussian_data):
        """Test that parallel gives same result as single-machine."""
        X, y, _ = gaussian_data

        # Single machine (one chunk)
        result_single = fit_glm_parallel([X], [y], family="gaussian")

        # Multiple chunks
        X_chunks = np.array_split(X, 10)
        y_chunks = np.array_split(y, 10)
        result_parallel = fit_glm_parallel(X_chunks, y_chunks, family="gaussian")

        # Should give identical results (IRLS is deterministic)
        np.testing.assert_allclose(result_parallel.coef_, result_single.coef_, rtol=1e-6)


class TestFitGLMParallelPoisson:
    """Tests for data-parallel Poisson regression."""

    def test_poisson_fit(self, poisson_data):
        """Test Poisson regression fitting."""
        X, y, _ = poisson_data
        X_chunks = np.array_split(X, 5)
        y_chunks = np.array_split(y, 5)

        result = fit_glm_parallel(X_chunks, y_chunks, family="poisson")

        assert result.family == "poisson"
        assert result.link == "log"

    def test_poisson_recovers_parameters(self, poisson_data):
        """Test Poisson parameter recovery."""
        X, y, beta_true = poisson_data
        X_chunks = np.array_split(X, 5)
        y_chunks = np.array_split(y, 5)

        result = fit_glm_parallel(X_chunks, y_chunks, family="poisson")

        # Should be reasonably close to true parameters
        np.testing.assert_allclose(result.coef_, beta_true, atol=0.2)


class TestDataParallelIRLS:
    """Tests for DataParallelIRLS class."""

    def test_init(self):
        """Test initialization."""
        irls = DataParallelIRLS(family="gaussian")

        assert irls.family == "gaussian"
        assert irls.link == "identity"

    def test_custom_link(self):
        """Test custom link function."""
        irls = DataParallelIRLS(family="poisson", link="identity")

        assert irls.link == "identity"

    def test_sufficient_stats_shape(self, gaussian_data):
        """Test sufficient statistics have correct shape."""
        X, y, _ = gaussian_data
        irls = DataParallelIRLS(family="gaussian")

        p = X.shape[1]
        beta = np.zeros(p)

        XtWX, XtWz = irls._compute_sufficient_stats(X, y, beta)

        assert XtWX.shape == (p, p)
        assert XtWz.shape == (p,)


class TestParallelResult:
    """Tests for ParallelResult methods."""

    def test_predict(self, gaussian_data):
        """Test prediction method."""
        X, y, _ = gaussian_data
        X_chunks = np.array_split(X, 5)
        y_chunks = np.array_split(y, 5)

        result = fit_glm_parallel(X_chunks, y_chunks, family="gaussian")

        y_pred = result.predict(X)
        assert y_pred.shape == y.shape

    def test_summary(self, gaussian_data):
        """Test summary method."""
        X, y, _ = gaussian_data
        X_chunks = np.array_split(X, 5)
        y_chunks = np.array_split(y, 5)

        result = fit_glm_parallel(X_chunks, y_chunks, family="gaussian")

        summary = result.summary()

        assert "n_obs" in summary
        assert "n_chunks" in summary
        assert "converged" in summary
        assert "n_iter" in summary


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_chunk(self, gaussian_data):
        """Test with single data chunk."""
        X, y, _ = gaussian_data

        result = fit_glm_parallel([X], [y], family="gaussian")

        assert result.n_chunks_ == 1
        assert result.converged_

    def test_many_small_chunks(self, gaussian_data):
        """Test with many small chunks."""
        X, y, _ = gaussian_data
        X_chunks = np.array_split(X, 50)
        y_chunks = np.array_split(y, 50)

        result = fit_glm_parallel(X_chunks, y_chunks, family="gaussian")

        assert result.n_chunks_ == 50
        assert result.converged_

    def test_unequal_chunk_sizes(self, gaussian_data):
        """Test with unequal chunk sizes."""
        X, y, _ = gaussian_data

        # Create chunks of different sizes
        X_chunks = [X[:100], X[100:350], X[350:]]
        y_chunks = [y[:100], y[100:350], y[350:]]

        result = fit_glm_parallel(X_chunks, y_chunks, family="gaussian")

        assert result.converged_
        assert result.n_obs_ == len(y)
