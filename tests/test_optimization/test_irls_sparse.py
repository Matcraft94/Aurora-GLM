"""Tests for sparse matrix support in IRLS.

This module tests the sparse matrix support in the IRLS algorithm,
verifying correctness against dense implementations and testing
performance characteristics.

Tests cover:
1. Basic correctness - sparse vs dense give same results
2. Different sparse formats (CSR, CSC, COO)
3. GLM families (Gaussian, Poisson, Binomial)
4. High-dimensional categorical features
5. Edge cases (very sparse, nearly dense)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aurora.core.optimization.irls import irls, _is_sparse, _sparse_weighted_lstsq


# Check for scipy sparse
try:
    from scipy import sparse
    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_gaussian_data():
    """Simple Gaussian regression data for testing."""
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p)
    beta_true = np.array([1.0, -0.5, 0.3, 0.0, 0.2])
    y = X @ beta_true + np.random.randn(n) * 0.5
    return X, y, beta_true


@pytest.fixture
def sparse_categorical_data():
    """High-dimensional sparse categorical data."""
    np.random.seed(123)
    n = 500
    n_categories = 50

    # Create one-hot encoded categorical features
    categories = np.random.randint(0, n_categories, size=n)
    X_dense = np.zeros((n, n_categories))
    X_dense[np.arange(n), categories] = 1.0

    # Add some continuous predictors
    X_continuous = np.random.randn(n, 3)
    X_dense = np.hstack([X_continuous, X_dense])

    # True coefficients
    p = X_dense.shape[1]
    beta_true = np.random.randn(p) * 0.5

    # Response
    y = X_dense @ beta_true + np.random.randn(n) * 0.3

    return X_dense, y, beta_true


class MockIdentityLink:
    """Mock identity link function for Gaussian."""
    def inverse(self, eta):
        return eta

    def derivative(self, mu):
        return np.ones_like(mu)


class MockLogLink:
    """Mock log link function for Poisson."""
    def inverse(self, eta):
        return np.exp(np.clip(eta, -20, 20))

    def derivative(self, mu):
        return 1.0 / np.clip(mu, 1e-10, None)


class MockLogitLink:
    """Mock logit link function for Binomial."""
    def inverse(self, eta):
        eta = np.clip(eta, -20, 20)
        return 1.0 / (1.0 + np.exp(-eta))

    def derivative(self, mu):
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return 1.0 / (mu * (1 - mu))


# =============================================================================
# Test sparse detection
# =============================================================================

@pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse not available")
class TestSparseDetection:
    """Test sparse matrix detection."""

    def test_is_sparse_csr(self):
        """Test detection of CSR matrix."""
        X = sparse.csr_matrix(np.eye(10))
        assert _is_sparse(X) is True

    def test_is_sparse_csc(self):
        """Test detection of CSC matrix."""
        X = sparse.csc_matrix(np.eye(10))
        assert _is_sparse(X) is True

    def test_is_sparse_coo(self):
        """Test detection of COO matrix."""
        X = sparse.coo_matrix(np.eye(10))
        assert _is_sparse(X) is True

    def test_is_sparse_dense(self):
        """Test that dense arrays are not sparse."""
        X = np.eye(10)
        assert _is_sparse(X) is False

    def test_is_sparse_list(self):
        """Test that lists are not sparse."""
        X = [[1, 0], [0, 1]]
        assert _is_sparse(X) is False


# =============================================================================
# Test sparse weighted least squares
# =============================================================================

@pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse not available")
class TestSparseWeightedLstSq:
    """Test the sparse weighted least squares solver."""

    def test_unweighted_lstsq(self):
        """Test with unit weights (standard OLS)."""
        np.random.seed(42)
        n, p = 50, 5
        X_dense = np.random.randn(n, p)
        X_sparse = sparse.csr_matrix(X_dense)

        beta_true = np.array([1.0, -0.5, 0.3, 0.1, -0.2])
        y = X_dense @ beta_true + np.random.randn(n) * 0.1

        # Unit weights
        w = np.ones(n)

        # Sparse solution
        beta_sparse = _sparse_weighted_lstsq(X_sparse, w, y)

        # Dense solution (reference)
        beta_dense, _, _, _ = np.linalg.lstsq(X_dense, y, rcond=None)

        assert_allclose(beta_sparse, beta_dense, rtol=1e-10)

    def test_weighted_lstsq(self):
        """Test with non-uniform weights."""
        np.random.seed(42)
        n, p = 50, 5
        X_dense = np.random.randn(n, p)
        X_sparse = sparse.csr_matrix(X_dense)

        beta_true = np.array([1.0, -0.5, 0.3, 0.1, -0.2])
        y = X_dense @ beta_true + np.random.randn(n) * 0.3

        # Random positive weights
        w = np.abs(np.random.randn(n)) + 0.1

        # Sparse solution
        beta_sparse = _sparse_weighted_lstsq(X_sparse, w, y)

        # Dense weighted least squares (reference)
        sqrt_w = np.sqrt(w)
        X_weighted = X_dense * sqrt_w[:, None]
        y_weighted = y * sqrt_w
        beta_dense, _, _, _ = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)

        assert_allclose(beta_sparse, beta_dense, rtol=1e-8)

    def test_sparse_design_matrix(self):
        """Test with genuinely sparse design matrix."""
        np.random.seed(42)
        n, p = 100, 20

        # Create sparse matrix with ~10% non-zeros
        density = 0.1
        X_sparse = sparse.random(n, p, density=density, format='csr')
        X_dense = X_sparse.toarray()

        # Ensure non-singular
        X_sparse = X_sparse + sparse.eye(n, p) * 0.1
        X_dense = X_sparse.toarray()

        y = np.random.randn(n)
        w = np.ones(n)

        # Both should give same result
        beta_sparse = _sparse_weighted_lstsq(X_sparse, w, y)

        # Dense reference
        sqrt_w = np.sqrt(w)
        X_weighted = X_dense * sqrt_w[:, None]
        y_weighted = y * sqrt_w
        beta_dense, _, _, _ = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)

        assert_allclose(beta_sparse, beta_dense, rtol=1e-8)


# =============================================================================
# Test IRLS with sparse matrices
# =============================================================================

@pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse not available")
class TestIRLSSparse:
    """Test IRLS algorithm with sparse design matrices."""

    def test_sparse_vs_dense_gaussian(self, simple_gaussian_data):
        """Sparse and dense IRLS should give same results for Gaussian."""
        X_dense, y, beta_true = simple_gaussian_data
        X_sparse = sparse.csr_matrix(X_dense)

        link = MockIdentityLink()
        variance_fn = lambda mu: np.ones_like(mu)
        loss_fn = lambda beta: 0.5 * np.sum((y - X_dense @ beta) ** 2)

        init_params = np.zeros(X_dense.shape[1])

        # Dense reference: standard weighted least squares
        # For Gaussian with identity link and unit variance, this is OLS
        beta_dense, _, _, _ = np.linalg.lstsq(X_dense, y, rcond=None)

        # Sparse IRLS
        result_sparse = irls(
            loss_fn=loss_fn,
            init_params=init_params,
            design_matrix=X_sparse,
            response=y,
            link=link,
            variance_fn=variance_fn,
            max_iter=50,
            tol=1e-8,
        )

        assert result_sparse.success
        assert_allclose(result_sparse.x, beta_dense, rtol=1e-6)

    def test_sparse_poisson(self):
        """Test sparse IRLS for Poisson regression."""
        np.random.seed(42)
        n, p = 100, 5
        X_dense = np.random.randn(n, p) * 0.5
        X_dense[:, 0] = 1.0  # Intercept
        X_sparse = sparse.csr_matrix(X_dense)

        beta_true = np.array([0.5, 0.3, -0.2, 0.1, 0.0])
        mu_true = np.exp(X_dense @ beta_true)
        y = np.random.poisson(mu_true).astype(float)

        link = MockLogLink()
        variance_fn = lambda mu: np.clip(mu, 1e-10, None)

        def loss_fn(beta):
            mu = np.exp(np.clip(X_dense @ beta, -20, 20))
            # Poisson deviance (avoid log(0))
            y_safe = np.clip(y, 1e-10, None)
            return 2 * np.sum(y * np.log(y_safe / mu) - (y - mu))

        init_params = np.zeros(p)
        init_params[0] = np.log(np.mean(y) + 0.1)

        # Sparse IRLS
        result_sparse = irls(
            loss_fn=loss_fn,
            init_params=init_params,
            design_matrix=X_sparse,
            response=y,
            link=link,
            variance_fn=variance_fn,
            max_iter=50,
            tol=1e-8,
        )

        assert result_sparse.success
        # Check that coefficients are reasonably close to true values
        # With random Poisson data and n=100, we expect estimation error
        assert_allclose(result_sparse.x, beta_true, atol=0.5)

    def test_high_dimensional_categorical(self, sparse_categorical_data):
        """Test with high-dimensional one-hot encoded categorical features."""
        X_dense, y, beta_true = sparse_categorical_data
        X_sparse = sparse.csr_matrix(X_dense)

        # Verify sparsity
        sparsity = 1.0 - (X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]))
        assert sparsity > 0.9, f"Expected >90% sparsity, got {sparsity*100:.1f}%"

        link = MockIdentityLink()
        variance_fn = lambda mu: np.ones_like(mu)
        loss_fn = lambda beta: 0.5 * np.sum((y - X_dense @ beta) ** 2)

        init_params = np.zeros(X_dense.shape[1])

        # Sparse IRLS
        result_sparse = irls(
            loss_fn=loss_fn,
            init_params=init_params,
            design_matrix=X_sparse,
            response=y,
            link=link,
            variance_fn=variance_fn,
            max_iter=50,
            tol=1e-8,
        )

        assert result_sparse.success

        # Check that solution is close to true coefficients
        # (with some tolerance due to noise)
        assert_allclose(result_sparse.x, beta_true, atol=0.3)

    def test_different_sparse_formats(self, simple_gaussian_data):
        """Test that different sparse formats give same results."""
        X_dense, y, beta_true = simple_gaussian_data

        link = MockIdentityLink()
        variance_fn = lambda mu: np.ones_like(mu)
        loss_fn = lambda beta: 0.5 * np.sum((y - X_dense @ beta) ** 2)

        init_params = np.zeros(X_dense.shape[1])

        formats = ['csr', 'csc', 'coo']
        results = {}

        for fmt in formats:
            if fmt == 'csr':
                X_sparse = sparse.csr_matrix(X_dense)
            elif fmt == 'csc':
                X_sparse = sparse.csc_matrix(X_dense)
            else:
                X_sparse = sparse.coo_matrix(X_dense)

            result = irls(
                loss_fn=loss_fn,
                init_params=init_params,
                design_matrix=X_sparse,
                response=y,
                link=link,
                variance_fn=variance_fn,
                max_iter=50,
                tol=1e-8,
            )
            results[fmt] = result.x

        # All formats should give same result
        assert_allclose(results['csr'], results['csc'], rtol=1e-10)
        assert_allclose(results['csr'], results['coo'], rtol=1e-10)

    def test_sparse_with_offset(self, simple_gaussian_data):
        """Test sparse IRLS with offset term."""
        X_dense, y, beta_true = simple_gaussian_data
        X_sparse = sparse.csr_matrix(X_dense)

        # Create offset
        np.random.seed(99)  # Separate seed for offset
        offset = np.random.randn(len(y)) * 0.5

        link = MockIdentityLink()
        variance_fn = lambda mu: np.ones_like(mu)

        # Loss function that accounts for offset
        def loss_fn_with_offset(beta):
            return 0.5 * np.sum((y - X_dense @ beta - offset) ** 2)

        init_params = np.zeros(X_dense.shape[1])

        # Sparse IRLS with offset
        result = irls(
            loss_fn=loss_fn_with_offset,
            init_params=init_params,
            design_matrix=X_sparse,
            response=y,
            link=link,
            variance_fn=variance_fn,
            offset=offset,
            max_iter=50,
            tol=1e-8,
        )

        assert result.success

        # Reference: OLS solution for y - offset = X @ beta
        # This is equivalent to fitting y ~ X + offset
        y_adj = y - offset
        beta_ref, _, _, _ = np.linalg.lstsq(X_dense, y_adj, rcond=None)

        assert_allclose(result.x, beta_ref, rtol=1e-5)

    def test_sparse_convergence_callback(self, simple_gaussian_data):
        """Test that callback works with sparse IRLS."""
        X_dense, y, beta_true = simple_gaussian_data
        X_sparse = sparse.csr_matrix(X_dense)

        link = MockIdentityLink()
        variance_fn = lambda mu: np.ones_like(mu)
        loss_fn = lambda beta: 0.5 * np.sum((y - X_dense @ beta) ** 2)

        init_params = np.zeros(X_dense.shape[1])

        # Track iterations
        iterations = []
        losses = []

        def callback(it, params, loss):
            iterations.append(it)
            losses.append(loss)

        result = irls(
            loss_fn=loss_fn,
            init_params=init_params,
            design_matrix=X_sparse,
            response=y,
            link=link,
            variance_fn=variance_fn,
            max_iter=50,
            tol=1e-8,
            callback=callback,
        )

        assert result.success
        assert len(iterations) == result.nit
        # Loss should decrease (at least not increase much)
        assert losses[-1] <= losses[0] + 1e-6


# =============================================================================
# Test edge cases
# =============================================================================

@pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse not available")
class TestSparseEdgeCases:
    """Test edge cases for sparse IRLS."""

    def test_very_sparse_matrix(self):
        """Test with extremely sparse matrix (>99% zeros)."""
        np.random.seed(42)
        n, p = 1000, 100

        # Very sparse matrix (~1% non-zeros)
        density = 0.01
        X_sparse = sparse.random(n, p, density=density, format='csr')

        # Ensure some structure
        X_sparse = X_sparse + sparse.eye(n, p) * 0.01
        X_dense = X_sparse.toarray()

        # Simple response
        beta_true = np.random.randn(p) * 0.1
        y = X_dense @ beta_true + np.random.randn(n) * 0.5

        link = MockIdentityLink()
        variance_fn = lambda mu: np.ones_like(mu)
        loss_fn = lambda beta: 0.5 * np.sum((y - X_dense @ beta) ** 2)

        init_params = np.zeros(p)

        result = irls(
            loss_fn=loss_fn,
            init_params=init_params,
            design_matrix=X_sparse,
            response=y,
            link=link,
            variance_fn=variance_fn,
            max_iter=50,
            tol=1e-6,
        )

        assert result.success

    def test_small_problem(self):
        """Test sparse IRLS on very small problem."""
        np.random.seed(42)
        n, p = 10, 3
        X_dense = np.random.randn(n, p)
        X_sparse = sparse.csr_matrix(X_dense)

        y = np.random.randn(n)

        link = MockIdentityLink()
        variance_fn = lambda mu: np.ones_like(mu)
        loss_fn = lambda beta: 0.5 * np.sum((y - X_dense @ beta) ** 2)

        init_params = np.zeros(p)

        result = irls(
            loss_fn=loss_fn,
            init_params=init_params,
            design_matrix=X_sparse,
            response=y,
            link=link,
            variance_fn=variance_fn,
            max_iter=50,
            tol=1e-8,
        )

        assert result.success

    def test_max_iterations(self):
        """Test that max_iter limit is respected."""
        np.random.seed(42)
        n, p = 50, 5
        X_dense = np.random.randn(n, p)
        X_sparse = sparse.csr_matrix(X_dense)

        y = np.random.randn(n)

        link = MockIdentityLink()
        variance_fn = lambda mu: np.ones_like(mu)
        loss_fn = lambda beta: 0.5 * np.sum((y - X_dense @ beta) ** 2)

        init_params = np.zeros(p)

        # Very tight tolerance - won't converge in 2 iterations
        result = irls(
            loss_fn=loss_fn,
            init_params=init_params,
            design_matrix=X_sparse,
            response=y,
            link=link,
            variance_fn=variance_fn,
            max_iter=2,
            tol=1e-15,
        )

        assert result.nit <= 2
        # May or may not converge with only 2 iterations

    def test_validation_errors(self):
        """Test that proper errors are raised for invalid inputs."""
        np.random.seed(42)
        X_sparse = sparse.csr_matrix(np.eye(10))
        y = np.random.randn(10)

        link = MockIdentityLink()
        variance_fn = lambda mu: np.ones_like(mu)
        loss_fn = lambda beta: 0.0

        # Missing required arguments
        with pytest.raises(ValueError, match="IRLS requires"):
            irls(
                loss_fn=loss_fn,
                init_params=np.zeros(10),
                design_matrix=X_sparse,
                response=y,
                link=None,  # Missing link
                variance_fn=variance_fn,
            )

        # Invalid link (no methods)
        class BadLink:
            pass

        with pytest.raises(TypeError, match="link must expose"):
            irls(
                loss_fn=loss_fn,
                init_params=np.zeros(10),
                design_matrix=X_sparse,
                response=y,
                link=BadLink(),
                variance_fn=variance_fn,
            )


# =============================================================================
# Test numerical stability
# =============================================================================

@pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse not available")
class TestSparseNumericalStability:
    """Test numerical stability of sparse IRLS."""

    def test_extreme_weights(self):
        """Test with extreme IRLS weights."""
        np.random.seed(42)
        n, p = 100, 5
        X_dense = np.random.randn(n, p)
        X_sparse = sparse.csr_matrix(X_dense)

        # Poisson-like response with some zeros
        beta_true = np.array([0.0, 0.3, -0.2, 0.1, 0.0])
        mu_true = np.exp(X_dense @ beta_true)
        y = np.random.poisson(mu_true)

        link = MockLogLink()
        variance_fn = lambda mu: np.clip(mu, 1e-10, None)
        loss_fn = lambda beta: np.sum((y - np.exp(X_dense @ beta)) ** 2)

        init_params = np.zeros(p)

        result = irls(
            loss_fn=loss_fn,
            init_params=init_params,
            design_matrix=X_sparse,
            response=y,
            link=link,
            variance_fn=variance_fn,
            max_iter=50,
            tol=1e-6,
        )

        # Should complete without NaN/Inf
        assert np.all(np.isfinite(result.x))

    def test_ill_conditioned_design(self):
        """Test with ill-conditioned design matrix."""
        np.random.seed(42)
        n, p = 100, 10

        # Create correlated columns (ill-conditioned)
        X_base = np.random.randn(n, 5)
        noise = np.random.randn(n, 5) * 0.01
        X_dense = np.hstack([X_base, X_base + noise])
        X_sparse = sparse.csr_matrix(X_dense)

        y = np.random.randn(n)

        link = MockIdentityLink()
        variance_fn = lambda mu: np.ones_like(mu)
        loss_fn = lambda beta: 0.5 * np.sum((y - X_dense @ beta) ** 2)

        init_params = np.zeros(p)

        # Should still complete (may use lstsq fallback)
        result = irls(
            loss_fn=loss_fn,
            init_params=init_params,
            design_matrix=X_sparse,
            response=y,
            link=link,
            variance_fn=variance_fn,
            max_iter=50,
            tol=1e-6,
        )

        # Should complete without NaN/Inf
        assert np.all(np.isfinite(result.x))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
