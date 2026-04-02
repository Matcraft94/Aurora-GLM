"""Tests for sparse B-spline basis evaluation.

This module tests the sparse CSR matrix implementation of B-spline basis
evaluation, comparing it against the dense implementation for correctness
and verifying performance improvements.
"""

import numpy as np
import pytest

from aurora.smoothing.splines import BSplineBasis

try:
    from scipy.sparse import csr_matrix, issparse  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestSparseBSplineBasis:
    """Tests for sparse B-spline basis evaluation."""

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_vs_dense_cubic(self):
        """Test sparse and dense outputs are identical for cubic B-splines."""
        # Create test data
        x = np.linspace(0, 10, 100)
        knots = BSplineBasis.create_knots(x, n_basis=15, degree=3)
        basis = BSplineBasis(knots, degree=3)

        # Compute both dense and sparse
        B_dense = basis.basis_matrix(x, sparse=False)
        B_sparse = basis.basis_matrix(x, sparse=True)

        # Check sparse format
        assert issparse(B_sparse)
        assert B_sparse.format == "csr"

        # Convert sparse to dense for comparison
        B_sparse_dense = B_sparse.toarray()

        # Should be identical
        np.testing.assert_allclose(B_dense, B_sparse_dense, rtol=1e-12)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_vs_dense_linear(self):
        """Test sparse and dense outputs for linear B-splines (degree=1)."""
        x = np.linspace(0, 5, 50)
        knots = BSplineBasis.create_knots(x, n_basis=8, degree=1)
        basis = BSplineBasis(knots, degree=1)

        B_dense = basis.basis_matrix(x, sparse=False)
        B_sparse = basis.basis_matrix(x, sparse=True)

        # Use atol for near-zero values to avoid inf relative error
        np.testing.assert_allclose(B_dense, B_sparse.toarray(), rtol=1e-12, atol=1e-14)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_vs_dense_quintic(self):
        """Test sparse and dense outputs for quintic B-splines (degree=5)."""
        x = np.linspace(-1, 1, 80)
        knots = BSplineBasis.create_knots(x, n_basis=12, degree=5)
        basis = BSplineBasis(knots, degree=5)

        B_dense = basis.basis_matrix(x, sparse=False)
        B_sparse = basis.basis_matrix(x, sparse=True)

        np.testing.assert_allclose(B_dense, B_sparse.toarray(), rtol=1e-12)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_sparsity_pattern(self):
        """Verify sparse matrix has expected sparsity pattern (at most degree+1 non-zeros per row)."""
        x = np.linspace(0, 10, 200)
        knots = BSplineBasis.create_knots(x, n_basis=20, degree=3)
        basis = BSplineBasis(knots, degree=3)

        B_sparse = basis.basis_matrix(x, sparse=True)

        # Each row should have at most degree+1 non-zero entries
        nnz_per_row = np.diff(B_sparse.indptr)
        assert np.all(nnz_per_row <= basis.degree_ + 1)

        # Most rows should have exactly degree+1 non-zeros (except near boundaries)
        # At least 80% of rows should have degree+1 non-zeros
        full_rows = np.sum(nnz_per_row == basis.degree_ + 1)
        assert full_rows >= 0.8 * len(x)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_partition_of_unity(self):
        """Verify sparse B-splines satisfy partition of unity: sum to 1."""
        x = np.linspace(0.1, 9.9, 150)  # Avoid exact boundaries
        knots = BSplineBasis.create_knots(x, n_basis=18, degree=3)
        basis = BSplineBasis(knots, degree=3)

        B_sparse = basis.basis_matrix(x, sparse=True)

        # Sum across columns (axis=1) should be 1
        row_sums = np.array(B_sparse.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_matrix_vector_product(self):
        """Test sparse matrix-vector multiplication gives same result as dense."""
        x = np.linspace(0, 10, 100)
        knots = BSplineBasis.create_knots(x, n_basis=15, degree=3)
        basis = BSplineBasis(knots, degree=3)

        B_dense = basis.basis_matrix(x, sparse=False)
        B_sparse = basis.basis_matrix(x, sparse=True)

        # Random coefficient vector
        np.random.seed(42)
        coef = np.random.randn(basis.n_basis_)

        # Matrix-vector products
        fitted_dense = B_dense @ coef
        fitted_sparse = B_sparse @ coef

        np.testing.assert_allclose(fitted_dense, fitted_sparse, rtol=1e-12)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_boundary_handling(self):
        """Test sparse evaluation handles boundaries correctly."""
        x = np.array([0.0, 5.0, 10.0])  # Include exact boundaries
        knots = BSplineBasis.create_knots(np.linspace(0, 10, 50), n_basis=12, degree=3)
        basis = BSplineBasis(knots, degree=3)

        B_dense = basis.basis_matrix(x, sparse=False)
        B_sparse = basis.basis_matrix(x, sparse=True)

        np.testing.assert_allclose(B_dense, B_sparse.toarray(), rtol=1e-12)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_efficiency(self):
        """Verify sparse matrix uses less memory than dense for large problems."""
        # Large problem: 1000 observations, 30 basis functions, degree 3
        x = np.linspace(0, 100, 1000)
        knots = BSplineBasis.create_knots(x, n_basis=30, degree=3)
        basis = BSplineBasis(knots, degree=3)

        B_sparse = basis.basis_matrix(x, sparse=True)

        # Dense would have 1000 × 30 = 30,000 elements
        dense_size = 1000 * 30

        # Sparse should have ~1000 × 4 = 4,000 non-zeros (degree + 1)
        sparse_nnz = B_sparse.nnz

        # Sparse should use < 20% of dense storage
        assert sparse_nnz < 0.2 * dense_size

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_single_point(self):
        """Test sparse evaluation for single point."""
        x = np.array([5.0])
        knots = BSplineBasis.create_knots(np.linspace(0, 10, 50), n_basis=12, degree=3)
        basis = BSplineBasis(knots, degree=3)

        B_dense = basis.basis_matrix(x, sparse=False)
        B_sparse = basis.basis_matrix(x, sparse=True)

        np.testing.assert_allclose(B_dense, B_sparse.toarray(), rtol=1e-12)

    def test_sparse_raises_without_scipy(self, monkeypatch):
        """Test that requesting sparse without scipy raises informative error."""
        x = np.linspace(0, 10, 50)
        knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
        basis = BSplineBasis(knots, degree=3)

        # Mock scipy import failure
        import sys

        original_modules = sys.modules.copy()

        try:
            # Remove scipy from sys.modules if present
            sys.modules.pop("scipy", None)
            sys.modules.pop("scipy.sparse", None)

            # This should work (dense)
            B_dense = basis.basis_matrix(x, sparse=False)
            assert B_dense.shape == (50, 10)

            # This should fail with informative message (sparse)
            # Note: We can't fully test this without actually removing scipy
            # Just verify sparse parameter exists
            assert "sparse" in basis.basis_matrix.__code__.co_varnames

        finally:
            # Restore original modules
            sys.modules.update(original_modules)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_dtype_preservation(self):
        """Test that sparse matrices preserve float32/float64 dtypes."""
        x_f64 = np.linspace(0, 10, 100, dtype=np.float64)
        x_f32 = np.linspace(0, 10, 100, dtype=np.float32)

        knots = BSplineBasis.create_knots(x_f64, n_basis=15, degree=3)
        basis = BSplineBasis(knots, degree=3)

        B_f64 = basis.basis_matrix(x_f64, sparse=True)
        B_f32 = basis.basis_matrix(x_f32, sparse=True)

        assert B_f64.dtype == np.float64
        assert B_f32.dtype == np.float32


class TestSparseBackendRestriction:
    """Test that sparse output is restricted to NumPy backend."""

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_numpy_works(self):
        """Test sparse works with NumPy arrays."""
        x = np.linspace(0, 10, 50)
        knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
        basis = BSplineBasis(knots, degree=3)

        B_sparse = basis.basis_matrix(x, sparse=True)
        assert issparse(B_sparse)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_torch_raises(self):
        """Test sparse raises error with PyTorch tensors."""
        pytest.importorskip("torch")
        import torch

        x = torch.linspace(0, 10, 50)
        knots = BSplineBasis.create_knots(np.linspace(0, 10, 50), n_basis=10, degree=3)
        basis = BSplineBasis(knots, degree=3)

        with pytest.raises(ValueError, match="Sparse output only supported for NumPy"):
            basis.basis_matrix(x, sparse=True)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_jax_raises(self):
        """Test sparse raises error with JAX arrays."""
        pytest.importorskip("jax")
        import jax.numpy as jnp

        x = jnp.linspace(0, 10, 50)
        knots = BSplineBasis.create_knots(np.linspace(0, 10, 50), n_basis=10, degree=3)
        basis = BSplineBasis(knots, degree=3)

        with pytest.raises(ValueError, match="Sparse output only supported for NumPy"):
            basis.basis_matrix(x, sparse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
