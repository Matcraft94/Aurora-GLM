"""Tests for aurora.core.linalg module.

Tests linear algebra primitives: qr_decomposition, safe_cholesky, svd, etc.
"""
from __future__ import annotations

import numpy as np
import pytest

from aurora.core.linalg import (
    qr_decomposition,
    safe_cholesky,
    svd,
    eigh,
    solve_cholesky,
    solve_qr,
    log_determinant,
    matrix_rank,
    condition_number,
    woodbury_inverse,
)


class TestQRDecomposition:
    """Tests for qr_decomposition function."""

    def test_qr_basic(self):
        """Test basic QR decomposition."""
        A = np.random.randn(10, 5)
        Q, R = qr_decomposition(A)
        
        assert Q is not None
        assert R is not None
        assert Q.shape == (10, 5)
        assert R.shape == (5, 5)

    def test_qr_orthogonality(self):
        """Test Q is orthogonal."""
        A = np.random.randn(10, 5)
        Q, R = qr_decomposition(A)
        
        # Q^T Q should be identity
        QtQ = Q.T @ Q
        np.testing.assert_array_almost_equal(QtQ, np.eye(5), decimal=10)

    def test_qr_upper_triangular(self):
        """Test R is upper triangular."""
        A = np.random.randn(10, 5)
        Q, R = qr_decomposition(A)
        
        # R should be upper triangular
        assert np.allclose(R, np.triu(R))

    def test_qr_reconstruction(self):
        """Test A = QR."""
        A = np.random.randn(10, 5)
        Q, R = qr_decomposition(A)
        
        np.testing.assert_array_almost_equal(Q @ R, A, decimal=10)

    def test_qr_square_matrix(self):
        """Test QR on square matrix."""
        A = np.random.randn(5, 5)
        Q, R = qr_decomposition(A)
        
        assert Q.shape == (5, 5)
        assert R.shape == (5, 5)
        np.testing.assert_array_almost_equal(Q @ R, A, decimal=10)

    def test_qr_with_pivoting(self):
        """Test QR with column pivoting."""
        A = np.random.randn(10, 5)
        result = qr_decomposition(A, pivoting=True)
        
        if len(result) == 3:
            Q, R, P = result
            assert P is not None
        else:
            Q, R = result


class TestSafeCholesky:
    """Tests for safe_cholesky function."""

    def test_cholesky_basic(self):
        """Test basic Cholesky decomposition."""
        A = np.array([[4, 2], [2, 3]])
        L = safe_cholesky(A)
        
        assert L is not None
        assert L.shape == (2, 2)

    def test_cholesky_reconstruction(self):
        """Test A = LL^T."""
        A = np.array([[4, 2], [2, 3]])
        L = safe_cholesky(A)
        
        np.testing.assert_array_almost_equal(L @ L.T, A, decimal=10)

    def test_cholesky_lower_triangular(self):
        """Test L is lower triangular."""
        A = np.array([[4, 2], [2, 3]])
        L = safe_cholesky(A)
        
        assert np.allclose(L, np.tril(L))

    def test_cholesky_positive_diagonal(self):
        """Test L has positive diagonal."""
        A = np.array([[4, 2], [2, 3]])
        L = safe_cholesky(A)
        
        assert all(np.diag(L) > 0)

    def test_cholesky_random_pd_matrix(self):
        """Test Cholesky on random positive definite matrix."""
        n = 10
        A = np.random.randn(n, n)
        A = A @ A.T + np.eye(n)  # Make positive definite
        
        L = safe_cholesky(A)
        np.testing.assert_array_almost_equal(L @ L.T, A, decimal=8)

    def test_cholesky_near_singular(self):
        """Test Cholesky handles near-singular matrices."""
        n = 5
        A = np.random.randn(n, n)
        A = A @ A.T + 1e-10 * np.eye(n)  # Nearly singular
        
        # Should either succeed with regularization or raise appropriate error
        try:
            L = safe_cholesky(A)
            assert L is not None
        except np.linalg.LinAlgError:
            pass  # Acceptable to fail on nearly singular


class TestSVD:
    """Tests for svd function."""

    def test_svd_basic(self):
        """Test basic SVD."""
        A = np.random.randn(10, 5)
        U, S, Vt = svd(A)
        
        assert U is not None
        assert S is not None
        assert Vt is not None

    def test_svd_shapes(self):
        """Test SVD shapes."""
        A = np.random.randn(10, 5)
        U, S, Vt = svd(A)
        
        assert U.shape[0] == 10
        assert len(S) == 5
        assert Vt.shape[1] == 5

    def test_svd_reconstruction(self):
        """Test A = U S V^T."""
        A = np.random.randn(10, 5)
        U, S, Vt = svd(A)
        
        reconstructed = U[:, :len(S)] @ np.diag(S) @ Vt
        np.testing.assert_array_almost_equal(reconstructed, A, decimal=10)

    def test_svd_singular_values_sorted(self):
        """Test singular values are sorted in descending order."""
        A = np.random.randn(10, 5)
        U, S, Vt = svd(A)
        
        assert all(S[i] >= S[i+1] for i in range(len(S)-1))

    def test_svd_singular_values_non_negative(self):
        """Test singular values are non-negative."""
        A = np.random.randn(10, 5)
        U, S, Vt = svd(A)
        
        assert all(S >= 0)


class TestEigh:
    """Tests for eigh function (symmetric eigendecomposition)."""

    def test_eigh_basic(self):
        """Test basic symmetric eigendecomposition."""
        A = np.array([[4, 2], [2, 3]])
        eigenvalues, eigenvectors = eigh(A)
        
        assert len(eigenvalues) == 2
        assert eigenvectors.shape == (2, 2)

    def test_eigh_reconstruction(self):
        """Test A = V D V^T."""
        A = np.array([[4, 2], [2, 3]])
        eigenvalues, eigenvectors = eigh(A)
        
        reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        np.testing.assert_array_almost_equal(reconstructed, A, decimal=10)

    def test_eigh_orthogonal_eigenvectors(self):
        """Test eigenvectors are orthogonal."""
        n = 5
        A = np.random.randn(n, n)
        A = (A + A.T) / 2  # Make symmetric
        
        eigenvalues, eigenvectors = eigh(A)
        
        VtV = eigenvectors.T @ eigenvectors
        np.testing.assert_array_almost_equal(VtV, np.eye(n), decimal=10)


class TestSolveCholesky:
    """Tests for solve_cholesky function."""

    def test_solve_cholesky_basic(self):
        """Test solving Ax = b via Cholesky factor."""
        A = np.array([[4, 2], [2, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)
        
        # First compute Cholesky factor
        L = safe_cholesky(A)
        x = solve_cholesky(L, b)
        
        np.testing.assert_array_almost_equal(A @ x, b, decimal=10)

    def test_solve_cholesky_multiple_rhs(self):
        """Test solving with multiple right-hand sides."""
        n = 5
        A = np.random.randn(n, n)
        A = A @ A.T + np.eye(n)
        B = np.random.randn(n, 3)
        
        L = safe_cholesky(A)
        X = solve_cholesky(L, B)
        
        np.testing.assert_array_almost_equal(A @ X, B, decimal=8)


class TestSolveQR:
    """Tests for solve_qr function."""

    def test_solve_qr_overdetermined(self):
        """Test solving overdetermined system via QR."""
        A = np.random.randn(10, 5)
        b = np.random.randn(10)
        
        Q, R = qr_decomposition(A)
        x = solve_qr(Q, R, b)
        
        # Should give least squares solution
        assert len(x) == 5

    def test_solve_qr_square(self):
        """Test solving square system via QR."""
        A = np.random.randn(5, 5) + np.eye(5)  # Make non-singular
        b = np.random.randn(5)
        
        Q, R = qr_decomposition(A)
        x = solve_qr(Q, R, b)
        
        np.testing.assert_array_almost_equal(A @ x, b, decimal=8)


class TestLogDeterminant:
    """Tests for log_determinant function."""

    def test_log_det_basic(self):
        """Test basic log determinant."""
        A = np.array([[4, 2], [2, 3]])
        log_det = log_determinant(A)
        
        expected = np.log(np.linalg.det(A))
        np.testing.assert_almost_equal(log_det, expected, decimal=10)

    def test_log_det_positive_definite(self):
        """Test log determinant of positive definite matrix."""
        n = 5
        A = np.random.randn(n, n)
        A = A @ A.T + np.eye(n)
        
        log_det = log_determinant(A)
        
        assert np.isfinite(log_det)


class TestMatrixRank:
    """Tests for matrix_rank function."""

    def test_rank_full(self):
        """Test rank of full rank matrix."""
        A = np.random.randn(5, 5)
        A = A + np.eye(5)  # Ensure full rank
        
        rank = matrix_rank(A)
        assert rank == 5

    def test_rank_deficient(self):
        """Test rank of rank-deficient matrix."""
        A = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]])
        
        rank = matrix_rank(A)
        assert rank < 3


class TestConditionNumber:
    """Tests for condition_number function."""

    def test_condition_number_basic(self):
        """Test basic condition number."""
        A = np.eye(5)
        cond = condition_number(A)
        
        # Identity matrix has condition number 1
        np.testing.assert_almost_equal(cond, 1.0, decimal=10)

    def test_condition_number_ill_conditioned(self):
        """Test condition number of ill-conditioned matrix."""
        A = np.array([[1, 1], [1, 1.0001]])
        cond = condition_number(A)
        
        # Should have large condition number
        assert cond > 100


class TestWoodburyInverse:
    """Tests for woodbury_inverse function."""

    def test_woodbury_basic(self):
        """Test Woodbury matrix identity."""
        n, k = 10, 3
        A = np.eye(n) * 2  # Easy to invert
        A_inv = np.eye(n) / 2  # Inverse of A
        U = np.random.randn(n, k)
        C = np.eye(k)
        V = np.random.randn(k, n)
        
        # (A + UCV)^-1 via Woodbury, using A_inv
        inv_woodbury = woodbury_inverse(A_inv, U, C, V)
        
        # Direct inverse
        M = A + U @ C @ V
        inv_direct = np.linalg.inv(M)
        
        np.testing.assert_array_almost_equal(inv_woodbury, inv_direct, decimal=8)


class TestLinalgIntegration:
    """Integration tests for linear algebra functions."""

    def test_solve_least_squares_comparison(self):
        """Compare QR solve vs normal equations."""
        A = np.random.randn(100, 10)
        b = np.random.randn(100)
        
        # Via QR
        Q, R = qr_decomposition(A)
        x_qr = solve_qr(Q, R, b)
        
        # Via normal equations with Cholesky
        AtA = A.T @ A
        Atb = A.T @ b
        L = safe_cholesky(AtA)
        x_chol = solve_cholesky(L, Atb)
        
        np.testing.assert_array_almost_equal(x_qr, x_chol, decimal=8)

    def test_svd_vs_eigh_for_symmetric(self):
        """Compare SVD and eigh for symmetric matrix."""
        n = 5
        A = np.random.randn(n, n)
        A = (A + A.T) / 2  # Symmetric
        
        U, S, Vt = svd(A)
        eigenvalues, eigenvectors = eigh(A)
        
        # Singular values should match absolute eigenvalues (for symmetric)
        np.testing.assert_array_almost_equal(
            np.sort(S)[::-1], 
            np.sort(np.abs(eigenvalues))[::-1], 
            decimal=8
        )
