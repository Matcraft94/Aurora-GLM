# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

#
# """Tests for aurora.core.linalg module.

from __future__ import annotations

import numpy as np
import pytest

from aurora.core.linalg import (
    log_determinant,
    lstsq,
    quadratic_form,
    qr_decomposition,
    safe_cholesky,
    safe_inverse,
    solve_cholesky,
    solve_qr,
    solve_triangular,
    woodbury_inverse,
)


# ---------------------------------------------------------------------------
# qr_decomposition
# ---------------------------------------------------------------------------


class TestQRDecomposition:
    def test_basic(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Q, R = qr_decomposition(A)
        assert Q.shape == (3, 2)
        assert R.shape == (2, 2)
        np.testing.assert_allclose(A, Q @ R, atol=1e-10)

    def test_complete_mode(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Q, R = qr_decomposition(A, mode="complete")
        assert Q.shape == (3, 3)
        assert R.shape == (3, 2)

    def test_pivoting(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Q, R, P = qr_decomposition(A, pivoting=True)
        np.testing.assert_allclose(A[:, P], Q @ R, atol=1e-10)

    def test_square_matrix(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        Q, R = qr_decomposition(A)
        np.testing.assert_allclose(A, Q @ R, atol=1e-10)

    def test_r_mode(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        R = qr_decomposition(A, mode="r")
        assert R.shape == (2, 2)

    def test_identity_reconstruction(self):
        A = np.eye(3)
        Q, R = qr_decomposition(A)
        np.testing.assert_allclose(A, Q @ R, atol=1e-10)


# ---------------------------------------------------------------------------
# safe_cholesky
# ---------------------------------------------------------------------------


class TestSafeCholesky:
    def test_positive_definite(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        L = safe_cholesky(A)
        np.testing.assert_allclose(A, L @ L.T, atol=1e-10)

    def test_near_singular(self):
        """Should still succeed with jitter."""
        A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]])
        L = safe_cholesky(A)
        assert L is not None
        assert np.allclose(np.tril(L), L)

    def test_lower_false(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        L = safe_cholesky(A, lower=False)
        np.testing.assert_allclose(A, L.T @ L, atol=1e-10)

    def test_identity(self):
        A = np.eye(3) * 2.0
        L = safe_cholesky(A)
        np.testing.assert_allclose(A, L @ L.T, atol=1e-10)


# ---------------------------------------------------------------------------
# solve_cholesky
# ---------------------------------------------------------------------------


class TestSolveCholesky:
    def test_basic(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 4.0])
        L = safe_cholesky(A)
        x = solve_cholesky(L, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-8)

    def test_multiple_rhs(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        B = np.array([[5.0, 4.0], [1.0, 2.0]])
        L = safe_cholesky(A)
        X = solve_cholesky(L, B)
        np.testing.assert_allclose(A @ X, B, atol=1e-8)


# ---------------------------------------------------------------------------
# solve_triangular
# ---------------------------------------------------------------------------


class TestSolveTriangular:
    def test_upper_triangular(self):
        R = np.array([[2.0, 1.0], [0.0, 3.0]])
        b = np.array([5.0, 6.0])
        x = solve_triangular(R, b, lower=False)
        np.testing.assert_allclose(R @ x, b, atol=1e-10)

    def test_lower_triangular(self):
        L = np.array([[2.0, 0.0], [1.0, 3.0]])
        b = np.array([4.0, 8.0])
        x = solve_triangular(L, b, lower=True)
        np.testing.assert_allclose(L @ x, b, atol=1e-10)

    def test_identity(self):
        I = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        x = solve_triangular(I, b, lower=False)
        np.testing.assert_allclose(x, b, atol=1e-10)


# ---------------------------------------------------------------------------
# solve_qr
# ---------------------------------------------------------------------------


class TestSolveQR:
    def test_overdetermined(self):
        A = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        b = np.array([1.0, 2.0, 3.0])
        Q, R = qr_decomposition(A)
        x = solve_qr(Q, R, b)
        # Check A @ x ≈ b in least-squares sense
        np.testing.assert_allclose(A.T @ A @ x, A.T @ b, atol=1e-8)


# ---------------------------------------------------------------------------
# safe_inverse
# ---------------------------------------------------------------------------


class TestSafeInverse:
    def test_invertible(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        A_inv = safe_inverse(A)
        np.testing.assert_allclose(A @ A_inv, np.eye(2), atol=1e-8)

    def test_near_singular(self):
        A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-8]])
        A_inv = safe_inverse(A)
        assert A_inv is not None
        # Should be close to true inverse
        np.testing.assert_allclose(A @ A_inv, np.eye(2), atol=1e-4)


# ---------------------------------------------------------------------------
# woodbury_inverse
# ---------------------------------------------------------------------------


class TestWoodburyInverse:
    def test_rank1_update(self):
        n = 5
        A_inv = np.eye(n) * 0.5
        U = np.random.randn(n, 1)
        V = U.T.copy()
        C_inv = np.array([[1.0]])
        result = woodbury_inverse(A_inv, U, C_inv, V)
        assert result.shape == (n, n)

    def test_identity_update(self):
        n = 3
        A_inv = np.eye(n)
        U = np.eye(n)
        C_inv = np.eye(n)
        V = np.eye(n)
        result = woodbury_inverse(A_inv, U, C_inv, V)
        # (I + I)^-1 = 0.5*I
        np.testing.assert_allclose(result, 0.5 * np.eye(n), atol=1e-8)


# ---------------------------------------------------------------------------
# quadratic_form
# ---------------------------------------------------------------------------


class TestQuadraticForm:
    def test_identity(self):
        A = np.eye(3)
        x = np.array([1.0, 2.0, 3.0])
        assert quadratic_form(x, A) == pytest.approx(14.0)

    def test_diagonal(self):
        A = np.diag([1.0, 2.0, 3.0])
        x = np.array([1.0, 1.0, 1.0])
        assert quadratic_form(x, A) == pytest.approx(6.0)

    def test_general(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        x = np.array([1.0, 2.0])
        expected = x @ A @ x
        np.testing.assert_allclose(quadratic_form(x, A), expected)


# ---------------------------------------------------------------------------
# log_determinant
# ---------------------------------------------------------------------------


class TestLogDeterminant:
    def test_identity(self):
        A = np.eye(3)
        result = log_determinant(A)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_diagonal(self):
        A = np.diag([2.0, 3.0, 4.0])
        result = log_determinant(A)
        np.testing.assert_allclose(result, np.log(24.0), atol=1e-10)

    def test_positive_definite(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        det = np.linalg.det(A)
        result = log_determinant(A)
        np.testing.assert_allclose(result, np.log(det), atol=1e-8)


# ---------------------------------------------------------------------------
# lstsq
# ---------------------------------------------------------------------------


class TestLstsq:
    def test_simple(self):
        A = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        b = np.array([2.0, 3.0, 4.0])
        x, residuals, rank, sv = lstsq(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-8)

    def test_exact_system(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        x, _, _, _ = lstsq(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-10)
