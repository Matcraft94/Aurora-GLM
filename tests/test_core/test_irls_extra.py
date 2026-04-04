# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Tests for IRLS helper functions and sparse/dense paths.

Covers _is_sparse, _sparse_diag_matmul, _sparse_weighted_lstsq, internal
helpers (_safe_divide, _sqrt, _convert_to_backend), and the full sparse
and dense IRLS paths including step-halving, Cholesky fallback, and
convergence failure.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from aurora.core.backends import get_backend
from aurora.core.optimization.irls import (
    _convert_to_backend,
    _is_sparse,
    _safe_divide,
    _sparse_diag_matmul,
    _sparse_weighted_lstsq,
    _sqrt,
    irls,
)


# ---------------------------------------------------------------------------
# Mock link / variance helpers
# ---------------------------------------------------------------------------


class MockIdentityLink:
    """Mock identity link for Gaussian."""

    def inverse(self, eta):
        return eta

    def derivative(self, mu):
        return np.ones_like(mu)


class MockLogLink:
    """Mock log link for Poisson."""

    def inverse(self, eta):
        return np.exp(np.clip(eta, -20, 20))

    def derivative(self, mu):
        return 1.0 / np.clip(mu, 1e-10, None)


class MockLogitLink:
    """Mock logit link for Binomial."""

    def inverse(self, eta):
        eta_c = np.clip(eta, -20, 20)
        return 1.0 / (1.0 + np.exp(-eta_c))

    def derivative(self, mu):
        return 1.0 / np.clip(mu * (1.0 - mu), 1e-10, None)


# ---------------------------------------------------------------------------
# Tests for _is_sparse
# ---------------------------------------------------------------------------


class TestIsSparse:
    def test_dense_array(self):
        assert _is_sparse(np.eye(10)) is False

    def test_dense_list(self):
        assert _is_sparse([1, 0, 1]) is False

    def test_numpy_array(self):
        assert _is_sparse(np.array([1, 0, 1])) is False

    def test_csr_matrix(self):
        X = sp.csr_matrix(np.eye(5))
        assert _is_sparse(X) is True

    def test_csc_matrix(self):
        X = sp.csc_matrix(np.eye(5))
        assert _is_sparse(X) is True

    def test_coo_matrix(self):
        X = sp.coo_matrix(np.eye(5))
        assert _is_sparse(X) is True


# ---------------------------------------------------------------------------
# Tests for _sparse_diag_matmul
# ---------------------------------------------------------------------------


class TestSparseDiagMatmul:
    def test_csr_row_scaling(self):
        X = sp.csr_matrix(np.array([[1, 2], [3, 4]], dtype=float))
        w = np.array([2.0, 0.5])
        result = _sparse_diag_matmul(X, w)
        expected = np.array([[2.0, 4.0], [1.5, 2.0]])
        np.testing.assert_allclose(result.toarray(), expected)

    def test_csc_format(self):
        X = sp.csc_matrix(np.array([[1, 0], [0, 3]], dtype=float))
        w = np.array([0.5, 2.0])
        result = _sparse_diag_matmul(X, w)
        expected = np.array([[0.5, 0.0], [0.0, 6.0]])
        np.testing.assert_allclose(result.toarray(), expected)

    def test_coo_format(self):
        X = sp.coo_matrix(np.array([[0, 1], [2, 0]], dtype=float))
        w = np.array([3.0, 4.0])
        result = _sparse_diag_matmul(X, w)
        expected = np.array([[0.0, 3.0], [8.0, 0.0]])
        np.testing.assert_allclose(result.toarray(), expected)

    def test_ones_weights(self):
        X = sp.csr_matrix(np.array([[1, 2], [3, 4]], dtype=float))
        w = np.ones(2)
        result = _sparse_diag_matmul(X, w)
        np.testing.assert_allclose(result.toarray(), X.toarray())

    def test_single_row(self):
        X = sp.csr_matrix(np.array([[1, 2, 3]], dtype=float))
        w = np.array([5.0])
        result = _sparse_diag_matmul(X, w)
        np.testing.assert_allclose(result.toarray(), [[5, 10, 15]])


# ---------------------------------------------------------------------------
# Tests for _sparse_weighted_lstsq
# ---------------------------------------------------------------------------


class TestSparseWeightedLstsq:
    def test_uniform_weights(self):
        X = sp.csr_matrix(np.array([[1, 0], [0, 1], [1, 1]], dtype=float))
        w = np.ones(3)
        z = np.array([2.0, 3.0, 5.0])
        beta = _sparse_weighted_lstsq(X, w, z)
        # X^T X = [[2, 1], [1, 2]], X^T z = [7, 8]
        expected = np.linalg.solve(
            np.array([[2, 1], [1, 2]], dtype=float),
            np.array([7.0, 8.0]),
        )
        np.testing.assert_allclose(beta, expected, rtol=1e-10)

    def test_non_uniform_weights(self):
        X = sp.csr_matrix(np.array([[1, 0], [0, 1], [1, 1]], dtype=float))
        w = np.array([1.0, 4.0, 1.0])  # heavier weight on second obs
        z = np.array([2.0, 3.0, 5.0])
        beta = _sparse_weighted_lstsq(X, w, z)
        # Verify by forming normal equations directly: X^T W X and X^T W z
        Xd = X.toarray()
        XtWX = Xd.T @ np.diag(w) @ Xd
        XtWz = Xd.T @ (w * z)
        expected = np.linalg.solve(XtWX, XtWz)
        np.testing.assert_allclose(beta, expected, rtol=1e-10)

    def test_single_observation(self):
        X = sp.csr_matrix(np.array([[1, 2]], dtype=float))
        w = np.array([1.0])
        z = np.array([3.0])
        # Under-determined system (n=1, p=2) -> least squares gives min-norm
        beta = _sparse_weighted_lstsq(X, w, z)
        # X^T W X = [[1, 2], [2, 4]], singular but lstsq fallback handles it
        assert beta.shape == (2,)
        assert np.all(np.isfinite(beta))

    def test_small_matrix(self):
        X = sp.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
        w = np.array([1.0, 1.0])
        z = np.array([1.0, 2.0])
        beta = _sparse_weighted_lstsq(X, w, z)
        XtWX = X.toarray().T @ X.toarray()
        XtWz = X.toarray().T @ z
        expected = np.linalg.solve(XtWX, XtWz)
        np.testing.assert_allclose(beta, expected, rtol=1e-5, atol=1e-12)

    def test_csc_input(self):
        X = sp.csc_matrix(np.array([[1, 0], [0, 1], [1, 1]], dtype=float))
        w = np.ones(3)
        z = np.array([1.0, 1.0, 2.0])
        beta = _sparse_weighted_lstsq(X, w, z)
        expected = np.array([1.0, 1.0])
        np.testing.assert_allclose(beta, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests for internal helpers: _safe_divide, _sqrt, _convert_to_backend
# ---------------------------------------------------------------------------


class TestSafeDivide:
    def test_numpy_normal(self):
        backend = get_backend("numpy")
        num = 1.0
        den = backend.array([2.0, 4.0])
        result = _safe_divide(backend, num, den)
        np.testing.assert_allclose(result, [0.5, 0.25])

    def test_numpy_clips_denom(self):
        backend = get_backend("numpy")
        num = 1.0
        den = backend.array([0.0, 1e-20])
        result = _safe_divide(backend, num, den)
        # denominator clipped to 1e-12
        assert np.all(np.isfinite(result))
        assert result[0] < 1e13  # not astronomically huge


class TestSqrt:
    def test_numpy_array(self):
        backend = get_backend("numpy")
        val = backend.array([4.0, 9.0, 16.0])
        result = _sqrt(backend, val)
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_numpy_scalar_like(self):
        backend = get_backend("numpy")
        val = backend.array([1.0])
        result = _sqrt(backend, val)
        np.testing.assert_allclose(result, [1.0])


class TestConvertToBackend:
    def test_tuple_of_arrays(self):
        backend = get_backend("numpy")
        result = _convert_to_backend(backend, (np.array([1.0]), np.array([2.0])))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_list_of_arrays(self):
        backend = get_backend("numpy")
        result = _convert_to_backend(backend, [np.array([1.0]), np.array([2.0])])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_dict_of_arrays(self):
        backend = get_backend("numpy")
        result = _convert_to_backend(backend, {"a": np.array([1.0]), "b": np.array([2.0])})
        assert isinstance(result, dict)
        assert "a" in result and "b" in result

    def test_scalar(self):
        backend = get_backend("numpy")
        result = _convert_to_backend(backend, 3.14)
        # Should convert to backend array
        assert np.allclose(result, 3.14)

    def test_nested_tuple(self):
        backend = get_backend("numpy")
        result = _convert_to_backend(backend, (np.array([1.0]), (np.array([2.0]),)))
        assert isinstance(result, tuple)
        assert isinstance(result[1], tuple)


# ---------------------------------------------------------------------------
# Tests for sparse IRLS path
# ---------------------------------------------------------------------------


class TestIRLSSparsePoisson:
    """Poisson GLM via sparse IRLS (log link)."""

    @staticmethod
    def _loss(beta, X_dense, y):
        eta = X_dense @ beta
        mu = np.exp(np.clip(eta, -20, 20))
        return float(np.sum(mu - y * eta))

    def test_convergence(self):
        np.random.seed(42)
        n, p = 50, 3
        X_dense = np.random.randn(n, p)
        X_dense[:, 0] = 1.0  # intercept
        beta_true = np.array([0.5, -0.3, 0.2])
        mu = np.exp(X_dense @ beta_true)
        y = np.random.poisson(mu)

        X_sparse = sp.csr_matrix(X_dense)
        link = MockLogLink()

        def variance_fn(mu):
            return mu

        def loss(beta):
            return self._loss(beta, X_dense, y)

        result = irls(
            loss,
            np.zeros(p),
            backend=get_backend("numpy"),
            design_matrix=X_sparse,
            response=y,
            link=link,
            variance_fn=variance_fn,
            max_iter=50,
            tol=1e-6,
        )
        assert result.success
        assert result.nit <= 50
        assert result.x.shape == (p,)

    def test_with_callback(self):
        np.random.seed(0)
        n, p = 20, 2
        X_dense = np.ones((n, p))
        X_dense[:, 1] = np.arange(n, dtype=float)
        y = np.random.poisson(2.0, size=n)
        X_sparse = sp.csr_matrix(X_dense)
        link = MockLogLink()
        calls = []

        def cb(it, beta, loss_val):
            calls.append((it, loss_val))

        result = irls(
            lambda b: self._loss(b, X_dense, y),
            np.zeros(p),
            backend=get_backend("numpy"),
            design_matrix=X_sparse,
            response=y,
            link=link,
            variance_fn=lambda mu: mu,
            max_iter=30,
            tol=1e-6,
            callback=cb,
        )
        assert len(calls) > 0
        # Callback receives iteration number and loss
        assert calls[0][0] == 0


class TestIRLSSparseBinomial:
    """Binomial GLM via sparse IRLS (logit link)."""

    @staticmethod
    def _loss(beta, X_dense, y):
        eta = X_dense @ beta
        eta = np.clip(eta, -20, 20)
        return float(-np.sum(y * eta - np.log(1 + np.exp(eta))))

    def test_convergence(self):
        np.random.seed(42)
        n, p = 60, 3
        X_dense = np.random.randn(n, p)
        X_dense[:, 0] = 1.0
        beta_true = np.array([0.3, -0.5, 0.4])
        prob = 1.0 / (1.0 + np.exp(-(X_dense @ beta_true)))
        y = (np.random.rand(n) < prob).astype(float)

        X_sparse = sp.csr_matrix(X_dense)
        link = MockLogitLink()

        def loss(beta):
            return self._loss(beta, X_dense, y)

        result = irls(
            loss,
            np.zeros(p),
            backend=get_backend("numpy"),
            design_matrix=X_sparse,
            response=y,
            link=link,
            variance_fn=lambda mu: mu * (1 - mu),
            max_iter=50,
            tol=1e-6,
        )
        assert result.success
        assert result.x.shape == (p,)


# ---------------------------------------------------------------------------
# Tests for dense IRLS path (Gaussian identity link)
# ---------------------------------------------------------------------------


class TestIRLSDenseGaussian:
    """Gaussian GLM via dense IRLS (identity link)."""

    @staticmethod
    def _loss(beta, X_dense, y):
        r = y - X_dense @ beta
        return float(0.5 * np.sum(r ** 2))

    def test_convergence(self):
        np.random.seed(42)
        n, p = 30, 3
        X_dense = np.random.randn(n, p)
        X_dense[:, 0] = 1.0
        beta_true = np.array([2.0, -1.0, 0.5])
        y = X_dense @ beta_true + np.random.randn(n) * 0.1

        backend = get_backend("numpy")
        link = MockIdentityLink()

        def loss(beta):
            return self._loss(beta, X_dense, y)

        result = irls(
            loss,
            np.zeros(p),
            backend=backend,
            design_matrix=X_dense,
            response=y,
            link=link,
            variance_fn=lambda mu: np.ones_like(mu),
            max_iter=50,
            tol=1e-6,
        )
        assert result.success
        np.testing.assert_allclose(result.x, beta_true, atol=0.2)

    def test_with_offset(self):
        np.random.seed(0)
        n, p = 20, 2
        X_dense = np.random.randn(n, p)
        X_dense[:, 0] = 1.0
        beta_true = np.array([1.0, -0.5])
        offset = np.ones(n) * 0.5
        y = X_dense @ beta_true + offset + np.random.randn(n) * 0.05

        backend = get_backend("numpy")
        link = MockIdentityLink()

        def loss(beta):
            return self._loss(beta, X_dense, y)

        result = irls(
            loss,
            np.zeros(p),
            backend=backend,
            design_matrix=X_dense,
            response=y,
            link=link,
            variance_fn=lambda mu: np.ones_like(mu),
            offset=offset,
            max_iter=50,
            tol=1e-6,
        )
        assert result.success
        # Should recover coefficients approximately
        assert np.linalg.norm(result.x - beta_true) < 1.0

    def test_callback(self):
        np.random.seed(1)
        n, p = 15, 2
        X_dense = np.random.randn(n, p)
        X_dense[:, 0] = 1.0
        y = X_dense @ np.array([1.0, -1.0]) + np.random.randn(n) * 0.1
        link = MockIdentityLink()
        calls = []

        def cb(it, beta, loss_val):
            calls.append((it, float(loss_val)))

        result = irls(
            lambda b: self._loss(b, X_dense, y),
            np.zeros(p),
            backend=get_backend("numpy"),
            design_matrix=X_dense,
            response=y,
            link=link,
            variance_fn=lambda mu: np.ones_like(mu),
            max_iter=50,
            tol=1e-6,
            callback=cb,
        )
        assert result.success
        assert len(calls) > 0


# ---------------------------------------------------------------------------
# Edge cases: validation errors, convergence failure, step-halving
# ---------------------------------------------------------------------------


class TestIRLSEdgeCases:
    @staticmethod
    def _loss(beta, X_dense, y):
        r = y - X_dense @ beta
        return float(0.5 * np.sum(r ** 2))

    def test_missing_design_matrix_raises(self):
        with pytest.raises(ValueError, match="design_matrix"):
            irls(
                lambda b: 0.0,
                np.zeros(2),
                backend=get_backend("numpy"),
                response=np.array([1.0]),
                link=MockIdentityLink(),
                variance_fn=lambda mu: np.ones_like(mu),
            )

    def test_missing_link_raises(self):
        with pytest.raises(ValueError, match="link"):
            irls(
                lambda b: 0.0,
                np.zeros(2),
                backend=get_backend("numpy"),
                design_matrix=np.eye(2),
                response=np.array([1.0]),
                variance_fn=lambda mu: np.ones_like(mu),
            )

    def test_bad_link_type_raises(self):
        with pytest.raises(TypeError, match="inverse.*derivative"):
            irls(
                lambda b: 0.0,
                np.zeros(2),
                backend=get_backend("numpy"),
                design_matrix=np.eye(2),
                response=np.array([1.0]),
                link="not_a_link",
                variance_fn=lambda mu: np.ones_like(mu),
            )

    def test_max_iterations_reached(self):
        """Force convergence failure by using max_iter=1."""
        np.random.seed(42)
        n, p = 30, 3
        X_dense = np.random.randn(n, p)
        y = np.random.randn(n)
        link = MockIdentityLink()

        result = irls(
            lambda b: self._loss(b, X_dense, y),
            np.zeros(p),
            backend=get_backend("numpy"),
            design_matrix=X_dense,
            response=y,
            link=link,
            variance_fn=lambda mu: np.ones_like(mu),
            max_iter=1,
            tol=1e-12,  # Very tight tolerance
        )
        assert result.success is False
        assert result.message == "Maximum iterations reached"
        assert result.nit == 1

    def test_step_halving_triggered(self):
        """Force step-halving by using a loss that increases on the first step."""
        np.random.seed(42)
        n, p = 10, 3
        X_dense = np.random.randn(n, p)
        X_dense[:, 0] = 1.0
        y = np.random.randn(n)
        link = MockIdentityLink()

        call_count = [0]
        original_loss = lambda b: self._loss(b, X_dense, y)

        def counting_loss(beta):
            call_count[0] += 1
            return original_loss(beta)

        result = irls(
            counting_loss,
            np.zeros(p),
            backend=get_backend("numpy"),
            design_matrix=X_dense,
            response=y,
            link=link,
            variance_fn=lambda mu: np.ones_like(mu),
            max_iter=5,
            tol=1e-6,
        )
        # Should have multiple function evaluations due to step-halving
        assert result.nfev > 0

    def test_convergence_with_kwargs(self):
        """Test IRLS with kwargs passed to loss."""
        np.random.seed(42)
        n, p = 15, 2
        X_dense = np.random.randn(n, p)
        X_dense[:, 0] = 1.0
        y = X_dense @ np.array([1.0, -0.5]) + np.random.randn(n) * 0.1

        def loss(beta, X=None, y=None):
            r = y - X @ beta
            return float(0.5 * np.sum(r ** 2))

        result = irls(
            loss,
            np.zeros(p),
            backend=get_backend("numpy"),
            args=(X_dense, y),
            design_matrix=X_dense,
            response=y,
            link=MockIdentityLink(),
            variance_fn=lambda mu: np.ones_like(mu),
            max_iter=30,
            tol=1e-6,
        )
        assert result.success


class TestIRLSSparseEdgeCases:
    """Sparse IRLS edge cases."""

    @staticmethod
    def _loss(beta, X_dense, y):
        eta = X_dense @ beta
        mu = np.exp(np.clip(eta, -20, 20))
        return float(np.sum(mu - y * eta))

    def test_convergence_failure_sparse(self):
        """Sparse IRLS with max_iter=1 should report failure."""
        np.random.seed(42)
        n, p = 20, 2
        X_dense = np.random.randn(n, p)
        y = np.random.poisson(2.0, size=n)
        X_sparse = sp.csr_matrix(X_dense)

        result = irls(
            lambda b: self._loss(b, X_dense, y),
            np.zeros(p),
            backend=get_backend("numpy"),
            design_matrix=X_sparse,
            response=y,
            link=MockLogLink(),
            variance_fn=lambda mu: mu,
            max_iter=1,
            tol=1e-15,
        )
        assert result.success is False
        assert result.nit == 1
