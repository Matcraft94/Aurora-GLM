# SPDX-License-Identifier: MIT
"""Tests for aurora.core.autodiff.products module.

Covers backend-specific branches for hvp(), jvp(), vjp() using mocked
namespace objects when the real backend is unavailable, plus direct
exercising of internal helpers.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aurora.core.autodiff.products import (
    _hvp_jax,
    _hvp_numerical,
    _hvp_torch,
    hvp,
    jvp,
    vjp,
)

# ---------------------------------------------------------------------------
# Optional backend detection
# ---------------------------------------------------------------------------
try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    HAS_JAX = False


def _quadratic(x):
    """f(x) = 0.5 * sum(x^2).  grad = x, hess = I."""
    return 0.5 * np.sum(x**2)


def _quadratic_matrix(x, A):
    """f(x) = 0.5 * x^T A x."""
    return 0.5 * x @ A @ x


def _vector_fn(x):
    """Vector-valued: [x0^2 + x1, x0*x1, x1^3]."""
    return np.array([x[0] ** 2 + x[1], x[0] * x[1], x[1] ** 3])


# ===========================================================================
# hvp() — high-level dispatch with mocked backends
# ===========================================================================


class TestHVPDispatch:
    """Test hvp() dispatching to the correct backend via detect_backend."""

    def test_numpy_dispatch(self):
        x = np.array([1.0, 2.0])
        v = np.array([0.5, -0.5])
        with patch("aurora.core.autodiff.products.detect_backend", return_value="numpy"):
            Hv = hvp(_quadratic, x, v)
        assert np.all(np.isfinite(Hv))
        assert Hv.shape == x.shape

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_dispatch(self):
        def f(x):
            return 0.5 * torch.sum(x**2)

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        v = torch.tensor([0.5, -0.5], dtype=torch.float64)
        with patch("aurora.core.autodiff.products.detect_backend", return_value="torch"):
            Hv = hvp(f, x, v)
        assert_allclose(Hv.detach().numpy(), v.detach().numpy(), atol=1e-4)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_dispatch(self):
        def f(x):
            return 0.5 * jnp.sum(x**2)

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.5, -0.5])
        with patch("aurora.core.autodiff.products.detect_backend", return_value="jax"):
            Hv = hvp(f, x, v)
        assert_allclose(np.asarray(Hv), np.array([0.5, -0.5]), atol=1e-5)


# ===========================================================================
# _hvp_numerical additional edge cases
# ===========================================================================


class TestHVPNumericalEdgeCases:
    """Additional coverage for _hvp_numerical."""

    def test_single_element(self):
        def f(x):
            return 0.5 * np.sum(x**2)

        x = np.array([3.0])
        v = np.array([1.0])
        Hv = _hvp_numerical(f, x, v)
        assert_allclose(Hv, [1.0], rtol=0.5, atol=0.5)

    def test_with_kwargs(self):
        def f(x, scale=1.0):
            return 0.5 * scale * np.sum(x**2)

        x = np.array([1.0, 2.0])
        v = np.array([1.0, 0.0])
        Hv = _hvp_numerical(f, x, v, scale=3.0)
        # Hessian = 3*I, so Hv = 3*v
        assert_allclose(Hv, [3.0, 0.0], rtol=0.5, atol=0.5)

    def test_zero_vector(self):
        x = np.array([1.0, 2.0, 3.0])
        v = np.zeros(3)
        Hv = _hvp_numerical(_quadratic, x, v)
        assert_allclose(Hv, np.zeros(3), atol=1e-12)


# ===========================================================================
# _hvp_torch directly
# ===========================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestHVPTorchDirect:
    """Cover _hvp_torch directly."""

    def test_quadratic_identity(self):
        def f(x):
            return 0.5 * torch.sum(x**2)

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        v = torch.tensor([0.5, -0.5], dtype=torch.float64)
        Hv = _hvp_torch(f, x, v)
        assert_allclose(Hv.detach().numpy(), v.numpy(), atol=1e-5)

    def test_matrix_quadratic(self):
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)

        def f(x):
            return 0.5 * x @ A @ x

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        v = torch.tensor([0.5, -0.5], dtype=torch.float64)
        Hv = _hvp_torch(f, x, v)
        assert_allclose(Hv.detach().numpy(), (A @ v).numpy(), atol=1e-5)

    def test_with_extra_args(self):
        def f(x, scale):
            return 0.5 * scale * torch.sum(x**2)

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        v = torch.tensor([0.5, -0.5], dtype=torch.float64)
        Hv = _hvp_torch(f, x, v, 3.0)
        # Hessian = 3*I, so Hv = 3*v
        assert_allclose(Hv.detach().numpy(), (3.0 * v).numpy(), atol=1e-4)


# ===========================================================================
# _hvp_jax directly
# ===========================================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestHVPJAXDirect:
    """Cover _hvp_jax directly."""

    def test_quadratic(self):
        def f(x):
            return 0.5 * jnp.sum(x**2)

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.5, -0.5])
        Hv = _hvp_jax(f, x, v)
        assert_allclose(np.asarray(Hv), np.array([0.5, -0.5]), atol=1e-5)

    def test_matrix_quadratic(self):
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])

        def f(x):
            return 0.5 * x @ A @ x

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.5, -0.5])
        Hv = _hvp_jax(f, x, v)
        assert_allclose(np.asarray(Hv), np.asarray(A @ v), atol=1e-5)

    def test_with_kwargs(self):
        def f(x, scale=1.0):
            return 0.5 * scale * jnp.sum(x**2)

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.5, -0.5])
        Hv = _hvp_jax(f, x, v, scale=3.0)
        assert_allclose(np.asarray(Hv), np.array([1.5, -1.5]), atol=1e-5)


# ===========================================================================
# jvp() — high-level dispatch with mocked backends
# ===========================================================================


class TestJVPDispatch:
    """Test jvp() dispatching via detect_backend."""

    def test_numpy_dispatch(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 2.0])
        v = np.array([0.5, -0.5])
        with patch("aurora.core.autodiff.products.detect_backend", return_value="numpy"):
            f_x, tangent = jvp(lambda x: A @ x, x, v)
        assert_allclose(f_x, A @ x, rtol=1e-5)
        assert_allclose(tangent, A @ v, rtol=1e-3)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_dispatch(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)

        def f(x):
            return A @ x

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        v = torch.tensor([0.5, -0.5], dtype=torch.float64)
        with patch("aurora.core.autodiff.products.detect_backend", return_value="torch"):
            f_x, tangent = jvp(f, x, v)
        assert_allclose(f_x.detach().numpy(), (A @ x).numpy(), rtol=1e-5)
        assert_allclose(tangent.detach().numpy(), (A @ v).numpy(), rtol=1e-4)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_dispatch(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        def f(x):
            return A @ x

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.5, -0.5])
        with patch("aurora.core.autodiff.products.detect_backend", return_value="jax"):
            f_x, tangent = jvp(f, x, v)
        assert_allclose(np.asarray(f_x), np.array([5.0, 11.0]), rtol=1e-5)
        assert_allclose(np.asarray(tangent), np.array([-0.5, -0.5]), rtol=1e-5)


# ===========================================================================
# jvp() — additional numpy edge cases
# ===========================================================================


class TestJVPNumPyEdgeCases:
    """Additional numpy-path jvp() coverage."""

    def test_scalar_output(self):
        def f(x):
            return np.array([np.sum(x**2)])

        x = np.array([1.0, 2.0])
        v = np.array([1.0, 0.0])
        f_x, tangent = jvp(f, x, v)
        assert_allclose(f_x, [5.0], rtol=1e-5)
        assert_allclose(tangent, [2.0], rtol=1e-2)

    def test_with_extra_args(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])

        def f(x, A):
            return A @ x

        x = np.array([1.0, 2.0])
        v = np.array([0.0, 1.0])
        f_x, tangent = jvp(f, x, v, A)
        assert_allclose(f_x, A @ x, rtol=1e-5)
        assert_allclose(tangent, A @ v, rtol=1e-3)

    def test_single_element(self):
        def f(x):
            return np.array([x[0] ** 2])

        x = np.array([3.0])
        v = np.array([1.0])
        f_x, tangent = jvp(f, x, v)
        assert_allclose(f_x, [9.0], rtol=1e-5)
        assert_allclose(tangent, [6.0], rtol=1e-2)


# ===========================================================================
# vjp() — high-level dispatch with mocked backends
# ===========================================================================


class TestVJPDispatch:
    """Test vjp() dispatching via detect_backend."""

    def test_numpy_dispatch(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 2.0])
        v = np.array([0.5, -0.5])
        with patch("aurora.core.autodiff.products.detect_backend", return_value="numpy"):
            f_x, cotangent = vjp(lambda x: A @ x, x, v)
        assert_allclose(f_x, A @ x, rtol=1e-5)
        assert_allclose(cotangent, v @ A, rtol=1e-3)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_dispatch(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)

        def f(x):
            return A @ x

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        v = torch.tensor([0.5, -0.5], dtype=torch.float64)
        with patch("aurora.core.autodiff.products.detect_backend", return_value="torch"):
            f_x, cotangent = vjp(f, x, v)
        assert_allclose(f_x.detach().numpy(), (A @ x).numpy(), rtol=1e-5)
        assert_allclose(cotangent.detach().numpy(), (v @ A).numpy(), rtol=1e-4)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_dispatch(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        def f(x):
            return A @ x

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.5, -0.5])
        with patch("aurora.core.autodiff.products.detect_backend", return_value="jax"):
            f_x, cotangent = vjp(f, x, v)
        assert_allclose(np.asarray(f_x), np.array([5.0, 11.0]), rtol=1e-5)
        assert_allclose(np.asarray(cotangent), np.asarray(v @ A), rtol=1e-5)


# ===========================================================================
# vjp() — additional numpy edge cases
# ===========================================================================


class TestVJPNumPyEdgeCases:
    """Additional numpy-path vjp() coverage."""

    def test_nonlinear_vector_fn(self):
        def f(x):
            return np.array([x[0] ** 2, x[0] * x[1]])

        x = np.array([2.0, 3.0])
        v = np.array([1.0, 0.0])
        f_x, cotangent = vjp(f, x, v)
        assert_allclose(f_x, [4.0, 6.0], rtol=1e-5)
        # J = [[4, 0], [3, 2]], v^T @ J = [4, 0]
        assert_allclose(cotangent, [4.0, 0.0], rtol=1e-3)

    def test_with_extra_args(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])

        def f(x, A):
            return A @ x

        x = np.array([1.0, 2.0])
        v = np.array([1.0, 0.0])
        f_x, cotangent = vjp(f, x, v, A)
        assert_allclose(f_x, A @ x, rtol=1e-5)
        assert_allclose(cotangent, v @ A, rtol=1e-3)

    def test_single_output(self):
        def f(x):
            return np.array([np.sum(x)])

        x = np.array([1.0, 2.0, 3.0])
        v = np.array([1.0])
        f_x, cotangent = vjp(f, x, v)
        assert_allclose(f_x, [6.0], rtol=1e-5)
        assert_allclose(cotangent, np.ones(3), rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
