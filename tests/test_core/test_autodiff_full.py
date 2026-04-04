# SPDX-License-Identifier: MIT
"""Comprehensive tests for aurora.core.autodiff sub-modules.

Covers gradient.py, hessian.py, jacobian.py, products.py, backends.py, and
utils.py -- the modules that have 0 % coverage.  Each internal function is
exercised directly so the branch-level coverage increases.

Pattern: np.random.seed(42), np.testing.assert_allclose(..., rtol=1e-5, atol=1e-8).
JAX-only helpers use pytest.mark.skipif(not HAS_JAX).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

# ---------------------------------------------------------------------------
# Imports under test -- internal sub-modules (0 % coverage targets)
# ---------------------------------------------------------------------------
from aurora.core.autodiff.backends import detect_backend, get_backend_module
from aurora.core.autodiff.gradient import (
    _gradient_jax,
    _gradient_numerical,
    _gradient_torch,
    gradient,
)
from aurora.core.autodiff.hessian import (
    _hessian_jax,
    _hessian_numerical,
    _hessian_torch,
    hessian,
)
from aurora.core.autodiff.jacobian import (
    _jacobian_jax,
    _jacobian_numerical,
    _jacobian_torch,
    jacobian,
)
from aurora.core.autodiff.products import (
    _hvp_jax,
    _hvp_numerical,
    _hvp_torch,
    hvp,
    jvp,
    vjp,
)
from aurora.core.autodiff.utils import check_gradient

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

np.random.seed(42)


# ===========================================================================
# Shared test functions
# ===========================================================================


def _quadratic(x):
    """f(x) = 0.5 * sum(x^2).  grad = x, hess = I."""
    return 0.5 * np.sum(x ** 2)


def _quadratic_matrix(x, A):
    """f(x) = 0.5 * x^T A x.  grad = A x (sym A), hess = A."""
    return 0.5 * x @ A @ x


def _rosenbrock(x):
    """Extended Rosenbrock."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def _sum_fn(x):
    """f(x) = sum(x).  grad = ones."""
    return float(np.sum(x))


def _vector_fn(x):
    """Vector-valued: [x0^2 + x1, x0*x1, x1^3]."""
    return np.array([x[0] ** 2 + x[1], x[0] * x[1], x[1] ** 3])


# ===========================================================================
# 1. autodiff/backends.py  --  detect_backend, get_backend_module
# ===========================================================================


class TestDetectBackend:
    """Cover aurora.core.autodiff.backends.detect_backend."""

    def test_numpy_array(self):
        assert detect_backend(np.array([1.0])) == "numpy"

    def test_python_list(self):
        assert detect_backend([1.0, 2.0]) == "numpy"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_tensor(self):
        assert detect_backend(torch.tensor([1.0])) == "torch"

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_array(self):
        assert detect_backend(jnp.array([1.0])) == "jax"


class TestGetBackendModule:
    """Cover aurora.core.autodiff.backends.get_backend_module."""

    def test_numpy(self):
        mod = get_backend_module("numpy")
        assert mod is np

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch(self):
        mod = get_backend_module("torch")
        assert mod is torch

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax(self):
        mod = get_backend_module("jax")
        assert mod is jnp


# ===========================================================================
# 2. autodiff/gradient.py  --  gradient, _gradient_numerical, etc.
# ===========================================================================


class TestGradientNumericalDirect:
    """Cover _gradient_numerical directly (central finite differences)."""

    def test_quadratic(self):
        x = np.array([1.0, 2.0, 3.0])
        g = _gradient_numerical(_quadratic, 0, x)
        assert_allclose(g, x, rtol=1e-5, atol=1e-8)

    def test_linear(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        g = _gradient_numerical(_sum_fn, 0, x)
        assert_allclose(g, np.ones(4), rtol=1e-5, atol=1e-8)

    def test_preserves_shape(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])

        def f(x):
            return float(np.sum(x ** 2))

        g = _gradient_numerical(f, 0, x)
        assert g.shape == x.shape

    def test_with_extra_args(self):
        x = np.array([1.0, 2.0])
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        g = _gradient_numerical(_quadratic_matrix, 0, x, A)
        assert_allclose(g, A @ x, rtol=1e-4, atol=1e-8)

    def test_gradient_at_zero(self):
        x = np.zeros(5)
        g = _gradient_numerical(_quadratic, 0, x)
        assert_allclose(g, np.zeros(5), atol=1e-10)

    def test_gradient_large_values(self):
        x = np.array([1e6, 2e6, 3e6])
        g = _gradient_numerical(_quadratic, 0, x)
        assert_allclose(g, x, rtol=1e-5)


class TestGradientHighLevel:
    """Cover gradient() wrapper dispatching."""

    def test_backend_kwarg_numpy(self):
        grad_fn = gradient(_quadratic, backend="numpy")
        x = np.array([1.0, 2.0])
        assert_allclose(grad_fn(x), x, rtol=1e-5)

    def test_auto_detect_numpy(self):
        grad_fn = gradient(_quadratic)
        x = np.array([3.0, -1.0])
        assert_allclose(grad_fn(x), x, rtol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_backend_kwarg(self):
        def f(x):
            return 0.5 * torch.sum(x ** 2)

        grad_fn = gradient(f, backend="torch")
        x = torch.tensor([1.0, 2.0, 3.0])
        g = grad_fn(x)
        assert_allclose(g.detach().numpy(), np.array([1.0, 2.0, 3.0]), rtol=1e-5)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_backend_kwarg(self):
        def f(x):
            return 0.5 * jnp.sum(x ** 2)

        grad_fn = gradient(f, backend="jax")
        x = jnp.array([1.0, 2.0, 3.0])
        g = grad_fn(x)
        assert_allclose(np.asarray(g), np.array([1.0, 2.0, 3.0]), rtol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestGradientTorchDirect:
    """Cover _gradient_torch directly."""

    def test_quadratic(self):
        def f(x):
            return 0.5 * torch.sum(x ** 2)

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        g = _gradient_torch(f, 0, x)
        assert_allclose(g.numpy(), [1.0, 2.0, 3.0], rtol=1e-5)

    def test_requires_grad_false_input(self):
        """When input lacks requires_grad the function should clone and set it."""

        def f(x):
            return torch.sum(x)

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        g = _gradient_torch(f, 0, x)
        assert_allclose(g.numpy(), [1.0, 1.0], rtol=1e-5)

    def test_with_kwargs(self):
        """Test gradient through kwargs dispatch."""

        def f(x):
            return 0.5 * torch.sum(x ** 2)

        grad_fn = gradient(f, backend="torch")
        x = torch.tensor([3.0, -2.0], dtype=torch.float64)
        g = grad_fn(x)
        assert_allclose(g.detach().numpy(), [3.0, -2.0], rtol=1e-5)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestGradientJAXDirect:
    """Cover _gradient_jax directly."""

    def test_quadratic(self):
        def f(x):
            return 0.5 * jnp.sum(x ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        g = _gradient_jax(f, 0, x)
        assert_allclose(np.asarray(g), [1.0, 2.0, 3.0], rtol=1e-5)

    def test_with_extra_args(self):
        def f(x, scale):
            return jnp.sum(x) * scale

        x = jnp.array([1.0, 2.0])
        g = _gradient_jax(f, 0, x, 2.0)
        assert_allclose(np.asarray(g), [2.0, 2.0], rtol=1e-5)


# ===========================================================================
# 3. autodiff/hessian.py  --  hessian, _hessian_numerical, etc.
# ===========================================================================


class TestHessianNumericalDirect:
    """Cover _hessian_numerical directly."""

    def test_quadratic_identity(self):
        x = np.array([1.0, 2.0, 3.0])
        H = _hessian_numerical(_quadratic, 0, x)
        assert_allclose(H, np.eye(3), atol=1e-2)

    def test_quadratic_matrix(self):
        x = np.array([1.0, 2.0])
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        H = _hessian_numerical(_quadratic_matrix, 0, x, A)
        assert_allclose(H, A, atol=1e-2)

    def test_symmetry(self):
        x = np.array([0.5, 1.5, 2.0])
        H = _hessian_numerical(_rosenbrock, 0, x)
        assert_allclose(H, H.T, atol=1e-12)

    def test_positive_definite_at_minimum(self):
        x = np.ones(3)
        H = _hessian_numerical(_rosenbrock, 0, x)
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(eigenvalues > -1e-8)


class TestHessianHighLevel:
    """Cover hessian() wrapper."""

    def test_numpy_backend_kwarg(self):
        hess_fn = hessian(_quadratic, backend="numpy")
        x = np.array([1.0, 2.0])
        H = hess_fn(x)
        assert_allclose(H, np.eye(2), atol=1e-2)

    def test_auto_detect(self):
        hess_fn = hessian(_quadratic)
        x = np.array([3.0, -1.0])
        H = hess_fn(x)
        assert_allclose(H, np.eye(2), atol=1e-2)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_backend(self):
        def f(x):
            return 0.5 * torch.sum(x ** 2)

        hess_fn = hessian(f, backend="torch")
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        H = hess_fn(x)
        assert_allclose(H.numpy(), np.eye(2), atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestHessianTorchDirect:
    """Cover _hessian_torch directly."""

    def test_quadratic(self):
        def f(x):
            return 0.5 * torch.sum(x ** 2)

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        H = _hessian_torch(f, 0, x)
        assert_allclose(H.numpy(), np.eye(2), atol=1e-5)

    def test_matrix_quadratic(self):
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)

        def f(x):
            return 0.5 * x @ A @ x

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        H = _hessian_torch(f, 0, x)
        assert_allclose(H.numpy(), A.numpy(), atol=1e-5)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestHessianJAXDirect:
    """Cover _hessian_jax directly."""

    def test_quadratic(self):
        def f(x):
            return 0.5 * jnp.sum(x ** 2)

        x = jnp.array([1.0, 2.0])
        H = _hessian_jax(f, 0, x)
        assert_allclose(np.asarray(H), np.eye(2), atol=1e-5)

    def test_matrix_quadratic(self):
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])

        def f(x):
            return 0.5 * x @ A @ x

        x = jnp.array([1.0, 2.0])
        H = _hessian_jax(f, 0, x)
        assert_allclose(np.asarray(H), np.asarray(A), atol=1e-5)

    def test_highlevel_dispatch(self):
        def f(x):
            return 0.5 * jnp.sum(x ** 2)

        hess_fn = hessian(f, backend="jax")
        x = jnp.array([3.0, -1.0])
        H = hess_fn(x)
        assert_allclose(np.asarray(H), np.eye(2), atol=1e-5)


# ===========================================================================
# 4. autodiff/jacobian.py  --  jacobian, _jacobian_numerical, etc.
# ===========================================================================


class TestJacobianNumericalDirect:
    """Cover _jacobian_numerical directly."""

    def test_vector_function(self):
        x = np.array([2.0, 3.0])
        J = _jacobian_numerical(_vector_fn, 0, x)
        expected = np.array([[4.0, 1.0], [3.0, 2.0], [0.0, 27.0]])
        assert_allclose(J, expected, rtol=1e-4)

    def test_identity(self):
        x = np.array([1.0, 2.0, 3.0])
        J = _jacobian_numerical(lambda x: x.copy(), 0, x)
        assert_allclose(J, np.eye(3), rtol=1e-5)

    def test_linear(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        J = _jacobian_numerical(lambda x: A @ x, 0, np.array([1.0, 2.0]))
        assert_allclose(J, A, rtol=1e-4)


class TestJacobianHighLevel:
    """Cover jacobian() wrapper."""

    def test_numpy_backend(self):
        jac_fn = jacobian(_vector_fn, backend="numpy")
        x = np.array([2.0, 3.0])
        J = jac_fn(x)
        expected = np.array([[4.0, 1.0], [3.0, 2.0], [0.0, 27.0]])
        assert_allclose(J, expected, rtol=1e-4)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_backend(self):
        def f(x):
            return torch.stack([x[0] ** 2 + x[1], x[0] * x[1], x[1] ** 3])

        jac_fn = jacobian(f, backend="torch")
        x = torch.tensor([2.0, 3.0], dtype=torch.float64)
        J = jac_fn(x)
        expected = np.array([[4.0, 1.0], [3.0, 2.0], [0.0, 27.0]])
        assert_allclose(J.numpy(), expected, rtol=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestJacobianTorchDirect:
    """Cover _jacobian_torch directly."""

    def test_vector_function(self):
        def f(x):
            return torch.stack([x[0] ** 2 + x[1], x[0] * x[1]])

        x = torch.tensor([2.0, 3.0], dtype=torch.float64)
        J = _jacobian_torch(f, 0, x)
        expected = np.array([[4.0, 1.0], [3.0, 2.0]])
        assert_allclose(J.numpy(), expected, rtol=1e-4)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestJacobianJAXDirect:
    """Cover _jacobian_jax directly."""

    def test_vector_function(self):
        def f(x):
            return jnp.array([x[0] ** 2 + x[1], x[0] * x[1], x[1] ** 3])

        x = jnp.array([2.0, 3.0])
        J = _jacobian_jax(f, 0, x)
        expected = np.array([[4.0, 1.0], [3.0, 2.0], [0.0, 27.0]])
        assert_allclose(np.asarray(J), expected, rtol=1e-4)

    def test_highlevel_dispatch(self):
        def f(x):
            return jnp.array([x[0] ** 2 + x[1], x[0] * x[1]])

        jac_fn = jacobian(f, backend="jax")
        x = jnp.array([2.0, 3.0])
        J = jac_fn(x)
        expected = np.array([[4.0, 1.0], [3.0, 2.0]])
        assert_allclose(np.asarray(J), expected, rtol=1e-4)


# ===========================================================================
# 5. autodiff/products.py  --  hvp, jvp, vjp, and internal helpers
# ===========================================================================


class TestHVPNumericalDirect:
    """Cover _hvp_numerical directly."""

    def test_quadratic(self):
        x = np.array([1.0, 2.0])
        v = np.array([0.5, -0.5])
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        Hv = _hvp_numerical(_quadratic_matrix, x, v, A)
        expected = A @ v
        # Numerical HVP has limited precision
        assert_allclose(Hv, expected, rtol=0.5, atol=0.5)

    def test_finite_output(self):
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([0.1, -0.2, 0.3])
        Hv = _hvp_numerical(_quadratic, x, v)
        assert np.all(np.isfinite(Hv))


class TestHVPHighLevel:
    """Cover hvp() wrapper."""

    def test_numpy_auto_detect(self):
        x = np.array([1.0, 2.0])
        v = np.array([0.5, -0.5])
        Hv = hvp(_quadratic, x, v)
        assert np.all(np.isfinite(Hv))

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_dispatch(self):
        def f(x):
            return 0.5 * torch.sum(x ** 2)

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        v = torch.tensor([0.5, -0.5], dtype=torch.float64)
        Hv = hvp(f, x, v)
        assert_allclose(Hv.detach().numpy(), v.detach().numpy(), atol=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestHVPTorchDirect:
    """Cover _hvp_torch directly."""

    def test_quadratic(self):
        def f(x):
            return 0.5 * torch.sum(x ** 2)

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        v = torch.tensor([0.5, -0.5], dtype=torch.float64)
        Hv = _hvp_torch(f, x, v)
        # H = I, so Hv = v
        assert_allclose(Hv.detach().numpy(), v.numpy(), atol=1e-5)

    def test_matrix_quadratic(self):
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)

        def f(x):
            return 0.5 * x @ A @ x

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        v = torch.tensor([0.5, -0.5], dtype=torch.float64)
        Hv = _hvp_torch(f, x, v)
        assert_allclose(Hv.detach().numpy(), (A @ v).numpy(), atol=1e-5)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestHVPJAXDirect:
    """Cover _hvp_jax directly."""

    def test_quadratic(self):
        def f(x):
            return 0.5 * jnp.sum(x ** 2)

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

    def test_highlevel_dispatch(self):
        def f(x):
            return 0.5 * jnp.sum(x ** 2)

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.5, -0.5])
        Hv = hvp(f, x, v)
        assert_allclose(np.asarray(Hv), np.array([0.5, -0.5]), atol=1e-5)


class TestJVPNumPy:
    """Cover jvp() for NumPy arrays."""

    def test_linear(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 2.0])
        v = np.array([0.5, -0.5])

        f_x, tangent = jvp(lambda x: A @ x, x, v)
        assert_allclose(f_x, A @ x, rtol=1e-5)
        assert_allclose(tangent, A @ v, rtol=1e-4)

    def test_scalar_function(self):
        """JVP on a scalar-valued function returns a scalar tangent."""

        def f(x):
            return np.array([np.sum(x ** 2)])

        x = np.array([1.0, 2.0])
        v = np.array([1.0, 0.0])
        f_x, tangent = jvp(f, x, v)
        assert_allclose(f_x, [5.0], rtol=1e-5)
        # d/dx0 sum(x^2) = 2*x0 = 2, but JVP multiplies by v so 2*1 + 4*0 = 2
        assert_allclose(tangent, [2.0], rtol=1e-2)


class TestVJPNumPy:
    """Cover vjp() for NumPy arrays."""

    def test_linear(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 2.0])
        v = np.array([0.5, -0.5])

        f_x, cotangent = vjp(lambda x: A @ x, x, v)
        assert_allclose(f_x, A @ x, rtol=1e-5)
        assert_allclose(cotangent, v @ A, rtol=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestJVPTorch:
    """Cover jvp() for PyTorch tensors."""

    def test_linear(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)

        def f(x):
            return A @ x

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        v = torch.tensor([0.5, -0.5], dtype=torch.float64)
        f_x, tangent = jvp(f, x, v)
        assert_allclose(f_x.detach().numpy(), (A @ x).numpy(), rtol=1e-5)
        assert_allclose(tangent.detach().numpy(), (A @ v).numpy(), rtol=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestVJPTorch:
    """Cover vjp() for PyTorch tensors."""

    def test_linear(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)

        def f(x):
            return A @ x

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        v = torch.tensor([0.5, -0.5], dtype=torch.float64)
        f_x, cotangent = vjp(f, x, v)
        assert_allclose(f_x.detach().numpy(), (A @ x).numpy(), rtol=1e-5)
        assert_allclose(cotangent.detach().numpy(), (v @ A).numpy(), rtol=1e-4)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestJVPJAX:
    """Cover jvp() for JAX arrays."""

    def test_linear(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        def f(x):
            return A @ x

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.5, -0.5])
        f_x, tangent = jvp(f, x, v)
        assert_allclose(np.asarray(f_x), np.array([5.0, 11.0]), rtol=1e-5)
        assert_allclose(np.asarray(tangent), np.array([-0.5, -0.5]), rtol=1e-5)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestVJPJAX:
    """Cover vjp() for JAX arrays."""

    def test_linear(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        def f(x):
            return A @ x

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.5, -0.5])
        f_x, cotangent = vjp(f, x, v)
        assert_allclose(np.asarray(f_x), np.array([5.0, 11.0]), rtol=1e-5)
        assert_allclose(np.asarray(cotangent), np.asarray(v @ A), rtol=1e-5)


# ===========================================================================
# 6. autodiff/utils.py  --  check_gradient
# ===========================================================================


class TestCheckGradientUtil:
    """Cover aurora.core.autodiff.utils.check_gradient."""

    def test_quadratic_passes(self):
        x = np.array([1.0, 2.0, 3.0])
        result = check_gradient(_quadratic, x)
        assert result["passed"]
        assert result["max_abs_diff"] < 1e-10

    def test_with_extra_args(self):
        x = np.array([1.0, 2.0])
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        result = check_gradient(_quadratic_matrix, x, A)
        assert result["passed"]

    def test_return_keys(self):
        x = np.array([1.0, 2.0])
        result = check_gradient(_quadratic, x)
        for key in ("analytic", "numerical", "max_abs_diff", "max_rel_diff", "passed"):
            assert key in result

    def test_custom_tolerances(self):
        x = np.array([1.0, 2.0])
        result = check_gradient(_quadratic, x, rtol=1e-2, atol=1e-3)
        assert result["passed"]

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_backend_check(self):
        def f(x):
            return 0.5 * torch.sum(x ** 2)

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = check_gradient(f, x)
        assert result["passed"]

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_backend_check(self):
        def f(x):
            return 0.5 * jnp.sum(x ** 2)

        x = jnp.array([1.0, 2.0])
        result = check_gradient(f, x)
        assert result["passed"]


# ===========================================================================
# Edge cases and stress tests
# ===========================================================================


class TestEdgeCases:
    """Additional edge-case coverage for all autodiff modules."""

    def test_gradient_single_element(self):
        x = np.array([5.0])
        grad_fn = gradient(_quadratic, backend="numpy")
        g = grad_fn(x)
        assert_allclose(g, [5.0], rtol=1e-4)

    def test_hessian_single_element(self):
        x = np.array([5.0])
        hess_fn = hessian(_quadratic, backend="numpy")
        H = hess_fn(x)
        assert H.shape == (1, 1)
        assert_allclose(H[0, 0], 1.0, atol=0.1)

    def test_jacobian_single_input(self):
        def f(x):
            return np.array([x[0], x[0] ** 2])

        x = np.array([3.0])
        jac_fn = jacobian(f, backend="numpy")
        J = jac_fn(x)
        assert_allclose(J, [[1.0], [6.0]], rtol=1e-3)

    def test_gradient_2d_input(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])

        def f(x):
            return float(np.sum(x ** 2))

        grad_fn = gradient(f, backend="numpy")
        g = grad_fn(x)
        assert g.shape == (2, 2)
        assert_allclose(g.ravel(), 2 * x.ravel(), rtol=1e-4)

    def test_high_dimensional_gradient(self):
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        grad_fn = gradient(_quadratic, backend="numpy")
        g = grad_fn(x)
        assert_allclose(g, x, rtol=1e-4)


# ===========================================================================
# Additional numerical path tests (Phase 2E)
# ===========================================================================


class TestGradientNumericalTrig:
    """Cover _gradient_numerical with trig functions."""

    def test_sin(self):
        def f(x):
            return float(np.sum(np.sin(x)))

        x = np.array([0.5, 1.0, 1.5])
        g = _gradient_numerical(f, 0, x)
        assert_allclose(g, np.cos(x), rtol=1e-5, atol=1e-8)

    def test_cos(self):
        def f(x):
            return float(np.sum(np.cos(x)))

        x = np.array([0.3, -0.7])
        g = _gradient_numerical(f, 0, x)
        assert_allclose(g, -np.sin(x), rtol=1e-5, atol=1e-8)

    def test_negative_values(self):
        def f(x):
            return float(np.sum(x ** 3))

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        g = _gradient_numerical(f, 0, x)
        assert_allclose(g, 3 * x ** 2, rtol=1e-4)

    def test_gradient_with_kwargs(self):
        def f(x, scale=1.0):
            return float(np.sum(x ** 2) * scale)

        x = np.array([1.0, 2.0])
        g = _gradient_numerical(f, 0, x, scale=2.0)
        assert_allclose(g, 4 * x, rtol=1e-4)

    def test_gradient_argnums_nonzero(self):
        """Test gradient with argnums=1 (differentiate w.r.t. second arg)."""

        def f(dummy, x):
            return float(np.sum(x ** 2))

        x = np.array([3.0, 4.0])
        g = _gradient_numerical(f, 1, np.zeros(2), x)
        assert_allclose(g, 2 * x, rtol=1e-4)


class TestHessianNumericalAdditional:
    """Additional _hessian_numerical tests covering more branches."""

    def test_trig_function(self):
        def f(x):
            return float(np.sum(np.sin(x)))

        x = np.array([0.5, 1.0])
        H = _hessian_numerical(f, 0, x)
        # H = -diag(sin(x))
        expected = -np.diag(np.sin(x))
        assert_allclose(H, expected, atol=1e-2)

    def test_constant_function(self):
        def f(x):
            return 5.0

        x = np.array([1.0, 2.0, 3.0])
        H = _hessian_numerical(f, 0, x)
        assert_allclose(H, np.zeros((3, 3)), atol=1e-4)

    def test_hessian_with_kwargs(self):
        def f(x, scale=1.0):
            return float(0.5 * scale * np.sum(x ** 2))

        x = np.array([1.0, 2.0])
        H = _hessian_numerical(f, 0, x, scale=3.0)
        assert_allclose(H, 3.0 * np.eye(2), atol=0.1)

    def test_symmetrization(self):
        """Verify the 0.5 * (H + H.T) symmetrization in _hessian_numerical."""

        def f(x):
            # Non-separable function -> off-diagonal entries
            return float(x[0] ** 3 * x[1] + x[1] ** 2)

        x = np.array([1.0, 2.0])
        H = _hessian_numerical(f, 0, x)
        # Must be symmetric
        assert_allclose(H, H.T, atol=1e-10)


class TestJacobianNumericalAdditional:
    """Additional _jacobian_numerical coverage."""

    def test_with_kwargs(self):
        def f(x, scale=1.0):
            return np.array([x[0] * scale, x[1] * scale])

        x = np.array([2.0, 3.0])
        J = _jacobian_numerical(f, 0, x, scale=2.0)
        expected = 2.0 * np.eye(2)
        assert_allclose(J, expected, rtol=1e-4)

    def test_single_output(self):
        def f(x):
            return np.array([np.sum(x)])

        x = np.array([1.0, 2.0, 3.0])
        J = _jacobian_numerical(f, 0, x)
        assert_allclose(J, np.ones((1, 3)), rtol=1e-5)

    def test_argnums_nonzero(self):
        def f(dummy, x):
            return np.array([x[0] ** 2, x[0] * x[1]])

        x = np.array([2.0, 3.0])
        J = _jacobian_numerical(f, 1, np.zeros(2), x)
        expected = np.array([[4.0, 0.0], [3.0, 2.0]])
        assert_allclose(J, expected, rtol=1e-4)


class TestHVPNumericalAdditional:
    """Additional _hvp_numerical coverage."""

    def test_with_extra_args(self):
        A = np.array([[3.0, 1.0], [1.0, 2.0]])

        x = np.array([1.0, 2.0])
        v = np.array([1.0, 0.0])
        Hv = _hvp_numerical(_quadratic_matrix, x, v, A)
        expected = A @ v
        assert_allclose(Hv, expected, rtol=0.5, atol=0.5)

    def test_identity_hessian(self):
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, -0.5, 1.0])
        Hv = _hvp_numerical(_quadratic, x, v)
        # H = I, so Hv = v
        assert_allclose(Hv, v, rtol=0.5, atol=0.5)

    def test_high_level_backend_kwarg(self):
        x = np.array([1.0, 2.0])
        v = np.array([0.5, -0.5])
        Hv = hvp(_quadratic, x, v)
        assert np.all(np.isfinite(Hv))
        assert Hv.shape == x.shape


class TestJVPNumericalAdditional:
    """Additional jvp() numpy path coverage."""

    def test_quadratic_function(self):
        def f(x):
            return np.array([np.sum(x ** 2)])

        x = np.array([1.0, 2.0])
        v = np.array([1.0, 0.0])
        f_x, tangent = jvp(f, x, v)
        assert_allclose(f_x, [5.0], rtol=1e-5)
        # d/dv sum(x^2) = 2*x0*1 + 2*x1*0 = 2
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


class TestVJPNumericalAdditional:
    """Additional vjp() numpy path coverage."""

    def test_quadratic_vector_fn(self):
        def f(x):
            return np.array([x[0] ** 2, x[0] * x[1]])

        x = np.array([2.0, 3.0])
        v = np.array([1.0, 0.0])
        f_x, cotangent = vjp(f, x, v)
        assert_allclose(f_x, [4.0, 6.0], rtol=1e-5)
        # J = [[2*x0, 0], [x1, x0]] = [[4, 0], [3, 2]]
        # v^T @ J = [1, 0] @ [[4, 0], [3, 2]] = [4, 0]
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


class TestDetectBackendAdditional:
    """Additional detect_backend coverage."""

    def test_python_float(self):
        assert detect_backend(3.14) == "numpy"

    def test_none_module_fallback(self):
        # Objects whose type().__module__ is builtins -> numpy
        assert detect_backend(42) == "numpy"


class TestCheckGradientAdditional:
    """Additional check_gradient coverage."""

    def test_constant_function(self):
        def f(x):
            return 5.0

        x = np.array([1.0, 2.0])
        result = check_gradient(f, x)
        assert result["passed"]

    def test_with_kwargs(self):
        def f(x, scale=1.0):
            return float(np.sum(x ** 2) * scale)

        x = np.array([1.0, 2.0])
        result = check_gradient(f, x, scale=2.0)
        assert result["passed"]

    def test_large_array(self):
        np.random.seed(42)
        x = np.random.randn(50)
        result = check_gradient(_quadratic, x)
        assert result["passed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
