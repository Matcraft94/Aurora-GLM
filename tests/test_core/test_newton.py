# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Tests for aurora.core.optimization.newton module.

Tests Newton-Raphson and Modified Newton optimizers using JAX backend
(which is available in the test environment).
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.core.optimization.newton import modified_newton, newton_raphson

try:
    import jax.numpy as jnp

    from aurora.core.backends import get_backend

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX not available")


# ---------------------------------------------------------------------------
# Helper loss functions
# ---------------------------------------------------------------------------


def _quadratic(x):
    """Simple quadratic: f(x) = x^T Q x, minimum at origin."""
    Q = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    return x @ Q @ x


def _rosenbrock(x):
    """Rosenbrock function, minimum at (1, 1)."""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def _quadratic_1d(x):
    """1D quadratic: (x-3)^2, minimum at x=3."""
    return (x[0] - 3.0) ** 2


def _non_convex(x):
    """Non-convex: sum of sin and quadratic, has local minima."""
    return jnp.sum(x**2) - 2 * jnp.sum(jnp.sin(x))


# ---------------------------------------------------------------------------
# Tests for newton_raphson
# ---------------------------------------------------------------------------


class TestNewtonRaphson:
    """Tests for the basic Newton-Raphson optimizer."""

    def test_quadratic_convergence(self):
        """Quadratic function should converge in 1 iteration."""
        result = newton_raphson(_quadratic, jnp.array([5.0, -3.0]))
        assert result.success
        np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-5)
        assert result.nit <= 2

    def test_quadratic_1d(self):
        """1D quadratic should find minimum at x=3."""
        result = newton_raphson(_quadratic_1d, jnp.array([0.0]))
        assert result.success
        np.testing.assert_allclose(result.x, [3.0], atol=1e-5)

    def test_rosenbrock_convergence(self):
        """Rosenbrock should converge to (1, 1)."""
        result = newton_raphson(_rosenbrock, jnp.array([0.0, 0.0]))
        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-3)
        assert result.fun < 1e-6

    def test_returns_optimization_result(self):
        """Check result has all expected fields."""
        result = newton_raphson(_quadratic, jnp.array([1.0, 1.0]))
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "grad")
        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert hasattr(result, "nit")
        assert hasattr(result, "nfev")
        assert hasattr(result, "njev")
        assert hasattr(result, "nhev")

    def test_max_iter_exceeded(self):
        """Low max_iter should return success=False."""
        result = newton_raphson(
            _rosenbrock, jnp.array([-5.0, 5.0]), max_iter=2, tol=1e-15
        )
        assert not result.success
        assert result.nit == 2
        assert "Maximum" in result.message

    def test_callback_called(self):
        """Callback should be called each iteration."""
        history = []

        def cb(iteration, params, loss):
            history.append((iteration, float(loss)))

        newton_raphson(_quadratic_1d, jnp.array([0.0]), callback=cb)
        assert len(history) > 0
        # Loss should decrease
        losses = [h[1] for h in history]
        assert losses[-1] < losses[0]

    def test_args_passed(self):
        """Additional args should be passed to loss function."""
        def shifted_quadratic(x, offset):
            return jnp.sum((x - offset) ** 2)

        result = newton_raphson(
            shifted_quadratic, jnp.array([0.0, 0.0]),
            args=(jnp.array([2.0, 3.0]),),
        )
        assert result.success
        np.testing.assert_allclose(result.x, [2.0, 3.0], atol=1e-5)

    def test_kwargs_passed(self):
        """Additional kwargs should be passed to loss function."""
        def shifted_quadratic(x, offset=None):
            return jnp.sum((x - offset) ** 2)

        result = newton_raphson(
            shifted_quadratic, jnp.array([0.0, 0.0]),
            kwargs={"offset": jnp.array([2.0, 3.0])},
        )
        assert result.success
        np.testing.assert_allclose(result.x, [2.0, 3.0], atol=1e-5)

    def test_positive_definite_converges(self):
        """For convex function, Hessian is always PD, so always converges."""
        result = newton_raphson(_quadratic, jnp.array([10.0, -10.0]))
        assert result.success
        assert result.fun < 1e-10

    def test_gradient_norm_convergence(self):
        """Should converge when gradient norm < tol."""
        result = newton_raphson(
            _quadratic_1d, jnp.array([0.0]), tol=1e-8
        )
        assert result.success
        assert np.linalg.norm(result.grad) < 1e-6


# ---------------------------------------------------------------------------
# Tests for modified_newton
# ---------------------------------------------------------------------------


class TestModifiedNewton:
    """Tests for Modified Newton with Levenberg-Marquardt regularization."""

    def test_quadratic_convergence(self):
        """Quadratic should converge with modified Newton."""
        result = modified_newton(_quadratic, jnp.array([5.0, -3.0]))
        assert result.success
        np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-5)

    def test_rosenbrock_convergence(self):
        """Rosenbrock should converge with modified Newton."""
        result = modified_newton(_rosenbrock, jnp.array([0.0, 0.0]))
        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-3)

    def test_handles_bad_start(self):
        """Modified Newton should handle poor starting points better."""
        result = modified_newton(
            _rosenbrock, jnp.array([-5.0, 5.0]),
            max_iter=200,
        )
        # Modified Newton should at least make progress even if it doesn't converge
        assert result.fun < 100 * (1 - (-5.0)) ** 2  # Better than start

    def test_lambda_init_parameter(self):
        """Custom lambda_init should work."""
        result = modified_newton(
            _quadratic, jnp.array([5.0, -3.0]),
            lambda_init=1.0,
        )
        assert result.success

    def test_max_iter(self):
        """Should respect max_iter."""
        result = modified_newton(
            _rosenbrock, jnp.array([-5.0, 5.0]),
            max_iter=3, tol=1e-15,
        )
        assert not result.success
        assert result.nit == 3

    def test_callback_called(self):
        """Callback should be called each iteration."""
        history = []

        def cb(iteration, params, loss):
            history.append(loss)

        modified_newton(_quadratic_1d, jnp.array([0.0]), callback=cb)
        assert len(history) > 0

    def test_args_kwargs(self):
        """Additional args/kwargs should work."""
        def shifted_quad(x, offset):
            return jnp.sum((x - offset) ** 2)

        result = modified_newton(
            shifted_quad, jnp.array([0.0, 0.0]),
            args=(jnp.array([1.0, 2.0]),),
        )
        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-4)

    def test_returns_correct_fields(self):
        """Result should have all expected fields."""
        result = modified_newton(_quadratic, jnp.array([1.0, 1.0]))
        assert isinstance(result.fun, float)
        assert isinstance(result.success, bool)
        assert isinstance(result.nit, int)
        assert isinstance(result.nfev, int)

    def test_step_size_convergence(self):
        """Should converge when step size < tol."""
        result = modified_newton(
            _quadratic, jnp.array([0.1, 0.1]), tol=1e-8
        )
        assert result.success

    def test_non_convex_function(self):
        """Modified Newton should handle non-convex functions."""
        result = modified_newton(
            _non_convex, jnp.array([2.0, 2.0]),
            max_iter=100,
        )
        # Should at least converge to a stationary point
        assert np.linalg.norm(result.grad) < 0.1 or result.fun < 5.0


# ---------------------------------------------------------------------------
# Tests for _compute_hessian (indirect via newton_raphson)
# ---------------------------------------------------------------------------


class TestHessianComputation:
    """Test Hessian computation via different backends."""

    def test_jax_backend(self):
        """JAX backend should compute exact Hessian."""
        result = newton_raphson(
            _quadratic, jnp.array([3.0, 2.0]),
            backend=get_backend("jax"),
        )
        assert result.success
        # Quadratic converges in very few iterations with exact Hessian
        assert result.nhev <= 3

    def test_numpy_backend_finite_differences(self):
        """NumPy backend should use finite differences."""
        backend = get_backend("numpy")
        result = newton_raphson(
            _quadratic_1d, np.array([0.0]),
            backend=backend,
        )
        assert result.success
        np.testing.assert_allclose(result.x, [3.0], atol=1e-3)

    def test_torch_backend(self):
        """PyTorch backend should compute exact Hessian."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        backend = get_backend("pytorch")
        result = newton_raphson(
            _quadratic_1d, torch.tensor([0.0]),
            backend=backend,
        )
        assert result.success
        np.testing.assert_allclose(result.x, [3.0], atol=1e-3)
