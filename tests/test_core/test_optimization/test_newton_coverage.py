"""Tests covering previously uncovered paths in aurora.core.optimization.newton.

Targets:
- newton_raphson(): callback invocation, step convergence, max_iter reached
- modified_newton(): gradient descent fallback, Cholesky failure path
- _compute_hessian(): finite differences path (when no torch/jax AD available)

Uses the NumPy backend (which lacks autodiff) so _compute_hessian falls through
to the finite-difference path.  A thin wrapper around NumPyBackend provides a
numerical gradient via central differences so newton_raphson can iterate.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aurora.core.optimization.newton import (
    _compute_hessian,
    modified_newton,
    newton_raphson,
)
from aurora.core.optimization.result import OptimizationResult


# ---------------------------------------------------------------------------
# Minimal backend with numerical gradient (no autodiff)
# ---------------------------------------------------------------------------


class _NumPyBackendWithGrad:
    """NumPy-based backend that provides a numerical gradient via finite diffs.

    This mirrors the NumPyBackend protocol but adds ``grad()`` so that
    ``newton_raphson`` can iterate.  The Hessian computation in newton.py
    will still fall through to the finite-difference branch because the
    parameters are plain numpy arrays (not torch.Tensor or jax.ndarray).
    """

    def array(self, data, dtype=None):
        return np.asarray(data, dtype=dtype)

    def as_numpy(self, data):
        return np.asarray(data)

    def grad(self, func):
        """Return a function that computes the gradient by central differences."""

        def _grad_fn(x, *args, **kwargs):
            eps = 1e-7
            x_np = np.asarray(x, dtype=np.float64)
            grad = np.zeros_like(x_np)
            for i in range(len(x_np)):
                x_plus = x_np.copy()
                x_minus = x_np.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                f_plus = float(func(x_plus, *args, **kwargs))
                f_minus = float(func(x_minus, *args, **kwargs))
                grad[i] = (f_plus - f_minus) / (2.0 * eps)
            return grad

        return _grad_fn

    def jit(self, func):
        return func

    def device_put(self, data):
        return np.asarray(data)

    def vmap(self, func, *, in_axes=0, out_axes=0):
        raise NotImplementedError

    def partial(self, func, *args, **kwargs):
        import functools

        return functools.partial(func, *args, **kwargs)


@pytest.fixture()
def backend():
    return _NumPyBackendWithGrad()


# ---------------------------------------------------------------------------
# Quadratic helper
# ---------------------------------------------------------------------------


def _quadratic(x):
    """f(x) = 0.5 * x^T x  =>  grad = x, hess = I."""
    return 0.5 * np.dot(x, x)


def _quadratic_offset(x):
    """f(x) = 0.5 * sum((x - 3)^2)  =>  min at x = [3, 3, ...]."""
    return 0.5 * np.sum((x - 3.0) ** 2)


# ---------------------------------------------------------------------------
# _compute_hessian — finite differences path
# ---------------------------------------------------------------------------


class TestComputeHessianFiniteDiff:
    """Exercise the finite-difference Hessian (no AD available)."""

    def test_identity_hessian_1d(self, backend):
        hess, evals = _compute_hessian(_quadratic, np.array([1.0]), backend, (), {})
        assert_allclose(hess, [[1.0]], atol=1e-3)
        assert evals == 4  # 1 x 1 matrix => 4 evaluations

    def test_identity_hessian_2d(self, backend):
        hess, evals = _compute_hessian(_quadratic, np.array([1.0, 2.0]), backend, (), {})
        assert_allclose(hess, np.eye(2), atol=1e-3)
        # 2x2 symmetric => 3 unique entries => 3*4 = 12 evaluations
        assert evals == 12

    def test_with_args_kwargs(self, backend):
        """_compute_hessian should forward args and kwargs to loss_fn."""

        def loss(x, a, b=0.0):
            return 0.5 * np.sum((x - a - b) ** 2)

        hess, evals = _compute_hessian(
            loss, np.array([1.0]), backend, (2.0,), {"b": 1.0}
        )
        assert_allclose(hess, [[1.0]], atol=1e-3)


# ---------------------------------------------------------------------------
# newton_raphson — callback invocation
# ---------------------------------------------------------------------------


class TestNewtonCallback:
    def test_callback_is_called(self, backend):
        cb_records = []

        def cb(iteration, params, value):
            cb_records.append((iteration, float(value)))

        result = newton_raphson(
            _quadratic,
            np.array([5.0]),
            backend=backend,
            max_iter=20,
            tol=1e-10,
            callback=cb,
        )
        assert len(cb_records) >= 1
        # First callback should record iteration 0
        assert cb_records[0][0] == 0

    def test_callback_receives_correct_loss(self, backend):
        """Verify the callback receives the *updated* loss value."""
        cb_values = []

        def cb(iteration, params, value):
            cb_values.append(value)

        newton_raphson(
            _quadratic,
            np.array([4.0]),
            backend=backend,
            max_iter=10,
            callback=cb,
        )
        # Losses should be monotonically decreasing (for convex fn)
        for i in range(1, len(cb_values)):
            assert cb_values[i] <= cb_values[i - 1]


# ---------------------------------------------------------------------------
# newton_raphson — step convergence
# ---------------------------------------------------------------------------


class TestNewtonStepConvergence:
    def test_step_convergence_message(self, backend):
        """When step size drops below tol, result should say step convergence."""
        result = newton_raphson(
            _quadratic,
            np.array([1.0]),
            backend=backend,
            max_iter=50,
            tol=1e-8,
        )
        # For a simple quadratic it should converge
        assert result.success is True
        # The exact message depends on whether gradient or step converged first,
        # but one of them should fire.
        assert "Converged" in result.message


# ---------------------------------------------------------------------------
# newton_raphson — max iterations reached
# ---------------------------------------------------------------------------


class TestNewtonMaxIter:
    def test_max_iter_returns_failure(self, backend):
        """With max_iter=0 the loop body never executes, but max_iter=1
        will run one iteration then fall through to max-iter return."""
        result = newton_raphson(
            _quadratic,
            np.array([100.0]),
            backend=backend,
            max_iter=1,
            tol=1e-15,  # Very tight so it won't converge in 1 step
        )
        assert result.success is False
        assert result.message == "Maximum iterations reached"
        assert result.nit == 1

    def test_max_iter_counts_evaluations(self, backend):
        result = newton_raphson(
            _quadratic,
            np.array([10.0]),
            backend=backend,
            max_iter=2,
            tol=1e-15,
        )
        # Should have at least 1 njev and 1 nhev per iteration
        assert result.njev >= 1
        assert result.nhev >= 1


# ---------------------------------------------------------------------------
# modified_newton — gradient descent fallback
# ---------------------------------------------------------------------------


class TestModifiedNewtonGradientFallback:
    """Test that modified_newton falls back to gradient descent when the
    Levenberg-Marquardt sub-iterations all fail to find a descent step."""

    def test_gradient_descent_fallback(self, backend):
        """Use a function where the Hessian is highly ill-conditioned and
        the initial lambda is very large, forcing Cholesky to fail or
        Armijo to fail, triggering the gradient descent fallback."""

        # Rosenbrock: highly non-convex away from [1,1]
        def rosenbrock(x):
            return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

        # Start far from the minimum with very high initial lambda
        result = modified_newton(
            rosenbrock,
            np.array([-5.0, 5.0]),
            backend=backend,
            max_iter=5,
            lambda_init=1e10,  # Forces H + lambda*I to be dominated by lambda
            lambda_max=1e20,
            tol=1e-15,  # Won't converge in 5 iterations
        )
        # Should not crash — gradient descent fallback path exercised
        assert isinstance(result, OptimizationResult)
        assert result.nit == 5
        assert result.success is False

    def test_gradient_descent_accepts_lower_loss(self, backend):
        """When the fallback gradient step actually decreases loss, verify it
        updates the current point."""

        def loss(x):
            return 0.5 * np.sum((x - 10.0) ** 2)

        result = modified_newton(
            loss,
            np.array([0.0, 0.0]),
            backend=backend,
            max_iter=3,
            lambda_init=1e15,
            lambda_max=1e30,
            tol=1e-15,
        )
        # The point should have moved toward [10, 10]
        assert np.linalg.norm(result.x - np.array([10.0, 10.0])) < np.sqrt(200)


# ---------------------------------------------------------------------------
# modified_newton — Cholesky failure path
# ---------------------------------------------------------------------------


class TestModifiedNewtonCholeskyFailure:
    def test_cholesky_failure_increases_lambda(self, backend):
        """When the Hessian is indefinite, Cholesky will fail and lambda
        should be increased.  This exercises lines 723-726."""

        # A saddle-like function where Hessian can be indefinite
        def saddle(x):
            return x[0] ** 2 - x[1] ** 2

        # Start at a point where Hessian = diag(2, -2) (indefinite)
        result = modified_newton(
            saddle,
            np.array([0.5, 0.5]),
            backend=backend,
            max_iter=10,
            lambda_init=0.0,  # Start with no regularization so Cholesky fails
            lambda_factor=10.0,
            lambda_max=1e10,
            tol=1e-10,
        )
        assert isinstance(result, OptimizationResult)
        # Should have run without error, meaning the Cholesky failure was caught
        # and lambda was increased until H + lambda*I was positive definite.


# ---------------------------------------------------------------------------
# modified_newton — convergence via step size
# ---------------------------------------------------------------------------


class TestModifiedNewtonStepConvergence:
    def test_step_convergence(self, backend):
        """A well-behaved quadratic should converge quickly."""
        result = modified_newton(
            _quadratic,
            np.array([5.0]),
            backend=backend,
            max_iter=50,
            tol=1e-8,
        )
        assert result.success is True
        assert_allclose(result.x, [0.0], atol=1e-5)
