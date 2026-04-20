# SPDX-License-Identifier: MIT
"""Additional coverage tests for aurora.core.autodiff.gradient module.

Covers backend dispatch branches with mocked detect_backend, _gradient_torch
non-Tensor return path, and edge cases for _gradient_numerical.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aurora.core.autodiff.gradient import (
    _gradient_numerical,
    _gradient_torch,
    gradient,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False


def _quadratic(x):
    return 0.5 * np.sum(x**2)


# ===========================================================================
# gradient() high-level dispatch with mocked backends
# ===========================================================================


class TestGradientDispatchMocked:
    """Cover gradient() dispatch via mocked detect_backend."""

    def test_numpy_dispatch_mocked(self):
        with patch("aurora.core.autodiff.gradient.detect_backend", return_value="numpy"):
            grad_fn = gradient(_quadratic, backend=None)
            x = np.array([1.0, 2.0])
            g = grad_fn(x)
        assert_allclose(g, x, rtol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_dispatch_mocked(self):
        def f(x):
            return 0.5 * torch.sum(x**2)

        with patch("aurora.core.autodiff.gradient.detect_backend", return_value="torch"):
            grad_fn = gradient(f, backend=None)
            x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
            g = grad_fn(x)
        assert_allclose(g.detach().numpy(), [1.0, 2.0, 3.0], rtol=1e-5)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_dispatch_mocked(self):
        def f(x):
            return 0.5 * jnp.sum(x**2)

        with patch("aurora.core.autodiff.gradient.detect_backend", return_value="jax"):
            grad_fn = gradient(f, backend=None)
            x = jnp.array([1.0, 2.0, 3.0])
            g = grad_fn(x)
        assert_allclose(np.asarray(g), [1.0, 2.0, 3.0], rtol=1e-5)


# ===========================================================================
# _gradient_torch — non-Tensor return path
# ===========================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestGradientTorchNonTensorReturn:
    """Cover the else branch where loss is not a torch.Tensor."""

    def test_scalar_return_uses_autograd_grad(self):
        call_count = [0]
        original_grad = torch.autograd.grad

        def fake_f(x):
            # Return a float, not a Tensor — triggers else branch
            return float(torch.sum(x**2).item())

        x = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)
        g = _gradient_torch(fake_f, 0, x)
        assert_allclose(g.detach().numpy(), [6.0, 8.0], rtol=1e-5)

    def test_requires_grad_false_clones(self):
        """Input without requires_grad should be cloned and tracked."""

        def f(x):
            return torch.sum(x**2)

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        g = _gradient_torch(f, 0, x)
        assert_allclose(g.numpy(), [2.0, 4.0], rtol=1e-5)


# ===========================================================================
# _gradient_numerical — additional edge cases
# ===========================================================================


class TestGradientNumericalAdditional:
    """Additional numerical gradient edge cases."""

    def test_with_kwargs(self):
        def f(x, scale=1.0):
            return float(np.sum(x**2) * scale)

        x = np.array([1.0, 2.0])
        g = _gradient_numerical(f, 0, x, scale=2.0)
        assert_allclose(g, 4 * x, rtol=1e-4)

    def test_2d_input(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])

        def f(x):
            return float(np.sum(x**2))

        g = _gradient_numerical(f, 0, x)
        assert g.shape == x.shape
        assert_allclose(g.ravel(), 2 * x.ravel(), rtol=1e-4)

    def test_single_element(self):
        x = np.array([5.0])
        g = _gradient_numerical(_quadratic, 0, x)
        assert_allclose(g, [5.0], rtol=1e-4)

    def test_argnums_nonzero(self):
        def f(dummy, x):
            return float(np.sum(x**2))

        x = np.array([3.0, 4.0])
        g = _gradient_numerical(f, 1, np.zeros(2), x)
        assert_allclose(g, [6.0, 8.0], rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
