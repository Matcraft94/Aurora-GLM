"""Tests for the Newton-Raphson optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.core.optimization import newton_raphson

torch = pytest.importorskip("torch")


@pytest.fixture(name="backend")
def _backend_fixture(pytorch_backend):
    return pytorch_backend


def test_newton_quadratic_converges(backend):
    def loss(x: torch.Tensor) -> torch.Tensor:
        return ((x - 2.0) ** 2).sum()

    result = newton_raphson(loss, np.array([10.0], dtype=np.float32), backend=backend, max_iter=25)

    assert result.success is True
    assert np.allclose(result.x, np.array([2.0]), atol=1e-6)


def test_newton_invokes_callback(backend):
    called = {"count": 0}

    def callback(iteration: int, params, value: float) -> None:
        called["count"] += 1

    def loss(x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum()

    newton_raphson(
        loss,
        np.array([3.0, -1.0], dtype=np.float32),
        backend=backend,
        max_iter=10,
        callback=callback,
    )

    assert called["count"] > 0


def test_newton_handles_singular_hessian(backend):
    def linear_loss(x: torch.Tensor) -> torch.Tensor:
        return x.sum()

    result = newton_raphson(
        linear_loss,
        np.array([1.0], dtype=np.float32),
        backend=backend,
        max_iter=5,
    )

    assert result.success is False
    assert "Hessian is singular" in result.message
