"""Tests for the L-BFGS optimizer."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.core.optimization import lbfgs

torch = pytest.importorskip("torch")


@pytest.fixture(name="backend")
def _backend_fixture(pytorch_backend):
    return pytorch_backend


def test_lbfgs_quadratic_converges(backend):
    def loss(x: torch.Tensor) -> torch.Tensor:
        return ((x - 3.0) ** 2).sum()

    result = lbfgs(loss, np.array([0.0], dtype=np.float32), backend=backend, max_iter=100)
    assert result.success is True
    assert np.allclose(result.x, np.array([3.0]), atol=1e-4)


def test_lbfgs_rosenbrock(backend):
    def rosenbrock(params: torch.Tensor) -> torch.Tensor:
        x, y = params[0], params[1]
        return (1 - x) ** 2 + 100.0 * (y - x**2) ** 2

    result = lbfgs(
        rosenbrock,
        np.array([-1.5, 2.0], dtype=np.float32),
        backend=backend,
        max_iter=1000,
        tol=1e-6,
    )

    assert result.success is True
    assert np.allclose(result.x, np.array([1.0, 1.0]), atol=1e-3)


def test_lbfgs_with_additional_arguments(backend):
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b = torch.tensor([5.0, 6.0], dtype=torch.float32)

    def loss(params: torch.Tensor, matrix: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = matrix @ params - target
        return (residual**2).sum()

    result = lbfgs(
        loss,
        np.zeros(2, dtype=np.float32),
        backend=backend,
        args=(A, b),
        max_iter=200,
    )

    assert result.success is True
    residual = (A @ torch.from_numpy(result.x) - b).detach().numpy()
    assert np.linalg.norm(residual) < 1e-4


def test_lbfgs_invokes_callback(backend):
    call_counter = {"n": 0}

    def callback(iteration: int, params, value: float) -> None:
        call_counter["n"] += 1

    def loss(x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum()

    lbfgs(
        loss,
        np.array([1.0, 1.0], dtype=np.float32),
        backend=backend,
        callback=callback,
        max_iter=5,
    )

    assert call_counter["n"] > 0


def test_lbfgs_reports_max_iterations(backend):
    def loss(x: torch.Tensor) -> torch.Tensor:
        return ((x - 100.0) ** 2).sum()

    result = lbfgs(
        loss,
        np.array([-100.0], dtype=np.float32),
        backend=backend,
        max_iter=1,
        tol=1e-12,
    )

    assert result.success is False
    assert result.nit == 1
