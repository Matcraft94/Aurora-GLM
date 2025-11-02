"""Tests for the IRLS optimizer."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.core.optimization import irls

torch = pytest.importorskip("torch")


class IdentityLink:
    def inverse(self, eta: torch.Tensor) -> torch.Tensor:
        return eta

    def derivative(self, mu: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(mu)


@pytest.fixture(name="backend")
def _backend_fixture(pytorch_backend):
    return pytorch_backend


@pytest.fixture(name="gaussian_problem")
def _gaussian_problem():
    X = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    return X, y


def test_irls_gaussian_identity_link_converges(backend, gaussian_problem):
    X, y = gaussian_problem

    def loss(beta: torch.Tensor, design: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = design @ beta - target
        return (residual**2).mean()

    result = irls(
        loss,
        np.zeros(X.shape[1], dtype=np.float32),
        backend=backend,
        args=(X, y),
        design_matrix=X,
        response=y,
        link=IdentityLink(),
        variance_fn=lambda mu: torch.ones_like(mu),
        max_iter=10,
        tol=1e-8,
    )

    assert result.success is True
    assert np.allclose(result.x, np.array([1.0, 1.0]), atol=1e-6)


def test_irls_calls_callback(backend, gaussian_problem):
    X, y = gaussian_problem
    calls = {"count": 0}

    def callback(iteration: int, params, value: float) -> None:
        calls["count"] += 1

    def loss(beta: torch.Tensor, design: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = design @ beta - target
        return (residual**2).mean()

    irls(
        loss,
        np.zeros(X.shape[1], dtype=np.float32),
        backend=backend,
        args=(X, y),
        design_matrix=X,
        response=y,
        link=IdentityLink(),
        variance_fn=lambda mu: torch.ones_like(mu),
        callback=callback,
        max_iter=3,
    )

    assert calls["count"] > 0


def test_irls_raises_without_required_arguments(pytorch_backend):
    with pytest.raises(ValueError, match="requires design_matrix"):
        irls(lambda x: x, np.array([0.0]), backend=pytorch_backend)
