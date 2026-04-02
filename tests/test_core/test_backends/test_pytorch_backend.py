"""Tests for the PyTorch backend implementation."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.core.backends import get_backend
from aurora.utils import BackendNotAvailableError

torch = pytest.importorskip("torch", reason="PyTorch not installed")


@pytest.fixture(name="backend")
def _backend_fixture():
    """Return the PyTorch backend or skip when unavailable."""
    try:
        return get_backend("pytorch")
    except BackendNotAvailableError:
        pytest.skip("PyTorch backend is not available")


def test_array_creates_tensor_on_default_device(backend):
    tensor = backend.array([1.0, 2.0, 3.0])
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type in {"cpu", "cuda"}


def test_as_numpy_roundtrip(backend):
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = backend.as_numpy(tensor)
    assert np.allclose(result, np.array([1.0, 2.0, 3.0]))


def test_grad_returns_correct_derivative(backend):
    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum()

    grad_fn = backend.grad(loss_fn)
    grad = grad_fn(torch.tensor([3.0, -2.0]))
    expected = torch.tensor([6.0, -4.0], device=grad.device)
    assert torch.allclose(grad, expected)


def test_jit_compiles_callable(backend):
    def scale(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    compiled = backend.jit(scale)
    tensor = backend.array([1.0, 2.0])
    result = compiled(tensor)
    assert torch.allclose(result, tensor * 2)


def test_device_put_moves_data_to_backend_device(backend):
    tensor = backend.device_put([5.0, 6.0])
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type in {"cpu", "cuda"}


def test_vmap_applies_function_over_batch(backend):
    def square(x: torch.Tensor) -> torch.Tensor:
        return x * x

    vmap_fn = backend.vmap(square)
    data = backend.array([0.0, 1.0, 2.0, 3.0])
    result = vmap_fn(data)
    expected = torch.tensor([0.0, 1.0, 4.0, 9.0], device=result.device)
    assert torch.allclose(result, expected)


def test_vmap_fallback_without_torch_vmap(monkeypatch, backend):
    monkeypatch.delattr(torch, "vmap", raising=False)

    def increment(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    vmap_fn = backend.vmap(increment)
    data = backend.array([1.0, 2.0, 3.0])
    result = vmap_fn(data)
    expected = torch.tensor([2.0, 3.0, 4.0], device=result.device)
    assert torch.allclose(result, expected)
