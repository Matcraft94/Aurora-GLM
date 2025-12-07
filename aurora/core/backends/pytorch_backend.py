# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""PyTorch numerical backend implementation."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from ...utils import BackendNotAvailableError

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


class PyTorchBackend:
    """PyTorch backend exposing Aurora protocol."""

    def __init__(self) -> None:
        if torch is None:  # pragma: no cover - runtime check
            raise BackendNotAvailableError("Install 'torch' to enable PyTorch backend.")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.get_default_dtype()

    def array(self, data: Any, dtype: Any | None = None):
        """Convert data to PyTorch tensor."""
        target_dtype = dtype or self._dtype
        if isinstance(data, torch.Tensor):
            tensor = data.to(self._device)
            if tensor.dtype != target_dtype:
                tensor = tensor.to(dtype=target_dtype)
            return tensor

        return torch.as_tensor(data, dtype=target_dtype, device=self._device)

    def as_numpy(self, data: Any):
        """Convert tensor to NumPy array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    def grad(self, func: Callable):
        """Create gradient function using torch.autograd."""

        def grad_fn(*args, **kwargs):  # noqa: ANN002 - backend protocol mirrors callable
            if not args:
                raise ValueError("Gradient requires at least one positional argument.")

            params = torch.as_tensor(
                args[0],
                dtype=self._dtype,
                device=self._device,
            ).clone()
            params.requires_grad_(True)

            result = func(params, *args[1:], **kwargs)
            if not isinstance(result, torch.Tensor):
                result = torch.as_tensor(result, device=self._device, dtype=self._dtype)
            if result.ndim != 0:
                raise ValueError(
                    "Gradient can only be computed for scalar-valued functions."
                )

            grad_tensor = torch.autograd.grad(
                result, params, create_graph=False, retain_graph=False
            )[0]
            return grad_tensor.detach()

        return grad_fn

    def jit(self, func: Callable):
        """JIT compile function using TorchScript."""
        if torch is None:  # pragma: no cover - defensive
            raise BackendNotAvailableError("Install 'torch' to enable PyTorch backend.")
        try:
            return torch.jit.script(func)
        except (RuntimeError, TypeError):
            return func

    def device_put(self, data: Any):
        """Move data to default device."""
        if isinstance(data, torch.Tensor):
            return data.to(self._device)
        return torch.as_tensor(data, device=self._device, dtype=self._dtype)

    def vmap(self, func: Callable, *, in_axes=0, out_axes=0):  # noqa: ANN001 - mirrors backend protocol
        """Vectorized map (similar to JAX vmap)."""
        if hasattr(torch, "vmap"):
            return torch.vmap(func, in_dims=in_axes, out_dims=out_axes)

        if in_axes != 0 or out_axes != 0:
            raise BackendNotAvailableError(
                "torch.vmap not available; fallback only supports in_axes=0 and out_axes=0."
            )

        def fallback(batch, *args, **kwargs):  # noqa: ANN001 - mirrors backend protocol
            outputs: list[Any] = []
            for item in torch.unbind(batch, dim=0):
                outputs.append(func(item, *args, **kwargs))
            if not outputs:
                return outputs
            first = outputs[0]
            if isinstance(first, torch.Tensor):
                return torch.stack(outputs, dim=0)
            return outputs

        return fallback

    def partial(self, func: Callable, *args: Any, **kwargs: Any) -> Callable:
        """Partial function application."""
        return partial(func, *args, **kwargs)


def create_backend() -> PyTorchBackend:
    """Factory for PyTorch backend."""
    return PyTorchBackend()


__all__ = ["PyTorchBackend", "create_backend"]
