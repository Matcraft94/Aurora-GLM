"""PyTorch numerical backend implementation."""
from __future__ import annotations

from typing import Any, Callable

from ...utils import BackendNotAvailableError

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


class PyTorchBackend:
    """Thin wrapper around PyTorch exposing the Aurora backend protocol."""

    def __init__(self, *, device: str | None = None, dtype: str | None = None) -> None:
        if torch is None:  # pragma: no cover - runtime check
            raise BackendNotAvailableError("Install 'torch' to enable the PyTorch backend.")

        self._device = torch.device(device) if device else torch.device("cpu")
        self._dtype = getattr(torch, dtype) if dtype else torch.float32

    def array(self, data: Any, dtype: Any | None = None):
        target_dtype = getattr(torch, dtype) if isinstance(dtype, str) else (dtype or self._dtype)
        return torch.as_tensor(data, dtype=target_dtype, device=self._device)

    def as_numpy(self, data: "torch.Tensor"):
        return data.detach().cpu().numpy()

    def grad(self, func: Callable):
        def wrapper(params, *args, **kwargs):  # noqa: ANN001 - mirrors backend protocol
            tensor_params = torch.as_tensor(
                params,
                dtype=self._dtype,
                device=self._device,
                requires_grad=True,
            )
            result = func(tensor_params, *args, **kwargs)
            if result.ndim != 0:
                raise ValueError("Gradient can only be computed for scalar-valued functions.")
            grad_tensor = torch.autograd.grad(result, tensor_params, create_graph=False, retain_graph=False)[0]
            return grad_tensor.detach()

        return wrapper

    def jit(self, func: Callable):
        # TorchScript tracing would require a warm input; keep things simple for now.
        return func

    def device_put(self, data: Any):
        tensor = torch.as_tensor(data, dtype=self._dtype)
        return tensor.to(self._device)


def create_backend(**kwargs: Any) -> PyTorchBackend:
    """Factory used by the backend registry."""
    return PyTorchBackend(**kwargs)


__all__ = ["PyTorchBackend", "create_backend"]
