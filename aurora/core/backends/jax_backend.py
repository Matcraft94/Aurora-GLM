"""JAX numerical backend implementation."""
from __future__ import annotations

from functools import partial
from typing import Any, Callable

from ...utils import BackendNotAvailableError

try:  # pragma: no cover - optional dependency
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    jax = None  # type: ignore
    jnp = None  # type: ignore


class JAXBackend:
    """Thin wrapper around JAX exposing the Aurora backend protocol."""

    def __init__(self) -> None:
        if jax is None or jnp is None:  # pragma: no cover - runtime check
            raise BackendNotAvailableError("Install 'jax' and 'jaxlib' to enable the JAX backend.")

    def array(self, data: Any, dtype: Any | None = None):
        return jnp.array(data, dtype=dtype)

    def as_numpy(self, data: Any):
        return jax.device_get(data)

    def grad(self, func: Callable):
        return jax.grad(func)

    def jit(self, func: Callable):
        return jax.jit(func)

    def device_put(self, data: Any):
        return jax.device_put(data)

    def vmap(self, func: Callable, *, in_axes=0, out_axes=0):  # noqa: ANN001 - mirrors jax.vmap signature
        return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)

    def partial(self, func: Callable, *args: Any, **kwargs: Any) -> Callable:
        return partial(func, *args, **kwargs)


def create_backend() -> JAXBackend:
    """Factory used by the backend registry."""
    return JAXBackend()


__all__ = ["JAXBackend", "create_backend"]
