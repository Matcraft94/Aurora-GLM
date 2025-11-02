"""Backend registry and abstraction layer for numerical computations."""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Callable, Protocol

from ...utils import BackendNotAvailableError


class Backend(Protocol):
    """Protocol describing the minimum surface area expected from a backend."""

    def array(self, data, dtype=None):  # noqa: ANN001 - backend dependent signature
        ...

    def as_numpy(self, data):  # noqa: ANN001 - backend dependent signature
        ...

    def grad(self, func: Callable):
        ...

    def jit(self, func: Callable):
        ...

    def device_put(self, data):  # noqa: ANN001 - backend dependent signature
        ...


BackendFactory = Callable[[], Backend]


_BACKENDS: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory, *, overwrite: bool = False) -> None:
    """Register a backend factory so it can be retrieved by name."""
    normalized = name.lower()
    if not overwrite and normalized in _BACKENDS:
        raise ValueError(f"Backend '{name}' is already registered. Pass overwrite=True to replace it.")
    _BACKENDS[normalized] = factory


def _load_builtin_backend(name: str) -> Backend:
    try:
        module: ModuleType = import_module(f"aurora.core.backends.{name}_backend")
    except ModuleNotFoundError as exc:  # pragma: no cover - lazy import path
        raise BackendNotAvailableError(
            f"Backend '{name}' is not available. Install optional dependencies and try again."
        ) from exc
    if not hasattr(module, "create_backend"):
        raise BackendNotAvailableError(
            f"Backend module 'aurora.core.backends.{name}_backend' does not expose create_backend()."
        )
    return module.create_backend()


def get_backend(name: str = "jax") -> Backend:
    """Retrieve a backend by name, loading built-ins on demand."""
    normalized = name.lower()

    if normalized not in _BACKENDS:
        register_backend(normalized, lambda: _load_builtin_backend(normalized))

    backend_factory = _BACKENDS[normalized]
    backend = backend_factory()
    return backend


def available_backends() -> tuple[str, ...]:
    """Return a tuple with the names of registered backends."""
    builtins = ("jax", "pytorch")
    return tuple(sorted(set(_BACKENDS) | set(builtins)))


__all__ = ["Backend", "available_backends", "get_backend", "register_backend"]
