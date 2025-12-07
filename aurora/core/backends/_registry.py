# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Backend registry for Aurora's multi-backend architecture.

This module manages the registration and retrieval of numerical backends.
It is an internal module - users should import from aurora.core.backends.

Thread Safety
-------------
The current implementation uses a module-level dictionary which is not
thread-safe for concurrent registration. For typical scientific computing
workflows (single-threaded initialization), this is acceptable.

Lazy Loading
------------
Built-in backends (JAX, PyTorch) are loaded on-demand when first requested,
minimizing import overhead for users who only need one backend.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from ...utils import BackendNotAvailableError

if TYPE_CHECKING:
    from ._protocol import Backend, BackendFactory


# Module-level registry for backend factories
_BACKENDS: dict[str, "BackendFactory"] = {}

# Built-in backends that can be loaded on demand
_BUILTIN_BACKENDS: frozenset[str] = frozenset(("jax", "pytorch"))


def register_backend(
    name: str,
    factory: "BackendFactory",
    *,
    overwrite: bool = False,
) -> None:
    """Register a backend factory for later retrieval.

    Parameters
    ----------
    name : str
        Identifier for the backend (case-insensitive).
    factory : BackendFactory
        Callable that returns a Backend instance.
    overwrite : bool, default=False
        If True, replace existing registration. Otherwise raise ValueError.

    Raises
    ------
    ValueError
        If backend is already registered and overwrite=False.

    Examples
    --------
    >>> def my_backend_factory():
    ...     return MyCustomBackend()
    >>> register_backend("custom", my_backend_factory)

    Notes
    -----
    Registration is typically done at module import time for built-in
    backends, or during application initialization for custom backends.
    """
    normalized = name.lower()
    if not overwrite and normalized in _BACKENDS:
        raise ValueError(
            f"Backend '{name}' is already registered. "
            f"Pass overwrite=True to replace it."
        )
    _BACKENDS[normalized] = factory


def _load_builtin_backend(name: str) -> "Backend":
    """Load a built-in backend module by name.

    Parameters
    ----------
    name : str
        Backend name: 'jax' or 'pytorch'.

    Returns
    -------
    Backend
        Initialized backend instance.

    Raises
    ------
    BackendNotAvailableError
        If backend module is not installed or doesn't expose create_backend().
    """
    try:
        module: ModuleType = import_module(f"aurora.core.backends.{name}_backend")
    except ModuleNotFoundError as exc:
        raise BackendNotAvailableError(
            f"Backend '{name}' is not available. "
            f"Install optional dependencies and try again."
        ) from exc

    if not hasattr(module, "create_backend"):
        raise BackendNotAvailableError(
            f"Backend module 'aurora.core.backends.{name}_backend' "
            f"does not expose create_backend()."
        )

    return module.create_backend()


def get_backend(name: str = "jax") -> "Backend":
    """Retrieve a backend by name, loading built-ins on demand.

    Parameters
    ----------
    name : str, default="jax"
        Backend identifier (case-insensitive).
        Built-in options: 'jax', 'pytorch'.

    Returns
    -------
    Backend
        Initialized backend instance ready for use.

    Raises
    ------
    BackendNotAvailableError
        If requested backend is not installed.

    Examples
    --------
    >>> backend = get_backend("pytorch")
    >>> tensor = backend.array([1.0, 2.0, 3.0])
    >>> grad_fn = backend.grad(my_loss_function)

    Notes
    -----
    Backends are instantiated fresh on each call. For performance-critical
    code, consider caching the backend instance.
    """
    normalized = name.lower()

    # Register built-in backends on first access (lazy loading)
    if normalized not in _BACKENDS and normalized in _BUILTIN_BACKENDS:
        register_backend(normalized, lambda n=normalized: _load_builtin_backend(n))

    if normalized not in _BACKENDS:
        available = ", ".join(sorted(available_backends()))
        raise BackendNotAvailableError(
            f"Unknown backend '{name}'. Available backends: {available}"
        )

    backend_factory = _BACKENDS[normalized]
    return backend_factory()


def available_backends() -> tuple[str, ...]:
    """Return names of all registered and built-in backends.

    Returns
    -------
    tuple of str
        Sorted tuple of backend names.

    Examples
    --------
    >>> available_backends()
    ('jax', 'pytorch')

    Notes
    -----
    This includes both explicitly registered backends and built-in
    backends that can be loaded on demand.
    """
    return tuple(sorted(set(_BACKENDS) | _BUILTIN_BACKENDS))


__all__ = [
    "register_backend",
    "get_backend",
    "available_backends",
]
