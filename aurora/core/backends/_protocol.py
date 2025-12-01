"""Protocol definitions for Aurora backend abstraction layer.

This module defines the interfaces that backend implementations must satisfy.
It is an internal module - users should import from aurora.core.backends.

Mathematical Context
--------------------
Aurora backends provide numerical primitives for statistical computing:
- Array operations (creation, manipulation)
- Automatic differentiation (gradients)
- JIT compilation for performance
- Device placement (CPU/GPU)

References
----------
.. [1] JAX: Composable transformations of Python+NumPy programs
       https://github.com/google/jax
.. [2] PyTorch: An Imperative Style, High-Performance Deep Learning Library
       https://pytorch.org/
"""
from __future__ import annotations

from typing import Callable, Protocol


class Backend(Protocol):
    """Protocol describing the minimum API expected from a numerical backend.

    All backend implementations (JAX, PyTorch, future backends) must satisfy
    this protocol to ensure consistent behavior across the Aurora framework.

    Methods
    -------
    array(data, dtype=None)
        Convert data to backend-native array format.
    as_numpy(data)
        Convert backend array to NumPy for interoperability.
    grad(func)
        Create gradient function for automatic differentiation.
    jit(func)
        JIT-compile function for performance optimization.
    device_put(data)
        Place data on appropriate device (CPU/GPU).

    Notes
    -----
    This protocol uses structural subtyping (PEP 544). Implementations
    do not need to explicitly inherit from Backend; they only need to
    provide the required methods with compatible signatures.

    Examples
    --------
    >>> class MyBackend:
    ...     def array(self, data, dtype=None): ...
    ...     def as_numpy(self, data): ...
    ...     def grad(self, func): ...
    ...     def jit(self, func): ...
    ...     def device_put(self, data): ...
    """

    def array(self, data, dtype=None):  # noqa: ANN001 - backend-dependent signature
        """Convert input data to backend-native array format."""
        ...

    def as_numpy(self, data):  # noqa: ANN001 - backend-dependent signature
        """Convert backend array to NumPy ndarray."""
        ...

    def grad(self, func: Callable) -> Callable:
        """Create gradient function for automatic differentiation."""
        ...

    def jit(self, func: Callable) -> Callable:
        """JIT-compile function for improved performance."""
        ...

    def device_put(self, data):  # noqa: ANN001 - backend-dependent signature
        """Place data on the appropriate compute device."""
        ...

    def vmap(self, func: Callable, *, in_axes=0, out_axes=0) -> Callable:  # noqa: ANN001
        """Vectorized map over batch dimensions.

        Similar to JAX's vmap, applies func over the leading axis of inputs.

        Parameters
        ----------
        func : Callable
            Function to vectorize.
        in_axes : int, default=0
            Axis to map over for inputs.
        out_axes : int, default=0
            Axis for outputs.

        Returns
        -------
        Callable
            Vectorized version of func.
        """
        ...

    def partial(self, func: Callable, *args, **kwargs) -> Callable:
        """Partial function application.

        Parameters
        ----------
        func : Callable
            Function to partially apply.
        *args
            Positional arguments to fix.
        **kwargs
            Keyword arguments to fix.

        Returns
        -------
        Callable
            Partially applied function.
        """
        ...


# Type alias for backend factory functions
BackendFactory = Callable[[], Backend]


__all__ = ["Backend", "BackendFactory"]
