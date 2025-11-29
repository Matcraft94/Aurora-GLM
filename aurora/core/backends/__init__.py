"""Backend abstraction layer for multi-platform numerical computing.

Aurora supports NumPy (CPU), PyTorch (GPU), and JAX (XLA) backends,
allowing the same statistical code to run on different platforms
without modification.

Quick Start
-----------
>>> from aurora.core.backends import get_backend
>>> backend = get_backend('pytorch')  # or 'jax'
>>> x = backend.array([1.0, 2.0, 3.0])
>>> grad_fn = backend.grad(my_loss_function)

Available Backends
------------------
- **jax**: Google JAX with XLA compilation (default)
- **pytorch**: PyTorch with CUDA support

Backend Operations
------------------
For lower-level numerical operations that work across backends:

>>> from aurora.core.backends import get_namespace, solve, cholesky
>>> xp, device = get_namespace('numpy')  # or 'torch', 'jax'
>>> solution = solve(A, b, xp)

Backward Compatibility
----------------------
This module maintains strict backward compatibility. All public functions
and classes exposed in v0.5.0 remain available with identical signatures.

See Also
--------
aurora.distributions._utils : Array namespace utilities for distributions
aurora.models : High-level model fitting interfaces
"""
from __future__ import annotations

# Protocol and type definitions
from ._protocol import Backend, BackendFactory

# Registry functions
from ._registry import (
    available_backends,
    get_backend,
    register_backend,
)

# Backend-agnostic numerical operations
from .operations import (
    get_namespace,
    to_backend_array,
    to_numpy,
    solve,
    cholesky,
    inv,
    det,
    slogdet,
    eigh,
    qr,
    lstsq,
    eye,
    zeros,
    ones,
    concatenate,
    stack,
    diag,
    trace,
    matmul,
    transpose,
)

__all__ = [
    # Protocol
    "Backend",
    "BackendFactory",
    # Registry
    "available_backends",
    "get_backend",
    "register_backend",
    # Operations - Namespace
    "get_namespace",
    "to_backend_array",
    "to_numpy",
    # Operations - Linear algebra
    "solve",
    "cholesky",
    "inv",
    "det",
    "slogdet",
    "eigh",
    "qr",
    "lstsq",
    # Operations - Array creation
    "eye",
    "zeros",
    "ones",
    # Operations - Array manipulation
    "concatenate",
    "stack",
    "diag",
    "trace",
    "matmul",
    "transpose",
]
