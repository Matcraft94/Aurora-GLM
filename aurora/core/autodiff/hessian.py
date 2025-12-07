# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Hessian computation for scalar-valued functions."""

from __future__ import annotations

from typing import Callable, Any
import numpy as np

from .backends import detect_backend
from .gradient import gradient

ArrayLike = Any
ScalarFunc = Callable[..., float]


def hessian(
    func: ScalarFunc,
    argnums: int = 0,
    backend: str | None = None,
) -> Callable:
    """Create a function that computes the Hessian of a scalar function.

    The Hessian is the matrix of second partial derivatives:
        H_ij = ∂²f / (∂x_i ∂x_j)

    Parameters
    ----------
    func : callable
        Scalar-valued function to differentiate twice.
    argnums : int, default=0
        Index of argument to differentiate with respect to.
    backend : str, optional
        Force a specific backend ('numpy', 'torch', 'jax').

    Returns
    -------
    hess_func : callable
        Function that returns the Hessian matrix.
        Signature: hess_func(params, *args, **kwargs) -> ndarray(p, p)

    Notes
    -----
    For large p, consider using `hvp` (Hessian-vector product) instead,
    which has O(p) memory complexity vs O(p²) for full Hessian.
    """

    def hess_func(*args, **kwargs):
        x = args[argnums]
        detected_backend = backend or detect_backend(x)

        if detected_backend == "jax":
            return _hessian_jax(func, argnums, *args, **kwargs)
        elif detected_backend == "torch":
            return _hessian_torch(func, argnums, *args, **kwargs)
        else:
            return _hessian_numerical(func, argnums, *args, **kwargs)

    return hess_func


def _hessian_jax(func: ScalarFunc, argnums: int, *args, **kwargs) -> ArrayLike:
    """Compute Hessian using JAX autodiff."""
    import jax

    hess_fn = jax.hessian(func, argnums=argnums)
    return hess_fn(*args, **kwargs)


def _hessian_torch(func: ScalarFunc, argnums: int, *args, **kwargs) -> ArrayLike:
    """Compute Hessian using PyTorch autograd."""
    from torch.autograd.functional import hessian as torch_hessian

    args = list(args)
    x = args[argnums]

    def wrapped_func(x_inner):
        args_inner = list(args)
        args_inner[argnums] = x_inner
        return func(*args_inner, **kwargs)

    return torch_hessian(wrapped_func, x)


def _hessian_numerical(func: ScalarFunc, argnums: int, *args, **kwargs) -> np.ndarray:
    """Compute Hessian using finite differences on the gradient.

    Uses central differences on the gradient to get second derivatives.
    """
    args = list(args)
    x = np.asarray(args[argnums], dtype=np.float64)
    n = x.size

    grad_fn = gradient(func, argnums=argnums, backend="numpy")

    args[argnums] = x

    eps = np.finfo(np.float64).eps
    h_base = eps ** (1 / 3)

    H = np.zeros((n, n), dtype=np.float64)
    x_flat = x.ravel()

    for j in range(n):
        h = h_base * max(abs(x_flat[j]), 1.0)

        x_plus = x_flat.copy()
        x_plus[j] += h
        args[argnums] = x_plus.reshape(x.shape)
        g_plus = grad_fn(*args, **kwargs).ravel()

        x_minus = x_flat.copy()
        x_minus[j] -= h
        args[argnums] = x_minus.reshape(x.shape)
        g_minus = grad_fn(*args, **kwargs).ravel()

        H[:, j] = (g_plus - g_minus) / (2 * h)

    H = 0.5 * (H + H.T)
    return H
