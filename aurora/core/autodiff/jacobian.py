"""Jacobian computation for vector-valued functions."""
from __future__ import annotations

from typing import Callable, Any
import numpy as np

from .backends import detect_backend

ArrayLike = Any
VectorFunc = Callable[..., ArrayLike]


def jacobian(
    func: VectorFunc,
    argnums: int = 0,
    backend: str | None = None,
) -> Callable:
    """Create a function that computes the Jacobian of a vector-valued function.

    The Jacobian is the matrix of first partial derivatives:
        J_ij = ∂f_i / ∂x_j

    Parameters
    ----------
    func : callable
        Vector-valued function to differentiate.
        Signature: func(params, *args, **kwargs) -> array(m,)
    argnums : int, default=0
        Index of argument to differentiate with respect to.
    backend : str, optional
        Force a specific backend.

    Returns
    -------
    jac_func : callable
        Function that returns the Jacobian matrix.
        Signature: jac_func(params, *args, **kwargs) -> ndarray(m, n)

    Notes
    -----
    For m outputs and n inputs:
    - Forward mode (JVP): O(n) passes, efficient when m >> n
    - Reverse mode (VJP): O(m) passes, efficient when n >> m
    """
    def jac_func(*args, **kwargs):
        x = args[argnums]
        detected_backend = backend or detect_backend(x)

        if detected_backend == 'jax':
            return _jacobian_jax(func, argnums, *args, **kwargs)
        elif detected_backend == 'torch':
            return _jacobian_torch(func, argnums, *args, **kwargs)
        else:
            return _jacobian_numerical(func, argnums, *args, **kwargs)

    return jac_func


def _jacobian_jax(func: VectorFunc, argnums: int, *args, **kwargs) -> ArrayLike:
    """Compute Jacobian using JAX autodiff."""
    import jax

    jac_fn = jax.jacfwd(func, argnums=argnums)
    return jac_fn(*args, **kwargs)


def _jacobian_torch(func: VectorFunc, argnums: int, *args, **kwargs) -> ArrayLike:
    """Compute Jacobian using PyTorch autograd."""
    import torch
    from torch.autograd.functional import jacobian as torch_jacobian

    args = list(args)
    x = args[argnums]

    def wrapped_func(x_inner):
        args_inner = list(args)
        args_inner[argnums] = x_inner
        return func(*args_inner, **kwargs)

    return torch_jacobian(wrapped_func, x)


def _jacobian_numerical(
    func: VectorFunc,
    argnums: int,
    *args,
    **kwargs
) -> np.ndarray:
    """Compute Jacobian using central finite differences."""
    args = list(args)
    x = np.asarray(args[argnums], dtype=np.float64)
    n = x.size
    x_flat = x.ravel()

    args[argnums] = x
    f0 = np.asarray(func(*args, **kwargs)).ravel()
    m = f0.size

    eps = np.finfo(np.float64).eps
    h_base = np.sqrt(eps)

    J = np.zeros((m, n), dtype=np.float64)

    for j in range(n):
        h = h_base * max(abs(x_flat[j]), 1.0)

        x_plus = x_flat.copy()
        x_plus[j] += h
        args[argnums] = x_plus.reshape(x.shape)
        f_plus = np.asarray(func(*args, **kwargs)).ravel()

        x_minus = x_flat.copy()
        x_minus[j] -= h
        args[argnums] = x_minus.reshape(x.shape)
        f_minus = np.asarray(func(*args, **kwargs)).ravel()

        J[:, j] = (f_plus - f_minus) / (2 * h)

    return J
