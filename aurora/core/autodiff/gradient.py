# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Gradient computation for scalar-valued functions.

This module provides gradient computation across backends (NumPy, PyTorch, JAX).
For NumPy, uses central finite differences. For JAX/PyTorch, uses autodiff.
"""

from __future__ import annotations

from typing import Callable, Any
import numpy as np

from .backends import detect_backend

ArrayLike = Any
ScalarFunc = Callable[..., float]


def gradient(
    func: ScalarFunc,
    argnums: int = 0,
    backend: str | None = None,
) -> Callable:
    """Create a function that computes the gradient of a scalar function.

    Automatically uses the appropriate autodiff method based on the
    computational backend (JAX, PyTorch) or falls back to numerical
    differentiation for NumPy.

    Parameters
    ----------
    func : callable
        Scalar-valued function to differentiate.
        Signature: func(params, *args, **kwargs) -> float
    argnums : int, default=0
        Index of argument to differentiate with respect to.
    backend : str, optional
        Force a specific backend ('numpy', 'torch', 'jax').
        If None, auto-detects from input array types.

    Returns
    -------
    grad_func : callable
        Function that returns the gradient.
        Signature: grad_func(params, *args, **kwargs) -> ndarray

    Examples
    --------
    >>> def quadratic(x):
    ...     return x @ x
    >>> grad_quad = gradient(quadratic)
    >>> grad_quad(np.array([1.0, 2.0, 3.0]))
    array([2., 4., 6.])

    Notes
    -----
    For JAX, uses `jax.grad` which computes exact gradients via reverse-mode AD.
    For PyTorch, uses `torch.autograd.grad` with gradient tracking.
    For NumPy, uses central finite differences with step size h = √ε × max(|x|, 1).
    """

    def grad_func(*args, **kwargs):
        x = args[argnums]
        detected_backend = backend or detect_backend(x)

        if detected_backend == "jax":
            return _gradient_jax(func, argnums, *args, **kwargs)
        elif detected_backend == "torch":
            return _gradient_torch(func, argnums, *args, **kwargs)
        else:
            return _gradient_numerical(func, argnums, *args, **kwargs)

    return grad_func


def _gradient_jax(func: ScalarFunc, argnums: int, *args, **kwargs) -> ArrayLike:
    """Compute gradient using JAX autodiff."""
    import jax

    grad_fn = jax.grad(func, argnums=argnums)
    return grad_fn(*args, **kwargs)


def _gradient_torch(func: ScalarFunc, argnums: int, *args, **kwargs) -> ArrayLike:
    """Compute gradient using PyTorch autograd."""
    import torch

    args = list(args)
    x = args[argnums]

    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)
        args[argnums] = x

    loss = func(*args, **kwargs)

    if isinstance(loss, torch.Tensor):
        loss.backward()
        grad = x.grad.clone()
        x.grad.zero_()
        return grad
    else:
        grad = torch.autograd.grad(loss, x)[0]
        return grad


def _gradient_numerical(func: ScalarFunc, argnums: int, *args, **kwargs) -> np.ndarray:
    """Compute gradient using central finite differences.

    Uses adaptive step size: h = sqrt(ε) × max(|x_i|, 1) for each component.

    Central difference formula:
        ∂f/∂x_i ≈ [f(x + h*e_i) - f(x - h*e_i)] / (2h)

    Error: O(h²) (second-order accurate)
    """
    args = list(args)
    x = np.asarray(args[argnums], dtype=np.float64)
    n = x.size
    x_flat = x.ravel()

    eps = np.finfo(np.float64).eps
    h_base = np.sqrt(eps)

    grad = np.zeros(n, dtype=np.float64)

    for i in range(n):
        h = h_base * max(abs(x_flat[i]), 1.0)

        x_plus = x_flat.copy()
        x_plus[i] += h
        args[argnums] = x_plus.reshape(x.shape)
        f_plus = func(*args, **kwargs)

        x_minus = x_flat.copy()
        x_minus[i] -= h
        args[argnums] = x_minus.reshape(x.shape)
        f_minus = func(*args, **kwargs)

        grad[i] = (f_plus - f_minus) / (2 * h)

    args[argnums] = x
    return grad.reshape(x.shape)
