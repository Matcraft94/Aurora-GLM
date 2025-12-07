# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Efficient matrix-vector products: HVP, JVP, VJP.

These functions compute products without forming full matrices,
providing O(p) memory complexity vs O(p²) for full matrices.
"""

from __future__ import annotations

from typing import Callable, Any, Tuple
import numpy as np

from .backends import detect_backend
from .gradient import gradient
from .jacobian import _jacobian_numerical, _jacobian_torch

ArrayLike = Any
ScalarFunc = Callable[..., float]
VectorFunc = Callable[..., ArrayLike]


def hvp(func: ScalarFunc, x: ArrayLike, v: ArrayLike, *args, **kwargs) -> ArrayLike:
    """Compute Hessian-vector product H @ v without forming full Hessian.

    This is memory efficient for large problems: O(p) vs O(p²) for full Hessian.

    The HVP can be computed as:
        H @ v = d/dε [∇f(x + εv)] |_{ε=0}

    Parameters
    ----------
    func : callable
        Scalar-valued function
    x : array-like
        Point at which to evaluate
    v : array-like
        Vector to multiply
    *args, **kwargs : additional arguments to func

    Returns
    -------
    Hv : array-like
        Product H(x) @ v
    """
    backend = detect_backend(x)

    if backend == "jax":
        return _hvp_jax(func, x, v, *args, **kwargs)
    elif backend == "torch":
        return _hvp_torch(func, x, v, *args, **kwargs)
    else:
        return _hvp_numerical(func, x, v, *args, **kwargs)


def _hvp_jax(func, x, v, *args, **kwargs):
    """JAX HVP using forward-over-reverse mode."""
    import jax

    def grad_func(x_inner):
        return jax.grad(lambda p: func(p, *args, **kwargs))(x_inner)

    _, hvp_result = jax.jvp(grad_func, (x,), (v,))
    return hvp_result


def _hvp_torch(func, x, v, *args, **kwargs):
    """PyTorch HVP using autograd."""
    import torch

    x = x.clone().detach().requires_grad_(True)
    loss = func(x, *args, **kwargs)
    grad = torch.autograd.grad(loss, x, create_graph=True)[0]
    hvp_result = torch.autograd.grad(grad, x, grad_outputs=v)[0]
    return hvp_result


def _hvp_numerical(func, x, v, *args, **kwargs):
    """Numerical HVP using finite differences on gradient."""
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    grad_fn = gradient(func, backend="numpy")

    eps = np.finfo(np.float64).eps
    h = np.sqrt(eps) * max(np.linalg.norm(x), 1.0)

    g_plus = grad_fn(x + h * v, *args, **kwargs)
    g_minus = grad_fn(x - h * v, *args, **kwargs)

    return (g_plus - g_minus) / (2 * h)


def jvp(
    func: VectorFunc, x: ArrayLike, v: ArrayLike, *args, **kwargs
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute Jacobian-vector product (forward mode).

    Computes (f(x), J(x) @ v) efficiently in a single forward pass.

    Parameters
    ----------
    func : callable
        Vector-valued function
    x : array-like
        Point at which to evaluate
    v : array-like
        Tangent vector
    *args, **kwargs : additional arguments to func

    Returns
    -------
    primals : array-like
        Function value f(x)
    tangents : array-like
        Directional derivative J(x) @ v
    """
    backend = detect_backend(x)

    if backend == "jax":
        import jax

        return jax.jvp(lambda p: func(p, *args, **kwargs), (x,), (v,))
    elif backend == "torch":
        J = _jacobian_torch(func, 0, x, *args, **kwargs)
        f_x = func(x, *args, **kwargs)
        return f_x, J @ v
    else:
        x = np.asarray(x, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        eps = np.finfo(np.float64).eps
        h = np.sqrt(eps) * max(np.linalg.norm(x), 1.0)

        f_x = func(x, *args, **kwargs)
        f_plus = func(x + h * v, *args, **kwargs)

        tangent = (np.asarray(f_plus) - np.asarray(f_x)) / h
        return f_x, tangent


def vjp(
    func: VectorFunc, x: ArrayLike, v: ArrayLike, *args, **kwargs
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute vector-Jacobian product (reverse mode).

    Computes (f(x), v^T @ J(x)) efficiently.

    Parameters
    ----------
    func : callable
        Vector-valued function
    x : array-like
        Point at which to evaluate
    v : array-like
        Cotangent vector (same shape as output)
    *args, **kwargs : additional arguments to func

    Returns
    -------
    primals : array-like
        Function value f(x)
    cotangents : array-like
        Vector-Jacobian product v^T @ J(x)
    """
    backend = detect_backend(x)

    if backend == "jax":
        import jax

        primals, vjp_fn = jax.vjp(lambda p: func(p, *args, **kwargs), x)
        cotangents = vjp_fn(v)[0]
        return primals, cotangents
    elif backend == "torch":

        x_t = x.clone().detach().requires_grad_(True)
        f_x = func(x_t, *args, **kwargs)
        f_x.backward(v)
        return f_x.detach(), x_t.grad
    else:
        J = _jacobian_numerical(func, 0, x, *args, **kwargs)
        f_x = func(x, *args, **kwargs)
        return f_x, v @ J
