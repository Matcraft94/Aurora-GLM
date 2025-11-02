"""Newton-Raphson optimization algorithm."""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..types import Array, OptimizationCallback
from .result import OptimizationResult


def newton_raphson(
    loss_fn: Callable,
    init_params: Array,
    *,
    backend=None,
    args: tuple = (),
    kwargs: dict | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    callback: OptimizationCallback | None = None,
) -> OptimizationResult:
    """Run the Newton-Raphson method with automatic differentiation support."""
    if kwargs is None:
        kwargs = {}

    if backend is None:
        from ..backends import get_backend

        backend = get_backend("jax")

    grad_fn = backend.grad(loss_fn)

    x = backend.array(init_params)
    nfev = 0
    njev = 0
    nhev = 0

    for iteration in range(max_iter):
        grad = grad_fn(x, *args, **kwargs)
        njev += 1

        grad_np = np.asarray(backend.as_numpy(grad), dtype=float)
        grad_norm = np.linalg.norm(grad_np)

        if grad_norm < tol:
            f_val = backend.as_numpy(loss_fn(x, *args, **kwargs))
            nfev += 1
            return OptimizationResult(
                x=backend.as_numpy(x),
                fun=float(f_val),
                grad=backend.as_numpy(grad),
                success=True,
                message="Converged: gradient norm below tolerance",
                nit=iteration,
                nfev=nfev,
                njev=njev,
                nhev=nhev,
            )

        hess_np, evals = _compute_hessian(loss_fn, x, backend, args, kwargs)
        nfev += evals
        nhev += 1

        try:
            step = np.linalg.solve(hess_np, grad_np)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - rare singular cases
            f_val = backend.as_numpy(loss_fn(x, *args, **kwargs))
            nfev += 1
            return OptimizationResult(
                x=backend.as_numpy(x),
                fun=float(f_val),
                grad=backend.as_numpy(grad),
                success=False,
                message=f"Hessian is singular: {exc}",
                nit=iteration,
                nfev=nfev,
                njev=njev,
                nhev=nhev,
            )

        x_np = np.asarray(backend.as_numpy(x), dtype=float)
        x_new_np = x_np - step
        x = backend.array(x_new_np, dtype=getattr(x, "dtype", None))

        f_val = backend.as_numpy(loss_fn(x, *args, **kwargs))
        nfev += 1

        if callback is not None:
            callback(iteration, backend.as_numpy(x), float(f_val))

        if np.linalg.norm(step) < tol:
            final_grad = backend.as_numpy(grad_fn(x, *args, **kwargs))
            njev += 1
            return OptimizationResult(
                x=backend.as_numpy(x),
                fun=float(f_val),
                grad=final_grad,
                success=True,
                message="Converged: step size below tolerance",
                nit=iteration + 1,
                nfev=nfev,
                njev=njev,
                nhev=nhev,
            )

    f_val = backend.as_numpy(loss_fn(x, *args, **kwargs))
    nfev += 1
    final_grad = backend.as_numpy(grad_fn(x, *args, **kwargs))
    njev += 1
    return OptimizationResult(
        x=backend.as_numpy(x),
        fun=float(f_val),
        grad=final_grad,
        success=False,
        message="Maximum iterations reached",
        nit=max_iter,
        nfev=nfev,
        njev=njev,
        nhev=nhev,
    )


def _compute_hessian(loss_fn, params, backend, args, kwargs):
    """Compute Hessian via autograd when available, otherwise finite differences."""
    params_np = np.asarray(backend.as_numpy(params), dtype=float)
    dim = params_np.size
    hessian = np.zeros((dim, dim), dtype=float)

    try:
        import torch  # type: ignore

        if isinstance(params, torch.Tensor):
            params_t = params.detach().clone().requires_grad_(True)
            result = loss_fn(params_t, *args, **kwargs)
            grad = torch.autograd.grad(result, params_t, create_graph=True)[0]
            rows = []
            for grad_component in grad:
                if not grad_component.requires_grad:
                    rows.append(torch.zeros_like(params_t))
                    continue
                second = torch.autograd.grad(
                    grad_component,
                    params_t,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if second is None:
                    second = torch.zeros_like(params_t)
                rows.append(second)
            hess_tensor = torch.stack(rows)
            return hess_tensor.detach().cpu().numpy(), 1
    except ImportError:  # pragma: no cover - optional dependency missing
        pass

    try:
        import jax
        import jax.numpy as jnp

        if isinstance(params, jnp.ndarray):  # type: ignore[attr-defined]
            hess = jax.hessian(lambda p: loss_fn(p, *args, **kwargs))(params)
            return np.asarray(hess), 1
    except ImportError:  # pragma: no cover - optional dependency missing
        pass

    eps = 1e-4
    evaluations = 0

    for i in range(dim):
        for j in range(i, dim):
            e_i = np.zeros(dim)
            e_j = np.zeros(dim)
            e_i[i] = eps
            e_j[j] = eps

            f_pp = backend.as_numpy(loss_fn(backend.array(params_np + e_i + e_j), *args, **kwargs))
            f_pm = backend.as_numpy(loss_fn(backend.array(params_np + e_i - e_j), *args, **kwargs))
            f_mp = backend.as_numpy(loss_fn(backend.array(params_np - e_i + e_j), *args, **kwargs))
            f_mm = backend.as_numpy(loss_fn(backend.array(params_np - e_i - e_j), *args, **kwargs))
            evaluations += 4

            value = float((f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps))
            hessian[i, j] = value
            hessian[j, i] = value

    return hessian, evaluations


__all__ = ["newton_raphson"]