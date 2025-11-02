"""L-BFGS optimization algorithm."""
from __future__ import annotations

from typing import Any, Callable

from ..types import Array, OptimizationCallback
from .result import OptimizationResult


def lbfgs(
    loss_fn: Callable,
    init_params: Array,
    *,
    backend=None,
    args: tuple = (),
    kwargs: dict | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    m: int = 10,
    line_search: str = "strong-wolfe",
    callback: OptimizationCallback | None = None,
) -> OptimizationResult:
    """L-BFGS (Limited-memory BFGS) optimization."""

    kwargs = kwargs or {}

    if backend is None:
        from ..backends import get_backend

        backend = get_backend("jax")

    converted_args = tuple(_convert_to_backend(backend, value) for value in args)
    converted_kwargs = {key: _convert_to_backend(backend, value) for key, value in kwargs.items()}

    grad_fn = backend.grad(loss_fn)
    x = backend.array(init_params)

    s_history: list[Any] = []
    y_history: list[Any] = []

    g = grad_fn(x, *converted_args, **converted_kwargs)

    nfev = 1
    njev = 1

    for iteration in range(max_iter):
        d = _lbfgs_direction(g, s_history, y_history, backend)

        alpha, f_new, g_new, ls_fev = _line_search(
            loss_fn,
            grad_fn,
            x,
            d,
            g,
            backend,
            args=converted_args,
            kwargs=converted_kwargs,
            method=line_search,
        )

        if alpha <= 0:
            failure_fun = loss_fn(x, *converted_args, **converted_kwargs)
            return OptimizationResult(
                x=backend.as_numpy(x),
                fun=float(failure_fun),
                grad=backend.as_numpy(g),
                success=False,
                message="Line search failed to find a descent direction",
                nit=iteration,
                nfev=nfev + ls_fev,
                njev=njev,
            )

        nfev += ls_fev
        njev += 1

        x_new = x + alpha * d
        s = x_new - x

        y = g_new - g

        if hasattr(backend, "as_numpy"):
            sy = backend.as_numpy((s * y).sum())
        else:
            sy = float((s * y).sum())

        if sy <= 0:
            s_history.clear()
            y_history.clear()
        else:
            s_history.append(s)
            y_history.append(y)
            if len(s_history) > m:
                s_history.pop(0)
                y_history.pop(0)

        x = x_new
        g = g_new

        if callback is not None:
            f_val = loss_fn(x, *converted_args, **converted_kwargs)
            callback(iteration, backend.as_numpy(x), float(backend.as_numpy(f_val)))
            nfev += 1

        grad_norm = backend.as_numpy((g * g).sum() ** 0.5)
        if grad_norm < tol:
            f_final = loss_fn(x, *converted_args, **converted_kwargs)
            nfev += 1
            return OptimizationResult(
                x=backend.as_numpy(x),
                fun=float(f_final),
                grad=backend.as_numpy(g),
                success=True,
                message="Converged: gradient norm below tolerance",
                nit=iteration + 1,
                nfev=nfev,
                njev=njev,
            )

    f_final = loss_fn(x, *converted_args, **converted_kwargs)
    nfev += 1
    return OptimizationResult(
        x=backend.as_numpy(x),
        fun=float(f_final),
        grad=backend.as_numpy(g),
        success=False,
        message="Maximum iterations reached",
        nit=max_iter,
        nfev=nfev,
        njev=njev,
    )


def _lbfgs_direction(g, s_history, y_history, backend):
    """Compute the L-BFGS search direction using the two-loop recursion."""
    q = g
    m = len(s_history)

    if m == 0:
        return -g

    alphas: list[Any] = []
    rhos: list[float] = []

    for i in range(m - 1, -1, -1):
        s_i = s_history[i]
        y_i = y_history[i]
        rho_i = 1.0 / (y_i * s_i).sum()
        alpha_i = rho_i * (s_i * q).sum()
        alphas.append(alpha_i)
        rhos.append(rho_i)
        q = q - alpha_i * y_i

    s_m = s_history[-1]
    y_m = y_history[-1]
    gamma = (s_m * y_m).sum() / (y_m * y_m).sum()
    r = gamma * q

    alphas.reverse()
    rhos.reverse()
    for i, alpha_i in enumerate(alphas):
        s_i = s_history[i]
        y_i = y_history[i]
        rho_i = rhos[i]
        beta_i = rho_i * (y_i * r).sum()
        r = r + s_i * (alpha_i - beta_i)

    return -r


def _line_search(
    loss_fn,
    grad_fn,
    x,
    d,
    g,
    backend,
    *,
    args=(),
    kwargs=None,
    method="strong-wolfe",
):
    """Perform a basic backtracking line search satisfying the Armijo condition."""
    if kwargs is None:
        kwargs = {}

    alpha = 1.0
    c1 = 1e-4
    rho = 0.9

    f_0 = loss_fn(x, *args, **kwargs)
    directional_derivative = (g * d).sum()

    fev = 1
    for _ in range(20):
        x_new = x + alpha * d
        f_new = loss_fn(x_new, *args, **kwargs)
        fev += 1

        if f_new <= f_0 + c1 * alpha * directional_derivative:
            g_new = grad_fn(x_new, *args, **kwargs)
            return alpha, f_new, g_new, fev

        alpha *= rho

    x_new = x + alpha * d
    f_new = loss_fn(x_new, *args, **kwargs)
    g_new = grad_fn(x_new, *args, **kwargs)
    return alpha, f_new, g_new, fev


def _convert_to_backend(backend, value):
    if isinstance(value, (tuple, list)):
        converted = [_convert_to_backend(backend, item) for item in value]
        return type(value)(converted)
    if isinstance(value, dict):
        return {key: _convert_to_backend(backend, item) for key, item in value.items()}
    try:
        return backend.array(value)
    except Exception:  # pragma: no cover - fallback when conversion is not applicable
        return value

__all__ = ["lbfgs"]
