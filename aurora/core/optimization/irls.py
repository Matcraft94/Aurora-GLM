"""Iteratively reweighted least squares optimizer."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..types import Array, OptimizationCallback
from .result import OptimizationResult


def irls(
    loss_fn: Callable,
    init_params: Array,
    *,
    backend=None,
    args: tuple = (),
    kwargs: dict | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    callback: OptimizationCallback | None = None,
    design_matrix: Array | None = None,
    response: Array | None = None,
    link: Any | None = None,
    variance_fn: Callable[[Array], Array] | None = None,
    offset: Array | None = None,
) -> OptimizationResult:
    """Run the IRLS procedure for generalized linear models."""
    if kwargs is None:
        kwargs = {}

    if backend is None:
        from ..backends import get_backend

        backend = get_backend("jax")

    if design_matrix is None or response is None or link is None or variance_fn is None:
        raise ValueError(
            "IRLS requires design_matrix, response, link, and variance_fn keyword arguments."
        )

    if not hasattr(link, "inverse") or not hasattr(link, "derivative"):
        raise TypeError("link must expose inverse() and derivative() methods")

    X = backend.array(design_matrix)
    y = backend.array(response)
    beta = backend.array(init_params)
    offset_arr = backend.array(offset) if offset is not None else y * 0

    converted_args = tuple(_convert_to_backend(backend, value) for value in args)
    converted_kwargs = {key: _convert_to_backend(backend, value) for key, value in kwargs.items()}

    nfev = 0

    def _to_backend(data):
        return backend.array(data, dtype=getattr(beta, "dtype", None))

    for iteration in range(max_iter):
        eta = X @ beta + offset_arr
        mu = link.inverse(eta)
        g_prime = link.derivative(mu)
        var = variance_fn(mu)

        weights = _safe_divide(backend, 1.0, var * (g_prime ** 2))
        z = eta + (y - mu) * g_prime

        sqrt_w = _sqrt(backend, weights)
        WX = X * sqrt_w.unsqueeze(-1) if hasattr(sqrt_w, "unsqueeze") else X * sqrt_w[:, None]
        Wz = z * sqrt_w

        beta_new = None
        try:  # Prefer backend-native linear algebra when available
            import torch  # type: ignore

            if isinstance(WX, torch.Tensor):
                lhs = WX.T @ WX
                rhs = WX.T @ Wz
                if rhs.ndim == 1:
                    rhs = rhs.unsqueeze(-1)
                beta_new = torch.linalg.solve(lhs, rhs).squeeze(-1)
                beta_new = beta_new.to(beta.device).type_as(beta)
        except ImportError:  # pragma: no cover - optional dependency missing
            beta_new = None

        if beta_new is None:
            WX_np = backend.as_numpy(WX)
            Wz_np = backend.as_numpy(Wz)
            lhs = WX_np.T @ WX_np
            rhs = WX_np.T @ Wz_np
            solution = np.linalg.solve(lhs, rhs)
            beta_new = _to_backend(solution)

        delta = beta_new - beta
        step_norm = backend.as_numpy((delta * delta).sum() ** 0.5)

        beta = beta_new

        nfev += 1
        loss_value = loss_fn(beta, *converted_args, **converted_kwargs)

        if callback is not None:
            callback(iteration, backend.as_numpy(beta), float(backend.as_numpy(loss_value)))

        if step_norm < tol:
            return OptimizationResult(
                x=backend.as_numpy(beta),
                fun=float(backend.as_numpy(loss_value)),
                grad=None,
                success=True,
                message="Converged: parameter change below tolerance",
                nit=iteration + 1,
                nfev=nfev,
                njev=0,
                nhev=0,
            )

    loss_value = loss_fn(beta, *converted_args, **converted_kwargs)
    nfev += 1
    return OptimizationResult(
        x=backend.as_numpy(beta),
        fun=float(backend.as_numpy(loss_value)),
        grad=None,
        success=False,
        message="Maximum iterations reached",
        nit=max_iter,
        nfev=nfev,
        njev=0,
        nhev=0,
    )


def _safe_divide(backend, numerator, denominator):
    eps = 1e-12
    if hasattr(denominator, "clamp_min"):
        denominator = denominator.clamp_min(eps)
        return numerator / denominator
    denom_np = backend.as_numpy(denominator)
    denom_np = np.clip(denom_np, eps, None)
    return numerator / backend.array(denom_np)


def _sqrt(backend, value):
    if hasattr(value, "sqrt"):
        return value.sqrt()
    value_np = backend.as_numpy(value)
    return backend.array(np.sqrt(value_np))


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


__all__ = ["irls"]
