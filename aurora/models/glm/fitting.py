"""Iteratively reweighted least squares fitting for GLMs."""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from ...core.types import Array
from ...distributions._utils import as_namespace_array, namespace
from ...distributions.base import Family, LinkFunction
from ...distributions.families import BinomialFamily, GammaFamily, GaussianFamily, PoissonFamily
from ...distributions.links import (
    CLogLogLink,
    IdentityLink,
    InverseLink,
    LogLink,
    LogitLink,
)
from ..base.result import GLMResult


_FAMILY_REGISTRY: dict[str, Callable[[], Family]] = {
    "gaussian": GaussianFamily,
    "poisson": PoissonFamily,
    "binomial": BinomialFamily,
    "gamma": GammaFamily,
}

_LINK_REGISTRY: dict[str, Callable[[], LinkFunction]] = {
    "identity": IdentityLink,
    "log": LogLink,
    "logit": LogitLink,
    "inverse": InverseLink,
    "cloglog": CLogLogLink,
}


def fit_glm(
    X: Array,
    y: Array,
    *,
    family: str | Family = "gaussian",
    link: str | LinkFunction | None = None,
    weights: Array | None = None,
    offset: Array | None = None,
    backend: str | None = None,
    max_iter: int = 25,
    tol: float = 1e-8,
    fit_intercept: bool = True,
) -> GLMResult:
    """Fit a Generalized Linear Model using IRLS."""

    del backend  # Multi-backend optimization hooks will be introduced later

    xp = namespace(X, y, weights, offset)
    X_arr = as_namespace_array(X, xp)

    if getattr(X_arr, "ndim", 1) == 1:
        if xp is np:
            X_arr = X_arr.reshape(-1, 1)
        else:
            X_arr = X_arr.unsqueeze(-1)

    y_arr = as_namespace_array(y, xp, like=X_arr)
    if getattr(y_arr, "ndim", 1) != 1:
        y_arr = y_arr.reshape(-1)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("Design matrix and response must share the same number of samples.")

    weights_arr = None
    if weights is not None:
        weights_arr = as_namespace_array(weights, xp, like=y_arr)
        if getattr(weights_arr, "ndim", 1) != 1:
            weights_arr = weights_arr.reshape(-1)

    offset_arr = None
    if offset is not None:
        offset_arr = as_namespace_array(offset, xp, like=y_arr)
        if getattr(offset_arr, "ndim", 1) != 1:
            offset_arr = offset_arr.reshape(-1)

    family_obj = _coerce_family(family)
    link_obj = _coerce_link(link, family_obj)

    X_design = X_arr
    if fit_intercept:
        intercept_column = _ones_column(xp, X_arr.shape[0], like=X_arr)
        X_design = _concat_columns(xp, intercept_column, X_arr)

    beta, eta_total, mu, n_iter, converged, deviance = _irls(
        xp,
        X_design,
        y_arr,
        family_obj,
        link_obj,
        weights_arr,
        offset_arr,
        max_iter,
        tol,
    )

    if fit_intercept:
        intercept = _to_python_float(beta[0])
        coef = beta[1:]
    else:
        intercept = None
        coef = beta

    deviance_value = _to_python_float(deviance)
    n_params = X_design.shape[1]
    n_obs = X_design.shape[0]

    if fit_intercept:
        X_null = _ones_column(xp, X_arr.shape[0], like=X_arr)
        beta_null, _, _, _, _, null_dev = _irls(
            xp,
            X_null,
            y_arr,
            family_obj,
            link_obj,
            weights_arr,
            offset_arr,
            max_iter,
            tol,
        )
        null_deviance = _to_python_float(null_dev)
    else:
        null_deviance = deviance_value

    aic = deviance_value + 2.0 * n_params
    bic = deviance_value + math.log(max(n_obs, 1)) * n_params

    result = GLMResult(
        coef_=coef,
        intercept_=intercept,
        family=family_obj,
        link=link_obj,
        mu_=mu,
        eta_=eta_total,
        deviance_=deviance_value,
        null_deviance_=null_deviance,
        aic_=aic,
        bic_=bic,
        n_iter_=n_iter,
        converged_=converged,
        _X=X_arr,
        _y=y_arr,
        _weights=weights_arr,
        _fit_intercept=fit_intercept,
    )

    return result


def _coerce_family(family: str | Family) -> Family:
    if isinstance(family, str):
        key = family.lower()
        try:
            return _FAMILY_REGISTRY[key]()
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown family: {family!r}") from exc
    if isinstance(family, Family):
        return family
    raise TypeError("family must be a string key or a Family instance")


def _coerce_link(link: str | LinkFunction | None, family: Family) -> LinkFunction:
    if link is None:
        return family.default_link
    if isinstance(link, str):
        key = link.lower()
        try:
            return _LINK_REGISTRY[key]()
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown link: {link!r}") from exc
    if isinstance(link, LinkFunction):
        return link
    raise TypeError("link must be None, a string key, or a LinkFunction instance")


def _irls(
    xp,
    X: Array,
    y: Array,
    family: Family,
    link: LinkFunction,
    weights: Array | None,
    offset: Array | None,
    max_iter: int,
    tol: float,
) -> tuple[Array, Array, Array, int, bool, float]:
    mu = family.initialize(y)
    mu = as_namespace_array(mu, xp, like=y)
    eta_total = link.link(mu)
    if offset is not None:
        eta_linear = eta_total - offset
    else:
        eta_linear = eta_total

    deviance_prev = _to_python_float(family.deviance(y, mu))
    converged = False
    beta = _zeros_vector(xp, X.shape[1], like=X)
    iteration = 0

    for iteration in range(1, max_iter + 1):
        deriv = link.derivative(mu)
        variance = family.variance(mu)
        denom = _clamp_positive(deriv * deriv * variance, xp)
        weight_core = _reciprocal(denom, xp)

        if weights is not None:
            weight_core = weight_core * weights

        sqrt_w = _sqrt(weight_core, xp)
        X_weighted = X * sqrt_w[..., None]

        eta_with_offset = eta_linear if offset is None else eta_linear + offset
        working_response = eta_with_offset + (y - mu) * deriv
        if offset is not None:
            working_response = working_response - offset

        z_weighted = working_response * sqrt_w

        beta = _weighted_least_squares(xp, X_weighted, z_weighted)
        beta = beta.reshape(-1)

        eta_linear = (X @ beta).reshape(-1)
        eta_with_offset = eta_linear if offset is None else eta_linear + offset
        mu = link.inverse(eta_with_offset)

        deviance_value = family.deviance(y, mu)
        deviance_curr = _to_python_float(deviance_value)

        dev_change = abs(deviance_curr - deviance_prev) / (abs(deviance_prev) + 0.1)
        if dev_change < tol:
            converged = True
            deviance_prev = deviance_curr
            eta_total = eta_with_offset
            break

        deviance_prev = deviance_curr
        eta_total = eta_with_offset

    return beta, eta_total, mu, iteration, converged, deviance_prev


def _weighted_least_squares(xp, X_weighted: Array, z_weighted: Array) -> Array:
    if xp is np:
        X_mat = np.asarray(X_weighted, dtype=np.float64)
        z_vec = np.asarray(z_weighted, dtype=np.float64)
        n_samples, n_features = X_mat.shape
        gram = np.zeros((n_features, n_features), dtype=np.float64)
        rhs = np.zeros(n_features, dtype=np.float64)

        for row in range(n_samples):
            xi = X_mat[row]
            zi = z_vec[row]
            for i in range(n_features):
                rhs[i] += xi[i] * zi
                gram[i, i] += xi[i] * xi[i]
                for j in range(i + 1, n_features):
                    val = xi[i] * xi[j]
                    gram[i, j] += val
                    gram[j, i] += val

        solution = _solve_normal_equation_numpy(gram, rhs)
        return solution.astype(X_mat.dtype, copy=False)

    if hasattr(X_weighted, "contiguous"):
        X_weighted = X_weighted.contiguous()
    transpose = getattr(X_weighted, "transpose")
    X_t = transpose(-1, -2)
    rhs = X_t @ z_weighted
    gram = X_t @ X_weighted

    dtype = getattr(gram, "dtype", None)
    device = getattr(gram, "device", None)
    eye_kwargs: dict[str, Any] = {}
    if dtype is not None:
        eye_kwargs["dtype"] = dtype
    if device is not None:
        eye_kwargs["device"] = device
    eye = getattr(xp, "eye")(gram.shape[-1], **eye_kwargs)
    ridge_scalar = getattr(xp, "tensor")(1e-8, **eye_kwargs)
    gram = gram + ridge_scalar * eye

    unsqueeze = getattr(rhs, "unsqueeze", None)
    if callable(unsqueeze):
        rhs_column = unsqueeze(-1)
    else:  # pragma: no cover - fallback for namespaces without unsqueeze
        rhs_column = rhs.reshape(-1, 1)

    try:
        solution = xp.linalg.solve(gram, rhs_column)
    except Exception:  # pragma: no cover - fallback to pseudoinverse on failure
        pinv = xp.linalg.pinv(gram)
        solution = pinv @ rhs_column

    squeeze = getattr(solution, "squeeze", None)
    if callable(squeeze):
        return squeeze(-1)
    return solution.reshape(-1)


def _solve_normal_equation_numpy(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = gram.shape[0]
    eye = np.eye(n, dtype=gram.dtype)
    jitter = 1e-8
    for _ in range(6):
        try:
            return _gaussian_elimination_solve_numpy(gram + jitter * eye, rhs)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return _gaussian_elimination_solve_numpy(gram + jitter * eye, rhs, allow_singular=True)


def _gaussian_elimination_solve_numpy(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    tol: float = 1e-12,
    allow_singular: bool = False,
) -> np.ndarray:
    A = np.array(matrix, dtype=np.float64, copy=True)
    b = np.array(rhs, dtype=np.float64, copy=True)
    n = A.shape[0]

    for k in range(n):
        pivot_idx = k + int(np.argmax(np.abs(A[k:, k])))
        pivot_val = A[pivot_idx, k]
        if abs(pivot_val) < tol:
            if allow_singular:
                pivot_val = tol
            else:
                raise np.linalg.LinAlgError("Matrix is singular to working precision")

        if pivot_idx != k:
            A[[k, pivot_idx]] = A[[pivot_idx, k]]
            b[[k, pivot_idx]] = b[[pivot_idx, k]]

        pivot_val = A[k, k]
        A[k, k:] = A[k, k:] / pivot_val
        b[k] = b[k] / pivot_val

        for i in range(k + 1, n):
            factor = A[i, k]
            if factor == 0.0:
                continue
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x = np.empty(n, dtype=A.dtype)
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(A[i, i + 1 :], x[i + 1 :])

    return x.astype(matrix.dtype, copy=False)


def _zeros_vector(xp, length: int, *, like: Array) -> Array:
    dtype = getattr(like, "dtype", None)
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    device = getattr(like, "device", None)
    if device is not None:
        kwargs["device"] = device
    shape = (length,)
    return xp.zeros(shape, **kwargs)


def _ones_column(xp, rows: int, *, like: Array) -> Array:
    dtype = getattr(like, "dtype", None)
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    device = getattr(like, "device", None)
    if device is not None:
        kwargs["device"] = device
    shape = (rows, 1)
    return xp.ones(shape, **kwargs)


def _concat_columns(xp, left: Array, right: Array) -> Array:
    if xp is np:
        return np.concatenate((left, right), axis=1)
    cat = getattr(xp, "cat")
    return cat((left, right), dim=1)


def _clamp_positive(value: Array, xp, eps: float = 1e-12) -> Array:
    if xp is np:
        return np.clip(value, eps, None)
    tensor = getattr(xp, "tensor")
    tensor_kwargs: dict[str, Any] = {}
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if dtype is not None:
        tensor_kwargs["dtype"] = dtype
    if device is not None:
        tensor_kwargs["device"] = device
    eps_tensor = tensor(eps, **tensor_kwargs)
    clamp = getattr(xp, "clamp")
    return clamp(value, min=eps_tensor)


def _reciprocal(value: Array, xp) -> Array:
    return 1.0 / value


def _sqrt(value: Array, xp) -> Array:
    if xp is np:
        return np.sqrt(value)
    sqrt = getattr(xp, "sqrt")
    return sqrt(value)


def _to_python_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


__all__ = ["fit_glm"]
