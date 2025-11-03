"""Wald hypothesis testing utilities."""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np

from ...models.base import GLMResult


def wald_test(
    result: GLMResult,
    contrast: Iterable[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    value: float | Iterable[float] | np.ndarray = 0.0,
    include_intercept: bool = True,
) -> dict[str, float]:
    """Perform a Wald test for (possibly multi-constraint) linear hypotheses.

    Parameters
    ----------
    result:
        Fitted GLM result providing coefficient estimates and covariance.
    contrast:
        Coefficient weights defining the hypothesis ``C beta = value``. Accepts
        one-dimensional vectors (single constraint) or matrices with one row per
        constraint.
    value:
        Hypothesised values for the linear combination(s). Can be a scalar or an
        iterable matching the number of constraints.
    include_intercept:
        Whether the contrast coefficients include the model intercept as the
        first element when present.

    Returns
    -------
    dict
        Mapping with keys ``statistic``, ``p_value`` and ``df``.
    """

    coef = _combine_parameters(result, include_intercept=include_intercept)
    cov = _full_covariance(result, include_intercept=include_intercept)

    contrast_mat = np.asarray(contrast, dtype=float)
    if contrast_mat.ndim == 1:
        contrast_mat = contrast_mat.reshape(1, -1)
    if contrast_mat.ndim != 2:
        raise ValueError("contrast must be a vector or matrix of coefficients")

    n_constraints, n_params = contrast_mat.shape
    if n_params != coef.shape[0]:
        raise ValueError("contrast dimension does not match number of parameters")

    value_arr = np.asarray(value, dtype=float)
    if value_arr.ndim == 0:
        value_vec = np.full((n_constraints,), float(value_arr))
    else:
        value_vec = value_arr.reshape(-1)
    if value_vec.shape[0] != n_constraints:
        raise ValueError("value dimension must match number of constraints")

    estimate = contrast_mat @ coef
    diff = estimate - value_vec

    cov_proj = contrast_mat @ cov @ contrast_mat.T
    cov_proj = _symmetrise_matrix(cov_proj)

    if n_constraints == 1:
        variance = float(np.clip(cov_proj[0, 0], 1e-12, None))
        z_score = float(diff[0] / math.sqrt(variance))
        p_value = 2.0 * (1.0 - NormalDist().cdf(abs(z_score)))
        statistic = z_score * z_score
        return {"statistic": statistic, "p_value": p_value, "df": 1.0}

    statistic = float(_quadratic_form(diff, cov_proj))
    df = float(n_constraints)
    p_value = _chi_square_sf(statistic, df)
    return {"statistic": statistic, "p_value": p_value, "df": df}


def _combine_parameters(result: GLMResult, *, include_intercept: bool) -> np.ndarray:
    coef = np.asarray(result.coef_, dtype=float)
    if include_intercept and result.intercept_ is not None:
        return np.concatenate(([float(result.intercept_)], coef))
    return coef


def _full_covariance(result: GLMResult, *, include_intercept: bool) -> np.ndarray:
    cov = np.asarray(result.coef_cov_, dtype=float)
    if include_intercept and result.intercept_ is not None:
        return cov
    if include_intercept and result.intercept_ is None:
        raise ValueError("Model does not include an intercept parameter.")
    if not include_intercept:
        if result.intercept_ is not None:
            return cov[1:, 1:]
    return cov


def _symmetrise_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _quadratic_form(diff: np.ndarray, covariance: np.ndarray) -> float:
    cov = np.asarray(covariance, dtype=float)
    diff_vec = np.asarray(diff, dtype=float).reshape(-1, 1)
    try:
        solve = np.linalg.solve(cov, diff_vec)
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(cov)
        solve = pinv @ diff_vec
    return float((diff_vec.T @ solve).item())


def _chi_square_sf(value: float, df: float) -> float:
    value = float(value)
    df = float(df)
    if value <= 0.0:
        return 1.0
    a = df / 2.0
    x = value / 2.0
    cdf = _regularized_gamma_p(a, x)
    sf = max(0.0, min(1.0, 1.0 - cdf))
    return sf


def _regularized_gamma_p(a: float, x: float, *, tol: float = 1e-12, max_iter: int = 1_000) -> float:
    if a <= 0.0:
        raise ValueError("shape parameter a must be positive")
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if x == 0.0:
        return 0.0

    if x < a + 1.0:
        # Series expansion for lower incomplete gamma
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(max_iter):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * tol:
                break
        log_part = -x + a * math.log(x) - math.lgamma(a)
        return float(total * math.exp(log_part))

    # Continued fraction for the complement; compute Q(a, x) then return 1 - Q
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break

    log_part = -x + a * math.log(x) - math.lgamma(a)
    q = math.exp(log_part) * h
    return float(1.0 - q)
