"""Wald hypothesis testing utilities."""
from __future__ import annotations

from statistics import NormalDist
from typing import Iterable

import numpy as np

from ...models.base import GLMResult


def wald_test(
    result: GLMResult,
    contrast: Iterable[float] | np.ndarray,
    *,
    value: float = 0.0,
    include_intercept: bool = True,
) -> dict[str, float]:
    """Perform a Wald test for a linear hypothesis of GLM coefficients.

    Parameters
    ----------
    result:
        Fitted GLM result providing coefficient estimates and covariance.
    contrast:
        Coefficient weights defining the hypothesis ``c^T beta = value``.
        Only single-constraint contrasts are currently supported.
    value:
        Hypothesised value for the linear combination. Defaults to ``0``.
    include_intercept:
        Whether the contrast coefficients include the model intercept as the
        first element when present.

    Returns
    -------
    dict
        Mapping with keys ``statistic``, ``p_value`` and ``df``.

    Notes
    -----
    The implementation currently supports only single-constraint contrasts
    (``df = 1``). Multi-parameter tests will raise ``NotImplementedError``.
    """

    coef = _combine_parameters(result, include_intercept=include_intercept)
    cov = _full_covariance(result, include_intercept=include_intercept)

    contrast_vec = np.asarray(list(contrast) if not isinstance(contrast, np.ndarray) else contrast, dtype=float)
    if contrast_vec.ndim == 2:
        if contrast_vec.shape[0] != 1:
            raise NotImplementedError("Multi-constraint Wald tests are not implemented yet.")
        contrast_vec = contrast_vec.reshape(-1)
    if contrast_vec.ndim != 1:
        raise ValueError("contrast must be a one-dimensional vector or a 1xP matrix")

    if contrast_vec.shape[0] != coef.shape[0]:
        raise ValueError("contrast length does not match number of parameters")

    estimate = float(np.dot(contrast_vec, coef))
    variance = float(np.dot(contrast_vec, cov @ contrast_vec))
    variance = max(variance, 1e-12)
    diff = estimate - value
    z_score = diff / np.sqrt(variance)
    p_value = 2.0 * (1.0 - NormalDist().cdf(abs(z_score)))

    statistic = z_score * z_score
    return {"statistic": statistic, "p_value": p_value, "df": 1.0}


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
