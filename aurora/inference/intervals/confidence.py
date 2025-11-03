"""Confidence interval utilities for GLM results."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Tuple

import numpy as np

from ...models.base import GLMResult


@dataclass(frozen=True)
class ConfidenceIntervalResult:
    """Container for confidence interval bounds."""

    lower: np.ndarray
    upper: np.ndarray
    intercept: Tuple[float, float] | None = None


def confidence_intervals(
    result: GLMResult,
    *,
    level: float = 0.95,
    method: str = "wald",
    include_intercept: bool = True,
) -> ConfidenceIntervalResult:
    """Compute confidence intervals for the coefficients of a fitted GLM."""

    if not 0.0 < level < 1.0:
        raise ValueError("level must be in the open interval (0, 1).")
    if method.lower() != "wald":
        raise NotImplementedError(f"Unsupported confidence interval method: {method!r}")

    quantile = NormalDist().inv_cdf(0.5 + level / 2.0)

    coef = np.asarray(result.coef_, dtype=float)
    std_errors = np.asarray(result.std_errors_, dtype=float)
    lower = coef - quantile * std_errors
    upper = coef + quantile * std_errors

    intercept_interval: Tuple[float, float] | None = None
    if include_intercept and result.intercept_ is not None:
        intercept_std = result.intercept_std_error_
        if intercept_std is None:
            raise RuntimeError("Intercept standard error is unavailable.")
        intercept_lower = float(result.intercept_ - quantile * intercept_std)
        intercept_upper = float(result.intercept_ + quantile * intercept_std)
        intercept_interval = (intercept_lower, intercept_upper)

    return ConfidenceIntervalResult(lower=lower, upper=upper, intercept=intercept_interval)
