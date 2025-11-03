"""GLM-specific validation metrics."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...distributions._utils import as_namespace_array, namespace
from ...distributions.base import Family
from ...models.base import GLMResult


def generalized_deviance(y_true: Any, mu_pred: Any, family: Family) -> float:
    """Compute the generalized deviance for predictions under a GLM family."""

    xp = namespace(y_true, mu_pred)
    y_arr = as_namespace_array(y_true, xp)
    mu_arr = as_namespace_array(mu_pred, xp, like=y_arr)
    dev = family.deviance(y_arr, mu_arr)
    return _to_float(dev)


def aic(deviance: float, n_params: int) -> float:
    """Compute Akaike's Information Criterion from deviance and parameter count."""

    return float(deviance + 2.0 * n_params)


def bic(deviance: float, n_params: int, n_samples: int) -> float:
    """Compute Bayesian Information Criterion from deviance and sample size."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive for BIC calculation")
    return float(deviance + math.log(n_samples) * n_params)


def pseudo_r2(result: GLMResult, *, method: str = "mcfadden") -> float:
    """Compute pseudo :math:`R^2` scores from a fitted GLM result."""

    method_key = method.lower()
    deviance = float(result.deviance_)
    null_deviance = float(result.null_deviance_)

    if null_deviance <= 0.0 and method_key in {"mcfadden", "deviance", "mcfadden_adj"}:
        raise ValueError("null deviance must be positive for McFadden-style pseudo R^2")

    if method_key in {"mcfadden", "deviance"}:
        ratio = 1.0 - deviance / null_deviance
        return _clip_unit_interval(ratio)

    if method_key in {"mcfadden_adj", "adj_mcfadden", "mcfadden_adjusted"}:
        n_params = result.coef_.shape[0] + (1 if result.intercept_ is not None else 0)
        adjusted = 1.0 - (deviance + 2.0 * n_params) / null_deviance
        return _clip_unit_interval(adjusted)

    n_obs = _num_observations(result)
    if n_obs <= 0:
        raise ValueError("Cannot compute pseudo R^2 with zero observations")

    if method_key in {"cox_snell", "coxsnell"}:
        delta = null_deviance - deviance
        cs = 1.0 - math.exp(-delta / n_obs)
        return _clip_unit_interval(cs)

    if method_key in {"nagelkerke", "cragg_uhler", "cragg-uhler"}:
        delta = null_deviance - deviance
        numerator = 1.0 - math.exp(-delta / n_obs)
        denom = 1.0 - math.exp(-null_deviance / n_obs)
        if denom <= 0.0:
            raise ValueError("Null deviance too small to compute Nagelkerke pseudo R^2")
        nk = numerator / denom
        return _clip_unit_interval(nk)

    raise NotImplementedError(f"Unsupported pseudo R^2 method: {method!r}")


def _to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(np.asarray(value, dtype=np.float64))


def _num_observations(result: GLMResult) -> int:
    if result._X is not None:
        return int(_to_numpy_array(result._X).shape[0])
    return int(_to_numpy_array(result.mu_).shape[0])


def _clip_unit_interval(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


__all__ = ["generalized_deviance", "aic", "bic", "pseudo_r2"]
