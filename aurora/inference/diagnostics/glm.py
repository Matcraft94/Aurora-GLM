"""Residual and influence diagnostics for fitted GLMs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...models.base import GLMResult
from ...distributions.families import BinomialFamily, GammaFamily, GaussianFamily, PoissonFamily


@dataclass(frozen=True)
class GLMDiagnosticResult:
    """Container bundling common GLM diagnostic measures."""

    response_residuals: np.ndarray
    pearson_residuals: np.ndarray
    deviance_residuals: np.ndarray
    working_residuals: np.ndarray
    studentized_residuals: np.ndarray
    leverage: np.ndarray
    cooks_distance: np.ndarray
    dfbetas: np.ndarray
    summary: np.ndarray
    summary_columns: tuple[str, ...]


def glm_diagnostics(result: GLMResult) -> GLMDiagnosticResult:
    """Compute residual and influence diagnostics for a fitted GLM."""

    if result._X is None or result._y is None:
        raise RuntimeError("GLMResult is missing design matrix or response data.")

    y = _to_numpy(result._y)
    mu = _to_numpy(result.mu_)
    eta = _to_numpy(result.eta_)
    X = _ensure_2d(_to_numpy(result._X))

    if y.shape[0] != X.shape[0]:
        raise RuntimeError("Design matrix and response have inconsistent shapes.")

    fit_intercept = bool(result._fit_intercept)
    if fit_intercept:
        intercept_col = np.ones((X.shape[0], 1), dtype=X.dtype)
        design = np.concatenate((intercept_col, X), axis=1)
    else:
        design = X

    deriv = _to_numpy(result.link.derivative(result.mu_))
    variance = _to_numpy(result.family.variance(result.mu_))
    weights = _prepare_weights(result, deriv, variance)

    response = y - mu
    sqrt_var = np.sqrt(np.clip(variance, 1e-12, None))
    pearson = response / sqrt_var
    working = response * deriv
    deviance = _deviance_residuals(result.family, y, mu)

    cov = np.asarray(result.coef_cov_, dtype=float)
    leverage = _hat_diagonal(design, weights, cov)

    p = design.shape[1]
    scale = _estimate_dispersion(result.deviance_, p, design.shape[0])
    denom = np.clip(1.0 - leverage, 1e-12, None)
    cooks = (pearson**2 / np.clip(scale, 1e-12, None)) * (leverage / (p * denom**2))

    studentized = pearson / np.sqrt(denom)
    dfbetas = _dfbetas(design, weights, cov, pearson, leverage)

    summary_cols = (
        "response_residual",
        "pearson_residual",
        "deviance_residual",
        "studentized_residual",
        "leverage",
        "cooks_distance",
    )
    summary = np.column_stack((response, pearson, deviance, studentized, leverage, cooks))

    return GLMDiagnosticResult(
        response_residuals=response,
        pearson_residuals=pearson,
        deviance_residuals=deviance,
        working_residuals=working,
        studentized_residuals=studentized,
        leverage=leverage,
        cooks_distance=cooks,
        dfbetas=dfbetas,
        summary=summary,
        summary_columns=summary_cols,
    )


def _to_numpy(value: Any | None) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _ensure_2d(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        return matrix.reshape(-1, 1)
    return matrix


def _prepare_weights(result: GLMResult, deriv: np.ndarray, variance: np.ndarray) -> np.ndarray:
    denom = np.clip(deriv * deriv * variance, 1e-12, None)
    base_weights = 1.0 / denom
    if result._weights is not None:
        obs_weights = _to_numpy(result._weights)
        if obs_weights.shape != base_weights.shape:
            obs_weights = obs_weights.reshape(base_weights.shape)
        base_weights = base_weights * obs_weights
    return np.clip(base_weights, 1e-12, None)


def _deviance_residuals(family: Any, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    residual = y - mu
    if isinstance(family, GaussianFamily):
        variance = float(getattr(family, "_variance", 1.0))
        return residual / np.sqrt(np.clip(variance, 1e-12, None))
    if isinstance(family, PoissonFamily):
        mu_safe = np.clip(mu, 1e-12, None)
        ratio = np.where(y == 0.0, 1.0, y / mu_safe)
        log_term = np.log(np.clip(ratio, 1e-12, None))
        contrib = 2.0 * (y * log_term - (y - mu_safe))
        return np.sign(residual) * np.sqrt(np.clip(contrib, 0.0, None))
    if isinstance(family, BinomialFamily):
        n_param = float(getattr(family, "_n", 1.0))
        n_arr = np.full_like(mu, n_param)
        eps = 1e-12
        y_safe = np.clip(y, eps, n_arr - eps)
        mu_safe = np.clip(mu, eps, n_arr - eps)
        term1 = y * np.log(y_safe / mu_safe)
        term2 = (n_arr - y) * np.log((n_arr - y_safe) / (n_arr - mu_safe))
        contrib = 2.0 * (term1 + term2)
        return np.sign(residual) * np.sqrt(np.clip(contrib, 0.0, None))
    if isinstance(family, GammaFamily):
        mu_safe = np.clip(mu, 1e-12, None)
        ratio = np.clip(y / mu_safe, 1e-12, None)
        contrib = 2.0 * ((y - mu_safe) / mu_safe - np.log(ratio))
        return np.sign(residual) * np.sqrt(np.clip(contrib, 0.0, None))
    raise NotImplementedError(f"Unsupported family for diagnostics: {type(family).__name__}")


def _hat_diagonal(X: np.ndarray, weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n_samples = X.shape[0]
    leverage = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        xi = X[i]
        wi = weights[i]
        cov_x = cov @ xi
        leverage[i] = wi * float(np.dot(xi, cov_x))
    return np.clip(leverage, 0.0, 1.0)


def _estimate_dispersion(deviance: float, p: int, n_obs: int) -> float:
    df = max(n_obs - p, 1)
    return deviance / df


def _dfbetas(
    design: np.ndarray,
    weights: np.ndarray,
    covariance: np.ndarray,
    pearson_residuals: np.ndarray,
    leverage: np.ndarray,
) -> np.ndarray:
    n_samples, n_params = design.shape
    cov = np.asarray(covariance, dtype=np.float64)
    diag_cov = np.clip(np.diag(cov), 1e-12, None)
    std_params = np.sqrt(diag_cov)

    dfbetas = np.zeros((n_samples, n_params), dtype=np.float64)
    for i in range(n_samples):
        xi = design[i]
        wi = float(weights[i])
        leverage_i = float(np.clip(leverage[i], 0.0, 1.0))
        denom = np.sqrt(np.clip(1.0 - leverage_i, 1e-12, None))
        pearson_i = float(pearson_residuals[i])
        adjustment = pearson_i * wi / np.clip(denom, 1e-12, None)
        influence = cov @ (xi * wi)
        dfbetas[i] = influence * adjustment / std_params
    return dfbetas


__all__ = ["GLMDiagnosticResult", "glm_diagnostics"]
