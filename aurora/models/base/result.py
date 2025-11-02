"""Dataclasses encapsulating fitted model state."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...core.types import Array, ArrayLike
from ...distributions._utils import as_namespace_array, namespace
from ...distributions.base import Family, LinkFunction


@dataclass(frozen=True)
class ModelResult:
    """Immutable container storing the outcome of a model fitting routine."""

    params: ArrayLike
    fitted_values: ArrayLike
    converged: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary representation."""
        return {
            "params": self.params,
            "converged": self.converged,
            "diagnostics": self.diagnostics,
        }

    def predict(self, design_matrix: ArrayLike, *, backend: str = "jax") -> ArrayLike:
        """Dispatch to the selected backend to generate predictions."""
        from ...core.backends import get_backend

        backend_impl = get_backend(backend)
        return backend_impl.array(design_matrix) @ backend_impl.array(self.params)


__all__ = ["ModelResult"]


@dataclass
class GLMResult:
    """Structured container describing the outcome of a GLM fit."""

    coef_: Array
    intercept_: float | None
    family: Family
    link: LinkFunction
    mu_: Array
    eta_: Array
    deviance_: float
    null_deviance_: float
    aic_: float
    bic_: float
    n_iter_: int
    converged_: bool
    _coef_cov: Array | None = None
    _std_errors: Array | None = None
    _p_values: Array | None = None
    _X: Array | None = None
    _y: Array | None = None
    _weights: Array | None = None
    _fit_intercept: bool = True
    _intercept_std_error: float | None = None
    _intercept_p_value: float | None = None

    @property
    def std_errors_(self) -> Array:
        """Return standard errors for the fitted coefficients."""

        if self._std_errors is None:
            self._compute_inference()
        return self._std_errors

    @property
    def p_values_(self) -> Array:
        """Return Wald p-values for the fitted coefficients."""

        if self._p_values is None:
            self._compute_inference()
        return self._p_values

    @property
    def coef_cov_(self) -> Array:
        """Return the covariance matrix of the fitted coefficients."""

        if self._coef_cov is None:
            self._compute_inference()
        return self._coef_cov

    def _compute_inference(self) -> None:
        """Compute covariance matrix, standard errors, and Wald p-values."""

        if self._X is None or self._y is None:
            raise RuntimeError("Design matrix and response are required for inference.")

        X_np = _to_numpy(self._X)
        xp = namespace(self.mu_)
        mu_xp = as_namespace_array(self.mu_, xp, like=self.mu_)
        deriv_xp = as_namespace_array(self.link.derivative(mu_xp), xp, like=mu_xp)
        variance_xp = as_namespace_array(self.family.variance(mu_xp), xp, like=mu_xp)
        deriv_np = _to_numpy(deriv_xp)
        variance_np = _to_numpy(variance_xp)

        weights_np = _to_numpy(self._weights) if self._weights is not None else None

        fisher = _fisher_information_numpy(
            X=X_np,
            deriv=deriv_np,
            variance=variance_np,
            weights=weights_np,
            fit_intercept=self._fit_intercept,
        )

        cov_full = _invert_information_numpy(fisher)
        self._coef_cov = cov_full

        diag = np.clip(np.diag(cov_full), 1e-12, None)
        std_full = np.sqrt(diag)

        coef_np = _to_numpy(self.coef_)
        coef_full = _combine_coefficients_numpy(self.intercept_, coef_np, self._fit_intercept)

        z_scores = np.divide(coef_full, std_full, out=np.zeros_like(std_full), where=std_full > 0)
        p_full = 2.0 * (1.0 - _standard_normal_cdf_numpy(np.abs(z_scores)))

        if self._fit_intercept and self.intercept_ is not None:
            self._intercept_std_error = float(std_full[0])
            self._intercept_p_value = float(p_full[0])
            self._std_errors = std_full[1:]
            self._p_values = p_full[1:]
        else:
            self._intercept_std_error = None
            self._intercept_p_value = None
            self._std_errors = std_full
            self._p_values = p_full

    @property
    def intercept_std_error_(self) -> float | None:
        """Standard error of the intercept parameter, if fitted."""

        if self.intercept_ is None:
            return None
        if self._intercept_std_error is None:
            self._compute_inference()
        return self._intercept_std_error

    @property
    def intercept_p_value_(self) -> float | None:
        """Wald p-value for the intercept parameter, if fitted."""

        if self.intercept_ is None:
            return None
        if self._intercept_p_value is None:
            self._compute_inference()
        return self._intercept_p_value

    def predict(
        self,
        X_new: Array,
        *,
        type: str = "response",
        backend: str | None = None,
        interval: str | None = None,
        level: float = 0.95,
    ) -> Array:
        """Generate predictions for new design matrices."""

        del backend  # Reserved for future multi-backend dispatching
        del level  # Level support will be introduced with intervals

        if interval is not None:
            raise NotImplementedError("Prediction intervals are not implemented yet.")

        xp = namespace(X_new, self._X, self._y)
        like = self._X if self._X is not None else self.mu_
        X_arr = as_namespace_array(X_new, xp, like=like)

        if xp is np:
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
        else:
            if getattr(X_arr, "ndim", 1) == 1:
                X_arr = X_arr.unsqueeze(0)

        coef_arr = as_namespace_array(self.coef_, xp, like=X_arr)
        if coef_arr.ndim == 1:
            coef_arr = coef_arr.reshape(-1, 1)

        eta = X_arr @ coef_arr
        eta = eta.squeeze(-1)

        if self.intercept_ is not None:
            eta = eta + self.intercept_

        if type == "link":
            return eta
        if type == "response":
            return self.link.inverse(eta)

        raise ValueError(f"Unknown prediction type: {type!r}")


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


def _fisher_information_numpy(
    *,
    X: np.ndarray,
    deriv: np.ndarray,
    variance: np.ndarray,
    weights: np.ndarray | None,
    fit_intercept: bool,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    deriv = np.asarray(deriv, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)

    denom = np.clip(deriv * deriv * variance, 1e-12, None)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64) / denom
    else:
        w = 1.0 / denom
    w = np.clip(w, 1e-12, None)

    if fit_intercept:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        X_design = np.concatenate((ones, X), axis=1)
    else:
        X_design = X

    return _weighted_gram_numpy(X_design, w)


def _invert_information_numpy(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    eye = np.eye(n, dtype=matrix.dtype)
    jitter = 1e-8
    for _ in range(6):
        try:
            return np.linalg.solve(matrix + jitter * eye, eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return np.linalg.pinv(matrix + jitter * eye)


def _combine_coefficients_numpy(
    intercept: float | None,
    coef: np.ndarray,
    fit_intercept: bool,
) -> np.ndarray:
    coef = np.asarray(coef, dtype=np.float64)
    if fit_intercept and intercept is not None:
        return np.concatenate(([float(intercept)], coef))
    return coef


def _standard_normal_cdf_numpy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def _weighted_gram_numpy(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n_samples, n_features = X.shape
    weights = np.asarray(weights, dtype=np.float64)
    gram = np.zeros((n_features, n_features), dtype=np.float64)
    for idx in range(n_samples):
        xi = X[idx]
        wi = weights[idx]
        for j in range(n_features):
            gram[j, j] += wi * xi[j] * xi[j]
            for k in range(j + 1, n_features):
                val = wi * xi[j] * xi[k]
                gram[j, k] += val
                gram[k, j] += val
    return gram


__all__.append("GLMResult")
