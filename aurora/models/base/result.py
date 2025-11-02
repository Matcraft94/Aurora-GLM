"""Dataclasses encapsulating fitted model state."""
from __future__ import annotations

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
        """Placeholder for future inference calculations."""

        raise NotImplementedError("Inference metrics are not implemented yet.")

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


__all__.append("GLMResult")
