"""Gamma distribution family implementation."""
from __future__ import annotations

import numpy as np

from ..base import Family, LinkFunction
from .._utils import as_namespace_array, ensure_positive, log_gamma, namespace
from ..links import InverseLink

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None  # type: ignore[assignment]


class GammaFamily(Family):
    """Gamma family with optional shape parameter and inverse link."""

    def __init__(self, shape: float = 1.0, link: LinkFunction | None = None) -> None:
        if shape <= 0:
            raise ValueError("shape must be positive")
        self._shape = shape
        self._link = link or InverseLink()

    def _shape_array(self, xp, like):
        shape_param = self._shape
        return ensure_positive(as_namespace_array(shape_param, xp, like=like), xp)

    def log_likelihood(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(y, mu)
        y_arr = ensure_positive(as_namespace_array(y, xp, like=mu), xp)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=y_arr), xp)
        shape_param = params.get("shape", self._shape)
        shape_arr = ensure_positive(as_namespace_array(shape_param, xp, like=mu_arr), xp)
        term1 = shape_arr * (xp.log(shape_arr) - xp.log(mu_arr))
        term2 = (shape_arr - 1.0) * xp.log(y_arr)
        term3 = -shape_arr * y_arr / mu_arr
        term4 = -log_gamma(shape_arr, xp)
        return (term1 + term2 + term3 + term4).sum()

    def deviance(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(y, mu)
        y_arr = ensure_positive(as_namespace_array(y, xp, like=mu), xp)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=y_arr), xp)
        ratio = y_arr / mu_arr
        return (2.0 * ((y_arr - mu_arr) / mu_arr - xp.log(ratio))).sum()

    def variance(self, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        shape_param = params.get("shape", self._shape)
        shape_arr = ensure_positive(as_namespace_array(shape_param, xp, like=mu_arr), xp)
        return (mu_arr**2) / shape_arr

    def initialize(self, y):  # noqa: ANN001 - match Family signature
        xp = namespace(y)
        y_arr = ensure_positive(as_namespace_array(y, xp, like=y), xp)
        return y_arr

    @property
    def default_link(self) -> LinkFunction:
        return self._link


__all__ = ["GammaFamily"]
