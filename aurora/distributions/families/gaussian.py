"""Gaussian distribution family implementation."""
from __future__ import annotations

import numpy as np

from ..base import Family, LinkFunction
from .._utils import as_namespace_array, namespace, ones_like
from ..links import IdentityLink

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


class GaussianFamily(Family):
    """Gaussian family with optional variance parameter and link."""

    def __init__(self, variance: float = 1.0, link: LinkFunction | None = None) -> None:
        self._variance = variance
        self._link = link or IdentityLink()

    def log_likelihood(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = as_namespace_array(mu, xp, like=y_arr)
        variance = params.get("variance", self._variance)
        var_arr = as_namespace_array(variance, xp, like=mu_arr)
        resid = y_arr - mu_arr
        return (-0.5 * (resid**2) / var_arr).sum()

    def deviance(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = as_namespace_array(mu, xp, like=y_arr)
        variance = params.get("variance", self._variance)
        var_arr = as_namespace_array(variance, xp, like=mu_arr)
        resid = y_arr - mu_arr
        return ((resid**2) / var_arr).sum()

    def variance(self, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(mu)
        mu_arr = as_namespace_array(mu, xp, like=mu)
        scale = params.get("variance", self._variance)
        scale_arr = as_namespace_array(scale, xp, like=mu_arr)
        if getattr(scale_arr, "shape", ()) == ():
            return ones_like(mu_arr) * scale_arr
        return scale_arr

    def initialize(self, y):  # noqa: ANN001 - match Family signature
        xp = namespace(y)
        return as_namespace_array(y, xp, like=y)

    @property
    def default_link(self) -> LinkFunction:
        return self._link


__all__ = ["GaussianFamily"]
