# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Gaussian distribution family implementation."""

from __future__ import annotations

from .._utils import as_namespace_array, namespace, ones_like
from ..base import Family, LinkFunction
from ..links import IdentityLink

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


class GaussianFamily(Family):
    """Gaussian (normal) distribution family for GLM.

    The Gaussian family is the default for continuous response variables.
    It uses the identity link function by default and has constant variance
    V(mu) = variance.

    Parameters
    ----------
    variance : float, default=1.0
        Known variance parameter. Set to a known value for weighted least
        squares; estimated from data when left at the default.
    link : LinkFunction, optional
        Link function. Defaults to IdentityLink(), which is the canonical
        link for the Gaussian family.

    Attributes
    ----------
    default_link : LinkFunction
        The link function for this family (IdentityLink by default).

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.families import GaussianFamily
    >>> from aurora.models.glm import fit_glm
    >>> family = GaussianFamily()
    >>> mu = np.array([1.0, 2.0, 3.0])
    >>> family.variance(mu)
    array([1., 1., 1.])
    >>> # Use with fit_glm
    >>> X = np.random.randn(100, 2)
    >>> y = X @ np.array([1.5, -0.5]) + np.random.randn(100)
    >>> result = fit_glm(X, y, family=family)

    See Also
    --------
    PoissonFamily : For count data.
    GammaFamily : For positive continuous data.
    BinomialFamily : For binary/proportion data.
    """

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
