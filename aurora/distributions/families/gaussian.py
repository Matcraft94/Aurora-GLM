# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Gaussian distribution family implementation."""

from __future__ import annotations

import math

from .._utils import as_namespace_array, namespace, ones_like
from ..base import Family, LinkFunction
from ..links import IdentityLink

_LOG_2PI = math.log(2.0 * math.pi)

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
    variance : float or None, default=None
        Known variance parameter. Set to a known value for weighted least
        squares; estimated from data when None (the default).
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

    def __init__(self, variance: float | None = None, link: LinkFunction | None = None) -> None:
        self._variance = variance
        self._link = link or IdentityLink()

    def log_likelihood(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = as_namespace_array(mu, xp, like=y_arr)
        resid = y_arr - mu_arr
        n = y_arr.shape[0] if hasattr(y_arr, "shape") and y_arr.shape else len(y_arr)
        # Prior weights (R convention): llf = Σ wᵢℓᵢ, with n_eff = Σ wᵢ in the
        # normalization constant and σ̂² = Σ wᵢrᵢ² / Σ wᵢ when the variance is
        # estimated. Matches statsmodels freq_weights.
        w_arr = params.get("weights")
        if w_arr is not None:
            w_arr = as_namespace_array(w_arr, xp, like=mu_arr)
            n_eff = float(w_arr.sum())
            weighted_resid2 = w_arr * (resid**2)
        else:
            n_eff = float(n)
            weighted_resid2 = resid**2
        if "variance" in params:
            var_arr = as_namespace_array(params["variance"], xp, like=mu_arr)
        elif self._variance is not None:
            var_arr = as_namespace_array(self._variance, xp, like=mu_arr)
        else:
            rss = float(weighted_resid2.sum())
            var_arr = as_namespace_array(rss / max(n_eff, 1.0), xp, like=mu_arr)
        log_var = xp.log(var_arr)
        return (-0.5 * weighted_resid2 / var_arr).sum() - 0.5 * n_eff * (_LOG_2PI + log_var)

    def deviance(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = as_namespace_array(mu, xp, like=y_arr)
        # Unit deviance (R convention): RSS when no fixed variance is set.
        variance = params.get("variance", self._variance)
        if variance is None:
            variance = 1.0
        var_arr = as_namespace_array(variance, xp, like=mu_arr)
        resid = y_arr - mu_arr
        contrib = (resid**2) / var_arr
        w_arr = params.get("weights")
        if w_arr is not None:
            contrib = as_namespace_array(w_arr, xp, like=mu_arr) * contrib
        return contrib.sum()

    def variance(self, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(mu)
        mu_arr = as_namespace_array(mu, xp, like=mu)
        # Variance function V(μ) = 1 unless a fixed variance was given.
        scale = params.get("variance", self._variance)
        if scale is None:
            scale = 1.0
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
