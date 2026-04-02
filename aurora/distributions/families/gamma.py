# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Gamma distribution family implementation."""

from __future__ import annotations

from .._utils import as_namespace_array, ensure_positive, log_gamma, namespace
from ..base import Family, LinkFunction
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
    """Gamma distribution family for positive continuous data.

    The Gamma family is suitable for modeling strictly positive continuous
    responses such as durations, costs, or sizes. The variance function is
    V(mu) = mu^2 / shape. The canonical link is the inverse function.

    Parameters
    ----------
    shape : float, default=1.0
        Shape parameter (also called the dispersion parameter). Must be
        positive. Controls the relationship between mean and variance:
        Var(Y) = mu^2 / shape.
    link : LinkFunction, optional
        Link function. Defaults to InverseLink(), which is the canonical link
        for the Gamma family. Common alternatives include LogLink and
        IdentityLink.

    Attributes
    ----------
    default_link : LinkFunction
        The link function for this family (InverseLink by default).

    Raises
    ------
    ValueError
        If shape is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.families import GammaFamily
    >>> from aurora.models.glm import fit_glm
    >>> family = GammaFamily(shape=2.0)
    >>> mu = np.array([1.0, 2.0, 4.0])
    >>> family.variance(mu)  # mu^2 / shape
    array([0.5, 2. , 8. ])
    >>> # Use with fit_glm
    >>> X = np.random.randn(200, 2)
    >>> y = np.exp(X @ np.array([0.5, -0.3]) + np.random.randn(200))
    >>> result = fit_glm(X, y, family=GammaFamily(link=None))

    See Also
    --------
    GaussianFamily : For continuous data.
    InverseGaussianFamily : For positive skewed data with V(mu) = mu^3.
    PoissonFamily : For count data.
    """

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
