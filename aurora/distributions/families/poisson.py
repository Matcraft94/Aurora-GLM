# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Poisson distribution family implementation."""

from __future__ import annotations

import numpy as np

from ..base import Family, LinkFunction
from .._utils import as_namespace_array, ensure_positive, namespace
from ..links import LogLink

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None  # type: ignore[assignment]


class PoissonFamily(Family):
    """Poisson distribution family for count data.

    The Poisson family models count response variables where the variance
    equals the mean: V(mu) = mu. The canonical link is the log function.

    Parameters
    ----------
    link : LinkFunction, optional
        Link function. Defaults to LogLink(), which is the canonical link
        for the Poisson family.

    Attributes
    ----------
    default_link : LinkFunction
        The link function for this family (LogLink by default).

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.families import PoissonFamily
    >>> from aurora.models.glm import fit_glm
    >>> family = PoissonFamily()
    >>> mu = np.array([1.0, 2.0, 5.0])
    >>> family.variance(mu)  # V(mu) = mu
    array([1., 2., 5.])
    >>> # Use with fit_glm
    >>> X = np.random.randn(200, 2)
    >>> y = np.random.poisson(np.exp(X @ np.array([0.5, -0.3])))
    >>> result = fit_glm(X, y, family=family)

    See Also
    --------
    GaussianFamily : For continuous data.
    NegativeBinomialFamily : For overdispersed count data.
    BinomialFamily : For binary/proportion data.
    """

    def __init__(self, link: LinkFunction | None = None) -> None:
        self._link = link or LogLink()

    def log_likelihood(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=y_arr), xp)
        return (y_arr * xp.log(mu_arr) - mu_arr).sum()

    def deviance(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=y_arr), xp)
        if xp is torch:  # type: ignore[comparison-overlap]
            ones = torch.ones_like(mu_arr)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            ones = jnp.ones_like(mu_arr)
        else:
            ones = np.ones_like(mu_arr)
        ratio = xp.where(y_arr == 0, ones, y_arr / mu_arr)
        log_term = xp.log(ratio)
        return (2.0 * (y_arr * log_term - (y_arr - mu_arr))).sum()

    def variance(self, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return mu_arr

    def initialize(self, y):  # noqa: ANN001 - match Family signature
        xp = namespace(y)
        y_arr = as_namespace_array(y, xp, like=y)
        if xp is torch:  # type: ignore[comparison-overlap]
            return torch.clamp(y_arr + 0.1, min=0.1)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            return jnp.clip(y_arr + 0.1, 0.1, None)
        return np.clip(y_arr + 0.1, 0.1, None)

    @property
    def default_link(self) -> LinkFunction:
        return self._link


__all__ = ["PoissonFamily"]
