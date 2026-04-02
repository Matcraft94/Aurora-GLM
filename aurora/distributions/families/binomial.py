# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Binomial distribution family implementation."""

from __future__ import annotations

import numpy as np

from .._utils import as_namespace_array, clip_probability, namespace
from ..base import Family, LinkFunction
from ..links import LogitLink

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None  # type: ignore[assignment]


def _safe_log(value, xp, eps: float = 1e-12):
    """Compute log with numeric stability across backends.

    Parameters
    ----------
    value : array
        Input array
    xp : module
        Array namespace (np, torch, or jnp)
    eps : float, default=1e-12
        Minimum value for numerical stability

    Returns
    -------
    array
        log(max(value, eps))

    Notes
    -----
    Uses consistent epsilon (1e-12) across all backends for numerical stability.
    Creates tensor with correct dtype and device for PyTorch/JAX compatibility.
    """
    if xp is torch:  # type: ignore[comparison-overlap]
        eps_tensor = torch.tensor(eps, dtype=value.dtype, device=value.device)
        return torch.log(torch.clamp(value, min=eps_tensor))
    elif xp is jnp:  # type: ignore[comparison-overlap]
        return jnp.log(jnp.clip(value, eps, None))
    return np.log(np.clip(value, eps, None))


class BinomialFamily(Family):
    """Binomial distribution family for binary and proportion data.

    The Binomial family models binary outcomes or proportions. For binary
    data (0/1), use n=1 (default). For grouped binomial data, set n to
    the number of trials. The variance function is V(mu) = n * mu * (1 - mu/n).

    Parameters
    ----------
    n : float, default=1.0
        Number of trials. Use 1.0 for binary (Bernoulli) data. For grouped
        binomial responses, set to the number of trials per observation.
    link : LinkFunction, optional
        Link function. Defaults to LogitLink(), which is the canonical link
        for the Binomial family. Alternatives include ProbitLink and CLogLogLink.

    Attributes
    ----------
    default_link : LinkFunction
        The link function for this family (LogitLink by default).

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.families import BinomialFamily
    >>> from aurora.models.glm import fit_glm
    >>> family = BinomialFamily()
    >>> mu = np.array([0.2, 0.5, 0.8])
    >>> family.variance(mu)  # V(mu) = mu * (1 - mu)
    array([0.16, 0.25, 0.16])
    >>> # Use with fit_glm for logistic regression
    >>> X = np.random.randn(200, 2)
    >>> p = 1.0 / (1.0 + np.exp(-(X @ np.array([1.0, -0.5])))
    >>> y = np.random.binomial(1, p)
    >>> result = fit_glm(X, y, family=family)

    See Also
    --------
    GaussianFamily : For continuous data.
    PoissonFamily : For count data.
    BetaFamily : For continuous proportions in (0, 1).
    """

    def __init__(self, n: float = 1.0, link: LinkFunction | None = None) -> None:
        self._n = n
        self._link = link or LogitLink()

    def log_likelihood(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = as_namespace_array(mu, xp, like=y_arr)
        n_param = params.get("n", self._n)
        n_arr = as_namespace_array(n_param, xp, like=mu_arr)
        probability = clip_probability(mu_arr / n_arr, xp)
        term1 = y_arr * _safe_log(probability, xp)
        term2 = (n_arr - y_arr) * _safe_log(1.0 - probability, xp)
        return (term1 + term2).sum()

    def deviance(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        """Compute binomial deviance with consistent epsilon handling.

        Uses _safe_log() for consistency with log_likelihood().
        """
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = as_namespace_array(mu, xp, like=y_arr)
        n_param = params.get("n", self._n)
        n_arr = as_namespace_array(n_param, xp, like=mu_arr)

        eps = 1e-12

        # Clamp y and mu to valid range [eps, n-eps]
        if xp is torch:
            eps_tensor = torch.tensor(eps, dtype=mu_arr.dtype, device=mu_arr.device)
            y_safe = torch.clamp(y_arr, min=eps_tensor, max=n_arr - eps_tensor)
            mu_safe = torch.clamp(mu_arr, min=eps_tensor, max=n_arr - eps_tensor)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            y_safe = jnp.clip(y_arr, eps, n_arr - eps)
            mu_safe = jnp.clip(mu_arr, eps, n_arr - eps)
        else:
            y_safe = np.clip(y_arr, eps, n_arr - eps)
            mu_safe = np.clip(mu_arr, eps, n_arr - eps)

        # Use _safe_log for consistency with log_likelihood
        term1 = y_arr * _safe_log(y_safe / mu_safe, xp, eps=eps)
        term2 = (n_arr - y_arr) * _safe_log((n_arr - y_safe) / (n_arr - mu_safe), xp, eps=eps)

        return (2.0 * (term1 + term2)).sum()

    def variance(self, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(mu)
        mu_arr = as_namespace_array(mu, xp, like=mu)
        n_param = params.get("n", self._n)
        n_arr = as_namespace_array(n_param, xp, like=mu_arr)
        probability = clip_probability(mu_arr / n_arr, xp)
        return n_arr * probability * (1.0 - probability)

    def initialize(self, y):  # noqa: ANN001 - match Family signature
        xp = namespace(y)
        y_arr = as_namespace_array(y, xp, like=y)
        n_arr = as_namespace_array(self._n, xp, like=y_arr)
        p_init = clip_probability((y_arr + 0.5) / (n_arr + 1.0), xp)
        return n_arr * p_init

    @property
    def default_link(self) -> LinkFunction:
        return self._link


__all__ = ["BinomialFamily"]
