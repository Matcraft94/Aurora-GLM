# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Binomial distribution family implementation."""

from __future__ import annotations

import numpy as np

from .._utils import (
    as_namespace_array,
    clip_probability,
    log_factorial,
    log_gamma,
    namespace,
)
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

    This family follows the R ``binomial()`` convention: the response is a
    **proportion** in [0, 1] and the mean parameter μ is a probability. The
    variance function is V(μ) = μ(1 − μ) and the number of trials enters as
    a prior weight (McCullagh & Nelder 1989, §2.3).

    Usage conventions
    -----------------
    - **Binary (Bernoulli) data**: pass 0/1 responses; no weights needed.
    - **Grouped data**: pass observed proportions ``y = successes / trials``
      and the trial counts as weights. With ``fit_glm`` use
      ``fit_glm(X, proportions, family=BinomialFamily(), weights=trials)``.
      A constant number of trials can also be set via ``BinomialFamily(n=k)``,
      which ``fit_glm`` treats as ``weights=k`` when no explicit weights are
      given. When trial counts are supplied (either way), the log-likelihood
      includes the binomial coefficient term ``log C(n_i, n_i·y_i)`` so that
      ``log_likelihood_``/``aic_`` match R and statsmodels exactly.

    Parameters
    ----------
    n : float, default=1.0
        Constant number of trials per observation. Used as the prior weight
        in ``deviance``/``log_likelihood`` (and by ``fit_glm`` as
        ``weights=n``) when no explicit ``weights`` are passed. Leave at the
        default 1.0 for binary data or when passing per-observation trial
        counts through ``weights``.
    link : LinkFunction, optional
        Link function. Defaults to LogitLink(), which is the canonical link
        for the Binomial family. Alternatives include ProbitLink and CLogLogLink.

    Attributes
    ----------
    default_link : LinkFunction
        The link function for this family (LogitLink by default).
    n_trials : float
        The constant trial count ``n`` given at construction.

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
    >>> p = 1.0 / (1.0 + np.exp(-(X @ np.array([1.0, -0.5]))))
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

    @property
    def n_trials(self) -> float:
        """Constant number of trials per observation (default 1.0)."""
        return self._n

    def _weights_and_trials(self, xp, params, like):
        """Resolve prior weights / trial counts for deviance and log-likelihood.

        Returns ``(weights, trials)`` as namespace arrays (or None). Explicit
        ``weights`` take precedence, then an explicit ``n`` parameter, then the
        constant ``n`` given at construction (only when different from 1).
        """
        weights = params.get("weights")
        n_param = params.get("n")
        if weights is not None:
            w_arr = as_namespace_array(weights, xp, like=like)
            return w_arr, w_arr
        if n_param is not None:
            n_arr = as_namespace_array(n_param, xp, like=like)
            return n_arr, n_arr
        if self._n != 1.0:
            n_arr = as_namespace_array(self._n, xp, like=like)
            return n_arr, n_arr
        return None, None

    def log_likelihood(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=y_arr), xp)
        w_arr, n_arr = self._weights_and_trials(xp, params, mu_arr)

        contrib = y_arr * _safe_log(mu_arr, xp) + (1.0 - y_arr) * _safe_log(1.0 - mu_arr, xp)
        if w_arr is not None:
            contrib = w_arr * contrib
        total = contrib.sum()

        if n_arr is not None:
            # Binomial coefficient log C(n, n·y): makes the log-likelihood
            # (and hence AIC/BIC) match R's glm() and statsmodels exactly for
            # grouped binomial data. For Bernoulli data (n = 1) or 0/1
            # responses this term is exactly zero.
            k_arr = n_arr * y_arr
            log_binom_coef = (
                log_gamma(n_arr + 1.0, xp)
                - log_factorial(k_arr, xp)
                - log_gamma(n_arr - k_arr + 1.0, xp)
            )
            total = total + log_binom_coef.sum()
        return total

    def deviance(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        """Compute binomial deviance on the probability scale.

        Unit deviance d_i = 2 [y_i log(y_i/μ_i) + (1−y_i) log((1−y_i)/(1−μ_i))]
        with the convention 0·log 0 = 0; the total deviance is Σ w_i d_i
        (R convention, McCullagh & Nelder §2.3).
        """
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=y_arr), xp)
        w_arr, _ = self._weights_and_trials(xp, params, mu_arr)

        # _safe_log clips the ratio away from 0, so at y = 0 (resp. y = 1)
        # the product is exactly 0 * finite = 0, giving the 0 log 0 = 0 rule.
        term1 = y_arr * _safe_log(y_arr / mu_arr, xp)
        term2 = (1.0 - y_arr) * _safe_log((1.0 - y_arr) / (1.0 - mu_arr), xp)
        contrib = 2.0 * (term1 + term2)
        if w_arr is not None:
            contrib = w_arr * contrib
        return contrib.sum()

    def variance(self, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(mu)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=mu), xp)
        return mu_arr * (1.0 - mu_arr)

    def initialize(self, y):  # noqa: ANN001 - match Family signature
        xp = namespace(y)
        y_arr = as_namespace_array(y, xp, like=y)
        return clip_probability((y_arr + 0.5) / 2.0, xp)

    @property
    def default_link(self) -> LinkFunction:
        return self._link


__all__ = ["BinomialFamily"]
