"""Inverse Gaussian (Wald) distribution family implementation.

The Inverse Gaussian distribution is suitable for modeling positive continuous
data with right skew, such as failure times, reaction times, or durations.

Mathematical Framework
----------------------
The Inverse Gaussian distribution has density:

**Density**:
    f(y; μ, λ) = √(λ/(2πy³)) × exp{-λ(y-μ)²/(2μ²y)}

where:
    - y > 0 is the response
    - μ > 0 is the mean
    - λ > 0 is the shape (dispersion) parameter

**Properties**:
    - E[Y] = μ
    - Var(Y) = μ³/λ
    - Also known as the Wald distribution
    - Related to first passage time of Brownian motion

**Variance function**:
    V(μ) = μ³

**Canonical link**: Inverse square g(μ) = 1/μ²
**Common alternatives**: Log, Inverse, Identity

References
----------
- Tweedie, M. C. K. (1957). "Statistical properties of inverse Gaussian
  distributions." The Annals of Mathematical Statistics, 28(2), 362-377.
- Chhikara, R. S., & Folks, J. L. (1989). The Inverse Gaussian Distribution.
  Marcel Dekker.
- McCullagh, P., & Nelder, J. A. (1989). Generalized Linear Models (2nd ed.).
  Chapman and Hall/CRC.

See Also
--------
aurora.distributions.links.common.InverseSquareLink : Canonical link
aurora.distributions.links.common.LogLink : Common alternative
aurora.distributions.links.common.InverseLink : Alternative link
"""
from __future__ import annotations

import math

import numpy as np

from ..base import Family, LinkFunction
from .._utils import as_namespace_array, ensure_positive, namespace
from ..links import InverseSquareLink

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None  # type: ignore[assignment]


class InverseGaussianFamily(Family):
    """Inverse Gaussian (Wald) distribution for positive durations.

    Suitable for modeling:
    - Time-to-event data (failure times)
    - Reaction times
    - Right-skewed positive continuous data
    - Insurance claim amounts

    Parameters
    ----------
    lambda_ : float or 'estimate', default=1.0
        Shape parameter (controls dispersion).
        - lambda large: Low variance, concentrated
        - lambda small: High variance, dispersed
        - 'estimate': Estimate from data via method-of-moments

    link : LinkFunction or None, default=None
        Link function. If None, uses InverseSquareLink (canonical).
        Common alternatives: LogLink, InverseLink, IdentityLink.

    Attributes
    ----------
    lambda_ : float or str
        Shape parameter or 'estimate' for automatic estimation.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.families import InverseGaussianFamily
    >>> from aurora.distributions.links import LogLink

    >>> # Basic usage with canonical link
    >>> family = InverseGaussianFamily(lambda_=2.0)
    >>> mu = np.array([1.0, 2.0, 3.0])
    >>> family.variance(mu)  # μ³
    array([ 1.,  8., 27.])

    >>> # With log link (more common in practice)
    >>> family = InverseGaussianFamily(lambda_=2.0, link=LogLink())

    Notes
    -----
    The variance function is V(μ) = μ³, which means variance increases
    rapidly with the mean. This makes the inverse Gaussian suitable for
    data where larger values are more variable.

    The canonical link (inverse square) is rarely used in practice;
    the log or inverse link are more common choices.

    For failure time data, the inverse Gaussian arises naturally as the
    distribution of the first passage time for Brownian motion with drift.
    """

    def __init__(
        self,
        lambda_: float | str = 1.0,
        link: LinkFunction | None = None
    ) -> None:
        """Initialize Inverse Gaussian family.

        Parameters
        ----------
        lambda_ : float or 'estimate', default=1.0
            Shape parameter. Use 'estimate' for method-of-moments.
        link : LinkFunction or None
            Link function (default: InverseSquareLink).
        """
        if isinstance(lambda_, str):
            if lambda_ != 'estimate':
                raise ValueError("lambda_ must be a positive float or 'estimate'")
            self._lambda = lambda_
        else:
            if lambda_ <= 0:
                raise ValueError("lambda_ must be positive")
            self._lambda = float(lambda_)

        self._link = link or InverseSquareLink()

    @property
    def lambda_(self) -> float | str:
        """Return the shape parameter."""
        return self._lambda

    def _get_lambda(self, xp, like, y=None, mu=None, **params):
        """Get lambda value, estimating if needed."""
        lambda_param = params.get("lambda_", self._lambda)

        if lambda_param == 'estimate' and y is not None and mu is not None:
            # Method-of-moments estimation
            lambda_param = self._estimate_lambda_mm(y, mu, xp)
        elif lambda_param == 'estimate':
            lambda_param = 1.0  # Fallback

        if isinstance(lambda_param, str):
            lambda_param = 1.0

        return ensure_positive(as_namespace_array(lambda_param, xp, like=like), xp)

    def _estimate_lambda_mm(self, y, mu, xp):
        """Estimate lambda via method-of-moments.

        For Inverse Gaussian:
            Var(Y) = μ³/λ

        Using residuals r = y - μ:
            λ ≈ E[μ³] / E[(y-μ)² × μ]

        More stable: use the deviance-based estimator
            λ = n / Σ[(y-μ)² / (μ²y)]
        """
        if xp is torch:  # type: ignore[comparison-overlap]
            y_arr = y
            mu_arr = mu
            n = y.shape[0]
        elif xp is jnp:  # type: ignore[comparison-overlap]
            y_arr = y
            mu_arr = mu
            n = y.shape[0]
        else:
            y_arr = np.asarray(y)
            mu_arr = np.asarray(mu)
            n = len(y_arr)

        # Deviance-based estimator
        # From McCullagh & Nelder: φ = (1/n) Σ[(y-μ)² / (μ²y)]
        # λ = 1/φ
        mu_safe = ensure_positive(mu_arr, xp)
        y_safe = ensure_positive(y_arr, xp)

        dev_contrib = (y_safe - mu_safe)**2 / (mu_safe**2 * y_safe)

        if xp is torch:  # type: ignore[comparison-overlap]
            phi_est = torch.mean(dev_contrib)
            lambda_est = 1.0 / phi_est
            return max(float(lambda_est), 0.01)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            phi_est = jnp.mean(dev_contrib)
            lambda_est = 1.0 / phi_est
            return max(float(lambda_est), 0.01)
        else:
            phi_est = np.mean(dev_contrib)
            lambda_est = 1.0 / phi_est
            return max(lambda_est, 0.01)

    def log_likelihood(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        """Compute log-likelihood for Inverse Gaussian distribution.

        Parameters
        ----------
        y : array
            Observed positive values.
        mu : array
            Mean parameter (positive).
        **params : dict
            Optional 'lambda_' parameter override.

        Returns
        -------
        scalar
            Sum of log-likelihoods.

        Notes
        -----
        Log-likelihood:
            ℓ = 0.5 × [log(λ) - log(2π) - 3log(y)] - λ(y-μ)² / (2μ²y)
        """
        xp = namespace(y, mu)

        y_arr = ensure_positive(as_namespace_array(y, xp, like=mu), xp)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=y_arr), xp)

        lambda_val = self._get_lambda(xp, mu_arr, y=y_arr, mu=mu_arr, **params)

        # Log-likelihood
        # f(y; μ, λ) = √(λ/(2πy³)) × exp{-λ(y-μ)²/(2μ²y)}
        # log f = 0.5[log λ - log(2π) - 3 log y] - λ(y-μ)²/(2μ²y)

        # Use namespace-compatible constant for 2π
        two_pi = as_namespace_array(2 * np.pi, xp, like=y_arr)

        log_lik = (
            0.5 * (xp.log(lambda_val) - xp.log(two_pi) - 3 * xp.log(y_arr))
            - lambda_val * (y_arr - mu_arr)**2 / (2 * mu_arr**2 * y_arr)
        )

        if xp is torch:  # type: ignore[comparison-overlap]
            return torch.sum(log_lik)
        return np.sum(log_lik)

    def deviance(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        """Compute deviance for Inverse Gaussian distribution.

        The unit deviance is:
            d(y, μ) = (y - μ)² / (μ² y)

        Total deviance: D = Σ d(y, μ)

        Parameters
        ----------
        y : array
            Observed values.
        mu : array
            Fitted mean values.
        **params : dict
            Optional parameters (not used for deviance).

        Returns
        -------
        scalar
            Deviance value.

        Notes
        -----
        This is the "scaled deviance" without λ. The full deviance
        would be λ × D.
        """
        xp = namespace(y, mu)

        y_arr = ensure_positive(as_namespace_array(y, xp, like=mu), xp)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=y_arr), xp)

        # Unit deviance: (y - μ)² / (μ² y)
        unit_dev = (y_arr - mu_arr)**2 / (mu_arr**2 * y_arr)

        if xp is torch:  # type: ignore[comparison-overlap]
            return torch.sum(unit_dev)
        return np.sum(unit_dev)

    def variance(self, mu, **params):  # noqa: ANN001 - match Family signature
        """Compute variance function V(μ) = μ³.

        Parameters
        ----------
        mu : array
            Mean parameter (positive).
        **params : dict
            Optional parameters (not used).

        Returns
        -------
        array
            Variance function values.

        Notes
        -----
        The true variance is Var(Y) = μ³/λ, but the variance function
        (without dispersion) is V(μ) = μ³.
        """
        xp = namespace(mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)

        return mu_arr**3

    def full_variance(self, mu, **params):
        """Compute full variance: Var(Y) = μ³/λ.

        Unlike variance(), this includes the shape parameter.

        Parameters
        ----------
        mu : array
            Mean parameter.
        **params : dict
            Optional 'lambda_' parameter.

        Returns
        -------
        array
            Full variance values.
        """
        xp = namespace(mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        lambda_val = self._get_lambda(xp, mu_arr, **params)

        return mu_arr**3 / lambda_val

    def initialize(self, y):  # noqa: ANN001 - match Family signature
        """Initialize mean parameter from data.

        Parameters
        ----------
        y : array
            Observed positive values.

        Returns
        -------
        array
            Initial mean values (sample mean, broadcast to array shape).

        Notes
        -----
        Uses the sample mean as initial value, which is a natural
        choice for positive data.
        """
        xp = namespace(y)
        y_arr = ensure_positive(as_namespace_array(y, xp, like=y), xp)

        if xp is torch:  # type: ignore[comparison-overlap]
            mean_val = torch.mean(y_arr)
            return torch.full_like(y_arr, mean_val)
        else:
            mean_val = np.mean(y_arr)
            return np.full_like(y_arr, mean_val, dtype=y_arr.dtype)

    def estimate_lambda(self, y, mu):
        """Estimate shape parameter from data and fitted values.

        Parameters
        ----------
        y : array
            Observed values.
        mu : array
            Fitted mean values.

        Returns
        -------
        float
            Estimated shape parameter λ.
        """
        xp = namespace(y, mu)
        y_arr = ensure_positive(as_namespace_array(y, xp, like=mu), xp)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=y_arr), xp)

        return self._estimate_lambda_mm(y_arr, mu_arr, xp)

    @property
    def default_link(self) -> LinkFunction:
        """Return the default link function (InverseSquare)."""
        return self._link


# Alias for compatibility
WaldFamily = InverseGaussianFamily


__all__ = ["InverseGaussianFamily", "WaldFamily"]
