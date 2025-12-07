# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Beta distribution family implementation.

The Beta distribution is suitable for modeling continuous proportions in (0, 1),
such as success rates, conversion rates, or market shares.

Mathematical Framework
----------------------
The Beta distribution with mean parameterization uses:

**Density**:
    f(y; μ, φ) = Γ(φ) / [Γ(μφ)Γ((1-μ)φ)] × y^{μφ-1} × (1-y)^{(1-μ)φ-1}

where:
    - y ∈ (0, 1) is the response
    - μ ∈ (0, 1) is the mean
    - φ > 0 is the precision parameter (larger = less dispersed)

**Properties**:
    - E[Y] = μ
    - Var(Y) = μ(1-μ) / (φ + 1)
    - Shape parameters: α = μφ, β = (1-μ)φ

**Variance function**:
    V(μ) = μ(1-μ)

**Canonical link**: Logit (though probit and cloglog also common)

References
----------
- Ferrari, S., & Cribari-Neto, F. (2004). "Beta regression for modelling rates
  and proportions." Journal of Applied Statistics, 31(7), 799-815.
- Smithson, M., & Verkuilen, J. (2006). "A better lemon squeezer? Maximum-likelihood
  regression with beta-distributed dependent variables." Psychological Methods, 11(1), 54.

See Also
--------
aurora.distributions.links.common.LogitLink : Canonical link function
aurora.distributions.links.common.ProbitLink : Alternative link function
"""

from __future__ import annotations


import numpy as np

from ..base import Family, LinkFunction
from .._utils import (
    as_namespace_array,
    clip_probability,
    ensure_positive,
    namespace,
    log_gamma,
)
from ..links import LogitLink

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None  # type: ignore[assignment]


def _log_beta(a, b, xp):
    """Compute log(B(a, b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a+b))."""
    return log_gamma(a, xp) + log_gamma(b, xp) - log_gamma(a + b, xp)


class BetaFamily(Family):
    """Beta distribution for continuous proportions in (0, 1).

    Suitable for modeling:
    - Success rates, conversion rates
    - Market shares, proportions
    - Any continuous outcome bounded in (0, 1)

    Parameters
    ----------
    phi : float or 'estimate', default=1.0
        Precision parameter (inverse dispersion).
        - phi large: Low variance, concentrated around mean
        - phi small: High variance, dispersed
        - 'estimate': Estimate from data via method-of-moments

    link : LinkFunction or None, default=None
        Link function. If None, uses LogitLink (canonical).
        Also supports ProbitLink, CLogLogLink.

    Attributes
    ----------
    phi : float or str
        Precision parameter or 'estimate' for automatic estimation.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.families import BetaFamily
    >>> family = BetaFamily(phi=10.0)
    >>> mu = np.array([0.3, 0.5, 0.7])
    >>> family.variance(mu)  # μ(1-μ)/(φ+1)
    array([0.01909091, 0.02272727, 0.01909091])

    >>> # Method-of-moments phi estimation
    >>> y = np.random.beta(3, 7, size=100)  # True mean ≈ 0.3
    >>> family = BetaFamily(phi='estimate')
    >>> mu_init = family.initialize(y)

    Notes
    -----
    The precision φ controls the dispersion:
    - φ = 1: Uniform(0,1) when μ = 0.5
    - φ → ∞: Concentration at μ (degenerate)
    - φ → 0: Bimodal distribution at 0 and 1

    The variance function is V(μ) = μ(1-μ), same as binomial, but
    the actual variance also depends on φ: Var(Y) = V(μ)/(φ+1).
    """

    def __init__(
        self, phi: float | str = 1.0, link: LinkFunction | None = None
    ) -> None:
        """Initialize Beta family.

        Parameters
        ----------
        phi : float or 'estimate', default=1.0
            Precision parameter. Use 'estimate' for method-of-moments.
        link : LinkFunction or None
            Link function (default: LogitLink).
        """
        if isinstance(phi, str):
            if phi != "estimate":
                raise ValueError("phi must be a positive float or 'estimate'")
            self._phi = phi
        else:
            if phi <= 0:
                raise ValueError("phi must be positive")
            self._phi = float(phi)

        self._link = link or LogitLink()

    @property
    def phi(self) -> float | str:
        """Return the precision parameter."""
        return self._phi

    def _get_phi(self, xp, like, y=None, **params):
        """Get phi value, estimating if needed."""
        phi_param = params.get("phi", self._phi)

        if phi_param == "estimate" and y is not None:
            # Method-of-moments estimation
            phi_param = self._estimate_phi_mm(y, xp)

        if isinstance(phi_param, str):
            # Fallback if estimation not possible
            phi_param = 1.0

        return ensure_positive(as_namespace_array(phi_param, xp, like=like), xp)

    def _estimate_phi_mm(self, y, xp):
        """Estimate phi via method-of-moments.

        For Beta(α, β) with μ = α/(α+β) and φ = α+β:
            Var(Y) = μ(1-μ)/(φ+1)

        Solving for φ:
            φ = μ(1-μ)/Var(Y) - 1
        """
        if xp is torch:  # type: ignore[comparison-overlap]
            y_mean = torch.mean(y)
            y_var = torch.var(y)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            y_mean = jnp.mean(y)
            y_var = jnp.var(y)
        else:
            y_mean = np.mean(y)
            y_var = np.var(y)

        # Clamp mean away from boundaries (convert to float for computation)
        mu_est = float(np.clip(float(y_mean), 0.01, 0.99))
        var_est = float(max(float(y_var), 1e-10))

        # phi = mu(1-mu)/var - 1
        phi_est = mu_est * (1 - mu_est) / var_est - 1

        # Ensure phi is positive and reasonable
        return max(phi_est, 0.1)

    def log_likelihood(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        """Compute log-likelihood for Beta distribution.

        Parameters
        ----------
        y : array
            Observed proportions in (0, 1).
        mu : array
            Mean parameter in (0, 1).
        **params : dict
            Optional 'phi' parameter override.

        Returns
        -------
        scalar
            Sum of log-likelihoods.

        Notes
        -----
        Log-likelihood:
            ℓ = log Γ(φ) - log Γ(μφ) - log Γ((1-μ)φ)
                + (μφ - 1) log(y) + ((1-μ)φ - 1) log(1-y)
        """
        xp = namespace(y, mu)
        eps = 1e-10

        # Ensure y and mu are in valid range
        y_arr = clip_probability(as_namespace_array(y, xp, like=mu), xp, eps=eps)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=y_arr), xp, eps=eps)

        phi = self._get_phi(xp, mu_arr, y=y_arr, **params)

        # Shape parameters
        alpha = mu_arr * phi
        beta_param = (1.0 - mu_arr) * phi

        # Ensure positive
        alpha = ensure_positive(alpha, xp)
        beta_param = ensure_positive(beta_param, xp)

        # Log-likelihood: log f(y; α, β)
        # = log Γ(α+β) - log Γ(α) - log Γ(β) + (α-1)log(y) + (β-1)log(1-y)
        log_lik = (
            log_gamma(alpha + beta_param, xp)
            - log_gamma(alpha, xp)
            - log_gamma(beta_param, xp)
            + (alpha - 1.0) * xp.log(y_arr)
            + (beta_param - 1.0) * xp.log(1.0 - y_arr)
        )

        if xp is torch:  # type: ignore[comparison-overlap]
            return torch.sum(log_lik)
        return np.sum(log_lik)

    def deviance(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        """Compute deviance for Beta distribution.

        For Beta regression, the deviance is computed as:
            D = 2 × φ × Σ[y log(y/μ) + (1-y) log((1-y)/(1-μ))]

        This is the weighted KL-divergence scaled by 2φ.

        Parameters
        ----------
        y : array
            Observed proportions.
        mu : array
            Fitted mean values.
        **params : dict
            Optional 'phi' parameter override.

        Returns
        -------
        scalar
            Deviance value.

        Notes
        -----
        For the Beta distribution, deviance is always non-negative.
        When y = μ, the deviance is zero.
        """
        xp = namespace(y, mu)
        eps = 1e-10

        y_arr = clip_probability(as_namespace_array(y, xp, like=mu), xp, eps=eps)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=y_arr), xp, eps=eps)

        phi = self._get_phi(xp, mu_arr, y=y_arr, **params)

        # Unit deviance for Beta: 2 × [y log(y/μ) + (1-y) log((1-y)/(1-μ))]
        # This is the KL divergence
        term1 = y_arr * xp.log(y_arr / mu_arr)
        term2 = (1.0 - y_arr) * xp.log((1.0 - y_arr) / (1.0 - mu_arr))

        unit_dev = 2.0 * (term1 + term2)

        if xp is torch:  # type: ignore[comparison-overlap]
            return phi * torch.sum(unit_dev)
        return phi * np.sum(unit_dev)

    def variance(self, mu, **params):  # noqa: ANN001 - match Family signature
        """Compute variance function V(μ) = μ(1-μ).

        Note: Actual variance is V(μ)/(φ+1), but for GLM/IRLS
        we return the variance function without dispersion.

        Parameters
        ----------
        mu : array
            Mean parameter in (0, 1).
        **params : dict
            Optional 'phi' parameter override.

        Returns
        -------
        array
            Variance function values.

        Notes
        -----
        The true variance is Var(Y) = μ(1-μ)/(φ+1), but in the GLM
        framework we separate the variance function V(μ) from the
        dispersion parameter. Here we return V(μ) = μ(1-μ).
        """
        xp = namespace(mu)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=mu), xp)

        return mu_arr * (1.0 - mu_arr)

    def full_variance(self, mu, **params):
        """Compute full variance: Var(Y) = μ(1-μ)/(φ+1).

        Unlike variance(), this includes the precision parameter.

        Parameters
        ----------
        mu : array
            Mean parameter.
        **params : dict
            Optional 'phi' parameter.

        Returns
        -------
        array
            Full variance values.
        """
        xp = namespace(mu)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=mu), xp)
        phi = self._get_phi(xp, mu_arr, **params)

        return mu_arr * (1.0 - mu_arr) / (phi + 1.0)

    def initialize(self, y):  # noqa: ANN001 - match Family signature
        """Initialize mean parameter from data.

        Parameters
        ----------
        y : array
            Observed proportions.

        Returns
        -------
        array
            Initial mean values, clamped to (0.01, 0.99).

        Notes
        -----
        Uses sample mean clamped away from boundaries to avoid
        numerical issues with link function.
        """
        xp = namespace(y)
        y_arr = as_namespace_array(y, xp, like=y)

        # Clamp to valid range
        y_arr = clip_probability(y_arr, xp, eps=0.01)

        # Return mean of clamped values
        if xp is torch:  # type: ignore[comparison-overlap]
            mean_val = torch.mean(y_arr)
            return torch.full_like(y_arr, mean_val)
        else:
            mean_val = np.mean(y_arr)
            return np.full_like(y_arr, mean_val)

    def estimate_phi(self, y, mu=None):
        """Estimate precision parameter from data.

        Uses method-of-moments estimation based on sample variance.

        Parameters
        ----------
        y : array
            Observed proportions.
        mu : array, optional
            Fitted means. If None, uses sample mean.

        Returns
        -------
        float
            Estimated precision parameter φ.
        """
        xp = namespace(y)
        y_arr = clip_probability(as_namespace_array(y, xp, like=y), xp, eps=0.01)

        if mu is None:
            if xp is torch:  # type: ignore[comparison-overlap]
                mu_val = torch.mean(y_arr)
            else:
                mu_val = np.mean(y_arr)
        else:
            mu_arr = clip_probability(
                as_namespace_array(mu, xp, like=y_arr), xp, eps=0.01
            )
            if xp is torch:  # type: ignore[comparison-overlap]
                mu_val = torch.mean(mu_arr)
            else:
                mu_val = np.mean(mu_arr)

        return self._estimate_phi_mm(y_arr, xp)

    @property
    def default_link(self) -> LinkFunction:
        """Return the default link function (Logit)."""
        return self._link


__all__ = ["BetaFamily"]
