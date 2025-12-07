# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Tweedie distribution family for zero-inflated continuous data.

The Tweedie distribution is a member of the exponential dispersion family
that naturally handles data with exact zeros and continuous positive values.
It is particularly useful for insurance claims, rainfall, and healthcare costs.

References
----------
.. [1] Jørgensen, B. (1987). "Exponential dispersion models."
       Journal of the Royal Statistical Society: Series B, 49(2), 127-162.
.. [2] Smyth, G. K., & Jørgensen, B. (2002).
       "Fitting Tweedie's compound Poisson model to insurance claims data."
       ASTIN Bulletin, 32(1), 143-157.
.. [3] Dunn, P. K., & Smyth, G. K. (2005).
       "Series evaluation of Tweedie exponential dispersion model densities."
       Statistics and Computing, 15(4), 267-280.
.. [4] Dunn, P. K., & Smyth, G. K. (2008).
       "Evaluation of Tweedie exponential dispersion model densities by Fourier inversion."
       Statistics and Computing, 18(1), 73-86.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aurora.distributions.base import Family
from aurora.distributions.links import LogLink, IdentityLink, PowerLink
from aurora.distributions._utils import (
    namespace,
    as_namespace_array,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None  # type: ignore[assignment]


class TweedieFamily(Family):
    """Tweedie distribution for zero-inflated positive continuous data.

    The Tweedie distribution is a compound Poisson-Gamma distribution that
    naturally handles data with:
    - A point mass at zero (no events)
    - Continuous positive values (sum of event magnitudes)

    The variance function is: Var(Y) = φ × μ^p

    Special cases based on power parameter p:
    - p = 0: Gaussian
    - p = 1: Poisson (not for continuous data)
    - 1 < p < 2: Compound Poisson-Gamma (Tweedie proper)
    - p = 2: Gamma
    - p = 3: Inverse Gaussian

    Parameters
    ----------
    power : float, default=1.5
        Variance power parameter p. For compound Poisson-Gamma, must be in (1, 2).
        Common choices:
        - p = 1.5: Balanced, works well for many applications
        - p = 1.6-1.7: More right-skewed, common for insurance claims
        - p closer to 1: More mass at zero
        - p closer to 2: Less mass at zero, more Gamma-like

    link : str, default='log'
        Link function: 'log' (default), 'identity', or 'power'

    Notes
    -----
    The Tweedie distribution can be viewed as:
    - N ~ Poisson(λ) events occur
    - Each event has magnitude X_i ~ Gamma(α, β)
    - Total Y = X_1 + ... + X_N

    This naturally creates exact zeros (N=0) and continuous positives (N>0).

    The probability of zero is: P(Y=0) = exp(-λ)

    Examples
    --------
    >>> from aurora.models.glm import fit_glm
    >>> # Insurance claims (many zeros, heavy right tail)
    >>> result = fit_glm(X, claims, family='tweedie',
    ...                  family_params={'power': 1.7})

    >>> # Rainfall amounts (zeros for dry days)
    >>> result = fit_glm(X, rainfall, family='tweedie',
    ...                  family_params={'power': 1.5})

    References
    ----------
    .. [1] Jørgensen (1987). Exponential dispersion models.
    .. [2] Smyth & Jørgensen (2002). Fitting Tweedie's compound Poisson model.
    """

    name = "tweedie"

    # Valid range for responses (non-negative)
    valid_y_range = (0, np.inf)

    def __init__(self, power: float = 1.5, link: str = "log", phi: float = 1.0):
        """Initialize Tweedie family.

        Parameters
        ----------
        power : float
            Variance power parameter. Must be in (1, 2) for proper Tweedie.
        link : str
            Link function name.
        phi : float
            Dispersion parameter (scale).
        """
        if not (1 < power < 2):
            raise ValueError(
                f"For compound Poisson-Gamma Tweedie, power must be in (1, 2). "
                f"Got power={power}. Use Gamma (power=2) or Poisson (power=1) instead."
            )

        self.power = power
        self.phi = phi

        if link == "log":
            self._link = LogLink()
        elif link == "identity":
            self._link = IdentityLink()
        elif link.startswith("power"):
            # Parse power link, e.g., 'power0.5'
            try:
                link_power = float(link.replace("power", ""))
            except ValueError:
                link_power = 1 - power / 2  # Canonical power
            self._link = PowerLink(link_power)
        else:
            raise ValueError(
                f"Unsupported link: {link}. Use 'log', 'identity', or 'power'"
            )

    @property
    def default_link(self):
        """Default link function (log)."""
        return self._link

    def variance(self, mu: NDArray, **params) -> NDArray:
        """Variance function: V(μ) = μ^p.

        Parameters
        ----------
        mu : ndarray
            Mean values
        **params : dict
            May contain 'phi' (dispersion)

        Returns
        -------
        ndarray
            Variance at each observation (without phi factor)
        """
        xp = namespace(mu)
        mu_arr = as_namespace_array(mu, xp, like=mu)

        # Clip mu to avoid log(0)
        if xp is torch:  # type: ignore[comparison-overlap]
            mu_arr = torch.clamp(mu_arr, min=1e-10)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            mu_arr = jnp.clip(mu_arr, 1e-10, None)
        else:
            mu_arr = np.clip(mu_arr, 1e-10, None)

        return mu_arr**self.power

    def initialize(self, y: NDArray) -> NDArray:
        """Initialize mean using positive values.

        Parameters
        ----------
        y : ndarray
            Response values

        Returns
        -------
        ndarray
            Initial mean estimates
        """
        xp = namespace(y)
        y_arr = as_namespace_array(y, xp, like=y)

        # Use mean of positive values, or small value if all zeros
        # Convert to numpy for indexing, then determine mu_init
        if xp is torch:  # type: ignore[comparison-overlap]
            y_np = y_arr.cpu().numpy() if y_arr.is_cuda else y_arr.numpy()
        elif xp is jnp:  # type: ignore[comparison-overlap]
            y_np = np.array(y_arr)
        else:
            y_np = np.asarray(y_arr, dtype=float)

        positive_y = y_np[y_np > 0]
        if len(positive_y) > 0:
            mu_init = float(np.mean(positive_y))
        else:
            mu_init = 0.1

        # Return with correct backend
        if xp is torch:  # type: ignore[comparison-overlap]
            return torch.full_like(y_arr, mu_init, dtype=torch.float32)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            return jnp.full_like(y_arr, mu_init, dtype=jnp.float32)
        else:
            return np.full_like(y_arr, mu_init, dtype=float)

    def deviance(self, y: NDArray, mu: NDArray, **params) -> float:
        """Deviance for Tweedie distribution.

        The unit deviance for Tweedie with power p is:

        For p ∈ (1, 2):
            d(y, μ) = 2 × [y^(2-p)/((1-p)(2-p)) - y×μ^(1-p)/(1-p) + μ^(2-p)/(2-p)]

        For y = 0:
            d(0, μ) = 2 × μ^(2-p) / (2-p)

        Parameters
        ----------
        y : ndarray
            Observed values (including zeros)
        mu : ndarray
            Fitted mean values
        **params : dict
            Additional parameters

        Returns
        -------
        float
            Total deviance
        """
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = as_namespace_array(mu, xp, like=y_arr)

        # Clip mu to avoid issues
        if xp is torch:  # type: ignore[comparison-overlap]
            mu_arr = torch.clamp(mu_arr, min=1e-10)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            mu_arr = jnp.clip(mu_arr, 1e-10, None)
        else:
            mu_arr = np.clip(mu_arr, 1e-10, None)

        p = self.power

        # Unit deviance
        # For y > 0
        if xp is torch:  # type: ignore[comparison-overlap]
            d_pos = 2 * (
                y_arr ** (2 - p) / ((1 - p) * (2 - p))
                - y_arr * mu_arr ** (1 - p) / (1 - p)
                + mu_arr ** (2 - p) / (2 - p)
            )
        elif xp is jnp:  # type: ignore[comparison-overlap]
            d_pos = 2 * (
                y_arr ** (2 - p) / ((1 - p) * (2 - p))
                - y_arr * mu_arr ** (1 - p) / (1 - p)
                + mu_arr ** (2 - p) / (2 - p)
            )
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                d_pos = 2 * (
                    y_arr ** (2 - p) / ((1 - p) * (2 - p))
                    - y_arr * mu_arr ** (1 - p) / (1 - p)
                    + mu_arr ** (2 - p) / (2 - p)
                )

        # For y = 0
        d_zero = 2 * mu_arr ** (2 - p) / (2 - p)

        # Select based on y
        if xp is torch:  # type: ignore[comparison-overlap]
            d = torch.where(y_arr == 0, d_zero, d_pos)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            d = jnp.where(y_arr == 0, d_zero, d_pos)
        else:
            d = np.where(y_arr == 0, d_zero, d_pos)

        return float(xp.sum(d))

    def log_likelihood(self, y: NDArray, mu: NDArray, **params) -> float:
        """Approximate log-likelihood for Tweedie.

        The Tweedie density has no closed form, but can be expressed
        as an infinite series or computed via Fourier inversion.

        For fitting purposes, we use the quasi-likelihood relationship:
            -2 log L ≈ D/φ + constant

        Parameters
        ----------
        y : ndarray
            Observed values
        mu : ndarray
            Mean values
        **params : dict
            May contain 'phi'

        Returns
        -------
        float
            Approximate log-likelihood
        """
        phi = params.get("phi", self.phi)

        # Quasi-likelihood approximation
        deviance = self.deviance(y, mu, **params)

        # Approximate log-likelihood (ignoring constants)
        return -0.5 * deviance / phi

    def weights(self, y: NDArray, mu: NDArray, **params) -> NDArray:
        """IRLS weights for Tweedie.

        Weights are 1/V(μ) = 1/μ^p.

        Parameters
        ----------
        y : ndarray
            Observed values (not used directly)
        mu : ndarray
            Current mean estimates

        Returns
        -------
        ndarray
            IRLS weights
        """
        mu = np.maximum(mu, 1e-10)
        return 1.0 / self.variance(mu)

    def estimate_power(
        self,
        y: NDArray,
        mu: NDArray,
        power_range: tuple[float, float] = (1.1, 1.9),
        n_grid: int = 20,
    ) -> float:
        """Estimate optimal power parameter via profile likelihood.

        Parameters
        ----------
        y : ndarray
            Observed values
        mu : ndarray
            Fitted means (from current model)
        power_range : tuple
            Range of powers to search
        n_grid : int
            Number of grid points

        Returns
        -------
        float
            Estimated power parameter
        """
        y = np.asarray(y, dtype=float)
        mu = np.maximum(mu, 1e-10)

        powers = np.linspace(power_range[0], power_range[1], n_grid)
        deviances = []

        for p in powers:
            self.power = p
            dev = self.deviance(y, mu)
            deviances.append(dev)

        # Find minimum
        best_idx = np.argmin(deviances)
        best_power = powers[best_idx]

        return best_power

    def estimate_phi(self, y: NDArray, mu: NDArray, ddof: int = 1) -> float:
        """Estimate dispersion parameter.

        Uses Pearson estimator: φ = (1/n) Σ (y-μ)² / V(μ)

        Parameters
        ----------
        y : ndarray
            Observed values
        mu : ndarray
            Fitted means
        ddof : int
            Degrees of freedom adjustment

        Returns
        -------
        float
            Estimated dispersion
        """
        y = np.asarray(y, dtype=float)
        mu = np.maximum(mu, 1e-10)
        n = len(y)

        # Pearson residuals
        resid = (y - mu) / np.sqrt(self.variance(mu))

        # Estimate phi
        phi = np.sum(resid**2) / (n - ddof)

        return max(phi, 1e-8)

    def d_log_likelihood(self, y: NDArray, mu: NDArray, **params) -> NDArray:
        """First derivative of quasi-log-likelihood w.r.t. μ.

        Parameters
        ----------
        y : ndarray
            Observed values
        mu : ndarray
            Mean values

        Returns
        -------
        ndarray
            Gradient
        """
        phi = params.get("phi", self.phi)
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = as_namespace_array(mu, xp, like=y_arr)

        # Clip mu to avoid division by zero
        if xp is torch:  # type: ignore[comparison-overlap]
            mu_arr = torch.clamp(mu_arr, min=1e-10)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            mu_arr = jnp.clip(mu_arr, 1e-10, None)
        else:
            mu_arr = np.clip(mu_arr, 1e-10, None)

        p = self.power

        # d/dμ (-D/2φ) = (y - μ) / (φ μ^p)
        grad = (y_arr - mu_arr) / (phi * mu_arr**p)

        return grad

    def d2_log_likelihood(self, y: NDArray, mu: NDArray, **params) -> NDArray:
        """Second derivative of quasi-log-likelihood w.r.t. μ.

        Parameters
        ----------
        y : ndarray
            Observed values
        mu : ndarray
            Mean values

        Returns
        -------
        ndarray
            Negative Hessian diagonal (Fisher information)
        """
        phi = params.get("phi", self.phi)
        xp = namespace(mu)
        mu_arr = as_namespace_array(mu, xp, like=mu)

        # Clip mu to avoid division by zero
        if xp is torch:  # type: ignore[comparison-overlap]
            mu_arr = torch.clamp(mu_arr, min=1e-10)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            mu_arr = jnp.clip(mu_arr, 1e-10, None)
        else:
            mu_arr = np.clip(mu_arr, 1e-10, None)

        p = self.power

        # Expected Fisher information: -E[d²/dμ²] = 1/(φ μ^p)
        # (using expected information for stability)
        info = -1.0 / (phi * mu_arr**p)

        return info

    def probability_zero(self, mu: NDArray, **params) -> NDArray:
        """Probability of observing exactly zero.

        For Tweedie with 1 < p < 2:
            P(Y=0) = exp(-λ)

        where λ = μ^(2-p) / [φ(2-p)]

        Parameters
        ----------
        mu : ndarray
            Mean values

        Returns
        -------
        ndarray
            Probability of zero at each observation
        """
        phi = params.get("phi", self.phi)
        xp = namespace(mu)
        mu_arr = as_namespace_array(mu, xp, like=mu)

        # Clip mu to avoid issues
        if xp is torch:  # type: ignore[comparison-overlap]
            mu_arr = torch.clamp(mu_arr, min=1e-10)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            mu_arr = jnp.clip(mu_arr, 1e-10, None)
        else:
            mu_arr = np.clip(mu_arr, 1e-10, None)

        p = self.power

        # Poisson rate parameter
        lambda_ = mu_arr ** (2 - p) / (phi * (2 - p))

        return xp.exp(-lambda_)

    def __repr__(self) -> str:
        """String representation."""
        # Get link name from class name (e.g., LogLink -> log)
        link_name = self._link.__class__.__name__.replace("Link", "").lower()
        return f"TweedieFamily(power={self.power:.3g}, phi={self.phi:.3g}, link='{link_name}')"


class CompoundPoissonGammaFamily(TweedieFamily):
    """Alias for Tweedie with explicit compound Poisson-Gamma interpretation.

    This is identical to TweedieFamily but with more descriptive naming
    for users familiar with the compound Poisson-Gamma formulation.

    The model assumes:
    - N ~ Poisson(λ) events occur
    - Each event has magnitude G_i ~ Gamma(α, β)
    - Total Y = G_1 + ... + G_N (or 0 if N=0)

    Parameters are related to Tweedie by:
    - λ = μ^(2-p) / [φ(2-p)]  (Poisson rate)
    - α = (2-p)/(p-1)  (Gamma shape)
    - β = φ(p-1)μ^(p-1)  (Gamma rate)
    """

    name = "compound_poisson_gamma"

    def get_poisson_rate(self, mu: NDArray, **params) -> NDArray:
        """Get the underlying Poisson rate λ.

        Parameters
        ----------
        mu : ndarray
            Mean values

        Returns
        -------
        ndarray
            Poisson rate for event count
        """
        phi = params.get("phi", self.phi)
        xp = namespace(mu)
        mu_arr = as_namespace_array(mu, xp, like=mu)

        # Clip mu to avoid issues
        if xp is torch:  # type: ignore[comparison-overlap]
            mu_arr = torch.clamp(mu_arr, min=1e-10)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            mu_arr = jnp.clip(mu_arr, 1e-10, None)
        else:
            mu_arr = np.clip(mu_arr, 1e-10, None)

        p = self.power

        return mu_arr ** (2 - p) / (phi * (2 - p))

    def get_gamma_shape(self) -> float:
        """Get the Gamma shape parameter α."""
        return (2 - self.power) / (self.power - 1)

    def get_gamma_rate(self, mu: NDArray, **params) -> NDArray:
        """Get the Gamma rate parameter β.

        Parameters
        ----------
        mu : ndarray
            Mean values

        Returns
        -------
        ndarray
            Gamma rate for event magnitude
        """
        phi = params.get("phi", self.phi)
        xp = namespace(mu)
        mu_arr = as_namespace_array(mu, xp, like=mu)

        # Clip mu to avoid issues
        if xp is torch:  # type: ignore[comparison-overlap]
            mu_arr = torch.clamp(mu_arr, min=1e-10)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            mu_arr = jnp.clip(mu_arr, 1e-10, None)
        else:
            mu_arr = np.clip(mu_arr, 1e-10, None)

        p = self.power

        return phi * (p - 1) * mu_arr ** (p - 1)


__all__ = ["TweedieFamily", "CompoundPoissonGammaFamily"]
