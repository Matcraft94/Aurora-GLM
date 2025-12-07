# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Student's t distribution family for robust GLM.

This module implements the Student's t distribution for robust regression,
which is less sensitive to outliers than the Gaussian distribution.

The t-distribution has heavier tails than the Gaussian, making it suitable
for data with outliers or when robustness is desired.

References
----------
.. [1] Lange, K. L., Little, R. J., & Taylor, J. M. (1989).
       "Robust statistical modeling using the t distribution."
       Journal of the American Statistical Association, 84(408), 881-896.
.. [2] Fernandez, C., & Steel, M. F. (1999).
       "Multivariate Student-t regression models."
       Journal of the Royal Statistical Society: Series B, 61(3), 579-602.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import special

from aurora.distributions.base import Family
from aurora.distributions.links import IdentityLink, LogLink

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StudentTFamily(Family):
    """Student's t distribution family for robust regression.

    The Student's t distribution provides robust regression that is less
    sensitive to outliers compared to Gaussian regression. The degree of
    robustness is controlled by the degrees of freedom parameter.

    Parameters
    ----------
    df : float, default=5.0
        Degrees of freedom (ν). Controls tail heaviness:
        - df=1: Cauchy distribution (very heavy tails, no moments)
        - df=3-5: Heavy tails, robust to outliers
        - df=10: Moderate tails
        - df→∞: Converges to Gaussian

    link : str, default='identity'
        Link function: 'identity' or 'log' (for positive responses)

    Notes
    -----
    The t-distribution is NOT in the exponential family, so standard
    IRLS does not apply directly. We use iteratively reweighted least
    squares with adaptive weights that downweight outliers.

    The variance is ν/(ν-2) for ν > 2. For ν ≤ 2, variance is infinite.

    Examples
    --------
    >>> from aurora.models.glm import fit_glm
    >>> # Robust regression with df=5
    >>> result = fit_glm(X, y, family='student_t', family_params={'df': 5})

    >>> # Very robust (heavy tails)
    >>> result = fit_glm(X, y, family='student_t', family_params={'df': 3})

    References
    ----------
    .. [1] Lange et al. (1989). Robust statistical modeling using the t distribution.
    """

    name = "student_t"

    # Valid range for responses (no restrictions)
    valid_y_range = (-np.inf, np.inf)

    def __init__(self, df: float = 5.0, link: str = "identity"):
        """Initialize Student's t family.

        Parameters
        ----------
        df : float
            Degrees of freedom. Must be > 0.
        link : str
            Link function name.
        """
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")

        self.df = df
        self._scale = 1.0  # Scale parameter (estimated)

        if link == "identity":
            self._link = IdentityLink()
        elif link == "log":
            self._link = LogLink()
        else:
            raise ValueError(f"Unsupported link: {link}. Use 'identity' or 'log'")

    @property
    def default_link(self):
        """Default link function."""
        return self._link

    def variance(self, mu: NDArray, **params) -> NDArray:
        """Variance function.

        For Student's t with df > 2:
            Var(Y) = σ² × df/(df-2)

        Parameters
        ----------
        mu : ndarray
            Mean values (not used, variance is constant for location-scale t)
        **params : dict
            May contain 'scale' parameter

        Returns
        -------
        ndarray
            Variance at each observation
        """
        scale = params.get("scale", self._scale)

        if self.df <= 2:
            # Variance is infinite for df <= 2
            return np.full_like(mu, np.inf)

        return np.full_like(mu, scale**2 * self.df / (self.df - 2))

    def initialize(self, y: NDArray) -> NDArray:
        """Initialize mean using median (robust).

        Parameters
        ----------
        y : ndarray
            Response values

        Returns
        -------
        ndarray
            Initial mean estimates
        """
        # Use median for robustness
        return np.full_like(y, np.median(y))

    def log_likelihood(self, y: NDArray, mu: NDArray, **params) -> float:
        """Log-likelihood for t-distribution.

        Parameters
        ----------
        y : ndarray
            Observed values
        mu : ndarray
            Mean (location) parameter
        **params : dict
            Must contain 'scale' or uses default

        Returns
        -------
        float
            Total log-likelihood
        """
        scale = params.get("scale", self._scale)
        df = self.df

        # Standardized residuals
        z = (y - mu) / scale

        # Log-likelihood: sum of log(f(y_i))
        # f(y) = Γ((ν+1)/2) / [√(νπ)σΓ(ν/2)] × (1 + z²/ν)^{-(ν+1)/2}

        log_const = (
            special.gammaln((df + 1) / 2)
            - special.gammaln(df / 2)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
        )

        log_kernel = -(df + 1) / 2 * np.log(1 + z**2 / df)

        return np.sum(log_const + log_kernel)

    def deviance(self, y: NDArray, mu: NDArray, **params) -> float:
        """Deviance for t-distribution.

        The deviance is -2 × (log-likelihood - saturated log-likelihood).
        For the t-distribution, we use the saturated model with μ = y.

        Parameters
        ----------
        y : ndarray
            Observed values
        mu : ndarray
            Fitted mean values
        **params : dict
            Additional parameters

        Returns
        -------
        float
            Total deviance
        """
        ll_model = self.log_likelihood(y, mu, **params)
        ll_saturated = self.log_likelihood(y, y, **params)

        return -2 * (ll_model - ll_saturated)

    def weights(self, y: NDArray, mu: NDArray, **params) -> NDArray:
        """Compute IRLS weights for robust regression.

        The t-distribution uses adaptive weights that downweight outliers:
            w_i = (ν + 1) / (ν + z_i²)

        where z_i = (y_i - μ_i) / σ is the standardized residual.

        Parameters
        ----------
        y : ndarray
            Observed values
        mu : ndarray
            Current mean estimates
        **params : dict
            Must contain 'scale'

        Returns
        -------
        ndarray
            Weights for each observation
        """
        scale = params.get("scale", self._scale)
        df = self.df

        # Standardized residuals
        z = (y - mu) / scale

        # Adaptive weights: downweight large residuals
        weights = (df + 1) / (df + z**2)

        return weights

    def estimate_scale(self, y: NDArray, mu: NDArray, ddof: int = 1) -> float:
        """Estimate scale parameter robustly.

        Uses the Median Absolute Deviation (MAD) for robustness.

        Parameters
        ----------
        y : ndarray
            Observed values
        mu : ndarray
            Fitted mean values
        ddof : int
            Degrees of freedom adjustment

        Returns
        -------
        float
            Estimated scale parameter
        """
        residuals = y - mu

        # MAD-based robust scale estimate
        # For t-distribution: σ = MAD / 0.6745 × correction_factor
        mad = np.median(np.abs(residuals - np.median(residuals)))

        # Correction factor for t-distribution
        # Approximately 1 for large df, larger for small df
        if self.df > 2:
            correction = np.sqrt(self.df / (self.df - 2))
        else:
            correction = 1.0

        scale = mad / 0.6745 * correction

        # Ensure positive
        return max(scale, 1e-8)

    def d_log_likelihood(self, y: NDArray, mu: NDArray, **params) -> NDArray:
        """First derivative of log-likelihood w.r.t. μ.

        For Laplace approximation and gradient-based optimization.

        Parameters
        ----------
        y : ndarray
            Observed values
        mu : ndarray
            Mean values

        Returns
        -------
        ndarray
            Gradient of log-likelihood
        """
        scale = params.get("scale", self._scale)
        df = self.df

        z = (y - mu) / scale

        # d/dμ log f = (ν+1) × z / (σ(ν + z²))
        grad = (df + 1) * z / (scale * (df + z**2))

        return grad

    def d2_log_likelihood(self, y: NDArray, mu: NDArray, **params) -> NDArray:
        """Second derivative of log-likelihood w.r.t. μ.

        For Hessian computation in Laplace approximation.

        Parameters
        ----------
        y : ndarray
            Observed values
        mu : ndarray
            Mean values

        Returns
        -------
        ndarray
            Negative Hessian (Fisher information approximation)
        """
        scale = params.get("scale", self._scale)
        df = self.df

        z = (y - mu) / scale

        # d²/dμ² log f = -(ν+1) × (ν - z²) / (σ²(ν + z²)²)
        denom = (df + z**2) ** 2
        hess = -(df + 1) * (df - z**2) / (scale**2 * denom)

        return hess

    def __repr__(self) -> str:
        """String representation."""
        # Get link name from class name (e.g., IdentityLink -> identity)
        link_name = self._link.__class__.__name__.replace("Link", "").lower()
        return f"StudentTFamily(df={self.df}, link='{link_name}')"


class CauchyFamily(StudentTFamily):
    """Cauchy distribution (Student's t with df=1).

    The Cauchy distribution has extremely heavy tails and no finite moments.
    It is maximally robust to outliers but may have convergence issues.

    Use with caution - suitable only when extreme robustness is needed.

    Examples
    --------
    >>> from aurora.models.glm import fit_glm
    >>> result = fit_glm(X, y, family='cauchy')
    """

    name = "cauchy"

    def __init__(self, link: str = "identity"):
        """Initialize Cauchy family (t with df=1)."""
        super().__init__(df=1.0, link=link)

    def variance(self, mu: NDArray, **params) -> NDArray:
        """Variance is undefined (infinite) for Cauchy."""
        return np.full_like(mu, np.inf)

    def __repr__(self) -> str:
        """String representation."""
        # Get link name from class name (e.g., IdentityLink -> identity)
        link_name = self._link.__class__.__name__.replace("Link", "").lower()
        return f"CauchyFamily(link='{link_name}')"


__all__ = ["StudentTFamily", "CauchyFamily"]
