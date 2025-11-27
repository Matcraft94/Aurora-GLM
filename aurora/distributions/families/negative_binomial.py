"""Negative Binomial distribution family for overdispersed count data.

This module implements the Negative Binomial (NB2) distribution for modeling
count data with overdispersion (variance > mean), which is common in many
biological, ecological, and social science applications.

References
----------
.. [1] Hilbe, J. M. (2011). Negative Binomial Regression (2nd ed.). 
       Cambridge University Press.
.. [2] Ver Hoef, J. M., & Boveng, P. L. (2007). 
       "Quasi-Poisson vs. negative binomial regression."
       Environmetrics, 18(3), 255-269.
.. [3] Lawless, J. F. (1987). 
       "Negative binomial and mixed Poisson regression."
       Canadian Journal of Statistics, 15(3), 209-225.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import special

from aurora.distributions.base import Family
from aurora.distributions.links import LogLink, IdentityLink, SqrtLink

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NegativeBinomialFamily(Family):
    """Negative Binomial distribution for overdispersed count data.

    The Negative Binomial (NB2 parameterization) models count data where
    the variance exceeds the mean: Var(Y) = μ + μ²/θ

    This is an extension of Poisson regression for overdispersed data,
    commonly arising from:
    - Unobserved heterogeneity
    - Clustered/hierarchical data
    - Temporal/spatial correlation

    Parameters
    ----------
    theta : float or str, default=1.0
        Dispersion parameter (also called 'size' or 'k').
        - theta → ∞: Converges to Poisson
        - theta small: High overdispersion
        - 'estimate': Estimate from data during fitting

    link : str, default='log'
        Link function: 'log' (default), 'identity', or 'sqrt'

    Notes
    -----
    The NB2 parameterization is used:
        P(Y=y) = Γ(y+θ) / [Γ(θ)y!] × (θ/(θ+μ))^θ × (μ/(θ+μ))^y

    Properties:
        E[Y] = μ
        Var(Y) = μ + μ²/θ

    The negative binomial is a member of the exponential family when θ is fixed.

    Examples
    --------
    >>> from aurora.models.glm import fit_glm
    >>> # Fixed dispersion
    >>> result = fit_glm(X, y, family='negativebinomial', 
    ...                  family_params={'theta': 2.0})

    >>> # Estimate dispersion
    >>> result = fit_glm(X, y, family='negativebinomial',
    ...                  family_params={'theta': 'estimate'})

    References
    ----------
    .. [1] Hilbe (2011). Negative Binomial Regression.
    .. [2] Cameron & Trivedi (2013). Regression Analysis of Count Data.
    """

    name = 'negative_binomial'
    
    # Valid range for responses (non-negative integers)
    valid_y_range = (0, np.inf)

    def __init__(
        self, 
        theta: float | str = 1.0, 
        link: str = 'log'
    ):
        """Initialize Negative Binomial family.

        Parameters
        ----------
        theta : float or 'estimate'
            Dispersion parameter. If 'estimate', will be estimated from data.
        link : str
            Link function name.
        """
        if isinstance(theta, str):
            if theta != 'estimate':
                raise ValueError("theta must be a number or 'estimate'")
            self._theta = None
            self._estimate_theta = True
        else:
            if theta <= 0:
                raise ValueError("theta must be positive")
            self._theta = float(theta)
            self._estimate_theta = False
        
        if link == 'log':
            self._link = LogLink()
        elif link == 'identity':
            self._link = IdentityLink()
        elif link == 'sqrt':
            self._link = SqrtLink()
        else:
            raise ValueError(f"Unsupported link: {link}. Use 'log', 'identity', or 'sqrt'")

    @property
    def theta(self) -> float:
        """Dispersion parameter."""
        if self._theta is None:
            raise ValueError("theta has not been estimated yet")
        return self._theta

    @theta.setter
    def theta(self, value: float):
        """Set dispersion parameter."""
        if value <= 0:
            raise ValueError("theta must be positive")
        self._theta = value

    @property
    def default_link(self):
        """Default link function (log)."""
        return self._link

    def variance(self, mu: NDArray, **params) -> NDArray:
        """Variance function: V(μ) = μ + μ²/θ.

        Parameters
        ----------
        mu : ndarray
            Mean values
        **params : dict
            May contain 'theta' override

        Returns
        -------
        ndarray
            Variance at each observation
        """
        theta = params.get('theta', self._theta)
        if theta is None:
            raise ValueError("theta must be specified or estimated")
        
        return mu + mu**2 / theta

    def initialize(self, y: NDArray) -> NDArray:
        """Initialize mean with sample mean + small constant.

        Parameters
        ----------
        y : ndarray
            Response values (counts)

        Returns
        -------
        ndarray
            Initial mean estimates
        """
        mu = np.mean(y)
        return np.full_like(y, max(mu, 0.1), dtype=float)

    def log_likelihood(
        self, 
        y: NDArray, 
        mu: NDArray, 
        **params
    ) -> float:
        """Log-likelihood for Negative Binomial.

        Parameters
        ----------
        y : ndarray
            Observed counts
        mu : ndarray
            Mean (fitted) values
        **params : dict
            May contain 'theta'

        Returns
        -------
        float
            Total log-likelihood
        """
        theta = params.get('theta', self._theta)
        if theta is None:
            raise ValueError("theta must be specified")
        
        # Ensure y and mu are valid
        y = np.asarray(y, dtype=float)
        mu = np.maximum(mu, 1e-10)
        
        # Log-likelihood: 
        # log Γ(y+θ) - log Γ(θ) - log(y!) + θ log(θ/(θ+μ)) + y log(μ/(θ+μ))
        log_lik = (
            special.gammaln(y + theta) - 
            special.gammaln(theta) - 
            special.gammaln(y + 1) +
            theta * np.log(theta / (theta + mu)) +
            y * np.log(mu / (theta + mu))
        )
        
        return np.sum(log_lik)

    def deviance(
        self, 
        y: NDArray, 
        mu: NDArray, 
        **params
    ) -> float:
        """Deviance for Negative Binomial.

        Parameters
        ----------
        y : ndarray
            Observed counts
        mu : ndarray
            Fitted mean values
        **params : dict
            May contain 'theta'

        Returns
        -------
        float
            Total deviance
        """
        theta = params.get('theta', self._theta)
        if theta is None:
            raise ValueError("theta must be specified")
        
        y = np.asarray(y, dtype=float)
        mu = np.maximum(mu, 1e-10)
        
        # Unit deviance
        # d_i = 2[y log(y/μ) - (y+θ) log((y+θ)/(μ+θ))]
        
        # Handle y=0 case
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = np.where(y > 0, y * np.log(y / mu), 0)
            term2 = (y + theta) * np.log((y + theta) / (mu + theta))
        
        d = 2 * (term1 - term2)
        
        return np.sum(d)

    def estimate_theta(
        self, 
        y: NDArray, 
        mu: NDArray, 
        method: str = 'ml'
    ) -> float:
        """Estimate dispersion parameter theta.

        Parameters
        ----------
        y : ndarray
            Observed counts
        mu : ndarray
            Fitted means
        method : str, default='ml'
            Estimation method:
            - 'moments': Method of moments (fast)
            - 'ml': Maximum likelihood (more accurate)

        Returns
        -------
        float
            Estimated theta
        """
        if method == 'moments':
            return self._estimate_theta_moments(y, mu)
        elif method == 'ml':
            return self._estimate_theta_ml(y, mu)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _estimate_theta_moments(
        self, 
        y: NDArray, 
        mu: NDArray
    ) -> float:
        """Method of moments estimator for theta.

        Based on: Var(Y) = μ + μ²/θ
        Rearranging: θ = μ² / (Var(Y) - μ)
        """
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        
        # Pearson residuals squared
        var_y = np.var(y)
        mean_y = np.mean(y)
        
        # Var = μ + μ²/θ => θ = μ²/(Var - μ)
        excess_var = var_y - mean_y
        
        if excess_var <= 0:
            # No overdispersion detected, return large theta (near Poisson)
            return 1e6
        
        theta = mean_y**2 / excess_var
        
        # Ensure reasonable bounds
        return np.clip(theta, 0.01, 1e6)

    def _estimate_theta_ml(
        self, 
        y: NDArray, 
        mu: NDArray, 
        maxiter: int = 50
    ) -> float:
        """Maximum likelihood estimator for theta.

        Uses Newton-Raphson on the profile log-likelihood.
        """
        from scipy.optimize import brentq
        
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        n = len(y)
        
        def score(theta):
            """Score function for theta."""
            if theta <= 0:
                return np.inf
            
            # d/dθ log L
            psi_deriv = special.digamma(y + theta) - special.digamma(theta)
            term1 = np.sum(psi_deriv)
            term2 = n * np.log(theta / (theta + mu)).mean()
            term3 = n * (1 - mu / (theta + mu)).mean()
            
            return term1 + term2 + term3

        # Initial estimate from moments
        theta_init = self._estimate_theta_moments(y, mu)
        
        # Bracket search
        try:
            theta_lo, theta_hi = 0.01, 1000.0
            
            # Check if solution exists in bracket
            if score(theta_lo) * score(theta_hi) > 0:
                # Use moments estimate if no root in bracket
                return theta_init
            
            theta_ml = brentq(score, theta_lo, theta_hi, maxiter=maxiter)
            return theta_ml
        except (ValueError, RuntimeError):
            return theta_init

    def d_log_likelihood(
        self, 
        y: NDArray, 
        mu: NDArray, 
        **params
    ) -> NDArray:
        """First derivative of log-likelihood w.r.t. μ.

        Parameters
        ----------
        y : ndarray
            Observed counts
        mu : ndarray
            Mean values

        Returns
        -------
        ndarray
            Gradient
        """
        theta = params.get('theta', self._theta)
        if theta is None:
            raise ValueError("theta must be specified")
        
        mu = np.maximum(mu, 1e-10)
        
        # d/dμ log L = y/μ - (y+θ)/(μ+θ)
        grad = y / mu - (y + theta) / (mu + theta)
        
        return grad

    def d2_log_likelihood(
        self, 
        y: NDArray, 
        mu: NDArray, 
        **params
    ) -> NDArray:
        """Second derivative of log-likelihood w.r.t. μ.

        Parameters
        ----------
        y : ndarray
            Observed counts
        mu : ndarray
            Mean values

        Returns
        -------
        ndarray
            Negative Hessian diagonal
        """
        theta = params.get('theta', self._theta)
        if theta is None:
            raise ValueError("theta must be specified")
        
        mu = np.maximum(mu, 1e-10)
        
        # d²/dμ² log L = -y/μ² + (y+θ)/(μ+θ)²
        hess = -y / mu**2 + (y + theta) / (mu + theta)**2
        
        return hess

    def __repr__(self) -> str:
        """String representation."""
        if self._theta is None:
            theta_str = "'estimate'"
        else:
            theta_str = f"{self._theta:.4g}"
        return f"NegativeBinomialFamily(theta={theta_str}, link='{self._link.name}')"


# Alias for convenience
NegBinFamily = NegativeBinomialFamily


__all__ = ['NegativeBinomialFamily', 'NegBinFamily']
