# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Truncated Count Distributions.

Truncated distributions are used in hurdle models for modeling
Y | Y > 0 (positive counts only).

Mathematical Framework
----------------------
For a base distribution with PMF f(k), the zero-truncated version has:

    P(Y = k | Y > 0) = f(k) / [1 - f(0)]   for k > 0

This is simply the original PMF rescaled to sum to 1 over positive integers.

Truncated Mean
--------------
E[Y | Y > 0] = E[Y] / [1 - f(0)]

where E[Y] is the mean of the untruncated distribution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["TruncatedPoissonFamily", "TruncatedNegBinFamily"]


class TruncatedPoissonFamily:
    """Zero-truncated Poisson distribution.

    For positive counts only: P(Y = k | Y > 0) for k = 1, 2, ...

    Parameters
    ----------
    link : str, default='log'
        Link function for the mean parameter
    """

    name = "truncated_poisson"

    def __init__(self, link: str = "log"):
        self.link = link

    def log_likelihood(self, y: NDArray, mu: NDArray) -> float:
        """Log-likelihood for truncated Poisson.

        Parameters
        ----------
        y : ndarray
            Positive counts (must be > 0)
        mu : ndarray
            Poisson mean parameter λ

        Returns
        -------
        float
            Total log-likelihood
        """
        y = np.asarray(y, dtype=float)
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)

        if np.any(y <= 0):
            raise ValueError("y must contain only positive counts for truncated Poisson")

        # log P(Y=k | Y>0) = log f(k) - log(1 - f(0))
        # = k*log(μ) - μ - log(k!) - log(1 - exp(-μ))
        log_poisson = y * np.log(mu) - mu - gammaln(y + 1)
        log_truncation = np.log(1 - np.exp(-mu))

        return float(np.sum(log_poisson - log_truncation))

    def truncated_mean(self, mu: NDArray) -> NDArray:
        """Expected value of truncated Poisson: E[Y | Y > 0].

        E[Y | Y > 0] = μ / [1 - exp(-μ)]
        """
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
        return mu / (1 - np.exp(-mu))

    def variance(self, mu: NDArray) -> NDArray:
        """Variance function for truncated Poisson.

        Var(Y | Y > 0) = μ/[1-exp(-μ)] × [1 - μ×exp(-μ)/(1-exp(-μ))]
        """
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
        p0 = np.exp(-mu)
        trunc_mean = mu / (1 - p0)
        return trunc_mean * (1 - p0 * trunc_mean)

    def d_log_likelihood(self, y: NDArray, mu: NDArray) -> NDArray:
        """First derivative of log-likelihood w.r.t. μ.

        d/dμ log L = y/μ - 1 - exp(-μ)/(1-exp(-μ))
        """
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
        p0 = np.exp(-mu)
        return y / mu - 1 - p0 / (1 - p0)


class TruncatedNegBinFamily:
    """Zero-truncated Negative Binomial distribution.

    For positive counts only: P(Y = k | Y > 0) for k = 1, 2, ...

    Parameters
    ----------
    theta : float
        Dispersion parameter (size)
    link : str, default='log'
        Link function for the mean
    """

    name = "truncated_negative_binomial"

    def __init__(self, theta: float = 1.0, link: str = "log"):
        self.theta = theta
        self.link = link

    def log_likelihood(
        self, y: NDArray, mu: NDArray, theta: float | None = None
    ) -> float:
        """Log-likelihood for truncated Negative Binomial.

        Parameters
        ----------
        y : ndarray
            Positive counts
        mu : ndarray
            NB mean parameter
        theta : float, optional
            Dispersion (uses self.theta if None)

        Returns
        -------
        float
            Total log-likelihood
        """
        if theta is None:
            theta = self.theta

        y = np.asarray(y, dtype=float)
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)

        if np.any(y <= 0):
            raise ValueError("y must contain only positive counts")

        # NB log-likelihood
        log_nb = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1)
            + theta * np.log(theta / (theta + mu))
            + y * np.log(mu / (theta + mu))
        )

        # Truncation: log(1 - P(Y=0))
        p0 = (theta / (theta + mu)) ** theta
        log_truncation = np.log(1 - p0)

        return float(np.sum(log_nb - log_truncation))

    def truncated_mean(self, mu: NDArray, theta: float | None = None) -> NDArray:
        """Expected value of truncated NB: E[Y | Y > 0].

        E[Y | Y > 0] = μ / [1 - (θ/(θ+μ))^θ]
        """
        if theta is None:
            theta = self.theta

        mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
        p0 = (theta / (theta + mu)) ** theta
        return mu / (1 - p0)

    def variance(self, mu: NDArray, theta: float | None = None) -> NDArray:
        """Variance of truncated NB."""
        if theta is None:
            theta = self.theta

        mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
        p0 = (theta / (theta + mu)) ** theta

        # NB variance: μ + μ²/θ
        var_nb = mu + mu**2 / theta

        # Truncated variance is more complex
        trunc_mean = mu / (1 - p0)
        # Approximation
        return var_nb / (1 - p0) - p0 * trunc_mean**2 / (1 - p0)

    def prob_zero_untruncated(self, mu: NDArray, theta: float | None = None) -> NDArray:
        """P(Y=0) from the untruncated distribution."""
        if theta is None:
            theta = self.theta
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
        return (theta / (theta + mu)) ** theta
