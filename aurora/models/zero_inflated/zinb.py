# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Zero-Inflated Negative Binomial (ZINB) Model.

The Zero-Inflated Negative Binomial model handles count data with both
excess zeros AND overdispersion (variance exceeding mean beyond what
zero-inflation alone would explain).

Mathematical Framework
----------------------
ZINB combines a point mass at zero with a Negative Binomial distribution:

    P(Y = 0) = π + (1-π) × (θ/(θ+μ))^θ
    P(Y = k) = (1-π) × NB(k; μ, θ)   for k > 0

where:
- π ∈ [0, 1] is the zero-inflation probability
- μ > 0 is the NB mean parameter
- θ > 0 is the NB dispersion parameter

The NB probability mass function is:
    NB(k; μ, θ) = Γ(k+θ)/[Γ(θ)k!] × (θ/(θ+μ))^θ × (μ/(θ+μ))^k

Properties:
    E[Y] = (1-π) × μ
    Var[Y] = (1-π) × μ × (1 + πμ + μ/θ)

The variance has three sources:
- Poisson variation: μ
- Overdispersion: μ²/θ
- Zero-inflation: πμ²

When to Use ZINB vs ZIP
-----------------------
- Use ZIP when excess zeros are the only source of overdispersion
- Use ZINB when data has both excess zeros AND additional overdispersion
- Compare models using Vuong test or likelihood ratio test

References
----------
.. [1] Ridout, M., Demétrio, C. G., & Hinde, J. (1998).
       "Models for count data with many zeros."
       Proceedings of the XIXth International Biometric Conference.
.. [2] Zuur, A. F., et al. (2009).
       Mixed Effects Models and Extensions in Ecology with R.
       Springer, Chapter 11.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import brentq

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ZeroInflatedNegBinFamily", "fit_zinb", "ZINBResult"]


class ZeroInflatedNegBinFamily:
    """Zero-Inflated Negative Binomial distribution family.

    Combines a point mass at zero with a Negative Binomial distribution
    for count data with excess zeros and overdispersion.

    Parameters
    ----------
    theta : float, default=1.0
        Initial dispersion parameter (will be estimated during fitting)
    link_count : str, default='log'
        Link function for count model
    link_inflate : str, default='logit'
        Link function for inflation model
    """

    name = "zero_inflated_negative_binomial"

    def __init__(
        self,
        theta: float = 1.0,
        link_count: str = "log",
        link_inflate: str = "logit",
    ):
        self.theta = theta
        self.link_count = link_count
        self.link_inflate = link_inflate

    def log_likelihood(
        self, y: NDArray, mu: NDArray, pi: NDArray, theta: float | None = None
    ) -> float:
        """Compute ZINB log-likelihood.

        Parameters
        ----------
        y : ndarray
            Observed counts
        mu : ndarray
            NB mean from count model
        pi : ndarray
            Zero-inflation probability
        theta : float, optional
            Dispersion parameter (uses self.theta if None)

        Returns
        -------
        float
            Total log-likelihood
        """
        if theta is None:
            theta = self.theta

        y = np.asarray(y, dtype=float)
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
        pi = np.clip(np.asarray(pi, dtype=float), 1e-10, 1 - 1e-10)

        # P(Y=0 | NB) = (θ/(θ+μ))^θ
        p_zero_nb = (theta / (theta + mu)) ** theta

        # Zero observations: log[π + (1-π) × p_zero_nb]
        is_zero = y == 0
        log_lik_zero = np.log(pi + (1 - pi) * p_zero_nb)

        # Positive observations: log(1-π) + NB log-likelihood
        log_nb = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1)
            + theta * np.log(theta / (theta + mu))
            + y * np.log(mu / (theta + mu))
        )
        log_lik_pos = np.log(1 - pi) + log_nb

        log_lik = np.where(is_zero, log_lik_zero, log_lik_pos)
        return float(np.sum(log_lik))

    def e_step(
        self, y: NDArray, mu: NDArray, pi: NDArray, theta: float | None = None
    ) -> NDArray:
        """E-step: Compute posterior probability of structural zero.

        Parameters
        ----------
        y : ndarray
            Observed counts
        mu : ndarray
            NB mean
        pi : ndarray
            Zero-inflation probability
        theta : float, optional
            Dispersion parameter

        Returns
        -------
        z : ndarray
            Posterior probability of structural zero
        """
        if theta is None:
            theta = self.theta

        y = np.asarray(y, dtype=float)
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
        pi = np.clip(np.asarray(pi, dtype=float), 1e-10, 1 - 1e-10)

        # P(Y=0 | NB)
        p_zero_nb = (theta / (theta + mu)) ** theta

        # For y=0: P(structural | y=0) = π / [π + (1-π) × p_zero_nb]
        p_structural_given_zero = pi / (pi + (1 - pi) * p_zero_nb)

        # For y>0: must be from count process
        z = np.where(y == 0, p_structural_given_zero, 0.0)

        return z

    def expected_count(self, mu: NDArray, pi: NDArray) -> NDArray:
        """Compute marginal expected count E[Y] = (1-π)μ."""
        return (1 - pi) * mu

    def prob_zero(
        self, mu: NDArray, pi: NDArray, theta: float | None = None
    ) -> NDArray:
        """Compute P(Y=0) = π + (1-π)(θ/(θ+μ))^θ."""
        if theta is None:
            theta = self.theta
        p_zero_nb = (theta / (theta + mu)) ** theta
        return pi + (1 - pi) * p_zero_nb


def fit_zinb(
    X_count: NDArray,
    y: NDArray,
    X_inflate: NDArray | None = None,
    theta_init: float | str = "estimate",
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> "ZINBResult":
    """Fit Zero-Inflated Negative Binomial model via EM algorithm.

    Parameters
    ----------
    X_count : array-like, shape (n, p)
        Design matrix for count model (NB, log link)
    y : array-like, shape (n,)
        Response counts (non-negative integers)
    X_inflate : array-like, shape (n, q), optional
        Design matrix for inflation model (logistic).
        If None, uses intercept-only model.
    theta_init : float or 'estimate'
        Initial dispersion parameter. If 'estimate', uses method of moments.
    max_iter : int, default=100
        Maximum EM iterations
    tol : float, default=1e-6
        Convergence tolerance
    verbose : bool, default=False
        Print convergence information

    Returns
    -------
    ZINBResult
        Fitted ZINB model

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.zero_inflated import fit_zinb
    >>>
    >>> # Generate ZINB data
    >>> n = 500
    >>> X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
    >>> result = fit_zinb(X, y)
    """
    # Validate inputs
    X_count = np.asarray(X_count, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    if X_count.shape[0] != n:
        raise ValueError("X_count and y must have same number of observations")
    if np.any(y < 0) or not np.allclose(y, y.astype(int)):
        raise ValueError("y must contain non-negative integers")

    # Default inflation design: intercept only
    if X_inflate is None:
        X_inflate = np.ones((n, 1))
    else:
        X_inflate = np.asarray(X_inflate, dtype=float)

    # Dimensions
    p_count = X_count.shape[1]
    p_inflate = X_inflate.shape[1]

    # Initialize with NB fit
    from aurora.models.glm import fit_glm

    try:
        nb_result = fit_glm(
            X_count, y, family="negativebinomial", link="log",
            family_params={"theta": 1.0}
        )
        beta = nb_result.coef_.copy()
    except Exception:
        beta = np.zeros(p_count)
        beta[0] = np.log(max(y.mean(), 0.1))

    mu = np.exp(X_count @ beta)

    # Initialize theta
    if theta_init == "estimate":
        theta = _estimate_theta_moments(y, mu)
    else:
        theta = float(theta_init)

    # Initialize inflation
    p_zero_nb = (theta / (theta + mu)) ** theta
    expected_zeros = np.sum(p_zero_nb)
    observed_zeros = np.sum(y == 0)
    pi_init = max(0.01, min(0.99, (observed_zeros - expected_zeros) / n))
    gamma = np.zeros(p_inflate)
    gamma[0] = np.log(pi_init / (1 - pi_init))

    family = ZeroInflatedNegBinFamily(theta=theta)

    # EM algorithm
    prev_ll = -np.inf
    converged = False

    for iteration in range(max_iter):
        # Current pi
        eta_inflate = X_inflate @ gamma
        pi = 1 / (1 + np.exp(-eta_inflate))
        pi = np.clip(pi, 1e-10, 1 - 1e-10)

        # E-step
        z = family.e_step(y, mu, pi, theta)

        # Log-likelihood
        ll = family.log_likelihood(y, mu, pi, theta)

        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: log-lik = {ll:.4f}, theta = {theta:.4f}")

        # Check convergence
        if iteration > 0:
            rel_change = abs(ll - prev_ll) / (abs(prev_ll) + 1e-10)
            if rel_change < tol:
                converged = True
                break
        prev_ll = ll

        # M-step for count model: weighted NB GLM
        weights_count = 1 - z
        weights_count = np.maximum(weights_count, 1e-10)

        beta = _fit_weighted_negbin(X_count, y, weights_count, beta, theta)
        mu = np.exp(np.clip(X_count @ beta, -20, 20))

        # M-step for inflation model
        gamma = _fit_weighted_logistic(X_inflate, z, gamma)

        # M-step for theta (dispersion)
        theta = _estimate_theta_weighted(y, mu, weights_count, theta)
        theta = max(0.01, min(1e6, theta))
        family.theta = theta

    # Final predictions
    eta_inflate = X_inflate @ gamma
    pi = 1 / (1 + np.exp(-eta_inflate))
    mu = np.exp(np.clip(X_count @ beta, -20, 20))
    expected_count = (1 - pi) * mu
    ll = family.log_likelihood(y, mu, pi, theta)

    # Information criteria
    k = p_count + p_inflate + 1  # +1 for theta
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)

    return ZINBResult(
        coef_count_=beta,
        coef_inflate_=gamma,
        theta_=theta,
        mu_=mu,
        pi_=pi,
        expected_count_=expected_count,
        converged_=converged,
        n_iter_=iteration + 1,
        log_likelihood_=ll,
        aic_=aic,
        bic_=bic,
        X_count_=X_count,
        X_inflate_=X_inflate,
        y_=y,
    )


def _estimate_theta_moments(y: NDArray, mu: NDArray) -> float:
    """Method of moments estimate for theta."""
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)

    mean_y = np.mean(y)
    var_y = np.var(y)

    # Var = μ + μ²/θ  =>  θ = μ²/(Var - μ)
    excess_var = var_y - mean_y
    if excess_var <= 0:
        return 1e6  # Near Poisson

    theta = mean_y**2 / excess_var
    return float(np.clip(theta, 0.01, 1e6))


def _estimate_theta_weighted(
    y: NDArray, mu: NDArray, weights: NDArray, theta_init: float
) -> float:
    """ML estimate of theta given current mu and weights."""
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
    weights = np.asarray(weights, dtype=float)

    # Weighted score function for theta
    def score(log_theta: float) -> float:
        theta = np.exp(log_theta)
        # d/dθ [weighted log-lik]
        psi_deriv = digamma(y + theta) - digamma(theta)
        term1 = np.sum(weights * psi_deriv)
        term2 = np.sum(weights * np.log(theta / (theta + mu)))
        term3 = np.sum(weights * (1 - (y + theta) / (theta + mu)))
        return term1 + term2 + term3

    # Search for root
    try:
        log_theta_opt = brentq(score, np.log(0.01), np.log(1e6), maxiter=50)
        return np.exp(log_theta_opt)
    except (ValueError, RuntimeError):
        return theta_init


def _fit_weighted_negbin(
    X: NDArray,
    y: NDArray,
    weights: NDArray,
    beta_init: NDArray,
    theta: float,
    max_iter: int = 25,
) -> NDArray:
    """Fit weighted Negative Binomial GLM via IRLS."""
    beta = beta_init.copy()
    n, p = X.shape

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.exp(eta)

        # NB variance: V(μ) = μ + μ²/θ
        var_mu = mu + mu**2 / theta

        # Working weights: W = weights * μ² / V(μ)
        W = weights * mu**2 / np.maximum(var_mu, 1e-10)
        W = np.maximum(W, 1e-10)

        # Working response
        z = eta + (y - mu) / np.maximum(mu, 1e-10)

        # Weighted least squares
        XtW = X.T * W
        XtWX = XtW @ X
        XtWz = XtW @ z

        XtWX += 1e-8 * np.eye(p)

        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    return beta


def _fit_weighted_logistic(
    X: NDArray, y: NDArray, gamma_init: NDArray, max_iter: int = 25
) -> NDArray:
    """Fit logistic regression via IRLS."""
    gamma = gamma_init.copy()
    n, q = X.shape

    for _ in range(max_iter):
        eta = X @ gamma
        pi = 1 / (1 + np.exp(-eta))
        pi = np.clip(pi, 1e-10, 1 - 1e-10)

        W = pi * (1 - pi)
        W = np.maximum(W, 1e-10)

        z = eta + (y - pi) / W

        XtW = X.T * W
        XtWX = XtW @ X
        XtWz = XtW @ z

        XtWX += 1e-8 * np.eye(q)

        try:
            gamma_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(gamma_new - gamma)) < 1e-8:
            gamma = gamma_new
            break
        gamma = gamma_new

    return gamma


@dataclass
class ZINBResult:
    """Results from Zero-Inflated Negative Binomial model.

    Attributes
    ----------
    coef_count_ : ndarray
        Coefficients for count model (log link)
    coef_inflate_ : ndarray
        Coefficients for inflation model (logit link)
    theta_ : float
        Estimated dispersion parameter
    mu_ : ndarray
        Fitted NB mean at training points
    pi_ : ndarray
        Fitted zero-inflation probability
    expected_count_ : ndarray
        Marginal expected count: (1-π)μ
    converged_ : bool
        Whether EM converged
    n_iter_ : int
        Number of iterations
    log_likelihood_ : float
        Log-likelihood at convergence
    aic_ : float
        AIC
    bic_ : float
        BIC
    """

    coef_count_: NDArray
    coef_inflate_: NDArray
    theta_: float
    mu_: NDArray
    pi_: NDArray
    expected_count_: NDArray
    converged_: bool
    n_iter_: int
    log_likelihood_: float
    aic_: float
    bic_: float
    X_count_: NDArray
    X_inflate_: NDArray
    y_: NDArray

    def predict(
        self,
        X_count: NDArray | None = None,
        X_inflate: NDArray | None = None,
        type: str = "response",
    ) -> NDArray:
        """Predict from fitted ZINB model.

        Parameters
        ----------
        X_count : ndarray, optional
            New count design matrix
        X_inflate : ndarray, optional
            New inflation design matrix
        type : str
            'response', 'count', 'prob_zero', or 'prob_inflate'

        Returns
        -------
        ndarray
            Predicted values
        """
        if X_count is None:
            X_count = self.X_count_
        else:
            X_count = np.asarray(X_count)

        if X_inflate is None:
            X_inflate = self.X_inflate_
        else:
            X_inflate = np.asarray(X_inflate)

        mu = np.exp(np.clip(X_count @ self.coef_count_, -20, 20))
        pi = 1 / (1 + np.exp(-X_inflate @ self.coef_inflate_))

        if type == "response":
            return (1 - pi) * mu
        elif type == "count":
            return mu
        elif type == "prob_zero":
            p_zero_nb = (self.theta_ / (self.theta_ + mu)) ** self.theta_
            return pi + (1 - pi) * p_zero_nb
        elif type == "prob_inflate":
            return pi
        else:
            raise ValueError(f"Unknown type: {type}")

    def summary(self) -> dict:
        """Return model summary."""
        n = len(self.y_)
        k = len(self.coef_count_) + len(self.coef_inflate_) + 1

        return {
            "n_obs": n,
            "n_zeros": int(np.sum(self.y_ == 0)),
            "n_params": k,
            "theta": self.theta_,
            "log_likelihood": self.log_likelihood_,
            "aic": self.aic_,
            "bic": self.bic_,
            "converged": self.converged_,
            "n_iter": self.n_iter_,
            "coef_count": self.coef_count_,
            "coef_inflate": self.coef_inflate_,
            "mean_pi": float(self.pi_.mean()),
            "mean_mu": float(self.mu_.mean()),
        }
