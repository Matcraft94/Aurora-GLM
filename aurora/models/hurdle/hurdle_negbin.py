# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Hurdle Negative Binomial Model.

The Hurdle Negative Binomial model combines:
1. Binary GLM: P(Y > 0) via logistic regression
2. Truncated Negative Binomial: E[Y | Y > 0] for overdispersed positive counts

This is appropriate when data has:
- Excess zeros (compared to NB)
- Overdispersion in positive counts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import brentq

from .truncated import TruncatedNegBinFamily

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["fit_hurdle_negbin", "HurdleNegBinResult"]


def fit_hurdle_negbin(
    X_binary: NDArray,
    X_count: NDArray,
    y: NDArray,
    theta_init: float | str = "estimate",
    max_iter: int = 50,
    tol: float = 1e-6,
    verbose: bool = False,
) -> "HurdleNegBinResult":
    """Fit Hurdle Negative Binomial model.

    Two-stage fitting:
    1. Binary logistic GLM: P(Y > 0)
    2. Truncated NB GLM on positive counts: E[Y | Y > 0]

    Parameters
    ----------
    X_binary : array-like, shape (n, p)
        Design matrix for binary model
    X_count : array-like, shape (n, q)
        Design matrix for count model
    y : array-like, shape (n,)
        Response counts
    theta_init : float or 'estimate'
        Initial dispersion parameter
    max_iter : int, default=50
        Maximum iterations
    tol : float, default=1e-6
        Convergence tolerance
    verbose : bool, default=False
        Print progress

    Returns
    -------
    HurdleNegBinResult
        Fitted hurdle model
    """
    # Validate inputs
    X_binary = np.asarray(X_binary, dtype=float)
    X_count = np.asarray(X_count, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    if X_binary.shape[0] != n or X_count.shape[0] != n:
        raise ValueError("X matrices and y must have same number of observations")

    p_binary = X_binary.shape[1]
    p_count = X_count.shape[1]

    # Stage 1: Binary model
    y_binary = (y > 0).astype(float)

    if verbose:
        print("Stage 1: Fitting binary model...")

    gamma = _fit_logistic(X_binary, y_binary, max_iter=max_iter, tol=tol)
    eta_binary = X_binary @ gamma
    pi = 1 / (1 + np.exp(-eta_binary))

    # Stage 2: Truncated NB on positive counts
    mask_positive = y > 0
    n_positive = np.sum(mask_positive)

    if n_positive == 0:
        raise ValueError("No positive counts")

    y_pos = y[mask_positive]
    X_pos = X_count[mask_positive]

    if verbose:
        print(f"Stage 2: Fitting truncated NB on {n_positive} positive counts...")

    # Initialize theta
    if theta_init == "estimate":
        mu_init = np.mean(y_pos)
        var_init = np.var(y_pos)
        excess = var_init - mu_init
        theta = mu_init**2 / max(excess, 0.1) if excess > 0 else 10.0
    else:
        theta = float(theta_init)

    # Fit truncated NB with theta estimation
    beta, theta = _fit_truncated_negbin(
        X_pos, y_pos, theta_init=theta, max_iter=max_iter, tol=tol
    )

    mu_all = np.exp(np.clip(X_count @ beta, -20, 20))

    # Truncated family
    trunc_family = TruncatedNegBinFamily(theta=theta)

    # E[Y | Y > 0]
    trunc_mean = trunc_family.truncated_mean(mu_all, theta)

    # Marginal: E[Y] = π × E[Y | Y > 0]
    expected_count = pi * trunc_mean

    # Log-likelihood
    ll_binary = np.sum(
        y_binary * np.log(pi + 1e-10) + (1 - y_binary) * np.log(1 - pi + 1e-10)
    )
    ll_count = trunc_family.log_likelihood(y_pos, mu_all[mask_positive], theta)
    ll_total = ll_binary + ll_count

    # Information criteria
    k = p_binary + p_count + 1  # +1 for theta
    aic = -2 * ll_total + 2 * k
    bic = -2 * ll_total + k * np.log(n)

    return HurdleNegBinResult(
        coef_binary_=gamma,
        coef_count_=beta,
        theta_=theta,
        pi_=pi,
        mu_=mu_all,
        truncated_mean_=trunc_mean,
        expected_count_=expected_count,
        log_likelihood_=ll_total,
        log_likelihood_binary_=ll_binary,
        log_likelihood_count_=ll_count,
        aic_=aic,
        bic_=bic,
        n_obs_=n,
        n_positive_=int(n_positive),
        X_binary_=X_binary,
        X_count_=X_count,
        y_=y,
    )


def _fit_logistic(
    X: NDArray, y: NDArray, max_iter: int = 50, tol: float = 1e-6
) -> NDArray:
    """Fit logistic regression via IRLS."""
    n, p = X.shape
    gamma = np.zeros(p)

    for iteration in range(max_iter):
        eta = X @ gamma
        pi = 1 / (1 + np.exp(-eta))
        pi = np.clip(pi, 1e-10, 1 - 1e-10)

        W = pi * (1 - pi)
        z = eta + (y - pi) / W

        XtW = X.T * W
        XtWX = XtW @ X + 1e-8 * np.eye(p)
        XtWz = XtW @ z

        try:
            gamma_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(gamma_new - gamma)) < tol:
            gamma = gamma_new
            break
        gamma = gamma_new

    return gamma


def _fit_truncated_negbin(
    X: NDArray,
    y: NDArray,
    theta_init: float,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[NDArray, float]:
    """Fit truncated NB with theta estimation.

    Alternates between:
    1. Fixing theta, update beta via IRLS
    2. Fixing beta, update theta via profile likelihood
    """
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = np.log(max(y.mean(), 1))
    theta = theta_init

    for outer_iter in range(max_iter):
        # Update beta given theta
        beta_new = _update_beta_truncnb(X, y, beta, theta, max_iter=20, tol=tol)

        # Update theta given beta
        mu = np.exp(np.clip(X @ beta_new, -20, 20))
        theta_new = _update_theta_truncnb(y, mu, theta)

        # Check convergence
        if (np.max(np.abs(beta_new - beta)) < tol and
            abs(theta_new - theta) / (theta + 1e-10) < tol):
            beta = beta_new
            theta = theta_new
            break

        beta = beta_new
        theta = theta_new

    return beta, theta


def _update_beta_truncnb(
    X: NDArray,
    y: NDArray,
    beta_init: NDArray,
    theta: float,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> NDArray:
    """Update beta via IRLS for truncated NB."""
    n, p = X.shape
    beta = beta_init.copy()

    for iteration in range(max_iter):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.exp(eta)

        # P(Y=0) for untruncated NB
        p0 = (theta / (theta + mu)) ** theta

        # Truncated mean
        trunc_mean = mu / (1 - p0 + 1e-10)

        # NB variance: μ + μ²/θ
        var_nb = mu + mu**2 / theta

        # Working weights (approximate)
        W = mu**2 / (var_nb * (1 - p0) + 1e-10)
        W = np.maximum(W, 1e-10)

        # Working response
        z = eta + (y - trunc_mean) / (W / mu + 1e-10)

        XtW = X.T * W
        XtWX = XtW @ X + 1e-8 * np.eye(p)
        XtWz = XtW @ z

        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return beta


def _update_theta_truncnb(y: NDArray, mu: NDArray, theta_init: float) -> float:
    """Update theta via profile likelihood for truncated NB."""
    n = len(y)

    def neg_profile_ll(log_theta: float) -> float:
        theta = np.exp(log_theta)

        # P(Y=0) untruncated
        p0 = (theta / (theta + mu)) ** theta

        # Truncated NB log-likelihood
        log_nb = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1)
            + theta * np.log(theta / (theta + mu))
            + y * np.log(mu / (theta + mu))
        )
        log_trunc = np.log(1 - p0 + 1e-10)

        return -np.sum(log_nb - log_trunc)

    # Grid search + refinement
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        neg_profile_ll,
        bounds=(np.log(0.01), np.log(1000)),
        method="bounded",
    )

    return np.exp(result.x)


@dataclass
class HurdleNegBinResult:
    """Results from Hurdle Negative Binomial model.

    Attributes
    ----------
    coef_binary_ : ndarray
        Binary model coefficients
    coef_count_ : ndarray
        Count model coefficients
    theta_ : float
        Dispersion parameter
    pi_ : ndarray
        P(Y > 0)
    mu_ : ndarray
        NB parameter μ
    truncated_mean_ : ndarray
        E[Y | Y > 0]
    expected_count_ : ndarray
        Marginal E[Y]
    log_likelihood_ : float
        Total log-likelihood
    aic_ : float
        AIC
    bic_ : float
        BIC
    """

    coef_binary_: NDArray
    coef_count_: NDArray
    theta_: float
    pi_: NDArray
    mu_: NDArray
    truncated_mean_: NDArray
    expected_count_: NDArray
    log_likelihood_: float
    log_likelihood_binary_: float
    log_likelihood_count_: float
    aic_: float
    bic_: float
    n_obs_: int
    n_positive_: int
    X_binary_: NDArray
    X_count_: NDArray
    y_: NDArray

    def predict(
        self,
        X_binary: NDArray | None = None,
        X_count: NDArray | None = None,
        type: str = "response",
    ) -> NDArray:
        """Predict from fitted model."""
        if X_binary is None:
            X_binary = self.X_binary_
        else:
            X_binary = np.asarray(X_binary)

        if X_count is None:
            X_count = self.X_count_
        else:
            X_count = np.asarray(X_count)

        eta_binary = X_binary @ self.coef_binary_
        pi = 1 / (1 + np.exp(-eta_binary))

        eta_count = np.clip(X_count @ self.coef_count_, -20, 20)
        mu = np.exp(eta_count)
        p0 = (self.theta_ / (self.theta_ + mu)) ** self.theta_
        trunc_mean = mu / (1 - p0)

        if type == "response":
            return pi * trunc_mean
        elif type == "prob_positive":
            return pi
        elif type == "prob_zero":
            return 1 - pi
        elif type == "count":
            return trunc_mean
        elif type == "mu":
            return mu
        else:
            raise ValueError(f"Unknown type: {type}")

    def summary(self) -> dict:
        """Return model summary."""
        return {
            "n_obs": self.n_obs_,
            "n_zeros": self.n_obs_ - self.n_positive_,
            "n_positive": self.n_positive_,
            "theta": self.theta_,
            "n_params_binary": len(self.coef_binary_),
            "n_params_count": len(self.coef_count_),
            "log_likelihood": self.log_likelihood_,
            "aic": self.aic_,
            "bic": self.bic_,
            "coef_binary": self.coef_binary_,
            "coef_count": self.coef_count_,
            "mean_pi": float(self.pi_.mean()),
            "mean_mu": float(self.mu_.mean()),
        }
