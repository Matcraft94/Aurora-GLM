# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Hurdle Poisson Model.

The Hurdle Poisson model combines:
1. Binary GLM: P(Y > 0) via logistic regression
2. Truncated Poisson: E[Y | Y > 0] via log-linear model

Mathematical Framework
----------------------
    P(Y = 0) = 1 - π
    P(Y = k) = π × [λ^k exp(-λ) / k!] / [1 - exp(-λ)]   for k > 0

where:
- π = P(Y > 0) from logistic model
- λ is the Poisson parameter
- The truncated Poisson has mean λ/[1-exp(-λ)]

Marginal moments:
    E[Y] = π × λ / [1 - exp(-λ)]

Estimation is straightforward as the binary and count components
are separable - they can be fit independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln

from .truncated import TruncatedPoissonFamily

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["fit_hurdle_poisson", "HurdlePoissonResult"]


def fit_hurdle_poisson(
    X_binary: NDArray,
    X_count: NDArray,
    y: NDArray,
    max_iter: int = 50,
    tol: float = 1e-6,
    verbose: bool = False,
) -> "HurdlePoissonResult":
    """Fit Hurdle Poisson model.

    Two-stage fitting:
    1. Binary logistic GLM on all data: P(Y > 0)
    2. Truncated Poisson GLM on positive counts: E[Y | Y > 0]

    Parameters
    ----------
    X_binary : array-like, shape (n, p)
        Design matrix for binary model (logistic, all observations)
    X_count : array-like, shape (n, q)
        Design matrix for count model (log link, all observations).
        Only positive counts will be used for fitting.
    y : array-like, shape (n,)
        Response counts (non-negative integers)
    max_iter : int, default=50
        Maximum iterations for each GLM
    tol : float, default=1e-6
        Convergence tolerance
    verbose : bool, default=False
        Print fitting progress

    Returns
    -------
    HurdlePoissonResult
        Fitted hurdle model

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.hurdle import fit_hurdle_poisson
    >>>
    >>> # Generate hurdle Poisson data
    >>> n = 500
    >>> X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
    >>>
    >>> # True parameters
    >>> gamma_true = [-0.5, 0.8]  # Binary model
    >>> beta_true = [1.5, 0.3]   # Count model
    >>>
    >>> pi = 1 / (1 + np.exp(-X @ gamma_true))
    >>> is_positive = np.random.binomial(1, pi)
    >>> mu = np.exp(X @ beta_true)
    >>> y_pos = np.random.poisson(mu)
    >>> y_pos[y_pos == 0] = 1  # Truncate at zero
    >>> y = np.where(is_positive, y_pos, 0)
    >>>
    >>> # Fit model
    >>> result = fit_hurdle_poisson(X, X, y)
    >>> print(f"Binary coefs: {result.coef_binary_}")
    >>> print(f"Count coefs: {result.coef_count_}")
    """
    # Validate inputs
    X_binary = np.asarray(X_binary, dtype=float)
    X_count = np.asarray(X_count, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    if X_binary.shape[0] != n or X_count.shape[0] != n:
        raise ValueError("X matrices and y must have same number of observations")
    if np.any(y < 0) or not np.allclose(y, y.astype(int)):
        raise ValueError("y must contain non-negative integers")

    p_binary = X_binary.shape[1]
    p_count = X_count.shape[1]

    # Stage 1: Binary model P(Y > 0) on all data
    y_binary = (y > 0).astype(float)

    if verbose:
        print("Stage 1: Fitting binary model...")

    gamma = _fit_logistic(X_binary, y_binary, max_iter=max_iter, tol=tol)
    eta_binary = X_binary @ gamma
    pi = 1 / (1 + np.exp(-eta_binary))

    # Stage 2: Truncated Poisson on positive counts only
    mask_positive = y > 0
    n_positive = np.sum(mask_positive)

    if n_positive == 0:
        raise ValueError("No positive counts in data - cannot fit count model")

    y_pos = y[mask_positive]
    X_pos = X_count[mask_positive]

    if verbose:
        print(f"Stage 2: Fitting truncated Poisson on {n_positive} positive counts...")

    beta = _fit_truncated_poisson(X_pos, y_pos, max_iter=max_iter, tol=tol)
    mu_all = np.exp(np.clip(X_count @ beta, -20, 20))

    # Truncated family for computing results
    trunc_family = TruncatedPoissonFamily()

    # Compute expected values
    # E[Y | Y > 0] = μ / [1 - exp(-μ)]
    trunc_mean = trunc_family.truncated_mean(mu_all)

    # Marginal expected count: E[Y] = π × E[Y | Y > 0]
    expected_count = pi * trunc_mean

    # Log-likelihood
    ll_binary = np.sum(y_binary * np.log(pi + 1e-10) + (1 - y_binary) * np.log(1 - pi + 1e-10))
    ll_count = trunc_family.log_likelihood(y_pos, mu_all[mask_positive])
    ll_total = ll_binary + ll_count

    # Information criteria
    k = p_binary + p_count
    aic = -2 * ll_total + 2 * k
    bic = -2 * ll_total + k * np.log(n)

    return HurdlePoissonResult(
        coef_binary_=gamma,
        coef_count_=beta,
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

        # Working weights and response
        W = pi * (1 - pi)
        z = eta + (y - pi) / W

        # Weighted least squares
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


def _fit_truncated_poisson(
    X: NDArray, y: NDArray, max_iter: int = 50, tol: float = 1e-6
) -> NDArray:
    """Fit truncated Poisson via IRLS.

    For truncated Poisson, the score function is:
        d log L / d β = X'(y - μ̃)
    where μ̃ = μ / [1 - exp(-μ)] is the truncated mean.

    We use modified IRLS with adjusted working response.
    """
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = np.log(max(y.mean(), 1))  # Initialize at data mean

    for iteration in range(max_iter):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.exp(eta)

        # Truncated mean: μ / [1 - exp(-μ)]
        p0 = np.exp(-mu)
        trunc_mean = mu / (1 - p0 + 1e-10)

        # Working weights
        # W ∝ μ × [1 - exp(-μ) - μ exp(-μ)] / [1 - exp(-μ)]²
        numer = 1 - p0 - mu * p0
        denom = (1 - p0) ** 2 + 1e-10
        W = mu * numer / denom
        W = np.maximum(W, 1e-10)

        # Working response
        z = eta + (y - trunc_mean) / (W / mu + 1e-10)

        # Weighted least squares
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


@dataclass
class HurdlePoissonResult:
    """Results from Hurdle Poisson model.

    Attributes
    ----------
    coef_binary_ : ndarray
        Coefficients for binary model (logit link)
    coef_count_ : ndarray
        Coefficients for count model (log link)
    pi_ : ndarray
        Fitted P(Y > 0) at training points
    mu_ : ndarray
        Fitted Poisson parameter λ at training points
    truncated_mean_ : ndarray
        E[Y | Y > 0] = λ / [1 - exp(-λ)]
    expected_count_ : ndarray
        Marginal mean E[Y] = π × E[Y | Y > 0]
    log_likelihood_ : float
        Total log-likelihood
    log_likelihood_binary_ : float
        Binary component log-likelihood
    log_likelihood_count_ : float
        Count component log-likelihood
    aic_ : float
        AIC
    bic_ : float
        BIC
    n_obs_ : int
        Number of observations
    n_positive_ : int
        Number of positive counts
    """

    coef_binary_: NDArray
    coef_count_: NDArray
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
        """Predict from fitted hurdle model.

        Parameters
        ----------
        X_binary : ndarray, optional
            Design matrix for binary model
        X_count : ndarray, optional
            Design matrix for count model
        type : str
            'response' (marginal mean), 'prob_positive' (π),
            'count' (truncated mean), 'prob_zero' (1-π)

        Returns
        -------
        ndarray
            Predicted values
        """
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
        trunc_mean = mu / (1 - np.exp(-mu))

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
            "n_params_binary": len(self.coef_binary_),
            "n_params_count": len(self.coef_count_),
            "log_likelihood": self.log_likelihood_,
            "log_likelihood_binary": self.log_likelihood_binary_,
            "log_likelihood_count": self.log_likelihood_count_,
            "aic": self.aic_,
            "bic": self.bic_,
            "coef_binary": self.coef_binary_,
            "coef_count": self.coef_count_,
            "mean_pi": float(self.pi_.mean()),
            "mean_mu": float(self.mu_.mean()),
        }
