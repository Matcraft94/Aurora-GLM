# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Zero-Inflated Poisson (ZIP) Model.

The Zero-Inflated Poisson model handles count data with excess zeros by
combining a point mass at zero with a Poisson distribution.

Mathematical Framework
----------------------
ZIP is a mixture model with two components:

    P(Y = 0) = π + (1-π) × exp(-λ)
    P(Y = k) = (1-π) × [λ^k × exp(-λ) / k!]   for k > 0

where:
- π ∈ [0, 1] is the zero-inflation probability (structural zeros)
- λ > 0 is the Poisson mean parameter

Model Components:
- **Inflation model**: logit(π) = Z'γ (predicts structural zeros)
- **Count model**: log(λ) = X'β (predicts count given not structural zero)

Properties:
    E[Y] = (1-π) × λ
    Var[Y] = (1-π) × λ × (1 + πλ)

The variance exceeds the mean when π > 0, so ZIP handles "apparent"
overdispersion due to excess zeros.

Estimation
----------
ZIP is typically estimated via EM algorithm:

E-step: Compute posterior probability of structural zero for each y=0:
    z_i = P(structural | y_i=0) = π_i / [π_i + (1-π_i)exp(-λ_i)]

M-step: Update parameters using weighted likelihoods:
    - Count model: weighted Poisson GLM with weights (1-z_i)
    - Inflation model: weighted logistic regression with response z_i

References
----------
.. [1] Lambert, D. (1992).
       "Zero-inflated Poisson regression, with an application to defects
       in manufacturing."
       Technometrics, 34(1), 1-14.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ZeroInflatedPoissonFamily", "fit_zip", "ZIPResult"]


class ZeroInflatedPoissonFamily:
    """Zero-Inflated Poisson distribution family.

    This family represents a mixture of a point mass at zero and a
    Poisson distribution for count data with excess zeros.

    Parameters
    ----------
    link_count : str, default='log'
        Link function for count model: 'log', 'identity', 'sqrt'
    link_inflate : str, default='logit'
        Link function for inflation model: 'logit', 'probit', 'cloglog'
    """

    name = "zero_inflated_poisson"

    def __init__(self, link_count: str = "log", link_inflate: str = "logit"):
        self.link_count = link_count
        self.link_inflate = link_inflate

    def log_likelihood(
        self, y: NDArray, mu: NDArray, pi: NDArray
    ) -> float:
        """Compute ZIP log-likelihood.

        Parameters
        ----------
        y : ndarray
            Observed counts
        mu : ndarray
            Poisson mean (λ) from count model
        pi : ndarray
            Zero-inflation probability from inflation model

        Returns
        -------
        float
            Total log-likelihood
        """
        y = np.asarray(y, dtype=float)
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
        pi = np.clip(np.asarray(pi, dtype=float), 1e-10, 1 - 1e-10)

        # Zero observations
        is_zero = y == 0
        # log[π + (1-π)exp(-λ)]
        log_lik_zero = np.log(pi + (1 - pi) * np.exp(-mu))

        # Positive observations: log(1-π) + Poisson log-lik
        log_lik_pos = np.log(1 - pi) + y * np.log(mu) - mu - gammaln(y + 1)

        log_lik = np.where(is_zero, log_lik_zero, log_lik_pos)
        return float(np.sum(log_lik))

    def e_step(self, y: NDArray, mu: NDArray, pi: NDArray) -> NDArray:
        """E-step: Compute posterior probability of structural zero.

        For observations with y=0, computes:
            P(structural | y=0) = π / [π + (1-π)exp(-λ)]

        For y>0, returns 0 (must be from count process).

        Parameters
        ----------
        y : ndarray
            Observed counts
        mu : ndarray
            Poisson mean from count model
        pi : ndarray
            Zero-inflation probability

        Returns
        -------
        z : ndarray
            Posterior probability of structural zero
        """
        y = np.asarray(y, dtype=float)
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
        pi = np.clip(np.asarray(pi, dtype=float), 1e-10, 1 - 1e-10)

        # For y=0: P(structural | y=0) = π / [π + (1-π)exp(-λ)]
        p_zero_poisson = np.exp(-mu)
        p_structural_given_zero = pi / (pi + (1 - pi) * p_zero_poisson)

        # For y>0: must be from count process
        z = np.where(y == 0, p_structural_given_zero, 0.0)

        return z

    def expected_count(self, mu: NDArray, pi: NDArray) -> NDArray:
        """Compute marginal expected count E[Y] = (1-π)λ.

        Parameters
        ----------
        mu : ndarray
            Poisson mean (λ)
        pi : ndarray
            Zero-inflation probability

        Returns
        -------
        ndarray
            Marginal expected counts
        """
        return (1 - pi) * mu

    def prob_zero(self, mu: NDArray, pi: NDArray) -> NDArray:
        """Compute probability of zero: P(Y=0) = π + (1-π)exp(-λ).

        Parameters
        ----------
        mu : ndarray
            Poisson mean
        pi : ndarray
            Zero-inflation probability

        Returns
        -------
        ndarray
            Probability of observing zero
        """
        return pi + (1 - pi) * np.exp(-mu)


def fit_zip(
    X_count: NDArray,
    y: NDArray,
    X_inflate: NDArray | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> "ZIPResult":
    """Fit Zero-Inflated Poisson model via EM algorithm.

    Parameters
    ----------
    X_count : array-like, shape (n, p)
        Design matrix for count model (Poisson, log link)
    y : array-like, shape (n,)
        Response counts (non-negative integers)
    X_inflate : array-like, shape (n, q), optional
        Design matrix for inflation model (logistic).
        If None, uses intercept-only model.
    max_iter : int, default=100
        Maximum EM iterations
    tol : float, default=1e-6
        Convergence tolerance (relative change in log-likelihood)
    verbose : bool, default=False
        Print convergence information

    Returns
    -------
    ZIPResult
        Fitted ZIP model with coefficients and predictions

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.zero_inflated import fit_zip
    >>>
    >>> # Generate ZIP data
    >>> n = 500
    >>> X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
    >>> beta_true = [1.0, 0.5]
    >>> gamma_true = [-1.0]  # Intercept-only inflation
    >>>
    >>> mu_true = np.exp(X @ beta_true)
    >>> pi_true = 1 / (1 + np.exp(-gamma_true[0]))
    >>> structural_zero = np.random.binomial(1, pi_true, n)
    >>> y = np.where(structural_zero, 0, np.random.poisson(mu_true))
    >>>
    >>> # Fit ZIP
    >>> result = fit_zip(X, y)
    >>> print(f"Count coefficients: {result.coef_count_}")
    >>> print(f"Zero-inflation rate: {result.pi_.mean():.3f}")
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

    # Initialize with Poisson fit
    from aurora.models.glm import fit_glm

    try:
        poisson_result = fit_glm(X_count, y, family="poisson", link="log")
        beta = poisson_result.coef_.copy()
    except Exception:
        # Fallback initialization
        beta = np.zeros(p_count)
        beta[0] = np.log(max(y.mean(), 0.1))

    mu = np.exp(X_count @ beta)

    # Initialize inflation: estimate from excess zeros
    expected_zeros = np.sum(np.exp(-mu))
    observed_zeros = np.sum(y == 0)
    pi_init = max(0.01, min(0.99, (observed_zeros - expected_zeros) / n))
    gamma = np.zeros(p_inflate)
    gamma[0] = np.log(pi_init / (1 - pi_init))  # Logit scale

    family = ZeroInflatedPoissonFamily()

    # EM algorithm
    prev_ll = -np.inf
    converged = False

    for iteration in range(max_iter):
        # Current pi from logit model
        eta_inflate = X_inflate @ gamma
        pi = 1 / (1 + np.exp(-eta_inflate))
        pi = np.clip(pi, 1e-10, 1 - 1e-10)

        # E-step: posterior probability of structural zero
        z = family.e_step(y, mu, pi)

        # Current log-likelihood
        ll = family.log_likelihood(y, mu, pi)

        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: log-lik = {ll:.4f}")

        # Check convergence
        if iteration > 0:
            rel_change = abs(ll - prev_ll) / (abs(prev_ll) + 1e-10)
            if rel_change < tol:
                converged = True
                break
        prev_ll = ll

        # M-step for count model: weighted Poisson GLM
        # Weight = 1 - z (contribution from count process)
        weights_count = 1 - z
        weights_count = np.maximum(weights_count, 1e-10)

        beta = _fit_weighted_poisson(X_count, y, weights_count, beta)
        mu = np.exp(np.clip(X_count @ beta, -20, 20))

        # M-step for inflation model: weighted logistic regression
        # Response = z, weight = 1 (or could use variance weights)
        gamma = _fit_weighted_logistic(X_inflate, z, gamma)

    # Final predictions
    eta_inflate = X_inflate @ gamma
    pi = 1 / (1 + np.exp(-eta_inflate))
    mu = np.exp(np.clip(X_count @ beta, -20, 20))
    expected_count = (1 - pi) * mu
    ll = family.log_likelihood(y, mu, pi)

    # Information criteria
    k = p_count + p_inflate  # Number of parameters
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)

    return ZIPResult(
        coef_count_=beta,
        coef_inflate_=gamma,
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


def _fit_weighted_poisson(
    X: NDArray, y: NDArray, weights: NDArray, beta_init: NDArray, max_iter: int = 25
) -> NDArray:
    """Fit weighted Poisson GLM via IRLS.

    Parameters
    ----------
    X : ndarray (n, p)
        Design matrix
    y : ndarray (n,)
        Response
    weights : ndarray (n,)
        Observation weights
    beta_init : ndarray (p,)
        Initial coefficients

    Returns
    -------
    beta : ndarray (p,)
        Fitted coefficients
    """
    beta = beta_init.copy()
    n, p = X.shape

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.exp(eta)

        # Working weights: W = weights * mu
        W = weights * mu
        W = np.maximum(W, 1e-10)

        # Working response: z = eta + (y - mu) / mu
        z = eta + (y - mu) / np.maximum(mu, 1e-10)

        # Weighted least squares
        XtW = X.T * W
        XtWX = XtW @ X
        XtWz = XtW @ z

        # Ridge for stability
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
    """Fit logistic regression via IRLS.

    Parameters
    ----------
    X : ndarray (n, q)
        Design matrix
    y : ndarray (n,)
        Response (probabilities in [0, 1])
    gamma_init : ndarray (q,)
        Initial coefficients

    Returns
    -------
    gamma : ndarray (q,)
        Fitted coefficients
    """
    gamma = gamma_init.copy()
    n, q = X.shape

    for _ in range(max_iter):
        eta = X @ gamma
        pi = 1 / (1 + np.exp(-eta))
        pi = np.clip(pi, 1e-10, 1 - 1e-10)

        # Working weights: W = pi * (1 - pi)
        W = pi * (1 - pi)
        W = np.maximum(W, 1e-10)

        # Working response: z = eta + (y - pi) / (pi * (1 - pi))
        z = eta + (y - pi) / W

        # Weighted least squares
        XtW = X.T * W
        XtWX = XtW @ X
        XtWz = XtW @ z

        # Ridge for stability
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
class ZIPResult:
    """Results from Zero-Inflated Poisson model.

    Attributes
    ----------
    coef_count_ : ndarray
        Coefficients for count model (log link)
    coef_inflate_ : ndarray
        Coefficients for inflation model (logit link)
    mu_ : ndarray
        Fitted Poisson mean (λ) at training points
    pi_ : ndarray
        Fitted zero-inflation probability at training points
    expected_count_ : ndarray
        Marginal expected count: (1-π)λ
    converged_ : bool
        Whether EM algorithm converged
    n_iter_ : int
        Number of EM iterations
    log_likelihood_ : float
        Log-likelihood at convergence
    aic_ : float
        Akaike Information Criterion
    bic_ : float
        Bayesian Information Criterion
    """

    coef_count_: NDArray
    coef_inflate_: NDArray
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
        """Predict from fitted ZIP model.

        Parameters
        ----------
        X_count : ndarray, optional
            New count design matrix. If None, uses training data.
        X_inflate : ndarray, optional
            New inflation design matrix. If None, uses training data.
        type : str
            Type of prediction:
            - 'response': Marginal mean E[Y] = (1-π)λ
            - 'count': Conditional mean λ (given not structural zero)
            - 'prob_zero': P(Y = 0)
            - 'prob_inflate': π (probability of structural zero)

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
            return pi + (1 - pi) * np.exp(-mu)
        elif type == "prob_inflate":
            return pi
        else:
            raise ValueError(
                f"Unknown type: {type}. "
                "Use 'response', 'count', 'prob_zero', or 'prob_inflate'"
            )

    def summary(self) -> dict:
        """Return model summary.

        Returns
        -------
        dict
            Summary statistics and parameter estimates
        """
        n = len(self.y_)
        k = len(self.coef_count_) + len(self.coef_inflate_)

        return {
            "n_obs": n,
            "n_zeros": int(np.sum(self.y_ == 0)),
            "n_params": k,
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
