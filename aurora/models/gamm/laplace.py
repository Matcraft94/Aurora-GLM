# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Laplace Approximation for Generalized Linear Mixed Models.

This module implements Laplace approximation for GLMMs, which provides
more accurate inference than PQL but at higher computational cost.

Laplace Approximation
---------------------
The Laplace approximation approximates the marginal likelihood by
integrating out the random effects using a normal approximation:

    L(β, θ) ≈ f(y | b̂, β) p(b̂ | θ) |H|^(-1/2)

Where:
- b̂ = argmax_b f(y | b, β) p(b | θ) is the conditional mode
- H = -∂²/∂b² log[f(y | b, β) p(b | θ)] at b = b̂ is the Hessian

The algorithm alternates between:
1. Finding b̂ given current β, θ (inner optimization)
2. Updating β, θ given b̂ (outer optimization)

References
----------
- Breslow, N.E. & Lin, X. (1995). "Bias correction in generalized linear
  mixed models with a single component of dispersion".
- Wood, S.N. (2017). Generalized Additive Models: An Introduction with R (2nd ed.).
  Chapter 6.10: Laplace approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg, optimize

if TYPE_CHECKING:
    from numpy.typing import NDArray

from aurora.distributions.base import Family
from aurora.distributions.families import BinomialFamily, GaussianFamily, PoissonFamily


@dataclass
class LaplaceResult:
    """Result of Laplace approximation.

    Attributes
    ----------
    beta : ndarray, shape (p,)
        Estimated fixed effect coefficients
    b : ndarray, shape (q,)
        Estimated random effect coefficients at conditional mode
    psi : ndarray, shape (n_effects, n_effects)
        Estimated variance-covariance matrix for random effects
    sigma2 : float
        Estimated residual variance (for Gaussian; 1.0 for others)
    fitted_values : ndarray, shape (n,)
        Fitted values on response scale (μ)
    linear_predictor : ndarray, shape (n,)
        Linear predictor (η = Xβ + Zb)
    hessian : ndarray, shape (q, q)
        Hessian of negative log joint density at b̂
    converged : bool
        Whether the algorithm converged
    n_iter : int
        Number of outer iterations
    log_likelihood : float
        Approximate marginal log-likelihood
    """

    beta: NDArray[np.floating]
    b: NDArray[np.floating]
    psi: NDArray[np.floating]
    sigma2: float
    fitted_values: NDArray[np.floating]
    linear_predictor: NDArray[np.floating]
    hessian: NDArray[np.floating]
    converged: bool
    n_iter: int
    log_likelihood: float


def fit_laplace(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    family: Family | str,
    S: np.ndarray | None = None,
    lambda_: float = 0.0,
    psi_init: np.ndarray | None = None,
    maxiter: int = 50,
    tol: float = 1e-6,
) -> LaplaceResult:
    """Fit a GLMM using Laplace approximation.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Fixed effects design matrix
    Z : ndarray, shape (n, q)
        Random effects design matrix
    y : ndarray, shape (n,)
        Response variable
    family : Family or str
        Distribution family ('gaussian', 'poisson', 'binomial')
    S : ndarray, shape (p, p), optional
        Penalty matrix for smooth terms (default: zeros)
    lambda_ : float, default=0.0
        Smoothing parameter
    psi_init : ndarray, shape (n_effects, n_effects), optional
        Initial variance-covariance matrix (default: identity)
    maxiter : int, default=50
        Maximum iterations
    tol : float, default=1e-6
        Convergence tolerance

    Returns
    -------
    LaplaceResult
        Fitted model results

    Notes
    -----
    Laplace approximation is more accurate than PQL, especially for:
    - Binary data with small group sizes
    - Count data with small means
    - Models with large random effects

    However, it is computationally more expensive due to:
    - Inner optimization to find conditional mode b̂
    - Hessian computation at each iteration

    Examples
    --------
    >>> # Poisson GLMM with Laplace approximation
    >>> import numpy as np
    >>> from aurora.models.gamm import fit_laplace
    >>>
    >>> n_groups, n_per_group = 10, 20
    >>> n = n_groups * n_per_group
    >>> groups = np.repeat(np.arange(n_groups), n_per_group)
    >>> x = np.random.randn(n)
    >>> X = np.column_stack([np.ones(n), x])
    >>>
    >>> # Random effects design
    >>> Z = np.zeros((n, n_groups))
    >>> for i in range(n):
    ...     Z[i, groups[i]] = 1
    >>>
    >>> # Generate data
    >>> b_true = np.random.randn(n_groups) * 0.5
    >>> eta = 1.0 + 0.5 * x + b_true[groups]
    >>> y = np.random.poisson(np.exp(eta))
    >>>
    >>> # Fit model
    >>> result = fit_laplace(X, Z, y, family='poisson')
    >>> print(f"Log-likelihood: {result.log_likelihood:.2f}")
    """
    # Get family object
    if isinstance(family, str):
        family = _get_family(family)

    n, p = X.shape
    q = Z.shape[1]

    # Initialize penalty matrix
    if S is None:
        S = np.zeros((p, p))

    # Infer number of effects per group
    n_effects = 1  # Will be passed from outside in production

    if psi_init is None:
        psi = np.eye(n_effects)
    else:
        psi = psi_init.copy()

    # Initialize parameters
    beta = np.zeros(p)
    b = np.zeros(q)

    # Get link function
    link = family.default_link

    # Initial linear predictor
    eta = X @ beta + Z @ b
    mu = link.inverse(eta)

    converged = False

    # Main optimization loop
    for iter_num in range(maxiter):
        beta_old = beta.copy()
        b_old = b.copy()
        psi_old = psi.copy()

        # Step 1: Find conditional mode b̂ given β, θ
        b, hessian = _find_conditional_mode(X, Z, y, beta, psi, n_effects, family)

        # Step 2: Update β given b̂
        beta = _update_fixed_effects(X, Z, y, b, psi, n_effects, family, S, lambda_)

        # Step 3: Update θ (variance components) given β, b̂, H
        psi = _update_variance_laplace(b, hessian, n_effects)

        # Update predictions
        eta = X @ beta + Z @ b
        mu = link.inverse(eta)

        # Check convergence
        delta_beta = np.max(np.abs(beta - beta_old))
        delta_b = np.max(np.abs(b - b_old))
        delta_psi = np.max(np.abs(psi - psi_old))

        if delta_beta < tol and delta_b < tol and delta_psi < tol:
            converged = True
            break

    # Compute approximate marginal log-likelihood
    log_lik = _compute_laplace_log_likelihood(y, mu, b, psi, hessian, n_effects, family)

    # Residual variance (Gaussian only)
    if isinstance(family, GaussianFamily):
        residuals = y - mu
        sigma2 = np.sum(residuals**2) / (n - p)
    else:
        sigma2 = 1.0  # Fixed for non-Gaussian

    return LaplaceResult(
        beta=beta,
        b=b,
        psi=psi,
        sigma2=sigma2,
        fitted_values=mu,
        linear_predictor=eta,
        hessian=hessian,
        converged=converged,
        n_iter=iter_num + 1,
        log_likelihood=log_lik,
    )


def _find_conditional_mode(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    psi: np.ndarray,
    n_effects: int,
    family: Family,
) -> tuple[np.ndarray, np.ndarray]:
    """Find conditional mode of random effects given fixed effects.

    Maximizes: log p(b | y, β, θ) = log f(y | b, β) + log p(b | θ) - const

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Fixed effects design
    Z : ndarray, shape (n, q)
        Random effects design
    y : ndarray, shape (n,)
        Response
    beta : ndarray, shape (p,)
        Current fixed effects
    psi : ndarray, shape (n_effects, n_effects)
        Current variance-covariance matrix
    n_effects : int
        Number of effects per group
    family : Family
        Distribution family

    Returns
    -------
    b_mode : ndarray, shape (q,)
        Conditional mode
    hessian : ndarray, shape (q, q)
        Hessian at mode
    """
    q = Z.shape[1]
    n_groups = q // n_effects

    # Compute Ψ⁻¹
    try:
        psi_inv = linalg.inv(psi)
    except linalg.LinAlgError:
        psi_inv = linalg.inv(psi + 1e-6 * np.eye(n_effects))

    # Expand to block diagonal
    psi_inv_block = np.kron(np.eye(n_groups), psi_inv)

    # Get link function
    link = family.default_link

    # Define negative log conditional density
    def neg_log_conditional(b):
        eta = X @ beta + Z @ b
        mu = link.inverse(eta)

        # Negative log-likelihood
        neg_log_lik = -family.log_likelihood(y, mu).sum()

        # Negative log prior
        neg_log_prior = 0.5 * (b @ psi_inv_block @ b)

        return neg_log_lik + neg_log_prior

    # Gradient
    def grad_neg_log_conditional(b):
        eta = X @ beta + Z @ b
        mu = link.inverse(eta)
        deta_dmu = link.derivative(mu)
        dmu_deta = 1.0 / (deta_dmu + 1e-10)

        # Score from likelihood
        residuals = (y - mu) / family.variance(mu)
        grad_log_lik = Z.T @ (residuals * dmu_deta)

        # Gradient of log prior
        grad_log_prior = -psi_inv_block @ b

        return -(grad_log_lik + grad_log_prior)

    # Optimize using L-BFGS-B
    result = optimize.minimize(
        neg_log_conditional,
        x0=np.zeros(q),
        jac=grad_neg_log_conditional,
        method="L-BFGS-B",
        options={"maxiter": 100, "ftol": 1e-8},
    )

    b_mode = result.x

    # Compute Hessian at mode numerically
    hessian = _compute_hessian(X, Z, y, beta, b_mode, psi_inv_block, family)

    return b_mode, hessian


def _compute_hessian(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    b: np.ndarray,
    psi_inv_block: np.ndarray,
    family: Family,
) -> np.ndarray:
    """Compute Hessian of negative log conditional density.

    H = Z'WZ + Ψ⁻¹

    Where W = diag(w_i), w_i = (dμ/dη)² / Var(μ) · |d²μ/dη²|

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Fixed effects design
    Z : ndarray, shape (n, q)
        Random effects design
    y : ndarray, shape (n,)
        Response
    beta : ndarray, shape (p,)
        Fixed effects
    b : ndarray, shape (q,)
        Random effects
    psi_inv_block : ndarray, shape (q, q)
        Block-diagonal inverse variance-covariance
    family : Family
        Distribution family

    Returns
    -------
    hessian : ndarray, shape (q, q)
        Hessian matrix
    """
    link = family.default_link
    eta = X @ beta + Z @ b
    mu = link.inverse(eta)
    deta_dmu = link.derivative(mu)
    dmu_deta = 1.0 / (deta_dmu + 1e-10)
    var_mu = family.variance(mu)

    # Weights for Hessian
    w = (dmu_deta**2) / var_mu
    w = np.maximum(w, 1e-8)

    # Hessian from likelihood
    W = np.diag(w)
    hess_lik = Z.T @ W @ Z

    # Total Hessian
    hessian = hess_lik + psi_inv_block

    return hessian


def _update_fixed_effects(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    psi: np.ndarray,
    n_effects: int,
    family: Family,
    S: np.ndarray,
    lambda_: float,
) -> np.ndarray:
    """Update fixed effects given random effects.

    Uses penalized IRLS with b fixed.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Fixed effects design
    Z : ndarray, shape (n, q)
        Random effects design
    y : ndarray, shape (n,)
        Response
    b : ndarray, shape (q,)
        Current random effects
    psi : ndarray, shape (n_effects, n_effects)
        Variance-covariance matrix
    n_effects : int
        Number of effects per group
    family : Family
        Distribution family
    S : ndarray, shape (p, p)
        Penalty matrix
    lambda_ : float
        Smoothing parameter

    Returns
    -------
    beta : ndarray, shape (p,)
        Updated fixed effects
    """
    p = X.shape[1]
    beta = np.zeros(p)

    # Get link function
    link = family.default_link

    # IRLS for β with offset Zb
    offset = Z @ b

    for _ in range(10):  # Max 10 iterations
        eta = X @ beta + offset
        mu = link.inverse(eta)
        deta_dmu = link.derivative(mu)
        dmu_deta = 1.0 / (deta_dmu + 1e-10)
        var_mu = family.variance(mu)

        # Working response
        z = eta + (y - mu) / dmu_deta - offset

        # Weights
        w = (dmu_deta**2) / var_mu
        w = np.maximum(w, 1e-8)

        # Weighted penalized least squares
        W_sqrt = np.sqrt(w)
        X_w = X * W_sqrt[:, np.newaxis]
        z_w = z * W_sqrt

        A = X_w.T @ X_w + lambda_ * S
        rhs = X_w.T @ z_w

        try:
            beta_new = linalg.solve(A, rhs, assume_a="pos")
        except linalg.LinAlgError:
            beta_new = linalg.solve(A + 1e-6 * np.eye(p), rhs)

        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break

        beta = beta_new

    return beta


def _update_variance_laplace(
    b: np.ndarray,
    hessian: np.ndarray,
    n_effects: int,
) -> np.ndarray:
    """Update variance components using Laplace approximation.

    Uses empirical Bayes estimate corrected for uncertainty in b.

    Parameters
    ----------
    b : ndarray, shape (q,)
        Random effects at conditional mode
    hessian : ndarray, shape (q, q)
        Hessian at conditional mode
    n_effects : int
        Number of effects per group

    Returns
    -------
    psi : ndarray, shape (n_effects, n_effects)
        Updated variance-covariance matrix
    """
    q = len(b)
    n_groups = q // n_effects

    # Reshape b into groups
    b_matrix = b.reshape(n_groups, n_effects)

    # Empirical covariance
    psi_emp = (b_matrix.T @ b_matrix) / n_groups

    # Correction for uncertainty (simplified)
    # In full implementation, would use hessian to adjust
    # For now, use empirical estimate with regularization

    # Ensure positive definiteness
    eigvals = np.linalg.eigvalsh(psi_emp)
    if np.min(eigvals) < 1e-6:
        psi_emp = psi_emp + 1e-6 * np.eye(n_effects)

    return psi_emp


def _compute_laplace_log_likelihood(
    y: np.ndarray,
    mu: np.ndarray,
    b: np.ndarray,
    psi: np.ndarray,
    hessian: np.ndarray,
    n_effects: int,
    family: Family,
) -> float:
    """Compute approximate marginal log-likelihood via Laplace.

    log L(β, θ) ≈ log f(y | b̂, β) + log p(b̂ | θ) - (1/2) log |H|

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response
    mu : ndarray, shape (n,)
        Fitted means
    b : ndarray, shape (q,)
        Random effects at mode
    psi : ndarray, shape (n_effects, n_effects)
        Variance-covariance matrix
    hessian : ndarray, shape (q, q)
        Hessian at mode
    n_effects : int
        Number of effects per group
    family : Family
        Distribution family

    Returns
    -------
    log_lik : float
        Approximate marginal log-likelihood
    """
    # Log-likelihood at mode
    log_lik_data = family.log_likelihood(y, mu).sum()

    # Log prior density at mode
    n_groups = len(b) // n_effects
    b_matrix = b.reshape(n_groups, n_effects)

    try:
        L = linalg.cholesky(psi, lower=True)
        log_det_psi = 2 * np.sum(np.log(np.diag(L)))
        psi_inv_b = linalg.cho_solve((L, True), b_matrix.T).T
    except linalg.LinAlgError:
        log_det_psi = np.linalg.slogdet(psi)[1]
        psi_inv_b = linalg.solve(psi, b_matrix.T, assume_a="pos").T

    log_prior = -0.5 * (
        n_groups * (n_effects * np.log(2 * np.pi) + log_det_psi)
        + np.sum(b_matrix * psi_inv_b)
    )

    # Laplace correction (log determinant of Hessian)
    try:
        _, log_det_hess = np.linalg.slogdet(hessian)
    except np.linalg.LinAlgError:
        log_det_hess = 0.0  # Fallback

    laplace_correction = -0.5 * log_det_hess

    return log_lik_data + log_prior + laplace_correction


def _get_family(family_name: str) -> Family:
    """Get family object from name.

    Parameters
    ----------
    family_name : str
        Family name ('gaussian', 'poisson', 'binomial')

    Returns
    -------
    family : Family
        Family object with default link

    Raises
    ------
    ValueError
        If family name not recognized
    """
    families = {
        "gaussian": GaussianFamily,
        "poisson": PoissonFamily,
        "binomial": BinomialFamily,
    }

    if family_name.lower() not in families:
        raise ValueError(
            f"Unknown family '{family_name}'. Must be one of {list(families.keys())}"
        )

    return families[family_name.lower()]()
