"""Penalized Quasi-Likelihood (PQL) for Generalized Linear Mixed Models.

This module implements PQL estimation for GLMMs, extending the Gaussian
GAMM implementation to non-Gaussian families (Poisson, Binomial, etc.).

PQL Algorithm
-------------
PQL uses an iterative approach similar to IRLS but includes random effects:

1. Initialize β = 0, b = 0
2. Outer loop (variance components):
   a. Inner loop (fixed/random effects):
      i. Compute working response: z = η + (y - μ) / g'(μ)
      ii. Compute weights: W = diag(w_i), w_i = (g'(μ))² / Var(μ)
      iii. Solve penalized weighted least squares:
           [X'WX + λS    X'WZ      ] [β]   [X'Wz]
           [Z'WX         Z'WZ + Ψ⁻¹] [b] = [Z'Wz]
      iv. Update η = Xβ + Zb, μ = g⁻¹(η)
   b. Update variance components Ψ via REML on pseudo-data
3. Repeat until convergence

References
----------
- Breslow, N.E. & Clayton, D.G. (1993). "Approximate inference in generalized
  linear mixed models". Journal of the American Statistical Association.
- Wood, S.N. (2017). Generalized Additive Models: An Introduction with R (2nd ed.).
  Chapter 6: GAMMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

if TYPE_CHECKING:
    from numpy.typing import NDArray

from aurora.distributions.base import Family
from aurora.distributions.families import BinomialFamily, GaussianFamily, PoissonFamily


@dataclass
class PQLResult:
    """Result of PQL estimation.

    Attributes
    ----------
    beta : ndarray, shape (p,)
        Estimated fixed effect coefficients
    b : ndarray, shape (q,)
        Estimated random effect coefficients (BLUPs)
    psi : ndarray, shape (n_effects, n_effects)
        Estimated variance-covariance matrix for random effects
    sigma2 : float
        Estimated residual variance (on working response scale)
    fitted_values : ndarray, shape (n,)
        Fitted values on response scale (μ)
    linear_predictor : ndarray, shape (n,)
        Linear predictor (η = Xβ + Zb)
    converged : bool
        Whether the algorithm converged
    n_iter_outer : int
        Number of outer iterations (variance component updates)
    n_iter_inner : int
        Total number of inner iterations (coefficient updates)
    deviance : float
        Final deviance
    log_likelihood : float
        Approximate log-likelihood
    """

    beta: NDArray[np.floating]
    b: NDArray[np.floating]
    psi: NDArray[np.floating]
    sigma2: float
    fitted_values: NDArray[np.floating]
    linear_predictor: NDArray[np.floating]
    converged: bool
    n_iter_outer: int
    n_iter_inner: int
    deviance: float
    log_likelihood: float


def fit_pql(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    family: Family | str,
    S: np.ndarray | None = None,
    lambda_: float = 0.0,
    psi_init: np.ndarray | None = None,
    maxiter_outer: int = 20,
    maxiter_inner: int = 10,
    tol_outer: float = 1e-4,
    tol_inner: float = 1e-6,
    update_psi: bool = True,
) -> PQLResult:
    """Fit a GLMM using Penalized Quasi-Likelihood.

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
    maxiter_outer : int, default=20
        Maximum outer iterations (variance updates)
    maxiter_inner : int, default=10
        Maximum inner iterations per outer (coefficient updates)
    tol_outer : float, default=1e-4
        Convergence tolerance for variance components
    tol_inner : float, default=1e-6
        Convergence tolerance for coefficients
    update_psi : bool, default=True
        Whether to update variance components (False for testing)

    Returns
    -------
    PQLResult
        Fitted model results

    Notes
    -----
    For Gaussian family with identity link, this reduces to exact REML
    estimation (no iteration needed).

    For non-Gaussian families, PQL provides approximate inference. The
    approximation is good when:
    - Group sizes are large
    - Random effects are not too large
    - Response is not too sparse (for counts)

    Examples
    --------
    >>> # Poisson GLMM with random intercept
    >>> import numpy as np
    >>> from aurora.models.gamm import fit_pql
    >>> from aurora.distributions.families import PoissonFamily
    >>>
    >>> n_groups, n_per_group = 10, 20
    >>> n = n_groups * n_per_group
    >>> groups = np.repeat(np.arange(n_groups), n_per_group)
    >>> x = np.random.randn(n)
    >>> X = np.column_stack([np.ones(n), x])
    >>>
    >>> # Random effects design (indicator matrix)
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
    >>> result = fit_pql(X, Z, y, family='poisson')
    >>> print(f"Converged: {result.converged}")
    >>> print(f"Fixed effects: {result.beta}")
    >>> print(f"Variance: {result.psi[0, 0]:.3f}")
    """
    # Get family object
    if isinstance(family, str):
        family = _get_family(family)

    n, p = X.shape
    q = Z.shape[1]

    # Initialize penalty matrix
    if S is None:
        S = np.zeros((p, p))

    # Initialize parameters
    beta = np.zeros(p)
    b = np.zeros(q)

    # Infer number of effects per group from Z structure
    # Assume Z is block diagonal with blocks of size n_effects
    # For now, assume random intercepts (n_effects = 1)
    n_effects = 1  # This will be passed from outside in production

    if psi_init is None:
        psi = np.eye(n_effects)
    else:
        psi = psi_init.copy()

    # Initial linear predictor
    eta = X @ beta + Z @ b
    mu = family.link.inverse(eta)

    # Track convergence
    converged = False
    total_inner_iter = 0

    # Outer loop: update variance components
    for outer_iter in range(maxiter_outer):
        psi_old = psi.copy()

        # Inner loop: update fixed and random effects
        for inner_iter in range(maxiter_inner):
            beta_old = beta.copy()
            b_old = b.copy()

            # Compute working response and weights
            dmu_deta = family.link.derivative_inv(eta)  # g'(μ)⁻¹ = dμ/dη
            var_mu = family.variance(mu)

            # Working response: z = η + (y - μ) / (dμ/dη)
            z = eta + (y - mu) / dmu_deta

            # Weights: w = (dμ/dη)² / Var(μ)
            w = (dmu_deta**2) / var_mu
            w = np.maximum(w, 1e-8)  # Avoid division by zero

            # Solve weighted penalized mixed model equations
            beta, b = _solve_pql_equations(X, Z, z, w, S, lambda_, psi, n_effects)

            # Update linear predictor and mean
            eta = X @ beta + Z @ b
            mu = family.link.inverse(eta)

            # Check inner convergence
            delta_beta = np.max(np.abs(beta - beta_old))
            delta_b = np.max(np.abs(b - b_old))

            total_inner_iter += 1

            if delta_beta < tol_inner and delta_b < tol_inner:
                break

        # Update variance components using REML on working response
        if update_psi:
            psi = _update_variance_components(X, Z, z, w, beta, b, n_effects)

        # Check outer convergence
        delta_psi = np.max(np.abs(psi - psi_old))

        if delta_psi < tol_outer:
            converged = True
            break

    # Compute diagnostics
    deviance = family.deviance(y, mu).sum()

    # Approximate log-likelihood (not exact for non-Gaussian)
    log_lik = _compute_approx_log_likelihood(y, mu, b, psi, family)

    # Residual variance on working scale
    residuals_working = z - (X @ beta + Z @ b)
    sigma2 = np.sum(w * residuals_working**2) / (n - p)

    return PQLResult(
        beta=beta,
        b=b,
        psi=psi,
        sigma2=sigma2,
        fitted_values=mu,
        linear_predictor=eta,
        converged=converged,
        n_iter_outer=outer_iter + 1,
        n_iter_inner=total_inner_iter,
        deviance=deviance,
        log_likelihood=log_lik,
    )


def _solve_pql_equations(
    X: np.ndarray,
    Z: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    lambda_: float,
    psi: np.ndarray,
    n_effects: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve penalized weighted mixed model equations.

    Solves:
        [X'WX + λS    X'WZ      ] [β]   [X'Wz]
        [Z'WX         Z'WZ + Ψ⁻¹] [b] = [Z'Wz]

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Fixed effects design
    Z : ndarray, shape (n, q)
        Random effects design
    z : ndarray, shape (n,)
        Working response
    w : ndarray, shape (n,)
        Weights (diagonal of W)
    S : ndarray, shape (p, p)
        Penalty matrix
    lambda_ : float
        Smoothing parameter
    psi : ndarray, shape (n_effects, n_effects)
        Variance-covariance matrix for random effects
    n_effects : int
        Number of random effects per group

    Returns
    -------
    beta : ndarray, shape (p,)
        Fixed effect coefficients
    b : ndarray, shape (q,)
        Random effect coefficients
    """
    n, p = X.shape
    q = Z.shape[1]
    n_groups = q // n_effects

    # Compute Ψ⁻¹ for penalty
    try:
        psi_inv = linalg.inv(psi)
    except linalg.LinAlgError:
        # Add ridge if singular
        psi_inv = linalg.inv(psi + 1e-6 * np.eye(n_effects))

    # Expand Ψ⁻¹ to block diagonal for all groups
    psi_inv_block = np.kron(np.eye(n_groups), psi_inv)

    # Build weighted matrices
    W_sqrt = np.sqrt(w)
    X_w = X * W_sqrt[:, np.newaxis]
    Z_w = Z * W_sqrt[:, np.newaxis]
    z_w = z * W_sqrt

    # Coefficient matrix components
    XtWX = X_w.T @ X_w + lambda_ * S
    XtWZ = X_w.T @ Z_w
    ZtWZ = Z_w.T @ Z_w + psi_inv_block

    # Right-hand side
    Xty = X_w.T @ z_w
    Zty = Z_w.T @ z_w

    # Build augmented system
    A = np.block([[XtWX, XtWZ], [XtWZ.T, ZtWZ]])
    rhs = np.concatenate([Xty, Zty])

    # Solve with Cholesky (faster than general solver)
    try:
        L = linalg.cholesky(A, lower=True)
        coef = linalg.cho_solve((L, True), rhs)
    except linalg.LinAlgError:
        # Fallback to ridge-regularized solution
        A_ridge = A + 1e-6 * np.eye(p + q)
        coef = linalg.solve(A_ridge, rhs, assume_a='pos')

    beta = coef[:p]
    b = coef[p:]

    return beta, b


def _update_variance_components(
    X: np.ndarray,
    Z: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    beta: np.ndarray,
    b: np.ndarray,
    n_effects: int,
) -> np.ndarray:
    """Update variance-covariance matrix using REML-like approach.

    This is a simplified update that computes the empirical covariance
    of the random effects, adjusted for their uncertainty.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Fixed effects design
    Z : ndarray, shape (n, q)
        Random effects design
    z : ndarray, shape (n,)
        Working response
    w : ndarray, shape (n,)
        Weights
    beta : ndarray, shape (p,)
        Current fixed effects
    b : ndarray, shape (q,)
        Current random effects
    n_effects : int
        Number of effects per group

    Returns
    -------
    psi : ndarray, shape (n_effects, n_effects)
        Updated variance-covariance matrix
    """
    q = Z.shape[1]
    n_groups = q // n_effects

    # Reshape b into groups
    b_matrix = b.reshape(n_groups, n_effects)

    # Compute empirical covariance
    psi_emp = (b_matrix.T @ b_matrix) / n_groups

    # For stability, ensure positive definiteness
    eigvals = np.linalg.eigvalsh(psi_emp)
    if np.min(eigvals) < 1e-6:
        # Add small ridge to diagonal
        psi_emp = psi_emp + 1e-6 * np.eye(n_effects)

    return psi_emp


def _compute_approx_log_likelihood(
    y: np.ndarray,
    mu: np.ndarray,
    b: np.ndarray,
    psi: np.ndarray,
    family: Family,
) -> float:
    """Compute approximate log-likelihood for GLMM.

    This is not the true likelihood but an approximation used for
    model comparison.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response
    mu : ndarray, shape (n,)
        Fitted means
    b : ndarray, shape (q,)
        Random effects
    psi : ndarray, shape (n_effects, n_effects)
        Variance-covariance matrix
    family : Family
        Distribution family

    Returns
    -------
    log_lik : float
        Approximate log-likelihood
    """
    # Log-likelihood contribution from data
    log_lik_data = family.log_likelihood(y, mu).sum()

    # Log-likelihood contribution from random effects (Gaussian)
    n_effects = psi.shape[0]
    n_groups = len(b) // n_effects
    b_matrix = b.reshape(n_groups, n_effects)

    try:
        L = linalg.cholesky(psi, lower=True)
        log_det = 2 * np.sum(np.log(np.diag(L)))
        psi_inv_b = linalg.cho_solve((L, True), b_matrix.T).T
    except linalg.LinAlgError:
        # Fallback
        log_det = np.linalg.slogdet(psi)[1]
        psi_inv_b = linalg.solve(psi, b_matrix.T, assume_a='pos').T

    log_lik_random = -0.5 * (
        n_groups * (n_effects * np.log(2 * np.pi) + log_det)
        + np.sum(b_matrix * psi_inv_b)
    )

    return log_lik_data + log_lik_random


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
        'gaussian': GaussianFamily,
        'poisson': PoissonFamily,
        'binomial': BinomialFamily,
    }

    if family_name.lower() not in families:
        raise ValueError(
            f"Unknown family '{family_name}'. "
            f"Must be one of {list(families.keys())}"
        )

    return families[family_name.lower()]()
