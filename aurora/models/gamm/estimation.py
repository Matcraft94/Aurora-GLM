"""REML estimation for variance components in mixed models.

This module implements Restricted Maximum Likelihood (REML) estimation
for variance components in linear mixed models, following the approach
used in lme4 and nlme.

Mathematical Background
-----------------------
For a linear mixed model:
    y = Xβ + Zb + ε
    b ~ N(0, Ψ)
    ε ~ N(0, σ²I)

The REML log-likelihood is:
    l_REML = -0.5 * [log|V| + log|X'V⁻¹X| + y'Py]

where:
    V = ZΨZ' + σ²I
    P = V⁻¹ - V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹

REML provides unbiased estimates of variance components by accounting
for the loss of degrees of freedom from estimating fixed effects.

References
----------
- Bates, D. et al. (2015). Fitting Linear Mixed-Effects Models Using lme4.
  Journal of Statistical Software, 67(1).
- Pinheiro, J.C. & Bates, D.M. (2000). Mixed-Effects Models in S and S-PLUS.
  Springer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg, optimize

from aurora.models.gamm.covariance import CovarianceStructure, get_covariance_structure

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class REMLResult:
    """Result from REML variance component estimation.

    Attributes
    ----------
    psi : ndarray, shape (q, q)
        Estimated random effects covariance matrix Ψ.
    sigma2 : float
        Estimated residual variance σ².
    log_likelihood : float
        REML log-likelihood at optimum.
    theta : ndarray
        Optimized covariance parameters (internal parameterization).
    n_iterations : int
        Number of optimization iterations.
    converged : bool
        Whether optimization converged successfully.
    V : ndarray, shape (n, n)
        Marginal covariance matrix V = ZΨZ' + σ²I.
    P : ndarray, shape (n, n)
        REML projection matrix P = V⁻¹ - V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹.
    """

    psi: NDArray[np.floating]
    sigma2: float
    log_likelihood: float
    theta: NDArray[np.floating]
    n_iterations: int
    converged: bool
    V: NDArray[np.floating] | None = None
    P: NDArray[np.floating] | None = None


def compute_V_matrix(
    Z: np.ndarray,
    psi: np.ndarray,
    sigma2: float,
    n_effects: int | None = None,
) -> np.ndarray:
    """Compute marginal covariance matrix V = ZΨZ' + σ²I.

    Parameters
    ----------
    Z : ndarray, shape (n, q)
        Random effects design matrix.
    psi : ndarray, shape (q_effects, q_effects)
        Random effects covariance matrix (per group).
    sigma2 : float
        Residual variance.
    n_effects : int or None, optional
        Number of random effects per group. If provided and psi is smaller,
        will expand psi to block-diagonal structure.

    Returns
    -------
    V : ndarray, shape (n, n)
        Marginal covariance matrix.

    Notes
    -----
    For random intercept model, psi is (1,1) but Z may have q columns for q groups.
    This function handles the expansion from per-group covariance to full structure.
    """
    n = Z.shape[0]
    q = Z.shape[1]

    # If psi is the per-group covariance and smaller than Z's second dimension,
    # expand it to block-diagonal for all groups
    if n_effects is not None and psi.shape[0] == n_effects < q:
        # Number of groups
        n_groups = q // n_effects
        # Create block-diagonal Ψ_full
        psi_full = linalg.block_diag(*([psi] * n_groups))
    else:
        psi_full = psi

    V = Z @ psi_full @ Z.T + sigma2 * np.eye(n)
    return V


def compute_P_matrix(
    V: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Compute REML projection matrix P = V⁻¹ - V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹.

    Parameters
    ----------
    V : ndarray, shape (n, n)
        Marginal covariance matrix.
    X : ndarray, shape (n, p)
        Fixed effects design matrix.

    Returns
    -------
    P : ndarray, shape (n, n)
        REML projection matrix.
    """
    # Use Cholesky decomposition for numerical stability
    L = linalg.cholesky(V, lower=True)
    V_inv = linalg.cho_solve((L, True), np.eye(V.shape[0]))

    # Compute X'V⁻¹X
    XtV_inv = X.T @ V_inv
    XtV_invX = XtV_inv @ X

    # Compute (X'V⁻¹X)⁻¹
    L_XtV_invX = linalg.cholesky(XtV_invX, lower=True)
    XtV_invX_inv = linalg.cho_solve((L_XtV_invX, True), np.eye(XtV_invX.shape[0]))

    # P = V⁻¹ - V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹
    P = V_inv - V_inv @ X @ XtV_invX_inv @ XtV_inv

    return P


def reml_log_likelihood(
    y: np.ndarray,
    X: np.ndarray,
    V: np.ndarray,
    P: np.ndarray,
) -> float:
    """Compute REML log-likelihood.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response vector.
    X : ndarray, shape (n, p)
        Fixed effects design matrix.
    V : ndarray, shape (n, n)
        Marginal covariance matrix.
    P : ndarray, shape (n, n)
        REML projection matrix.

    Returns
    -------
    log_lik : float
        REML log-likelihood.

    Notes
    -----
    l_REML = -0.5 * [log|V| + log|X'V⁻¹X| + y'Py]
    """
    n = len(y)
    p = X.shape[1]

    # log|V| via Cholesky
    L_V = linalg.cholesky(V, lower=True)
    log_det_V = 2 * np.sum(np.log(np.diag(L_V)))

    # log|X'V⁻¹X| via Cholesky
    V_inv = linalg.cho_solve((L_V, True), np.eye(n))
    XtV_invX = X.T @ V_inv @ X
    L_XtV_invX = linalg.cholesky(XtV_invX, lower=True)
    log_det_XtV_invX = 2 * np.sum(np.log(np.diag(L_XtV_invX)))

    # y'Py
    yPy = y @ P @ y

    # REML log-likelihood
    log_lik = -0.5 * (log_det_V + log_det_XtV_invX + yPy)

    # Add constant term
    log_lik -= 0.5 * (n - p) * np.log(2 * np.pi)

    return log_lik


def reml_objective(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    cov_structure: CovarianceStructure,
    n_effects: int,
) -> float:
    """REML objective function (negative log-likelihood).

    Parameters
    ----------
    theta : ndarray
        Covariance parameters [psi_params..., log(sigma2)].
    y : ndarray, shape (n,)
        Response vector.
    X : ndarray, shape (n, p)
        Fixed effects design matrix.
    Z : ndarray, shape (n, q)
        Random effects design matrix.
    cov_structure : CovarianceStructure
        Covariance structure for Ψ.
    n_effects : int
        Number of random effects per group.

    Returns
    -------
    neg_log_lik : float
        Negative REML log-likelihood.
    """
    # Extract parameters
    n_psi_params = cov_structure.n_parameters(n_effects)
    psi_params = theta[:n_psi_params]
    log_sigma2 = theta[n_psi_params]
    sigma2 = np.exp(log_sigma2)

    try:
        # Construct Ψ
        psi = cov_structure.construct_psi(psi_params, n_effects)

        # Compute V and P
        V = compute_V_matrix(Z, psi, sigma2, n_effects=n_effects)
        P = compute_P_matrix(V, X)

        # Compute log-likelihood
        log_lik = reml_log_likelihood(y, X, V, P)

        return -log_lik

    except (np.linalg.LinAlgError, ValueError):
        # Return large value if matrix operations fail
        return 1e10


def estimate_variance_components(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    Z_info: list[dict],
    covariance: str = "unstructured",
    method: str = "L-BFGS-B",
    maxiter: int = 1000,
    tol: float = 1e-6,
    initial_psi: np.ndarray | None = None,
    initial_sigma2: float | None = None,
    store_matrices: bool = False,
) -> REMLResult:
    """Estimate variance components via REML.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response vector.
    X : ndarray, shape (n, p)
        Fixed effects design matrix.
    Z : ndarray, shape (n, q)
        Random effects design matrix.
    Z_info : list of dict
        Metadata about Z structure (from construct_Z_matrix).
    covariance : str, default='unstructured'
        Covariance structure: 'unstructured', 'diagonal', or 'identity'.
    method : str, default='L-BFGS-B'
        Optimization method for scipy.optimize.minimize.
    maxiter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    initial_psi : ndarray or None, optional
        Initial value for Ψ. If None, uses identity.
    initial_sigma2 : float or None, optional
        Initial value for σ². If None, uses variance of residuals.
    store_matrices : bool, default=False
        Whether to store V and P matrices in result (memory intensive).

    Returns
    -------
    result : REMLResult
        REML estimation result.

    Notes
    -----
    This function currently assumes a single random effect term with one
    covariance structure. Extension to multiple terms with different
    structures will be implemented in future milestones.

    Examples
    --------
    >>> # Random intercept model
    >>> n = 100
    >>> y = np.random.randn(n)
    >>> X = np.ones((n, 1))
    >>> groups = np.repeat(np.arange(20), 5)
    >>> Z = np.zeros((n, 20))
    >>> Z[np.arange(n), groups] = 1
    >>> Z_info = [{'n_effects': 1, 'n_groups': 20}]
    >>> result = estimate_variance_components(y, X, Z, Z_info)
    """
    n, p = X.shape
    _, q = Z.shape

    # Get covariance structure
    cov_structure = get_covariance_structure(covariance)

    # For now, assume single random effect term
    if len(Z_info) != 1:
        raise NotImplementedError(
            "Multiple random effect terms not yet supported. "
            "This will be implemented in Milestone 3."
        )

    n_effects = Z_info[0]["n_effects"]

    # Initialize parameters
    if initial_psi is None:
        initial_psi = np.eye(n_effects)

    if initial_sigma2 is None:
        # Use variance of OLS residuals as initial guess
        beta_ols = linalg.lstsq(X, y)[0]
        residuals = y - X @ beta_ols
        initial_sigma2 = np.var(residuals)

    # Extract initial covariance parameters
    psi_params_init = cov_structure.extract_params(initial_psi)
    log_sigma2_init = np.log(initial_sigma2)
    theta_init = np.concatenate([psi_params_init, [log_sigma2_init]])

    # Optimize
    result = optimize.minimize(
        reml_objective,
        theta_init,
        args=(y, X, Z, cov_structure, n_effects),
        method=method,
        options={"maxiter": maxiter, "ftol": tol},
    )

    # Extract optimized parameters
    n_psi_params = cov_structure.n_parameters(n_effects)
    psi_params_opt = result.x[:n_psi_params]
    log_sigma2_opt = result.x[n_psi_params]
    sigma2_opt = np.exp(log_sigma2_opt)

    # Construct final Ψ
    psi_opt = cov_structure.construct_psi(psi_params_opt, n_effects)

    # Compute final V and P if requested
    V_opt = None
    P_opt = None
    if store_matrices:
        V_opt = compute_V_matrix(Z, psi_opt, sigma2_opt, n_effects=n_effects)
        P_opt = compute_P_matrix(V_opt, X)

    return REMLResult(
        psi=psi_opt,
        sigma2=sigma2_opt,
        log_likelihood=-result.fun,
        theta=result.x,
        n_iterations=result.nit,
        converged=result.success,
        V=V_opt,
        P=P_opt,
    )


def estimate_fixed_effects(
    y: np.ndarray,
    X: np.ndarray,
    V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate fixed effects β via generalized least squares.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response vector.
    X : ndarray, shape (n, p)
        Fixed effects design matrix.
    V : ndarray, shape (n, n)
        Marginal covariance matrix.

    Returns
    -------
    beta : ndarray, shape (p,)
        Fixed effects estimates.
    cov_beta : ndarray, shape (p, p)
        Covariance matrix of β: (X'V⁻¹X)⁻¹.
    """
    # Use Cholesky for numerical stability
    L = linalg.cholesky(V, lower=True)
    V_inv = linalg.cho_solve((L, True), np.eye(V.shape[0]))

    # β = (X'V⁻¹X)⁻¹X'V⁻¹y
    XtV_inv = X.T @ V_inv
    XtV_invX = XtV_inv @ X
    L_XtV_invX = linalg.cholesky(XtV_invX, lower=True)
    XtV_invX_inv = linalg.cho_solve((L_XtV_invX, True), np.eye(XtV_invX.shape[0]))

    beta = XtV_invX_inv @ XtV_inv @ y
    cov_beta = XtV_invX_inv

    return beta, cov_beta


def estimate_random_effects(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    beta: np.ndarray,
    psi: np.ndarray,
    sigma2: float,
    n_effects: int | None = None,
) -> np.ndarray:
    """Estimate random effects b (BLUPs) via conditional expectation.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response vector.
    X : ndarray, shape (n, p)
        Fixed effects design matrix.
    Z : ndarray, shape (n, q)
        Random effects design matrix.
    beta : ndarray, shape (p,)
        Fixed effects estimates.
    psi : ndarray, shape (q_effects, q_effects)
        Random effects covariance matrix (per group).
    sigma2 : float
        Residual variance.
    n_effects : int or None, optional
        Number of random effects per group.

    Returns
    -------
    b : ndarray, shape (q,)
        Random effects estimates (BLUPs).

    Notes
    -----
    b = Ψ_full Z'V⁻¹(y - Xβ)
    where V = ZΨ_full Z' + σ²I
    """
    # Compute residuals
    residuals = y - X @ beta

    # Compute V
    V = compute_V_matrix(Z, psi, sigma2, n_effects=n_effects)

    # Expand psi to full dimension if needed
    q = Z.shape[1]
    if n_effects is not None and psi.shape[0] == n_effects < q:
        n_groups = q // n_effects
        psi_full = linalg.block_diag(*([psi] * n_groups))
    else:
        psi_full = psi

    # Compute V⁻¹
    L = linalg.cholesky(V, lower=True)
    V_inv = linalg.cho_solve((L, True), np.eye(V.shape[0]))

    # b = Ψ_full Z'V⁻¹(y - Xβ)
    b = psi_full @ Z.T @ V_inv @ residuals

    return b
