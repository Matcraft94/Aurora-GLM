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
    cov_structure: CovarianceStructure | list[CovarianceStructure],
    n_effects: int | list[int],
    Z_info: list[dict] | None = None,
) -> float:
    """REML objective function (negative log-likelihood).

    Parameters
    ----------
    theta : ndarray
        Covariance parameters [psi_params_1..., psi_params_2..., log(sigma2)].
    y : ndarray, shape (n,)
        Response vector.
    X : ndarray, shape (n, p)
        Fixed effects design matrix.
    Z : ndarray, shape (n, q)
        Random effects design matrix.
    cov_structure : CovarianceStructure or list of CovarianceStructure
        Covariance structure(s) for Ψ. If list, one per random effect term.
    n_effects : int or list of int
        Number of random effects per group. If list, one per random effect term.
    Z_info : list of dict, optional
        Information about Z structure (required for multiple terms).

    Returns
    -------
    neg_log_lik : float
        Negative REML log-likelihood.
    """
    try:
        # Handle single vs multiple random effects
        if isinstance(cov_structure, list):
            # Multiple random effects
            if Z_info is None:
                raise ValueError("Z_info required for multiple random effects")
            
            n_terms = len(Z_info)
            psi_blocks = []
            param_idx = 0
            
            for i in range(n_terms):
                n_eff = n_effects[i] if isinstance(n_effects, list) else n_effects
                cov = cov_structure[i] if isinstance(cov_structure, list) else cov_structure
                
                # Extract parameters for this term
                n_params = cov.n_parameters(n_eff)
                psi_params = theta[param_idx:param_idx + n_params]
                param_idx += n_params
                
                # Construct Ψ for this term
                psi_i = cov.construct_psi(psi_params, n_eff)
                
                # Expand to block diagonal for all groups in this term
                n_groups = Z_info[i]['n_groups']
                psi_block = linalg.block_diag(*([psi_i] * n_groups))
                psi_blocks.append(psi_block)
            
            # Combine into full block-diagonal Ψ
            psi_full = linalg.block_diag(*psi_blocks)
            
            # Extract sigma2
            log_sigma2 = theta[param_idx]
            sigma2 = np.exp(log_sigma2)
            
            # Compute V = ZΨZ' + σ²I
            V = Z @ psi_full @ Z.T + sigma2 * np.eye(len(y))
            
        else:
            # Single random effect (backward compatibility)
            n_psi_params = cov_structure.n_parameters(n_effects)
            psi_params = theta[:n_psi_params]
            log_sigma2 = theta[n_psi_params]
            sigma2 = np.exp(log_sigma2)

            # Construct Ψ
            psi = cov_structure.construct_psi(psi_params, n_effects)

            # Compute V
            V = compute_V_matrix(Z, psi, sigma2, n_effects=n_effects)

        # Compute P and log-likelihood
        P = compute_P_matrix(V, X)
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
    initial_psi: np.ndarray | list[np.ndarray] | None = None,
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
        Each dict contains: 'n_effects', 'n_groups', 'grouping', 'start_col', 'end_col'.
    covariance : str, default='unstructured'
        Covariance structure: 'unstructured', 'diagonal', or 'identity'.
        Applied to all random effect terms.
    method : str, default='L-BFGS-B'
        Optimization method for scipy.optimize.minimize.
    maxiter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    initial_psi : ndarray or list of ndarray or None, optional
        Initial value(s) for Ψ. If None, uses identity for each term.
        If list, one per random effect term. If single array, used for all terms.
    initial_sigma2 : float or None, optional
        Initial value for σ². If None, uses variance of residuals.
    store_matrices : bool, default=False
        Whether to store V and P matrices in result (memory intensive).

    Returns
    -------
    result : REMLResult
        REML estimation result. For multiple random effects, psi is block-diagonal
        containing all variance-covariance matrices.

    Examples
    --------
    >>> # Single random intercept model
    >>> n = 100
    >>> y = np.random.randn(n)
    >>> X = np.ones((n, 1))
    >>> groups = np.repeat(np.arange(20), 5)
    >>> Z = np.zeros((n, 20))
    >>> Z[np.arange(n), groups] = 1
    >>> Z_info = [{'n_effects': 1, 'n_groups': 20, 'grouping': 'subject',
    ...            'start_col': 0, 'end_col': 20}]
    >>> result = estimate_variance_components(y, X, Z, Z_info)
    
    >>> # Multiple random effects (crossed): (1 | school) + (1 | class)
    >>> # Z_info = [
    >>> #     {'n_effects': 1, 'n_groups': 10, 'grouping': 'school', 'start_col': 0, 'end_col': 10},
    >>> #     {'n_effects': 1, 'n_groups': 30, 'grouping': 'class', 'start_col': 10, 'end_col': 40}
    >>> # ]
    >>> # result = estimate_variance_components(y, X, Z, Z_info)
    """
    n, p = X.shape
    _, q = Z.shape

    # Number of random effect terms
    n_terms = len(Z_info)
    
    if n_terms == 0:
        raise ValueError("No random effects specified in Z_info")

    # Get covariance structure(s) from Z_info (can be different per term)
    # Fall back to covariance parameter if not specified in Z_info
    cov_structures = []
    for info in Z_info:
        cov_type = info.get('covariance', covariance)
        cov_structures.append(get_covariance_structure(cov_type))
    n_effects_list = [info["n_effects"] for info in Z_info]

    # Initialize parameters
    if initial_psi is None:
        initial_psi_list = [np.eye(n_eff) for n_eff in n_effects_list]
    elif isinstance(initial_psi, list):
        initial_psi_list = initial_psi
    else:
        # Single array provided, use for all terms
        initial_psi_list = [initial_psi for _ in range(n_terms)]

    if initial_sigma2 is None:
        # Use variance of OLS residuals as initial guess
        beta_ols = linalg.lstsq(X, y)[0]
        residuals = y - X @ beta_ols
        initial_sigma2 = np.var(residuals)

    # Extract initial covariance parameters for all terms
    theta_init_parts = []
    for i in range(n_terms):
        psi_params_init = cov_structures[i].extract_params(initial_psi_list[i])
        theta_init_parts.append(psi_params_init)
    
    # Add log(sigma2) at the end
    log_sigma2_init = np.log(initial_sigma2)
    theta_init = np.concatenate(theta_init_parts + [np.array([log_sigma2_init])])

    # Optimize
    result = optimize.minimize(
        reml_objective,
        theta_init,
        args=(y, X, Z, cov_structures, n_effects_list, Z_info),
        method=method,
        options={"maxiter": maxiter, "ftol": tol},
    )

    # Extract optimized parameters
    psi_per_term = []
    param_idx = 0

    for i in range(n_terms):
        n_params = cov_structures[i].n_parameters(n_effects_list[i])
        psi_params_opt = result.x[param_idx:param_idx + n_params]
        param_idx += n_params

        # Construct Ψ for this term (per-group covariance structure)
        psi_i = cov_structures[i].construct_psi(psi_params_opt, n_effects_list[i])
        psi_per_term.append(psi_i)

    # Extract sigma2
    log_sigma2_opt = result.x[param_idx]
    sigma2_opt = np.exp(log_sigma2_opt)

    # Return per-group covariance structure, not expanded block-diagonal
    # For single random effect term, return the per-group structure
    # For multiple terms, return block-diagonal of per-group structures
    if len(psi_per_term) == 1:
        psi_opt = psi_per_term[0]
    else:
        # For multiple random effect terms, combine their per-group structures
        psi_opt = linalg.block_diag(*psi_per_term)

    # Compute final V and P if requested
    V_opt = None
    P_opt = None
    if store_matrices:
        # Need to expand psi to block-diagonal for V computation
        if len(psi_per_term) == 1 and psi_opt.shape[0] == n_effects_list[0]:
            # Single term, expand per-group to full block-diagonal
            n_groups = Z_info[0]['n_groups']
            psi_full = linalg.block_diag(*([psi_opt] * n_groups))
        else:
            # Multiple terms or already correct size
            # For multiple terms, need to expand each and combine
            psi_blocks = []
            for i, psi_i in enumerate(psi_per_term):
                n_groups = Z_info[i]['n_groups']
                psi_blocks.append(linalg.block_diag(*([psi_i] * n_groups)))
            psi_full = linalg.block_diag(*psi_blocks) if len(psi_blocks) > 1 else psi_blocks[0]

        V_opt = Z @ psi_full @ Z.T + sigma2_opt * np.eye(n)
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
    Z_info: list[dict] | None = None,
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
    psi : ndarray
        Random effects covariance matrix. Can be:
        - shape (q_effects, q_effects): per-group covariance (single term)
        - shape (q, q): full block-diagonal for multiple terms
    sigma2 : float
        Residual variance.
    Z_info : list of dict or None, optional
        Information about Z structure. If provided and psi is per-group,
        will expand psi to block-diagonal.

    Returns
    -------
    b : ndarray, shape (q,)
        Random effects estimates (BLUPs).

    Notes
    -----
    b = Ψ_full Z'V⁻¹(y - Xβ)
    where V = ZΨ_full Z' + σ²I
    
    For multiple random effects, psi should already be block-diagonal.
    """
    # Compute residuals
    residuals = y - X @ beta
    
    q = Z.shape[1]
    n = len(y)
    
    # Check if psi needs expansion (backward compatibility)
    if Z_info is not None and len(Z_info) == 1 and psi.shape[0] < q:
        # Single term with per-group covariance
        n_effects = Z_info[0]["n_effects"]
        n_groups = Z_info[0]["n_groups"]
        if psi.shape[0] == n_effects:
            psi_full = linalg.block_diag(*([psi] * n_groups))
        else:
            psi_full = psi
    else:
        # Already in correct form (single group or already block-diagonal)
        psi_full = psi

    # Compute V = ZΨZ' + σ²I
    V = Z @ psi_full @ Z.T + sigma2 * np.eye(n)

    # Compute V⁻¹
    L = linalg.cholesky(V, lower=True)
    V_inv = linalg.cho_solve((L, True), np.eye(n))

    # b = Ψ_full Z'V⁻¹(y - Xβ)
    b = psi_full @ Z.T @ V_inv @ residuals

    return b
