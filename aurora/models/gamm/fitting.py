"""GAMM fitting for Gaussian family (Linear Mixed Models).

This module implements GAMM fitting that integrates:
- Smooth terms from GAM (Phase 3)
- Random effects (Phase 4 Milestones 1-2)
- REML estimation for both smoothing parameters and variance components

For Gaussian family, GAMM reduces to LMM which can be solved efficiently
using penalized least squares with REML.

References
----------
- Wood, S.N. (2017). Generalized Additive Models: An Introduction with R (2nd ed.).
  Chapter 6: GAMMs.
- Bates, D. et al. (2015). Fitting Linear Mixed-Effects Models Using lme4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

from aurora.models.gamm.estimation import (
    REMLResult,
    estimate_fixed_effects,
    estimate_random_effects,
    estimate_variance_components,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class GAMMResult:
    """Result from GAMM fitting.

    Attributes
    ----------
    coefficients : ndarray
        Combined coefficients [β_parametric, β_smooth, b_random].
    beta_parametric : ndarray
        Parametric (fixed) effect coefficients.
    beta_smooth : dict[str, ndarray]
        Smooth term coefficients by term name.
    random_effects : dict[str, ndarray]
        Random effect coefficients by grouping variable.
    variance_components : ndarray
        Variance-covariance matrix Ψ for random effects.
    residual_variance : float
        Residual variance σ².
    smoothing_parameters : dict[str, float] | None
        Smoothing parameters λ by smooth term (if estimated).
    edf_total : float
        Total effective degrees of freedom.
    edf_parametric : float
        EDF for parametric terms.
    edf_smooth : dict[str, float]
        EDF by smooth term.
    fitted_values : ndarray
        Fitted values η̂ = Xβ + Zb.
    residuals : ndarray
        Residuals y - η̂.
    log_likelihood : float
        Log-likelihood (or REML log-likelihood).
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    converged : bool
        Whether fitting converged.
    n_iterations : int
        Number of iterations.
    n_obs : int
        Number of observations.
    n_groups : int
        Number of groups.
    family : str
        Distribution family.
    """

    coefficients: NDArray[np.floating]
    beta_parametric: NDArray[np.floating]
    beta_smooth: dict[str, NDArray[np.floating]]
    random_effects: dict[str, NDArray[np.floating]]
    variance_components: NDArray[np.floating]
    residual_variance: float
    smoothing_parameters: dict[str, float] | None
    edf_total: float
    edf_parametric: float
    edf_smooth: dict[str, float]
    fitted_values: NDArray[np.floating]
    residuals: NDArray[np.floating]
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    n_iterations: int
    n_obs: int
    n_groups: int
    family: str

    # Internal storage for prediction
    _X_parametric: NDArray[np.floating] | None = None
    _X_smooth: dict[str, NDArray[np.floating]] | None = None
    _Z: NDArray[np.floating] | None = None
    _Z_info: list[dict] | None = None
    _y: NDArray[np.floating] | None = None


def solve_mixed_model_equations(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    S_smooth: np.ndarray | None = None,
    psi_inv: np.ndarray | None = None,
    lambda_smooth: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve penalized mixed model equations for Gaussian GAMM.

    Solves the augmented system:
        [X'X + λS    X'Z      ] [β]   [X'y]
        [Z'X         Z'Z + Ψ⁻¹] [b] = [Z'y]

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Fixed effects design matrix (parametric + smooth basis).
    Z : ndarray, shape (n, q)
        Random effects design matrix.
    y : ndarray, shape (n,)
        Response vector.
    S_smooth : ndarray, shape (p, p), optional
        Smoothing penalty matrix for smooth terms.
    psi_inv : ndarray, shape (q, q), optional
        Inverse of random effects covariance Ψ⁻¹.
    lambda_smooth : float, default=0.0
        Smoothing parameter.

    Returns
    -------
    beta : ndarray, shape (p,)
        Fixed effect coefficients.
    b : ndarray, shape (q,)
        Random effect coefficients (BLUPs).

    Notes
    -----
    Uses Cholesky decomposition for numerical stability.
    For large systems, could exploit block structure or sparsity.
    """
    n, p = X.shape
    q = Z.shape[1]

    # Build fixed effects equations
    XtX = X.T @ X
    if S_smooth is not None and lambda_smooth > 0:
        XtX += lambda_smooth * S_smooth

    XtZ = X.T @ Z
    ZtZ = Z.T @ Z

    # Add random effects penalty
    if psi_inv is not None:
        ZtZ = ZtZ + psi_inv

    # Right-hand side
    Xty = X.T @ y
    Zty = Z.T @ y

    # Construct augmented system
    A = np.block([[XtX, XtZ], [XtZ.T, ZtZ]])

    b_rhs = np.concatenate([Xty, Zty])

    # Solve via Cholesky
    try:
        L = linalg.cholesky(A, lower=True)
        coef = linalg.cho_solve((L, True), b_rhs)
    except np.linalg.LinAlgError:
        # Fallback to regularized solve if Cholesky fails
        ridge = 1e-6
        A_reg = A + ridge * np.eye(A.shape[0])
        coef = linalg.solve(A_reg, b_rhs, assume_a="pos")

    beta = coef[:p]
    b = coef[p:]

    return beta, b


def compute_edf(
    X: np.ndarray,
    Z: np.ndarray,
    S_smooth: np.ndarray | None,
    psi_inv: np.ndarray | None,
    lambda_smooth: float,
) -> tuple[float, float]:
    """Compute effective degrees of freedom for fixed and random effects.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Fixed effects design matrix.
    Z : ndarray, shape (n, q)
        Random effects design matrix.
    S_smooth : ndarray, shape (p, p), optional
        Smoothing penalty matrix.
    psi_inv : ndarray, shape (q, q), optional
        Inverse of random effects covariance.
    lambda_smooth : float
        Smoothing parameter.

    Returns
    -------
    edf_fixed : float
        Effective degrees of freedom for fixed effects.
    edf_random : float
        Effective degrees of freedom for random effects.

    Notes
    -----
    For penalized least squares, EDF = tr(H) where H is the hat matrix.
    For mixed models: EDF_fixed = tr[(X'X + λS)⁻¹X'X]
                      EDF_random = tr[(Z'Z + Ψ⁻¹)⁻¹Z'Z]
    """
    p = X.shape[1]
    q = Z.shape[1]

    # Fixed effects EDF
    XtX = X.T @ X
    if S_smooth is not None and lambda_smooth > 0:
        XtX_pen = XtX + lambda_smooth * S_smooth
    else:
        XtX_pen = XtX

    try:
        XtX_pen_inv = linalg.inv(XtX_pen)
        edf_fixed = np.trace(XtX_pen_inv @ XtX)
    except np.linalg.LinAlgError:
        edf_fixed = float(p)  # Fallback to nominal DF

    # Random effects EDF
    ZtZ = Z.T @ Z
    if psi_inv is not None:
        ZtZ_pen = ZtZ + psi_inv
    else:
        ZtZ_pen = ZtZ

    try:
        ZtZ_pen_inv = linalg.inv(ZtZ_pen)
        edf_random = np.trace(ZtZ_pen_inv @ ZtZ)
    except np.linalg.LinAlgError:
        edf_random = float(q)

    return edf_fixed, edf_random


def fit_gamm_gaussian(
    X_parametric: np.ndarray,
    X_smooth: dict[str, np.ndarray] | None,
    Z: np.ndarray,
    Z_info: list[dict],
    y: np.ndarray,
    S_smooth: dict[str, np.ndarray] | None = None,
    lambda_smooth: dict[str, float] | None = None,
    covariance: str = "unstructured",
    maxiter: int = 100,
    tol: float = 1e-6,
) -> GAMMResult:
    """Fit Gaussian GAMM (Linear Mixed Model with smooth terms).

    Parameters
    ----------
    X_parametric : ndarray, shape (n, p_para)
        Parametric fixed effects design matrix.
    X_smooth : dict[str, ndarray], optional
        Smooth term basis matrices by term name.
    Z : ndarray, shape (n, q)
        Random effects design matrix.
    Z_info : list of dict
        Metadata about Z structure from construct_Z_matrix.
    y : ndarray, shape (n,)
        Response vector.
    S_smooth : dict[str, ndarray], optional
        Penalty matrices by smooth term name.
    lambda_smooth : dict[str, float], optional
        Smoothing parameters by term name. If None, uses default λ=1.0.
    covariance : str, default='unstructured'
        Covariance structure for random effects.
    maxiter : int, default=100
        Maximum iterations (for future iterative methods).
    tol : float, default=1e-6
        Convergence tolerance.

    Returns
    -------
    result : GAMMResult
        Fitted GAMM result.

    Notes
    -----
    For Gaussian family, the model reduces to:
        y = X_para β_para + Σ_k X_k β_k + Zb + ε
        b ~ N(0, Ψ), ε ~ N(0, σ²I)

    This is solved in closed form via penalized least squares with REML
    estimation of variance components.

    Algorithm:
    1. Construct combined fixed effects matrix X = [X_para | X_smooth_1 | ...]
    2. Build combined penalty matrix S (block diagonal for smooth terms)
    3. Estimate variance components Ψ, σ² via REML
    4. Solve mixed model equations with estimated variances
    5. Compute EDF, AIC, BIC

    Examples
    --------
    >>> # Random intercept with linear predictor
    >>> n = 100
    >>> X_para = np.ones((n, 1))
    >>> Z = np.eye(n)[:, :10]  # 10 groups
    >>> y = X_para @ [2.0] + Z @ np.random.randn(10) + np.random.randn(n)
    >>> Z_info = [{'n_effects': 1, 'n_groups': 10}]
    >>> result = fit_gamm_gaussian(X_para, None, Z, Z_info, y)
    """
    n = len(y)

    # Combine fixed effects design matrices
    X_list = [X_parametric]
    p_parametric = X_parametric.shape[1]

    smooth_term_names = []
    smooth_start_cols = {}
    smooth_end_cols = {}

    if X_smooth is not None:
        col_idx = p_parametric
        for term_name, X_term in X_smooth.items():
            smooth_term_names.append(term_name)
            smooth_start_cols[term_name] = col_idx
            smooth_end_cols[term_name] = col_idx + X_term.shape[1]
            col_idx = smooth_end_cols[term_name]
            X_list.append(X_term)

    X_combined = np.column_stack(X_list)
    p_combined = X_combined.shape[1]

    # Build combined penalty matrix (block diagonal for smooth terms)
    S_combined = np.zeros((p_combined, p_combined))

    if S_smooth is not None and lambda_smooth is not None:
        for term_name in smooth_term_names:
            if term_name in S_smooth and term_name in lambda_smooth:
                start = smooth_start_cols[term_name]
                end = smooth_end_cols[term_name]
                S_term = S_smooth[term_name]
                lam = lambda_smooth[term_name]
                S_combined[start:end, start:end] = lam * S_term
    elif S_smooth is not None:
        # Use default λ=1.0 for all smooth terms
        for term_name in smooth_term_names:
            if term_name in S_smooth:
                start = smooth_start_cols[term_name]
                end = smooth_end_cols[term_name]
                S_term = S_smooth[term_name]
                S_combined[start:end, start:end] = S_term

    # Step 1: Estimate variance components via REML
    reml_result = estimate_variance_components(
        y=y,
        X=X_combined,
        Z=Z,
        Z_info=Z_info,
        covariance=covariance,
        maxiter=maxiter,
        tol=tol,
        store_matrices=False,
    )

    psi = reml_result.psi
    sigma2 = reml_result.sigma2
    n_effects = Z_info[0]["n_effects"]

    # Compute Ψ⁻¹ with proper expansion
    if psi.shape[0] == n_effects < Z.shape[1]:
        # Expand to block diagonal for all groups
        n_groups = Z.shape[1] // n_effects
        psi_inv_block = linalg.inv(psi)
        psi_inv = linalg.block_diag(*([psi_inv_block] * n_groups))
    else:
        psi_inv = linalg.inv(psi)

    # Step 2: Solve mixed model equations
    beta_combined, b = solve_mixed_model_equations(
        X=X_combined,
        Z=Z,
        y=y,
        S_smooth=S_combined if S_smooth is not None else None,
        psi_inv=psi_inv,
        lambda_smooth=1.0,  # Already incorporated in S_combined
    )

    # Step 3: Extract coefficient components
    beta_parametric = beta_combined[:p_parametric]

    beta_smooth = {}
    for term_name in smooth_term_names:
        start = smooth_start_cols[term_name]
        end = smooth_end_cols[term_name]
        beta_smooth[term_name] = beta_combined[start:end]

    # Step 4: Extract random effects by group
    from aurora.models.gamm.design import extract_random_effects

    random_effects = extract_random_effects(b, Z_info)

    # Step 5: Compute fitted values and residuals
    fitted_values = X_combined @ beta_combined + Z @ b
    residuals = y - fitted_values

    # Step 6: Compute effective degrees of freedom
    edf_fixed, edf_random = compute_edf(
        X=X_combined,
        Z=Z,
        S_smooth=S_combined if S_smooth is not None else None,
        psi_inv=psi_inv,
        lambda_smooth=1.0,
    )

    edf_smooth = {}
    if X_smooth is not None and S_smooth is not None:
        # Compute per-term EDF (approximation)
        for term_name in smooth_term_names:
            start = smooth_start_cols[term_name]
            end = smooth_end_cols[term_name]
            p_term = end - start
            if term_name in S_smooth and term_name in lambda_smooth:
                # EDF ≈ tr[(X_k'X_k + λS)⁻¹ X_k'X_k]
                X_term = X_smooth[term_name]
                S_term = S_smooth[term_name]
                lam = lambda_smooth.get(term_name, 1.0)
                XtX_term = X_term.T @ X_term
                XtX_pen = XtX_term + lam * S_term
                try:
                    XtX_pen_inv = linalg.inv(XtX_pen)
                    edf_smooth[term_name] = np.trace(XtX_pen_inv @ XtX_term)
                except np.linalg.LinAlgError:
                    edf_smooth[term_name] = float(p_term)
            else:
                edf_smooth[term_name] = float(p_term)

    edf_total = edf_fixed + edf_random

    # Step 7: Compute information criteria
    log_likelihood = reml_result.log_likelihood

    # AIC = -2*log(L) + 2*k where k = effective parameters
    k_params = edf_total
    aic = -2 * log_likelihood + 2 * k_params

    # BIC = -2*log(L) + k*log(n)
    bic = -2 * log_likelihood + k_params * np.log(n)

    # Store smoothing parameters
    smoothing_params = lambda_smooth if lambda_smooth is not None else None

    # Count groups
    n_groups = Z_info[0]["n_groups"]

    return GAMMResult(
        coefficients=beta_combined,
        beta_parametric=beta_parametric,
        beta_smooth=beta_smooth,
        random_effects=random_effects,
        variance_components=psi,
        residual_variance=sigma2,
        smoothing_parameters=smoothing_params,
        edf_total=edf_total,
        edf_parametric=float(p_parametric),
        edf_smooth=edf_smooth,
        fitted_values=fitted_values,
        residuals=residuals,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        converged=reml_result.converged,
        n_iterations=reml_result.n_iterations,
        n_obs=n,
        n_groups=n_groups,
        family="gaussian",
        _X_parametric=X_parametric,
        _X_smooth=X_smooth,
        _Z=Z,
        _Z_info=Z_info,
        _y=y,
    )


def predict_gamm(
    result: GAMMResult,
    X_parametric_new: np.ndarray,
    X_smooth_new: dict[str, np.ndarray] | None = None,
    Z_new: np.ndarray | None = None,
    include_random: bool = False,
) -> np.ndarray:
    """Make predictions from fitted GAMM.

    Parameters
    ----------
    result : GAMMResult
        Fitted GAMM result.
    X_parametric_new : ndarray, shape (n_new, p_para)
        Parametric design matrix for new data.
    X_smooth_new : dict[str, ndarray], optional
        Smooth term basis matrices for new data.
    Z_new : ndarray, shape (n_new, q), optional
        Random effects design matrix for new data.
    include_random : bool, default=False
        Whether to include random effects in predictions.

    Returns
    -------
    predictions : ndarray, shape (n_new,)
        Predicted values.

    Notes
    -----
    - If include_random=False: η̂ = X_para β̂_para + Σ_k X_k β̂_k (population-level)
    - If include_random=True: η̂ = X_para β̂_para + Σ_k X_k β̂_k + Z b̂ (conditional)

    For new groups not in training data, random effects default to 0.
    """
    # Parametric prediction
    pred = X_parametric_new @ result.beta_parametric

    # Add smooth terms
    if X_smooth_new is not None:
        for term_name, X_term_new in X_smooth_new.items():
            if term_name in result.beta_smooth:
                pred += X_term_new @ result.beta_smooth[term_name]

    # Add random effects if requested
    if include_random and Z_new is not None:
        # Flatten random effects coefficients
        b_flat = []
        for group_id in sorted(result.random_effects.keys()):
            # Handle both single random effect terms and multiple
            if isinstance(result.random_effects[group_id], dict):
                # Multiple terms (not yet implemented in extract_random_effects)
                for term_name, b_term in result.random_effects[group_id].items():
                    b_flat.extend(b_term)
            else:
                # Single term
                b_flat.extend(result.random_effects[group_id])

        b_array = np.array(b_flat)
        pred += Z_new @ b_array

    return pred
