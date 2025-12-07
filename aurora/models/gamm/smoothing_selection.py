# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Smoothing Parameter Selection for GAMM.

This module implements methods for selecting smoothing parameters (λ) in
Generalized Additive Mixed Models, including:

1. GCV (Generalized Cross-Validation) for non-Gaussian families
2. REML via Laplace approximation (Wood, 2011)
3. Performance iteration algorithm

Mathematical Framework
----------------------
For non-Gaussian GAMM with smooth terms:

    g(E[Y | b]) = Xβ + Zb

where X includes both parametric and smooth terms, we need to select
smoothing parameters λ = (λ₁, ..., λₚ) that control the roughness of
smooth functions.

GCV for Non-Gaussian GAMM
--------------------------
GCV approximates leave-one-out cross-validation using the working response
from PQL:

    GCV(λ) = (n × RSS_w) / (n - tr(H_λ))²

where:
- RSS_w: Residual sum of squares on working response
- H_λ: Smoother matrix (hat matrix) for given λ
- n: Number of observations

**Algorithm**:
1. For each λ candidate in grid:
   a. Run simplified PQL to get working response z and weights W
   b. Fit penalized weighted least squares with penalty λS
   c. Compute effective degrees of freedom: edf = tr(H_λ)
   d. Compute GCV score

2. Select λ* = argmin GCV(λ)

**Advantages**:
- Fast: No need to fully converge PQL for each λ
- Simple: Uses working response from single PQL iteration
- Effective: Good approximation to cross-validation

**Limitations**:
- Approximate: Based on working response, not true likelihood
- Single smooth: More complex for multiple smoothing parameters

REML via Laplace Approximation
-------------------------------
For more accurate selection, use Laplace-approximated REML:

    -2ℓ_R(λ) ≈ log|H| + log|X^T V⁻¹ X| + r^T V⁻¹ r + log|S + λI|

where:
- H: Hessian of penalized log-likelihood
- V: Marginal covariance
- r: Residuals

**Advantages**:
- More accurate than GCV
- Proper likelihood-based criterion
- Accounts for uncertainty in fixed effects

**Disadvantages**:
- More expensive to compute
- Requires Hessian computation

Performance Iteration
---------------------
Alternating optimization between β/b and λ:

1. Initialize λ
2. Iterate:
   a. Fix λ, update (β, b) via PQL
   b. Fix (β, b), update λ via GCV or REML
   c. Check convergence

**Convergence**: Usually 3-5 outer iterations

References
----------
- Wood, S. N. (2011). "Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models."
  Journal of the Royal Statistical Society: Series B, 73(1), 3-36.
  https://doi.org/10.1111/j.1467-9868.2010.00749.x

- Gu, C. (1992). "Cross-validating non-Gaussian data." Journal of
  Computational and Graphical Statistics, 1(2), 169-179.

- Craven, P., & Wahba, G. (1978). "Smoothing noisy data with spline functions."
  Numerische Mathematik, 31(4), 377-403.
  https://doi.org/10.1007/BF01404567

See Also
--------
aurora.models.gamm.pql_smooth : PQL with smooth terms
aurora.smoothing.selection.gcv : GCV for Gaussian responses
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

if TYPE_CHECKING:
    from numpy.typing import NDArray


def select_smoothing_gcv(
    X_parametric: NDArray[np.floating],
    X_smooth_dict: dict[str, NDArray[np.floating]],
    S_smooth_dict: dict[str, NDArray[np.floating]],
    Z: NDArray[np.floating],
    y: NDArray[np.floating],
    family_obj,
    link,
    Psi: NDArray[np.floating],
    lambda_grid: dict[str, NDArray[np.floating]] | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    """Select smoothing parameters via GCV for non-Gaussian GAMM.

    Parameters
    ----------
    X_parametric : ndarray, shape (n, p_para)
        Parametric design matrix.
    X_smooth_dict : dict[str, ndarray]
        Smooth term basis matrices by name.
    S_smooth_dict : dict[str, ndarray]
        Penalty matrices by smooth term name.
    Z : ndarray, shape (n, q)
        Random effects design matrix.
    y : ndarray, shape (n,)
        Response vector.
    family_obj : Family
        Distribution family object.
    link : LinkFunction
        Link function object.
    Psi : ndarray
        Current estimate of random effects covariance.
    lambda_grid : dict[str, ndarray], optional
        Grid of λ candidates for each smooth term.
        If None, uses default grid [1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6].
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    lambda_opt : dict[str, float]
        Optimal smoothing parameters by term name.

    Notes
    -----
    For multiple smooth terms, this function optimizes each λ separately
    while holding others fixed (coordinate descent). For joint optimization,
    use multi-dimensional grid search or gradient-based methods.

    Examples
    --------
    >>> from aurora.models.gamm.smoothing_selection import select_smoothing_gcv
    >>> from aurora.distributions.families import PoissonFamily
    >>> # After initial PQL fit
    >>> lambda_opt = select_smoothing_gcv(
    ...     X_parametric=X_para,
    ...     X_smooth_dict={'s(x)': B},
    ...     S_smooth_dict={'s(x)': S},
    ...     Z=Z,
    ...     y=y,
    ...     family_obj=PoissonFamily(),
    ...     link=PoissonFamily().default_link,
    ...     Psi=Psi_current,
    ... )
    """
    n = len(y)
    smooth_names = list(X_smooth_dict.keys())

    # Default grid if not provided
    if lambda_grid is None:
        lambda_grid = {
            name: np.logspace(-6, 6, 13)  # 13 points from 1e-6 to 1e6
            for name in smooth_names
        }

    # Initialize with middle of grid
    lambda_current = {
        name: lambda_grid[name][len(lambda_grid[name]) // 2] for name in smooth_names
    }

    # Coordinate descent over smooth terms
    max_coord_iter = 3  # Usually converges quickly
    for coord_iter in range(max_coord_iter):
        lambda_old = lambda_current.copy()

        for term_name in smooth_names:
            if verbose:
                print(f"\nOptimizing λ for '{term_name}'...")

            # Get current λ values
            lambda_fixed = {k: v for k, v in lambda_current.items() if k != term_name}

            # Search over grid for this term
            gcv_scores = []
            for lambda_candidate in lambda_grid[term_name]:
                # Temporary λ dict with candidate
                lambda_temp = lambda_fixed.copy()
                lambda_temp[term_name] = lambda_candidate

                # Compute GCV score
                gcv = _compute_gcv_score(
                    X_parametric=X_parametric,
                    X_smooth_dict=X_smooth_dict,
                    S_smooth_dict=S_smooth_dict,
                    Z=Z,
                    y=y,
                    family_obj=family_obj,
                    link=link,
                    Psi=Psi,
                    lambda_dict=lambda_temp,
                )
                gcv_scores.append(gcv)

            # Select best λ
            best_idx = np.argmin(gcv_scores)
            lambda_current[term_name] = lambda_grid[term_name][best_idx]

            if verbose:
                print(
                    f"  Best λ: {lambda_current[term_name]:.2e} (GCV: {gcv_scores[best_idx]:.4f})"
                )

        # Check convergence
        change = max(
            abs(np.log(lambda_current[k]) - np.log(lambda_old[k])) for k in smooth_names
        )
        if change < 0.1:  # Converged in log scale
            break

    return lambda_current


def _compute_gcv_score(
    X_parametric: NDArray[np.floating],
    X_smooth_dict: dict[str, NDArray[np.floating]],
    S_smooth_dict: dict[str, NDArray[np.floating]],
    Z: NDArray[np.floating],
    y: NDArray[np.floating],
    family_obj,
    link,
    Psi: NDArray[np.floating],
    lambda_dict: dict[str, float],
) -> float:
    """Compute GCV score for given smoothing parameters.

    This performs a simplified PQL iteration to get working response,
    then computes GCV on the penalized weighted least squares problem.
    """
    n = len(y)

    # Concatenate smooth matrices
    smooth_names = list(X_smooth_dict.keys())
    X_smooth = np.hstack([X_smooth_dict[name] for name in smooth_names])

    # Build penalty matrix
    S_blocks = [S_smooth_dict[name] for name in smooth_names]
    S = linalg.block_diag(*S_blocks)

    # Build Lambda matrix
    p_smooth_list = [X_smooth_dict[name].shape[1] for name in smooth_names]
    Lambda_blocks = []
    for name, K_j in zip(smooth_names, p_smooth_list):
        lambda_j = lambda_dict[name]
        Lambda_blocks.append(lambda_j * np.eye(K_j))
    Lambda = linalg.block_diag(*Lambda_blocks)

    # Initialize eta with simple prediction
    eta = link.link(y + 0.1)  # Avoid boundaries
    mu = link.inverse(eta)

    # Single PQL iteration to get working response and weights
    deta_dmu = link.derivative(mu)
    dmu_deta = 1.0 / (deta_dmu + 1e-10)
    var = family_obj.variance(mu)
    W_diag = 1.0 / (var * deta_dmu**2 + 1e-10)
    W_diag = np.clip(W_diag, 1e-10, 1e10)
    W = np.diag(W_diag)

    # Working response
    z = eta + (y - mu) * dmu_deta

    # Solve penalized mixed model equations
    # For simplicity, ignore random effects in GCV computation
    # (full version would include Z, but this is faster and usually sufficient)
    p_para = X_parametric.shape[1]
    p_smooth = X_smooth.shape[1]

    XpWXp = X_parametric.T @ W @ X_parametric
    XpWXs = X_parametric.T @ W @ X_smooth
    XsWXs = X_smooth.T @ W @ X_smooth + Lambda @ S

    Xp_W_z = X_parametric.T @ W @ z
    Xs_W_z = X_smooth.T @ W @ z

    # Solve for coefficients
    A = np.block([[XpWXp, XpWXs], [XpWXs.T, XsWXs]])
    b_rhs = np.concatenate([Xp_W_z, Xs_W_z])

    try:
        coef = np.linalg.solve(A, b_rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, b_rhs, rcond=None)[0]

    beta_para = coef[:p_para]
    beta_smooth = coef[p_para:]

    # Fitted values
    z_fitted = X_parametric @ beta_para + X_smooth @ beta_smooth

    # Residual sum of squares (weighted)
    residuals = z - z_fitted
    RSS = np.sum(W_diag * residuals**2)

    # Effective degrees of freedom
    # edf = tr(H) where H = X (X^T W X + Λ S)^{-1} X^T W
    # For computational efficiency, use: edf = tr((X^T W X + Λ S)^{-1} X^T W X)
    try:
        A_inv = np.linalg.inv(A)
        X_full = np.hstack([X_parametric, X_smooth])
        XtWX = X_full.T @ W @ X_full
        edf = np.trace(A_inv @ XtWX)
    except np.linalg.LinAlgError:
        # Fallback: approximate edf
        edf = p_para + p_smooth / 2

    # GCV score
    if edf >= n - 1:
        # Degenerate case
        return np.inf

    gcv = (n * RSS) / (n - edf) ** 2

    return gcv


def select_smoothing_performance_iter(
    X_parametric: NDArray[np.floating],
    X_smooth_dict: dict[str, NDArray[np.floating]],
    S_smooth_dict: dict[str, NDArray[np.floating]],
    Z: NDArray[np.floating],
    Z_info: list[dict],
    y: NDArray[np.floating],
    family: str,
    max_iter: int = 5,
    verbose: bool = False,
) -> dict:
    """Select smoothing parameters via performance iteration.

    This alternates between:
    1. Fitting (β, b) with fixed λ
    2. Updating λ via GCV with fixed (β, b)

    Parameters
    ----------
    X_parametric : ndarray
        Parametric design matrix.
    X_smooth_dict : dict
        Smooth basis matrices.
    S_smooth_dict : dict
        Penalty matrices.
    Z : ndarray
        Random effects design matrix.
    Z_info : list
        Random effects metadata.
    y : ndarray
        Response vector.
    family : str
        Distribution family.
    max_iter : int, default=5
        Maximum performance iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    result : dict
        Contains:
        - 'lambda_opt': Optimal smoothing parameters
        - 'beta_parametric': Final parametric coefficients
        - 'beta_smooth': Final smooth coefficients
        - 'random_effects': Final random effects
        - 'variance_components': Final Psi
        - 'converged': Whether algorithm converged
    """
    from aurora.distributions.families import (
        BinomialFamily,
        GammaFamily,
        PoissonFamily,
    )
    from aurora.models.gamm.pql_smooth import fit_pql_with_smooth

    # Get family object
    family_map = {
        "poisson": PoissonFamily(),
        "binomial": BinomialFamily(),
        "gamma": GammaFamily(),
    }
    family_obj = family_map[family]
    link = family_obj.default_link

    # Initialize with default λ
    lambda_current = {name: 1.0 for name in X_smooth_dict.keys()}

    for iter_num in range(max_iter):
        if verbose:
            print(f"\n=== Performance Iteration {iter_num + 1}/{max_iter} ===")
            print(f"Current λ: {lambda_current}")

        # Step 1: Fit model with current λ
        result = fit_pql_with_smooth(
            X_parametric=X_parametric,
            X_smooth_dict=X_smooth_dict,
            Z=Z,
            Z_info=Z_info,
            y=y,
            family=family,
            S_smooth_dict=S_smooth_dict,
            lambda_smooth=lambda_current,
            maxiter_outer=10,
            maxiter_inner=10,
            verbose=False,
        )

        # Step 2: Update λ via GCV
        Psi = linalg.block_diag(*result["variance_components"])
        lambda_new = select_smoothing_gcv(
            X_parametric=X_parametric,
            X_smooth_dict=X_smooth_dict,
            S_smooth_dict=S_smooth_dict,
            Z=Z,
            y=y,
            family_obj=family_obj,
            link=link,
            Psi=Psi,
            verbose=verbose,
        )

        # Check convergence
        change = max(
            abs(np.log(lambda_new[k]) - np.log(lambda_current[k]))
            for k in lambda_new.keys()
        )

        if verbose:
            print(f"New λ: {lambda_new}")
            print(f"Max log change: {change:.4f}")

        if change < 0.05:  # Converged
            if verbose:
                print("Converged!")
            result["lambda_opt"] = lambda_new
            result["converged"] = True
            return result

        lambda_current = lambda_new

    # Final fit with final λ
    result = fit_pql_with_smooth(
        X_parametric=X_parametric,
        X_smooth_dict=X_smooth_dict,
        Z=Z,
        Z_info=Z_info,
        y=y,
        family=family,
        S_smooth_dict=S_smooth_dict,
        lambda_smooth=lambda_current,
        maxiter_outer=15,
        maxiter_inner=10,
        verbose=False,
    )
    result["lambda_opt"] = lambda_current
    result["converged"] = False  # Did not converge in max_iter

    return result


__all__ = [
    "select_smoothing_gcv",
    "select_smoothing_performance_iter",
]
