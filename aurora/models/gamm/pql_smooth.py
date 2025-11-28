"""Penalized Quasi-Likelihood (PQL) with Smooth Terms for Non-Gaussian GAMM.

This module extends PQL to support smooth functions in non-Gaussian GAMMs,
enabling models like:

    g(E[Y | b]) = f₁(x₁) + f₂(x₂) + ... + Zb

where f_j are smooth functions represented via penalized splines.

Mathematical Framework
----------------------
The non-Gaussian GAMM with smooth terms combines:
1. Smooth functions via basis expansion: f_j(x) = Σₖ β_jk B_jk(x)
2. Random effects: b ~ N(0, Ψ)
3. Non-Gaussian responses: Y ~ ExpFamily(μ, φ)
4. Link function: g(μ) = η

PQL Algorithm with Smooth Terms
--------------------------------
**Outer loop** (variance components):
1. Initialize Ψ, λ (variance and smoothing parameters)
2. Iterate until convergence:
   a. Update (β, b) given (Ψ, λ) via penalized IRLS
   b. Update Ψ via REML
   c. Update λ via GCV or REML (optional)

**Inner loop** (penalized IRLS):
1. Compute working response z and weights W
2. Solve augmented mixed model equations:

   [X_p^T W X_p      X_p^T W X_s      X_p^T W Z    ] [β_p  ]   [X_p^T W z]
   [X_s^T W X_p  X_s^T W X_s + λS    X_s^T W Z    ] [β_s  ] = [X_s^T W z]
   [Z^T W X_p        Z^T W X_s        Z^T W Z + Ψ⁻¹] [b    ]   [Z^T W z]

   where:
   - X_p: parametric design matrix
   - X_s: smooth term basis matrices (concatenated)
   - S: penalty matrix (block-diagonal for all smooth terms)
   - λ: smoothing parameters (vector or single value)

3. Update η = X_p β_p + X_s β_s + Zb
4. Check convergence on ||β^(t+1) - β^(t)||

References
----------
- Wood, S. N. (2011). "Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models."
  Journal of the Royal Statistical Society: Series B, 73(1), 3-36.

- Wood, S. N. (2017). Generalized Additive Models: An Introduction with R (2nd ed.).
  CRC Press. Chapter 6: GAMMs.

See Also
--------
aurora.models.gamm.pql : PQL for non-Gaussian GAMM without smooth terms
aurora.models.gam.fitting : GAM fitting for Gaussian responses
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

from aurora.distributions.families import (
    BinomialFamily,
    GammaFamily,
    GaussianFamily,
    PoissonFamily,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def fit_pql_with_smooth(
    X_parametric: NDArray[np.floating],
    X_smooth_dict: dict[str, NDArray[np.floating]],
    Z: NDArray[np.floating],
    Z_info: list[dict],
    y: NDArray[np.floating],
    family: str,
    S_smooth_dict: dict[str, NDArray[np.floating]],
    lambda_smooth: dict[str, float] | None = None,
    Psi_init: list[NDArray[np.floating]] | None = None,
    maxiter_outer: int = 20,
    maxiter_inner: int = 10,
    tol_outer: float = 1e-4,
    tol_inner: float = 1e-6,
    verbose: bool = False,
) -> dict:
    """Fit non-Gaussian GAMM with smooth terms using PQL.

    Parameters
    ----------
    X_parametric : ndarray, shape (n, p_para)
        Parametric fixed effects design matrix.
    X_smooth_dict : dict[str, ndarray]
        Smooth term basis matrices {term_name: X_smooth}.
        Each X_smooth has shape (n, K_j) for K_j basis functions.
    Z : ndarray, shape (n, q)
        Random effects design matrix.
    Z_info : list[dict]
        Random effects metadata, one dict per random effect term.
        Each dict contains 'n_levels' and 'type' keys.
    y : ndarray, shape (n,)
        Response vector.
    family : str
        Distribution family: 'poisson', 'binomial', 'gamma'.
    S_smooth_dict : dict[str, ndarray]
        Penalty matrices for each smooth term {term_name: S}.
        Each S has shape (K_j, K_j).
    lambda_smooth : dict[str, float], optional
        Smoothing parameters for each smooth term. If None, uses GCV.
    Psi_init : list[ndarray], optional
        Initial variance-covariance matrices for random effects.
    maxiter_outer : int, default=20
        Maximum outer iterations for variance components.
    maxiter_inner : int, default=10
        Maximum inner iterations for (β, b) update.
    tol_outer : float, default=1e-4
        Convergence tolerance for variance components.
    tol_inner : float, default=1e-6
        Convergence tolerance for coefficients.
    verbose : bool, default=False
        Print iteration progress.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'beta_parametric': Parametric coefficients
        - 'beta_smooth': Dictionary of smooth coefficients by term
        - 'random_effects': Random effect estimates
        - 'variance_components': List of Ψ matrices
        - 'smoothing_parameters': Dictionary of λ by smooth term
        - 'fitted_values': Fitted linear predictor
        - 'converged': Whether algorithm converged
        - 'n_iterations_outer': Number of outer iterations
        - 'edf_smooth': EDF by smooth term

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.gamm.pql_smooth import fit_pql_with_smooth
    >>> from aurora.smoothing.splines.bspline import BSplineBasis
    >>> # Generate data
    >>> n = 200
    >>> x = np.linspace(0, 1, n)
    >>> groups = np.repeat(np.arange(20), 10)
    >>> # Create smooth basis
    >>> basis = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    >>> B = BSplineBasis(basis).basis_matrix(x)
    >>> X_smooth = {'s(x)': B}
    >>> S = {'s(x)': BSplineBasis(basis).penalty_matrix(order=2)}
    >>> # Random effects
    >>> Z = np.eye(n)[:, groups]
    >>> # Fit model
    >>> result = fit_pql_with_smooth(
    ...     X_parametric=np.ones((n, 1)),
    ...     X_smooth_dict=X_smooth,
    ...     Z=Z,
    ...     Z_info=[{'n_levels': 20, 'type': 'intercept'}],
    ...     y=y,
    ...     family='poisson',
    ...     S_smooth_dict=S,
    ... )
    """
    # Validate inputs
    n = len(y)
    if X_parametric.shape[0] != n:
        raise ValueError(f"X_parametric has {X_parametric.shape[0]} rows, expected {n}")
    if Z.shape[0] != n:
        raise ValueError(f"Z has {Z.shape[0]} rows, expected {n}")

    for name, X_s in X_smooth_dict.items():
        if X_s.shape[0] != n:
            raise ValueError(f"X_smooth['{name}'] has {X_s.shape[0]} rows, expected {n}")
        if name not in S_smooth_dict:
            raise ValueError(f"Missing penalty matrix for smooth term '{name}'")

    # Get family object
    family_map = {
        'poisson': PoissonFamily(),
        'binomial': BinomialFamily(),
        'gamma': GammaFamily(),
        'gaussian': GaussianFamily(),
    }
    if family not in family_map:
        raise ValueError(f"Unsupported family: {family}")
    family_obj = family_map[family]
    link = family_obj.default_link

    # Dimensions
    p_para = X_parametric.shape[1]
    smooth_names = list(X_smooth_dict.keys())
    p_smooth_list = [X_smooth_dict[name].shape[1] for name in smooth_names]
    p_smooth_total = sum(p_smooth_list)
    q = Z.shape[1]

    # Concatenate smooth matrices
    X_smooth = np.hstack([X_smooth_dict[name] for name in smooth_names])

    # Build block-diagonal penalty matrix
    S_blocks = [S_smooth_dict[name] for name in smooth_names]
    S = linalg.block_diag(*S_blocks)

    # Initialize smoothing parameters
    if lambda_smooth is None:
        # Use GCV to select smoothing parameters
        if verbose:
            print("Selecting smoothing parameters via GCV...")

        # Initial rough fit to get Psi estimate
        lambda_init = {name: 1.0 for name in smooth_names}
        auto_lambda = True
    else:
        lambda_init = lambda_smooth
        auto_lambda = False

    # Build diagonal Lambda matrix
    Lambda = _build_lambda_matrix(lambda_init, smooth_names, p_smooth_list)
    lambda_smooth = lambda_init.copy()  # Working copy

    # Initialize variance components
    if Psi_init is None:
        # Simple initialization: identity for each random effect
        Psi_list = []
        for info in Z_info:
            dim = info.get('dim', 1)  # Dimension of random effect (1 for intercept)
            Psi_list.append(np.eye(dim))
    else:
        Psi_list = Psi_init

    # Build block-diagonal Psi
    if len(Psi_list) == 1:
        Psi = Psi_list[0]
    else:
        Psi = linalg.block_diag(*Psi_list)

    # Initialize coefficients
    # Simple initialization: GLM without random effects
    eta = link.link(y + 0.1)  # Avoid boundaries
    beta_para = np.zeros(p_para)
    beta_smooth = np.zeros(p_smooth_total)
    b = np.zeros(q)

    # Outer loop: Update variance components
    converged_outer = False
    for iter_outer in range(maxiter_outer):
        Psi_old = Psi.copy()

        # Inner loop: Update (beta_para, beta_smooth, b) given Psi and Lambda
        converged_inner = False
        for iter_inner in range(maxiter_inner):
            # Clamp eta to prevent overflow in link.inverse (exp)
            eta = np.clip(eta, -700.0, 700.0)
            
            # Compute current predictions
            mu = link.inverse(eta)
            
            # Clamp mu to valid range for the family
            mu = np.clip(mu, 1e-10, 1e10)

            # Compute derivative dμ/dη with protection
            deta_dmu = link.derivative(mu)
            deta_dmu = np.clip(deta_dmu, 1e-10, None)
            dmu_deta = 1.0 / deta_dmu

            # Variance function with protection
            var = family_obj.variance(mu)
            var = np.clip(var, 1e-10, None)

            # Working weights with enhanced stability
            W_diag = dmu_deta**2 / var
            W_diag = np.clip(W_diag, 1e-10, 1e10)  # Numerical stability
            W = np.diag(W_diag)

            # Working response with NaN protection
            z = eta + (y - mu) * dmu_deta
            
            # Validate finite values and fall back if needed
            if not (np.all(np.isfinite(z)) and np.all(np.isfinite(W_diag))):
                import warnings
                warnings.warn("NaN/Inf detected in PQL smooth iteration, using regularization")
                z = np.where(np.isfinite(z), z, eta)
                W_diag = np.where(np.isfinite(W_diag), W_diag, 1e-6)
                W = np.diag(W_diag)

            # Solve augmented system
            # [X_p^T W X_p      X_p^T W X_s        X_p^T W Z    ] [β_p]   [X_p^T W z]
            # [X_s^T W X_p  X_s^T W X_s + λS    X_s^T W Z    ] [β_s] = [X_s^T W z]
            # [Z^T W X_p        Z^T W X_s          Z^T W Z + Ψ⁻¹] [b  ]   [Z^T W z]

            # Build blocks
            XpWXp = X_parametric.T @ W @ X_parametric
            XpWXs = X_parametric.T @ W @ X_smooth
            XpWZ = X_parametric.T @ W @ Z
            XsWXs = X_smooth.T @ W @ X_smooth + Lambda @ S
            XsWZ = X_smooth.T @ W @ Z
            ZWZ = Z.T @ W @ Z

            # Add random effects penalty
            try:
                Psi_inv = np.linalg.inv(Psi + 1e-6 * np.eye(Psi.shape[0]))
            except np.linalg.LinAlgError:
                Psi_inv = np.linalg.pinv(Psi + 1e-6 * np.eye(Psi.shape[0]))

            ZWZ_pen = ZWZ + Psi_inv

            # Right-hand side
            Xp_W_z = X_parametric.T @ W @ z
            Xs_W_z = X_smooth.T @ W @ z
            Z_W_z = Z.T @ W @ z

            # Assemble augmented system
            A = np.block([
                [XpWXp, XpWXs, XpWZ],
                [XpWXs.T, XsWXs, XsWZ],
                [XpWZ.T, XsWZ.T, ZWZ_pen]
            ])
            b_rhs = np.concatenate([Xp_W_z, Xs_W_z, Z_W_z])

            # Solve system
            try:
                coef_new = np.linalg.solve(A, b_rhs)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                coef_new = np.linalg.lstsq(A, b_rhs, rcond=None)[0]

            # Extract components
            beta_para_new = coef_new[:p_para]
            beta_smooth_new = coef_new[p_para:p_para + p_smooth_total]
            b_new = coef_new[p_para + p_smooth_total:]

            # Update linear predictor
            eta_new = X_parametric @ beta_para_new + X_smooth @ beta_smooth_new + Z @ b_new

            # Check convergence
            coef_change = np.linalg.norm(coef_new - np.concatenate([beta_para, beta_smooth, b]))
            if coef_change < tol_inner:
                converged_inner = True
                break

            # Update coefficients
            beta_para = beta_para_new
            beta_smooth = beta_smooth_new
            b = b_new
            eta = eta_new

        # Update variance components using REML
        # For simplicity, use method of moments estimate
        # More sophisticated: use REML optimization
        b_grouped = _split_random_effects(b, Z_info)
        Psi_list_new = []
        for i, b_group in enumerate(b_grouped):
            # Reshape to matrix (n_levels, dim)
            # Support both 'dim' and 'n_effects' keys for compatibility
            dim = Z_info[i].get('dim', Z_info[i].get('n_effects', 1))
            n_levels = len(b_group) // dim
            if len(b_group) % dim != 0:
                # Handle uneven split
                b_mat = b_group.reshape(-1, 1)
            else:
                b_mat = b_group.reshape(n_levels, dim)

            # Estimate variance-covariance
            Psi_new = np.cov(b_mat.T) if dim > 1 else np.array([[np.var(b_group)]])
            Psi_new = Psi_new + 1e-6 * np.eye(Psi_new.shape[0])  # Regularization
            Psi_list_new.append(Psi_new)

        # Update Psi
        if len(Psi_list_new) == 1:
            Psi = Psi_list_new[0]
        else:
            Psi = linalg.block_diag(*Psi_list_new)

        # Auto-select λ after a few iterations (when Psi estimate is stable)
        if auto_lambda and iter_outer == 2:
            if verbose:
                print("\nSelecting smoothing parameters via GCV...")

            from aurora.models.gamm.smoothing_selection import select_smoothing_gcv

            lambda_smooth = select_smoothing_gcv(
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

            # Rebuild Lambda with new smoothing parameters
            Lambda = _build_lambda_matrix(lambda_smooth, smooth_names, p_smooth_list)

            if verbose:
                print(f"Selected λ: {lambda_smooth}")

        # Check outer convergence
        Psi_change = np.linalg.norm(Psi - Psi_old, 'fro')
        if verbose:
            print(f"Outer iter {iter_outer + 1}: Psi change = {Psi_change:.6f}")

        if Psi_change < tol_outer:
            converged_outer = True
            break

    # Compute effective degrees of freedom for smooth terms
    edf_smooth_dict = {}
    offset = 0
    for i, name in enumerate(smooth_names):
        K_j = p_smooth_list[i]
        # Extract block for this smooth term
        idx = slice(offset, offset + K_j)
        S_j = S[idx, idx]
        lambda_j = lambda_smooth[name]

        # Smoother matrix for this term: (X_s^T W X_s + λS)^{-1} X_s^T W X_s
        XsWXs_j = X_smooth[:, idx].T @ W @ X_smooth[:, idx]
        A_j = XsWXs_j + lambda_j * S_j
        try:
            A_j_inv = np.linalg.inv(A_j)
            edf_j = np.trace(A_j_inv @ XsWXs_j)
        except np.linalg.LinAlgError:
            edf_j = np.nan

        edf_smooth_dict[name] = edf_j
        offset += K_j

    # Split beta_smooth into dictionary
    beta_smooth_dict = {}
    offset = 0
    for i, name in enumerate(smooth_names):
        K_j = p_smooth_list[i]
        beta_smooth_dict[name] = beta_smooth[offset:offset + K_j]
        offset += K_j

    # Build result dictionary
    result = {
        'beta_parametric': beta_para,
        'beta_smooth': beta_smooth_dict,
        'random_effects': b,
        'variance_components': Psi_list_new,
        'smoothing_parameters': lambda_smooth,
        'fitted_values': eta,
        'converged': converged_outer,
        'n_iterations_outer': iter_outer + 1,
        'edf_smooth': edf_smooth_dict,
    }

    return result


def _build_lambda_matrix(
    lambda_smooth: dict[str, float],
    smooth_names: list[str],
    p_smooth_list: list[int],
) -> NDArray[np.floating]:
    """Build block-diagonal Lambda matrix from smoothing parameters.

    Parameters
    ----------
    lambda_smooth : dict
        Smoothing parameters by term name.
    smooth_names : list
        Ordered list of smooth term names.
    p_smooth_list : list
        Number of basis functions for each smooth term.

    Returns
    -------
    Lambda : ndarray
        Block-diagonal matrix with λ_j I_K_j on diagonal.
    """
    blocks = []
    for name, K_j in zip(smooth_names, p_smooth_list):
        lambda_j = lambda_smooth[name]
        blocks.append(lambda_j * np.eye(K_j))
    return linalg.block_diag(*blocks)


def _split_random_effects(
    b: NDArray[np.floating],
    Z_info: list[dict],
) -> list[NDArray[np.floating]]:
    """Split random effects vector into groups.

    Parameters
    ----------
    b : ndarray
        Combined random effects vector.
    Z_info : list[dict]
        Random effects metadata.

    Returns
    -------
    b_grouped : list[ndarray]
        Random effects split by grouping variable.
    """
    b_grouped = []
    offset = 0
    for info in Z_info:
        # Support both 'n_levels' and 'n_groups' keys for compatibility
        n_levels = info.get('n_levels', info.get('n_groups', 1))
        dim = info.get('dim', info.get('n_effects', 1))
        size = n_levels * dim
        b_grouped.append(b[offset:offset + size])
        offset += size
    return b_grouped


__all__ = ['fit_pql_with_smooth']
