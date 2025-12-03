"""Penalized Quasi-Likelihood (PQL) for Generalized Linear Mixed Models.

This module implements PQL estimation for GLMMs, extending the Gaussian
GAMM implementation to non-Gaussian families (Poisson, Binomial, Gamma).

Mathematical Framework
----------------------
Consider a GLMM with observations y_ij for subject i, measurement j:

    g(E[Y_ij | b_i]) = X_ij^T β + Z_ij^T b_i

where:
    - g(·) is the link function
    - β are fixed effects coefficients
    - b_i ~ N(0, Ψ) are random effects
    - Y_ij follows an exponential family distribution

The goal is to estimate β and Ψ by maximizing the marginal likelihood:

    L(β, Ψ) = ∫ p(y | b, β) p(b | Ψ) db

Since this integral is intractable for non-Gaussian families, PQL uses an
approximation based on the quasi-likelihood and linearization.

PQL Algorithm (Breslow & Clayton, 1993)
----------------------------------------
The PQL algorithm alternates between two steps:

**Step 1: Update (β, b) given Ψ**

At iteration t, linearize the GLMM around current estimates:

1. Compute linear predictor:
   η_ij = X_ij^T β + Z_ij^T b_i

2. Compute mean and derivative:
   μ_ij = g^{-1}(η_ij)
   dμ/dη = [dη/dμ]^{-1} = [g'(μ_ij)]^{-1}

3. Working response (first-order Taylor approximation):
   z_ij = η_ij + (y_ij - μ_ij)(dμ/dη)|_{μ=μ_ij}
        = η_ij + (y_ij - μ_ij) / g'(μ_ij)

4. Working weights (inverse variance of pseudo-data):
   w_ij = [(dμ/dη)^2 / V(μ_ij)]|_{μ=μ_ij}
        = [g'(μ_ij)]^{-2} / V(μ_ij)

   where V(μ) is the variance function of the exponential family.

5. Solve weighted mixed model equations (Henderson, 1975):

   [X^T W X + λS    X^T W Z     ] [β]   [X^T W z]
   [Z^T W X         Z^T W Z + Ψ^{-1}] [b] = [Z^T W z]

   This gives updated estimates (β^{t+1}, b^{t+1}).

**Step 2: Update Ψ given (β, b)**

Update variance components using empirical covariance:

   Ψ^{t+1} = (1/m) Σ_{i=1}^m b_i b_i^T

where m is the number of subjects.

For numerical stability, we apply shrinkage (5% toward identity):

   Ψ = (1 - α)Ψ_emp + α(tr(Ψ_emp)/q)I_q

where q is the dimension of b_i and α = 0.05.

**Convergence**: Iterate until ||Ψ^{t+1} - Ψ^{t}|| / ||Ψ^{t}|| < ε

Statistical Properties
----------------------
**Asymptotic bias** (Breslow & Lin, 1995; Lin & Breslow, 1996):

- PQL estimates are asymptotically unbiased as cluster size n_i → ∞
- For small clusters or binary data with rare events, bias can be substantial
- Bias is O(1/n_i) for large n_i

**Compared to other methods**:

- Laplace approximation: More accurate but computationally expensive
- Adaptive Gaussian quadrature: Gold standard but prohibitive for large problems
- PQL: Fast and scalable, good for exploratory analysis

**When to use PQL**:

✓ Large cluster sizes (n_i > 5-10)
✓ Moderate random effect variance (σ_b < 1)
✓ Poisson data without excessive zeros
✓ Binomial data with balanced proportions (not too close to 0 or 1)

**When to avoid PQL**:

✗ Small cluster sizes (n_i < 5)
✗ Binary data with rare events (p < 0.1 or p > 0.9)
✗ Large random effect variance (σ_b > 1.5)
✗ High-precision inference required

Implementation Details
----------------------
**Numerical stability enhancements**:

1. Add small constant (1e-10) to avoid division by zero in dμ/dη
2. Shrinkage in variance component estimation
3. Enforce positive-definiteness of Ψ via eigenvalue thresholding
4. Cholesky decomposition with fallback to lstsq if singular

**Computational complexity**:

Per iteration:
    - Inner loop: O(n(p^2 + q^2) + q^3) for solving mixed model equations
    - Outer loop: O(q^3) for variance component update

Total: O(T_outer × T_inner × [n(p^2 + q^2) + q^3])

where T_outer ≈ 5-20, T_inner ≈ 5-10 typically.

References
----------
**Core algorithm**:

- Breslow, N. E., & Clayton, D. G. (1993). "Approximate inference in generalized
  linear mixed models." *Journal of the American Statistical Association*,
  88(421), 9-25. https://doi.org/10.1080/01621459.1993.10594284

- Schall, R. (1991). "Estimation in generalized linear models with random effects."
  *Biometrika*, 78(4), 719-727. https://doi.org/10.1093/biomet/78.4.719

**Bias correction**:

- Breslow, N. E., & Lin, X. (1995). "Bias correction in generalised linear mixed
  models with a single component of dispersion." *Biometrika*, 82(1), 81-91.
  https://doi.org/10.1093/biomet/82.1.81

- Lin, X., & Breslow, N. E. (1996). "Bias correction in generalized linear mixed
  models with multiple components of dispersion." *Journal of the American
  Statistical Association*, 91(435), 1007-1016.
  https://doi.org/10.1080/01621459.1996.10476971

**Extensions and applications**:

- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). CRC Press. Chapter 6: GAMMs.

- Pinheiro, J. C., & Bates, D. M. (2000). *Mixed-Effects Models in S and S-PLUS*.
  Springer. Chapter 7: Nonlinear mixed-effects models.

**Software comparison**:

- R package lme4: Uses Laplace approximation instead of PQL
  (Bates et al., 2015, JSS, 67(1))

- R package MASS::glmmPQL: Classic PQL implementation
  (Venables & Ripley, 2002)

- SAS PROC GLIMMIX: Offers both PQL and Laplace

See Also
--------
aurora.models.gamm.fitting.fit_gamm_gaussian : REML for Gaussian GAMM
aurora.models.gamm.laplace : Laplace approximation for GLMM (planned)
aurora.models.glm.fitting.fit_glm : GLM without random effects

Notes
-----
For details on the mathematical derivations and proofs, see REFERENCES.md
in the repository root.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

if TYPE_CHECKING:
    from numpy.typing import NDArray

from aurora.distributions.base import Family
from aurora.distributions.families import (
    BinomialFamily,
    GammaFamily,
    GaussianFamily,
    PoissonFamily,
)


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
        Whether to update variance components (False for fixed Ψ)

    Returns
    -------
    result : PQLResult
        Fitted PQL model

    Notes
    -----
    For Gaussian family, this reduces to standard LMM which could be solved
    more efficiently with REML. Use fit_gamm_gaussian() instead for Gaussian data.

    The algorithm iterates between:
    1. Updating (β, b) given Ψ via weighted least squares
    2. Updating Ψ given (β, b) via empirical covariance

    Convergence is determined by relative change in Ψ falling below tol_outer.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.gamm.pql import fit_pql
    >>> # Generate Poisson data with random intercepts
    >>> n_groups, n_per_group = 5, 20
    >>> n = n_groups * n_per_group
    >>> groups = np.repeat(np.arange(n_groups), n_per_group)
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> Z = np.eye(n_groups)[groups]  # Random intercept design
    >>> b_true = np.random.randn(n_groups) * 0.5
    >>> eta = X @ [1.0, 0.3] + Z @ b_true
    >>> y = np.random.poisson(np.exp(eta))
    >>> result = fit_pql(X, Z, y, family='poisson')
    >>> print(result.converged)
    True
    """
    # Get family object
    if isinstance(family, str):
        family_obj = _get_family(family)
    else:
        family_obj = family

    n, p = X.shape
    n, q = Z.shape

    # Initialize penalty matrix
    if S is None:
        S = np.zeros((p, p))

    # Initialize variance-covariance matrix
    # Infer n_effects from psi_init shape (Phase 1.3 fix)
    if psi_init is None:
        n_effects = 1  # Default to random intercept
        psi = np.eye(n_effects)
    else:
        n_effects = psi_init.shape[0]  # Infer from initialization
        psi = psi_init.copy()

    # Initialize coefficients
    beta = np.zeros(p)
    b = np.zeros(q)

    # Convergence tracking
    converged = False
    n_iter_inner_total = 0

    # Outer loop: update variance components
    for iter_outer in range(maxiter_outer):
        psi_old = psi.copy()

        # Inner loop: update fixed and random effects
        for iter_inner in range(maxiter_inner):
            beta_old = beta.copy()
            b_old = b.copy()

            # Compute linear predictor and fitted values
            eta = X @ beta + Z @ b
            mu = family_obj.default_link.inverse(eta)

            # Compute working response and weights
            # derivative() returns dη/dμ, we need dμ/dη = 1/(dη/dμ)
            deta_dmu = family_obj.default_link.derivative(mu)
            dmu_deta = 1.0 / (deta_dmu + 1e-10)  # Add small epsilon for stability
            var_mu = family_obj.variance(mu)
            
            # Ensure numerical stability - clip variance to avoid division issues
            var_mu = np.clip(var_mu, 1e-10, 1e10)
            dmu_deta = np.clip(dmu_deta, -1e10, 1e10)

            # Working response: z = η + (y - μ) / (dμ/dη)
            z = eta + (y - mu) / dmu_deta
            
            # Handle NaN/Inf in working response
            z = np.nan_to_num(z, nan=0.0, posinf=1e10, neginf=-1e10)

            # Weights: w = (dμ/dη)² / Var(μ)
            w = dmu_deta**2 / var_mu
            
            # Ensure weights are positive and finite
            w = np.clip(w, 1e-10, 1e10)
            w = np.nan_to_num(w, nan=1e-10, posinf=1e10, neginf=1e-10)

            # Solve weighted mixed model equations
            beta, b, sigma2 = _solve_pql_equations(X, Z, z, w, S, lambda_, psi)
            
            # Check for NaN in coefficients - indicates divergence
            if np.any(np.isnan(beta)) or np.any(np.isnan(b)):
                raise ValueError("PQL iteration diverged (NaN in coefficients). "
                                "Try different starting values or increase regularization.")

            # Check inner convergence
            beta_change = np.max(np.abs(beta - beta_old))
            b_change = np.max(np.abs(b - b_old))
            n_iter_inner_total += 1

            if beta_change < tol_inner and b_change < tol_inner:
                break

        # Update variance components (empirical covariance)
        if update_psi:
            psi = _update_variance_components(X, Z, z, w, beta, b, n_effects)

            # Check outer convergence
            psi_change = np.max(np.abs(psi - psi_old) / (np.abs(psi_old) + 1e-10))
            if psi_change < tol_outer:
                converged = True
                break

    # Compute final fitted values
    eta = X @ beta + Z @ b
    mu = family_obj.default_link.inverse(eta)

    # Compute deviance
    deviance = family_obj.deviance(y, mu)

    # Compute approximate log-likelihood
    # For PQL, this is approximate (not exact marginal likelihood)
    log_likelihood = np.sum(family_obj.log_likelihood(y, mu))

    return PQLResult(
        beta=beta,
        b=b,
        psi=psi,
        sigma2=sigma2,
        fitted_values=mu,
        linear_predictor=eta,
        converged=converged,
        n_iter_outer=iter_outer + 1,
        n_iter_inner=n_iter_inner_total,
        deviance=deviance,
        log_likelihood=log_likelihood,
    )


def _solve_pql_equations(
    X: np.ndarray,
    Z: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    lambda_: float,
    psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve weighted mixed model equations for PQL.

    Solves:
        [X'WX + λS    X'WZ      ] [β]   [X'Wz]
        [Z'WX         Z'WZ + Ψ⁻¹] [b] = [Z'Wz]

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Fixed effects design matrix
    Z : ndarray, shape (n, q)
        Random effects design matrix
    z : ndarray, shape (n,)
        Working response
    w : ndarray, shape (n,)
        Weights
    S : ndarray, shape (p, p)
        Penalty matrix
    lambda_ : float
        Smoothing parameter
    psi : ndarray, shape (n_effects, n_effects)
        Variance-covariance matrix

    Returns
    -------
    beta : ndarray, shape (p,)
        Fixed effect coefficients
    b : ndarray, shape (q,)
        Random effect coefficients
    sigma2 : float
        Residual variance estimate
    """
    n, p = X.shape
    n, q = Z.shape

    # Build weighted design matrices
    W_sqrt = np.sqrt(w)
    X_weighted = W_sqrt[:, None] * X
    Z_weighted = W_sqrt[:, None] * Z
    z_weighted = W_sqrt * z

    # Compute inverse of Ψ (handle block structure later in Phase 3)
    # For now, assume single block
    n_effects = psi.shape[0]
    n_groups = q // n_effects

    # Validate psi before inversion
    if not np.all(np.isfinite(psi)):
        import warnings
        warnings.warn("NaN/Inf in psi matrix, using identity")
        psi = np.eye(n_effects)
    
    # Ensure psi is positive definite before inversion
    try:
        psi_inv = linalg.inv(psi)
    except linalg.LinAlgError:
        # Regularize and retry
        psi_reg = psi + 1e-4 * np.eye(n_effects)
        try:
            psi_inv = linalg.inv(psi_reg)
        except linalg.LinAlgError:
            psi_inv = np.eye(n_effects)
    
    # Handle NaN/Inf in inverse
    if not np.all(np.isfinite(psi_inv)):
        psi_inv = np.eye(n_effects)

    # Expand Ψ⁻¹ to block-diagonal form
    psi_inv_expanded = np.kron(np.eye(n_groups), psi_inv)

    # Build augmented system
    #  [X'WX + λS    X'WZ     ]
    #  [Z'WX        Z'WZ + Ψ⁻¹]
    XWX = X_weighted.T @ X_weighted + lambda_ * S
    XWZ = X_weighted.T @ Z_weighted
    ZWX = Z_weighted.T @ X_weighted
    ZWZ = Z_weighted.T @ Z_weighted + psi_inv_expanded

    # Right-hand side
    XWz = X_weighted.T @ z_weighted
    ZWz = Z_weighted.T @ z_weighted

    # Combine into block matrix
    A = np.block([[XWX, XWZ], [ZWX, ZWZ]])
    rhs = np.concatenate([XWz, ZWz])

    # Solve via Cholesky decomposition
    try:
        L = linalg.cholesky(A, lower=True)
        coef = linalg.cho_solve((L, True), rhs)
    except linalg.LinAlgError:
        # Fall back to least squares if Cholesky fails
        coef = linalg.lstsq(A, rhs)[0]

    beta = coef[:p]
    b = coef[p:]

    # Estimate residual variance
    fitted = X @ beta + Z @ b
    residuals = z - fitted
    sigma2 = np.sum(w * residuals**2) / n

    return beta, b, sigma2


def _update_variance_components(
    X: np.ndarray,
    Z: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    beta: np.ndarray,
    b: np.ndarray,
    n_effects: int,
    method: str = "empirical",
) -> np.ndarray:
    """Update variance-covariance matrix for random effects.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Fixed effects design matrix
    Z : ndarray, shape (n, q)
        Random effects design matrix
    z : ndarray, shape (n,)
        Working response
    w : ndarray, shape (n,)
        Weights
    beta : ndarray, shape (p,)
        Current fixed effects
    b : ndarray, shape (q,)
        Current random effects
    n_effects : int
        Number of random effects per group
    method : str, default='empirical'
        Update method ('empirical' for now)

    Returns
    -------
    psi : ndarray, shape (n_effects, n_effects)
        Updated variance-covariance matrix

    Notes
    -----
    Phase 1.4 enhancement: Add shrinkage for numerical stability
    """
    q = Z.shape[1]
    n_groups = q // n_effects

    # Reshape b into groups × effects matrix
    b_matrix = b.reshape(n_groups, n_effects)
    
    # Handle NaN/Inf in random effects
    if not np.all(np.isfinite(b_matrix)):
        import warnings
        warnings.warn("NaN/Inf detected in random effects, using identity covariance")
        return np.eye(n_effects)

    if method == "empirical":
        # Compute empirical covariance
        psi_emp = (b_matrix.T @ b_matrix) / n_groups
        
        # Handle NaN/Inf in covariance matrix
        if not np.all(np.isfinite(psi_emp)):
            import warnings
            warnings.warn("NaN/Inf in empirical covariance, using identity")
            return np.eye(n_effects)

        # Phase 1.4: Add shrinkage toward identity (5%)
        shrinkage = 0.05
        trace_avg = np.trace(psi_emp) / n_effects
        if not np.isfinite(trace_avg) or trace_avg <= 0:
            trace_avg = 1.0
        psi_emp = (1 - shrinkage) * psi_emp + shrinkage * trace_avg * np.eye(
            n_effects
        )

        # Phase 1.4: Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(psi_emp)
        if not np.all(np.isfinite(eigvals)):
            return np.eye(n_effects)
        if np.min(eigvals) < 1e-6:
            psi_emp = psi_emp + (1e-6 - np.min(eigvals)) * np.eye(n_effects)

        return psi_emp

    else:
        raise ValueError(f"Unknown method: {method}")


def _get_family(family_name: str) -> Family:
    """Get family object from string name.

    Parameters
    ----------
    family_name : str
        Family name ('gaussian', 'poisson', 'binomial', 'gamma')

    Returns
    -------
    family : Family
        Family object

    Raises
    ------
    ValueError
        If family name is unknown
    """
    families = {
        "gaussian": GaussianFamily,
        "poisson": PoissonFamily,
        "binomial": BinomialFamily,
        "gamma": GammaFamily,
    }

    if family_name.lower() not in families:
        raise ValueError(
            f"Unknown family '{family_name}'. " f"Must be one of {list(families.keys())}"
        )

    return families[family_name.lower()]()


# ============================================================================
# High-Level Interface for GAMM
# ============================================================================


def fit_pql_gamm(
    X_parametric: np.ndarray,
    X_smooth: dict[str, np.ndarray] | None,
    Z: np.ndarray,
    Z_info: list[dict],
    y: np.ndarray,
    family: Family | str,
    S_smooth: dict[str, np.ndarray] | None = None,
    lambda_smooth: dict[str, float] | None = None,
    covariance: str = "unstructured",
    maxiter_outer: int = 20,
    maxiter_inner: int = 10,
    tol_outer: float = 1e-4,
    tol_inner: float = 1e-6,
    backend: str = "numpy",
    device: str | None = None,
):
    """Fit non-Gaussian GAMM using Penalized Quasi-Likelihood.

    This is the high-level interface that wraps fit_pql() and converts
    PQLResult to GAMMResult for consistency with Gaussian GAMM.

    Parameters
    ----------
    X_parametric : ndarray, shape (n, p_para)
        Parametric fixed effects design matrix
    X_smooth : dict[str, ndarray], optional
        Smooth term basis matrices (Phase 2: not yet supported)
    Z : ndarray, shape (n, q)
        Random effects design matrix
    Z_info : list of dict
        Metadata about Z structure from construct_Z_matrix()
    y : ndarray, shape (n,)
        Response vector
    family : Family or str
        Distribution family ('poisson', 'binomial')
    S_smooth : dict[str, ndarray], optional
        Penalty matrices for smooth terms (Phase 2)
    lambda_smooth : dict[str, float], optional
        Smoothing parameters (Phase 2)
    covariance : str, default='unstructured'
        Covariance structure for random effects
    maxiter_outer : int, default=20
        Maximum outer iterations (variance updates)
    maxiter_inner : int, default=10
        Maximum inner iterations per outer (coefficient updates)
    tol_outer : float, default=1e-4
        Convergence tolerance for variance components
    tol_inner : float, default=1e-6
        Convergence tolerance for coefficients
    backend : str, default='numpy'
        Computational backend (currently only 'numpy' supported for PQL)
    device : str, optional
        Device for computation (ignored for numpy backend)

    Returns
    -------
    result : GAMMResult
        Fitted GAMM result with PQL estimates

    Raises
    ------
    NotImplementedError
        If smooth terms are provided (Phase 2 feature)
        If multiple random effect terms are provided (Phase 3 feature)
        If non-numpy backend is requested

    Notes
    -----
    PQL Algorithm:
    1. Initialize β = 0, b = 0, Ψ = I
    2. Outer loop (variance components):
       a. Inner loop (fixed/random effects):
          - Compute working response: z = η + (y - μ)/g'(μ)
          - Compute weights: w = (g'(μ))²/Var(μ)
          - Solve weighted mixed model equations
       b. Update Ψ via empirical covariance of b
    3. Convert PQLResult to GAMMResult

    The approximation quality depends on:
    - Group sizes (larger is better, recommend n_per_group > 5)
    - Magnitude of random effects (smaller is better, recommend σ_b < 1)
    - Sparsity of response (fewer zeros is better for Poisson)

    References
    ----------
    Breslow & Clayton (1993). "Approximate inference in generalized
    linear mixed models". JASA 88(421).

    Examples
    --------
    >>> import pandas as pd
    >>> from aurora.models.gamm import fit_gamm
    >>> # Poisson GAMM with random intercepts
    >>> df = pd.DataFrame({
    ...     'count': [5, 3, 8, 2, 7, 4, 6, 9],
    ...     'x': [1.2, 0.8, 1.5, 0.5, 1.3, 0.9, 1.1, 1.6],
    ...     'group': [0, 0, 1, 1, 2, 2, 3, 3]
    ... })
    >>> result = fit_gamm(
    ...     formula='count ~ x + (1 | group)',
    ...     data=df,
    ...     family='poisson'
    ... )
    >>> print(result.converged)
    True
    """
    # Validate backend
    if backend != "numpy":
        raise NotImplementedError(
            f"Backend '{backend}' not yet supported for PQL. " "Use backend='numpy'."
        )

    # Get family object
    if isinstance(family, str):
        family_obj = _get_family(family)
    else:
        family_obj = family

    # Phase 1: No smooth terms yet
    if X_smooth is not None:
        raise NotImplementedError(
            "Smooth terms in PQL not yet implemented (Phase 2). "
            "Use X_parametric only for now."
        )

    X_combined = X_parametric
    p_parametric = X_parametric.shape[1]

    # Phase 1: Single random effect term only
    if len(Z_info) != 1:
        raise NotImplementedError(
            "Multiple random effect terms not yet supported in PQL (Phase 3). "
            "Use single random effect term (e.g., '(1|subject)')."
        )

    n_effects = Z_info[0]["n_effects"]

    # Build penalty matrix (Phase 2)
    S_combined = None
    lambda_combined = 0.0

    # Get covariance structure (Phase 2.3 / Phase 5)
    from aurora.models.gamm.covariance import get_covariance_structure

    cov_structure = get_covariance_structure(covariance)

    # Initialize Ψ based on covariance structure
    if covariance == "identity":
        psi_init = np.eye(n_effects)
    elif covariance == "diagonal":
        psi_init = np.eye(n_effects)
    else:  # unstructured
        psi_init = np.eye(n_effects)

    # Call core PQL fitting
    pql_result = fit_pql(
        X=X_combined,
        Z=Z,
        y=y,
        family=family_obj,
        S=S_combined,
        lambda_=lambda_combined,
        psi_init=psi_init,
        maxiter_outer=maxiter_outer,
        maxiter_inner=maxiter_inner,
        tol_outer=tol_outer,
        tol_inner=tol_inner,
        update_psi=True,
    )

    # Convert PQLResult to GAMMResult
    from aurora.models.gamm.design import extract_random_effects
    from aurora.models.gamm.fitting import GAMMResult

    # Extract random effects by group
    random_effects = extract_random_effects(pql_result.b, Z_info)

    # Compute EDF (simplified for now)
    n = len(y)
    q = Z.shape[1]
    edf_parametric = float(p_parametric)
    edf_random = float(q)  # Approximate
    edf_total = edf_parametric + edf_random

    # Compute AIC/BIC (using approximate log-likelihood)
    log_likelihood = pql_result.log_likelihood
    aic = -2 * log_likelihood + 2 * edf_total
    bic = -2 * log_likelihood + edf_total * np.log(n)

    # Count groups
    n_groups = Z_info[0]["n_groups"]

    return GAMMResult(
        coefficients=pql_result.beta,
        beta_parametric=pql_result.beta,
        beta_smooth={},  # Phase 2
        random_effects=random_effects,
        variance_components=[pql_result.psi],
        covariance_params=None,  # PQL doesn't use structured covariance (yet)
        residual_variance=pql_result.sigma2,
        smoothing_parameters=None,
        edf_total=edf_total,
        edf_parametric=edf_parametric,
        edf_smooth={},
        fitted_values=pql_result.fitted_values,
        residuals=y - pql_result.fitted_values,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        converged=pql_result.converged,
        n_iterations=pql_result.n_iter_outer,
        n_obs=n,
        n_groups=n_groups,
        family=(
            family
            if isinstance(family, str)
            else family.__class__.__name__.lower().replace("family", "")
        ),
        _X_parametric=X_parametric,
        _X_smooth=None,
        _Z=Z,
        _Z_info=Z_info,
        _y=y,
    )
