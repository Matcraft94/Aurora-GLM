# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Generalized Additive Mixed Models (GAMM) for Gaussian Responses.

Mathematical Framework
----------------------
A GAMM extends GAM by adding random effects to account for correlation
and hierarchical structure in data. For Gaussian responses, GAMM reduces
to a Linear Mixed Model (LMM) with smooth terms.

Model Structure
---------------
The Gaussian GAMM has the form:

    y_i = β₀ + f₁(x_{i1}) + f₂(x_{i2}) + ... + f_p(x_{ip})
          + X_i β_p + Z_i b + ε_i

where:
    - y_i: Response for observation i
    - f_j(·): Smooth functions (non-parametric effects)
    - X_i: Parametric covariates with fixed effects β_p
    - Z_i: Random effects design row
    - b ~ N(0, Ψ): Random effects (group-level deviations)
    - ε_i ~ N(0, σ²): Independent errors

**In matrix form**:

    y = Xβ + Zb + ε

where:
    - X = [X_fixed | B₁ | B₂ | ... | B_p]: Design matrix
      - X_fixed: Parametric covariates
      - B_j: Basis matrix for j-th smooth term
    - β = [β_p; β₁; β₂; ...; β_p]: Combined fixed effects
    - Z: Random effects design matrix
    - b ~ N(0, Ψ): Random effects vector
    - ε ~ N(0, σ²I): Error vector

Penalized Least Squares Formulation
------------------------------------
The smooth functions are represented using basis expansions:

    f_j(x) = Σₖ β_{jk} φ_{jk}(x)

where φ_{jk} are basis functions (typically B-splines).

To prevent overfitting, we penalize roughness:

    ℓ_p(β, b; λ, Ψ, σ²) = -½ [(y - Xβ - Zb)^T (y - Xβ - Zb) / σ²
                              + Σⱼ λⱼ β_j^T S_j β_j
                              + b^T Ψ⁻¹ b]

where:
    - λⱼ: Smoothing parameter for j-th smooth term
    - S_j: Penalty matrix for j-th smooth term
    - Ψ: Variance-covariance matrix of random effects

**Unified penalty interpretation**:
Random effects are equivalent to heavily penalized fixed effects:

    b ~ N(0, Ψ)  ⟺  Penalty b^T Ψ⁻¹ b with λ = 1

This unifies smoothing and random effects in a single framework.

REML Estimation
---------------
**Restricted Maximum Likelihood** (REML) provides unbiased estimates of
variance parameters by maximizing the likelihood of a linear transformation
of y that is invariant to β.

### REML Criterion

For fixed (λ, Ψ, σ²), the REML log-likelihood is:

    ℓ_R(λ, Ψ, σ²) = -½ [log|V| + log|X^T V⁻¹ X| + r^T V⁻¹ r]

where:
    - V = σ²I + ZΨZ^T: Marginal covariance of y
    - r = y - X β̂: Residuals with β̂ = (X^T V⁻¹ X)⁻¹ X^T V⁻¹ y

**Components**:
1. log|V|: Variance determinant (penalizes complexity)
2. log|X^T V⁻¹ X|: Fixed effects information (adjusts for β uncertainty)
3. r^T V⁻¹ r: Penalized residual sum of squares

### Optimization

**Two-stage optimization**:

**Outer loop**: Optimize variance parameters (λ, Ψ, σ²)
    - Use Newton-Raphson or L-BFGS on ℓ_R
    - Derivatives via automatic differentiation or finite differences

**Inner loop**: For given (λ, Ψ, σ²), solve for (β, b)
    - Penalized least squares (closed-form solution)
    - Or equivalently: Henderson's mixed model equations

### Henderson's Mixed Model Equations

For given variance parameters, the BLUP (Best Linear Unbiased Predictor)
satisfies:

    [ X^T X     X^T Z   ] [ β̂ ]   [ X^T y ]
    [ Z^T X   Z^T Z + Ψ⁻¹σ² ] [ b̂ ] = [ Z^T y ]

**Properties**:
- β̂ are generalized least squares estimates
- b̂ are empirical Bayes estimates (shrunk toward zero)
- System is sparse when Z is sparse (hierarchical data)

Smooth Terms as Penalized Fixed Effects
----------------------------------------
Each smooth term f_j(x) = B_j β_j with penalty λ_j β_j^T S_j β_j can be
incorporated into the mixed model equations.

**Augmented system** (Wood, 2011):

    [ X^T X + Λ     X^T Z   ] [ β̂ ]   [ X^T y ]
    [ Z^T X      Z^T Z + Ψ⁻¹σ² ] [ b̂ ] = [ Z^T y ]

where Λ = block_diag(0, λ₁S₁, λ₂S₂, ..., λ_pS_p) includes smoothing penalties.

**Interpretation**: Smooth terms are fixed effects with structured penalties,
while random effects are \"infinitely penalized\" fixed effects (λ → ∞).

Effective Degrees of Freedom
-----------------------------
EDF measures model complexity, accounting for both smoothing and random effects.

### For Smooth Terms

    edf_j = tr[(X^T V⁻¹ X + Λ)⁻¹ X^T V⁻¹ X]_j

where the trace is taken over the block corresponding to smooth term j.

**Interpretation**:
- edf_j = K_j (number of basis functions): No penalization
- edf_j = 1: Linear function (heavy penalization)
- edf_j ∈ (1, K_j): Intermediate smoothness

### For Random Effects

    edf_random = tr[Z(Z^T Z + Ψ⁻¹σ²)⁻¹ Z^T]

**Interpretation**: Number of \"independent\" random effects estimated
(shrinkage reduces EDF below nominal dimension).

### Total EDF

    edf_total = tr(H)

where H is the hat matrix:

    ŷ = H y,  H = [X  Z] [(X^T X + Λ)  X^T Z   ]⁻¹ [X^T]
                          [Z^T X      Z^T Z + Ψ⁻¹σ²]    [Z^T]

Computational Algorithms
------------------------
### Direct Solution (Small to Medium Problems)

For n < 10,000, solve Henderson's equations directly:

**Steps**:
1. Form augmented system (p + q) × (p + q)
2. Cholesky decomposition: A = LL^T
3. Solve: L(L^T x) = b via back-substitution

**Cost**: O((p+q)³) where p = dim(β), q = dim(b)

**Advantages**: Exact, stable, simple

**Disadvantages**: O(n²) for random effects (dense Z^T Z)

### Sparse Methods (Large Problems)

When Z is sparse (hierarchical/grouped data):

**Techniques**:
1. **Sparse Cholesky**: Exploit sparsity pattern of augmented system
   - Cost: O(nnz × fill) where fill depends on elimination ordering
   - Libraries: SuiteSparse, Eigen

2. **Iterative solvers**: Conjugate gradient (CG)
   - Cost: O(k × nnz) for k iterations
   - Preconditioner: Incomplete Cholesky

3. **Block updates**: Exploit structure (nested random effects)
   - Update each level separately
   - Leverage conditional independence

### Smoothing Parameter Selection

**GCV (Generalized Cross-Validation)**:

    GCV(λ) = (n / (n - edf)²) × RSS

**REML (Preferred)**:
- Jointly optimize λ and Ψ via ℓ_R
- More stable for small samples
- Accounts for uncertainty in fixed effects

**Grid search**:
- Evaluate ℓ_R on grid {λ_min, ..., λ_max}
- Computationally expensive for multiple smooths

**Gradient-based**:
- Compute ∇_λ ℓ_R via automatic differentiation
- Use L-BFGS or Newton-Raphson
- Faster convergence, requires derivatives

Model Selection
---------------
### Akaike Information Criterion (AIC)

    AIC = -2ℓ + 2 × edf_total

Penalizes complexity via EDF.

### Bayesian Information Criterion (BIC)

    BIC = -2ℓ + log(n) × edf_total

Stronger penalty for large n.

**REML-based AIC/BIC**:
Use REML log-likelihood ℓ_R instead of ML when comparing models
with different variance structures.

### Likelihood Ratio Tests

For nested models M₀ ⊂ M₁:

    LRT = 2(ℓ_R(M₁) - ℓ_R(M₀)) ~ χ²_Δedf

where Δedf = edf_total(M₁) - edf_total(M₀).

**Caveats**:
- Variance components on boundary (σ² = 0) → mixture of χ²
- Smoothing parameters: penalized likelihood, not nested

Numerical Stability
-------------------
**Challenges**:

1. **Ill-conditioning**: X^T X + Λ may have large condition number
   - Solution: QR decomposition instead of Cholesky
   - Scaling: Standardize columns of X

2. **Variance parameter boundaries**: σ², Ψ must be positive-definite
   - Reparameterize: log(σ²), Cholesky of Ψ
   - Constrained optimization

3. **Computational cost**: O((p+q)³) for large q
   - Sparse methods essential for large random effects
   - Approximations: Laplace, expectation propagation

Integration with GAM
--------------------
This module seamlessly integrates:

1. **Smooth terms from GAM**: Basis matrices B_j, penalty matrices S_j
2. **Random effects**: Design matrix Z, variance components Ψ
3. **Unified estimation**: REML optimizes {λ, Ψ, σ²} jointly
4. **Prediction**: ŷ_new = X_new β̂ + Z_new b̂

**Advantages of integration**:
- Smooth functions + random effects in one model
- Avoids two-stage estimation (fit smooth, then add random effects)
- Automatic smoothing parameter selection

Applications in Aurora-GLM
---------------------------
Gaussian GAMM is used for:

1. **Longitudinal data**: Repeated measures with smooth time trends
   - Example: Patient outcomes over time with patient random intercepts

2. **Spatial data**: Smooth spatial surfaces + area random effects
   - Example: Disease mapping with spatial smooth + regional effects

3. **Hierarchical smoothing**: Different smooths per group
   - Example: Growth curves varying by school

4. **Large datasets**: Sparse random effects structure
   - Example: Students within classes within schools

Implementation Notes
--------------------
**Variance parameter representation**:
- Variance components stored as list of covariance matrices
- Single random intercept: Ψ = [σ_b²] (1×1 matrix)
- Random intercept + slope: Ψ = 2×2 matrix
- Multiple groupings: List with one matrix per term

**Effective degrees of freedom**:
- Computed via hat matrix trace
- Accounts for both smoothing and shrinkage
- Used for AIC/BIC calculation

**REML vs ML**:
- REML for variance component estimation (unbiased)
- ML for fixed effects inference (conditional on variance)
- Both available, REML default

References
----------
**Core GAMM theory**:

- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). CRC Press. Chapter 6: GAMMs.
  https://doi.org/10.1201/9781315370279
  (Comprehensive treatment, unifies smoothing and random effects)

- Ruppert, D., Wand, M. P., & Carroll, R. J. (2003). *Semiparametric Regression*.
  Cambridge University Press. Chapter 11: Mixed Model Representation.
  (Mathematical foundations of penalized splines as mixed models)

**REML estimation**:

- Patterson, H. D., & Thompson, R. (1971). \"Recovery of inter-block information
  when block sizes are unequal.\" *Biometrika*, 58(3), 545-554.
  https://doi.org/10.1093/biomet/58.3.545
  (Original REML paper)

- Harville, D. A. (1977). \"Maximum likelihood approaches to variance component
  estimation and to related problems.\" *Journal of the American Statistical
  Association*, 72(358), 320-338.
  (ML vs REML comparison)

**Mixed model computation**:

- Henderson, C. R. (1975). \"Best linear unbiased estimation and prediction
  under a selection model.\" *Biometrics*, 31(2), 423-447.
  (Mixed model equations, BLUP theory)

- Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). \"Fitting linear
  mixed-effects models using lme4.\" *Journal of Statistical Software*, 67(1), 1-48.
  https://doi.org/10.18637/jss.v067.i01
  (Practical implementation in R, computational strategies)

**Smoothing parameter selection**:

- Wood, S. N. (2011). \"Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.\"
  *Journal of the Royal Statistical Society: Series B*, 73(1), 3-36.
  https://doi.org/10.1111/j.1467-9868.2010.00749.x
  (Efficient REML for GAM/GAMM, Newton method with stable computation)

- Wahba, G. (1985). \"A comparison of GCV and GML for choosing the smoothing
  parameter in the generalized spline smoothing problem.\" *Annals of Statistics*,
  13(4), 1378-1402.
  (Theory comparing GCV and REML/ML)

**Computational methods**:

- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.).
  Johns Hopkins University Press. Chapter 11: Least Squares Problems.
  (Numerical linear algebra for mixed models)

- Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM.
  https://doi.org/10.1137/1.9780898718881
  (Sparse Cholesky for large mixed models)

**Degrees of freedom**:

- Hodges, J. S., & Sargent, D. J. (2001). \"Counting degrees of freedom in
  hierarchical and other richly-parameterised models.\" *Biometrika*, 88(2), 367-379.
  https://doi.org/10.1093/biomet/88.2.367
  (EDF definitions for mixed models)

See Also
--------
aurora.models.gamm.pql : PQL for non-Gaussian GAMM
aurora.models.gamm.estimation : REML estimation functions
aurora.models.gam.fitting : GAM fitting (no random effects)
aurora.models.gamm.random_effects : Random effects structures

Notes
-----
For detailed mathematical derivations, see REFERENCES.md in the repository root.

Gaussian GAMM unifies smooth regression and mixed models in an elegant framework.
The connection between penalized regression and random effects (via REML) allows
automatic selection of both smoothing parameters and variance components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

from aurora.models.gamm.covariance import get_covariance_structure
from aurora.models.gamm.estimation import (
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
    variance_components : list[ndarray]
        List of variance-covariance matrices Ψ, one per random effect term.
        For single term models, this is a list with one element.
    covariance_params : list[ndarray] | None
        List of raw covariance parameters for each random effect term.
        For structured covariance (AR1, CS, etc.), contains transformed parameters.
        For example, AR1 params are [log(σ²), arctanh(ρ)].
        None for unstructured covariance.
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
    variance_components: list[NDArray[np.floating]]
    covariance_params: list[NDArray[np.floating]] | None
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

    def predict(self, include_random: bool = True) -> NDArray[np.floating]:
        """Make predictions using the fitted model.

        Parameters
        ----------
        include_random : bool, default=True
            Whether to include random effects in predictions.
            If True, returns fitted values (fixed + random effects).
            If False, returns only fixed effects (population-level predictions).

        Returns
        -------
        predictions : ndarray
            Model predictions.

        Examples
        --------
        >>> # Population-level predictions (fixed effects only)
        >>> pred_fixed = result.predict(include_random=False)
        >>>
        >>> # Individual predictions (fixed + random effects)
        >>> pred_full = result.predict(include_random=True)
        """
        if include_random:
            return self.fitted_values
        else:
            # Only fixed effects
            if self._X_parametric is not None:
                return self._X_parametric @ self.beta_parametric
            else:
                raise ValueError(
                    "Cannot compute fixed-only predictions: "
                    "_X_parametric not stored in result"
                )

    def summary(self) -> str:
        """Generate a formatted summary of the model fit.

        Returns
        -------
        summary_str : str
            Formatted summary string with fixed effects, random effects,
            and model fit statistics.

        Examples
        --------
        >>> result = fit_gamm(formula='y ~ x + (1|subject)', data=df)
        >>> print(result.summary())
        """
        lines = []
        lines.append("=" * 75)
        lines.append("Generalized Additive Mixed Model (GAMM)")
        lines.append("=" * 75)
        lines.append(f"Family: {self.family}")
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of groups: {self.n_groups}")
        lines.append("")

        # Fixed Effects
        lines.append("Fixed Effects (Parametric):")
        lines.append("-" * 75)
        lines.append(f"{'Parameter':<25} {'Estimate':>12}")
        lines.append("-" * 75)
        param_names = [f"β{i}" for i in range(len(self.beta_parametric))]
        for name, coef in zip(param_names, self.beta_parametric):
            lines.append(f"{name:<25} {coef:>12.4f}")
        lines.append("")

        # Random Effects
        lines.append("Random Effects:")
        lines.append("-" * 75)
        for i, (group_name, vc) in enumerate(
            zip(self.random_effects.keys(), self.variance_components)
        ):
            lines.append(f"Group: {group_name}")
            if vc.shape[0] == 1:
                # Single variance component
                lines.append(f"  Variance: {vc[0, 0]:.4f}")
                lines.append(f"  Std.Dev.: {np.sqrt(vc[0, 0]):.4f}")
            else:
                # Multiple components (e.g., random intercept + slope)
                lines.append("  Variance-Covariance Matrix:")
                for row in vc:
                    row_str = "    " + "  ".join([f"{val:>10.4f}" for val in row])
                    lines.append(row_str)
                lines.append("  Standard Deviations:")
                for j in range(vc.shape[0]):
                    lines.append(f"    Component {j}: {np.sqrt(vc[j, j]):.4f}")
                # Correlation matrix if multivariate
                if vc.shape[0] > 1:
                    corr = np.zeros_like(vc)
                    for ii in range(vc.shape[0]):
                        for jj in range(vc.shape[1]):
                            corr[ii, jj] = vc[ii, jj] / (
                                np.sqrt(vc[ii, ii]) * np.sqrt(vc[jj, jj])
                            )
                    lines.append("  Correlation Matrix:")
                    for row in corr:
                        row_str = "    " + "  ".join([f"{val:>10.4f}" for val in row])
                        lines.append(row_str)
            lines.append("")

        lines.append(
            f"Residual Standard Deviation: {np.sqrt(self.residual_variance):.4f}"
        )
        lines.append("")

        # Model Fit
        lines.append("Model Fit Statistics:")
        lines.append("-" * 75)
        lines.append(f"  Log-likelihood:      {self.log_likelihood:>12.2f}")
        lines.append(f"  AIC:                 {self.aic:>12.2f}")
        lines.append(f"  BIC:                 {self.bic:>12.2f}")
        lines.append(f"  Effective df (total): {self.edf_total:>11.2f}")
        lines.append(f"  Converged:           {str(self.converged):>12}")
        lines.append(f"  Iterations:          {self.n_iterations:>12}")
        lines.append("=" * 75)

        return "\n".join(lines)


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
    backend: str = "numpy",
    device: str | None = None,
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
    backend : str, default='numpy'
        Computational backend: 'numpy', 'torch', or 'jax'.
    device : str, optional
        Device for computation (for torch backend): 'cpu', 'cuda', etc.

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
    # Backend conversion
    # For now, we convert to the specified backend and back to numpy for internal operations
    # This allows benchmarking and sets up infrastructure for full backend support
    backend = backend.lower()

    if backend in ("torch", "pytorch"):
        try:
            import torch

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_device = torch.device(device)

            from scipy.sparse import issparse

            # Convert to torch tensors
            X_parametric_t = torch.tensor(
                X_parametric, dtype=torch.float64, device=torch_device
            )
            Z_t = torch.tensor(Z, dtype=torch.float64, device=torch_device)
            y_t = torch.tensor(y, dtype=torch.float64, device=torch_device)

            # Convert back to numpy for internal operations (future: full torch support)
            X_parametric = X_parametric_t.cpu().numpy()
            Z = Z_t.cpu().numpy()
            y = y_t.cpu().numpy()

            if X_smooth is not None:
                # Handle sparse matrices
                X_smooth = {
                    k: (
                        torch.tensor(
                            v.toarray(), dtype=torch.float64, device=torch_device
                        )
                        .cpu()
                        .numpy()
                        if issparse(v)
                        else torch.tensor(v, dtype=torch.float64, device=torch_device)
                        .cpu()
                        .numpy()
                    )
                    for k, v in X_smooth.items()
                }
            if S_smooth is not None:
                S_smooth = {
                    k: torch.tensor(v, dtype=torch.float64, device=torch_device)
                    .cpu()
                    .numpy()
                    for k, v in S_smooth.items()
                }
        except ImportError:
            pass  # Fall back to numpy

    elif backend == "jax":
        try:
            import jax.numpy as jnp
            from scipy.sparse import issparse

            # Convert to JAX arrays
            X_parametric_j = jnp.array(X_parametric)
            Z_j = jnp.array(Z)
            y_j = jnp.array(y)

            # Convert back to numpy for internal operations (future: full JAX support)
            X_parametric = np.asarray(X_parametric_j)
            Z = np.asarray(Z_j)
            y = np.asarray(y_j)

            if X_smooth is not None:
                # Handle sparse matrices
                X_smooth = {
                    k: np.asarray(jnp.array(v.toarray()))
                    if issparse(v)
                    else np.asarray(jnp.array(v))
                    for k, v in X_smooth.items()
                }
            if S_smooth is not None:
                S_smooth = {k: np.asarray(jnp.array(v)) for k, v in S_smooth.items()}
        except ImportError:
            pass  # Fall back to numpy

    n = len(y)

    # Combine fixed effects design matrices
    X_list = [X_parametric]
    p_parametric = X_parametric.shape[1]

    smooth_term_names = []
    smooth_start_cols = {}
    smooth_end_cols = {}

    if X_smooth is not None:
        from scipy.sparse import issparse

        col_idx = p_parametric
        for term_name, X_term in X_smooth.items():
            smooth_term_names.append(term_name)
            smooth_start_cols[term_name] = col_idx
            smooth_end_cols[term_name] = col_idx + X_term.shape[1]
            col_idx = smooth_end_cols[term_name]

            # Convert sparse to dense for GAMM (mixed model equations require dense)
            if issparse(X_term):
                X_list.append(X_term.toarray())
            else:
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

    # Expand psi to full block-diagonal form before inverting
    # psi from REML is per-group covariance structure
    if len(Z_info) == 1 and psi.shape[0] == Z_info[0]["n_effects"]:
        # Single random effect term, expand to block-diagonal
        n_groups = Z_info[0]["n_groups"]
        psi_full = linalg.block_diag(*([psi] * n_groups))
    elif len(Z_info) > 1:
        # Multiple terms, psi is block-diagonal of per-term structures
        # Need to expand each term
        psi_blocks = []
        term_offset = 0
        for info in Z_info:
            n_effects = info["n_effects"]
            n_groups = info["n_groups"]
            # Extract this term's per-group structure
            psi_term = psi[
                term_offset : term_offset + n_effects,
                term_offset : term_offset + n_effects,
            ]
            # Expand to block-diagonal for all groups
            psi_blocks.append(linalg.block_diag(*([psi_term] * n_groups)))
            term_offset += n_effects
        psi_full = linalg.block_diag(*psi_blocks)
    else:
        # Already in correct form
        psi_full = psi

    # Compute Ψ⁻¹
    psi_inv = linalg.inv(psi_full)

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

    # Step 4b: Convert variance_components to list format
    # psi from REML is per-group structure (or block-diagonal of per-group structures)
    # Convert to list of per-term covariance matrices
    if len(Z_info) == 1:
        # Single random effect term
        variance_components_list = [psi]
    else:
        # Multiple terms: psi is block-diagonal of per-term structures
        # Extract each term's structure
        variance_components_list = []
        offset = 0
        for info in Z_info:
            n_effects = info["n_effects"]
            psi_term = psi[offset : offset + n_effects, offset : offset + n_effects]
            variance_components_list.append(psi_term)
            offset += n_effects

    # Step 4c: Extract raw covariance parameters from theta
    # theta contains: [params_term1, params_term2, ..., log(sigma2)]
    # Split theta by number of parameters per term

    covariance_params_list = []
    theta = reml_result.theta
    param_idx = 0

    for info in Z_info:
        cov_type = info.get("covariance", "unstructured")
        n_effects = info["n_effects"]
        cov_structure = get_covariance_structure(cov_type)
        n_params = cov_structure.n_parameters(n_effects)

        # Extract parameters for this term
        params_term = theta[param_idx : param_idx + n_params]
        covariance_params_list.append(params_term)
        param_idx += n_params

    # Note: theta also contains log(sigma2) at the end, but we store sigma2 separately

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
            if (
                term_name in S_smooth
                and lambda_smooth is not None
                and term_name in lambda_smooth
            ):
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

    # For marginal likelihood (REML), AIC should count:
    # - Fixed effects (use EDF for penalized terms like smooths)
    # - Variance component parameters (NOT random effects themselves)
    # - Residual variance
    #
    # Count variance component parameters
    n_variance_params = 0
    for info in Z_info:
        cov_type = info.get("covariance", covariance)
        n_effects = info["n_effects"]
        cov_structure = get_covariance_structure(cov_type)
        n_variance_params += cov_structure.n_parameters(n_effects)

    # Add residual variance parameter
    n_variance_params += 1

    # AIC = -2*log(L) + 2*k where k = fixed effects (EDF) + variance parameters
    k_params = edf_fixed + n_variance_params
    aic = -2 * log_likelihood + 2 * k_params

    # BIC = -2*log(L) + k*log(n)
    bic = -2 * log_likelihood + k_params * np.log(n)

    # Store smoothing parameters
    smoothing_params = lambda_smooth if lambda_smooth is not None else None

    # Count total groups across all random effect terms
    n_groups = sum(info["n_groups"] for info in Z_info)

    return GAMMResult(
        coefficients=beta_combined,
        beta_parametric=beta_parametric,
        beta_smooth=beta_smooth,
        random_effects=random_effects,
        variance_components=variance_components_list,
        covariance_params=covariance_params_list,
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
        # extract_random_effects returns {'grouping_var': {group_id: array, ...}}
        b_flat = []
        for grouping_var in sorted(result.random_effects.keys()):
            group_effects = result.random_effects[grouping_var]
            # group_effects is a dict: {group_id: array of effects}
            for group_id in sorted(group_effects.keys()):
                b_group = group_effects[group_id]
                # Extend with all effects for this group
                if isinstance(b_group, np.ndarray):
                    b_flat.extend(b_group.ravel())
                else:
                    b_flat.append(b_group)

        b_array = np.array(b_flat)
        pred += Z_new @ b_array

    return pred
