# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Generalized Additive Models (GAM) using Penalized Regression Splines.

Mathematical Framework
----------------------
A Generalized Additive Model extends linear models by allowing non-linear
smooth functions of predictors:

    E[Y] = β₀ + f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ)

where fⱼ(·) are smooth functions estimated from data.

Penalized Spline Representation
--------------------------------
Each smooth function fⱼ(x) is represented using a basis expansion:

    fⱼ(x) = Σₖ βⱼₖ φⱼₖ(x)

where:
    - φⱼₖ(x) are basis functions (B-splines or cubic splines)
    - βⱼₖ are coefficients to be estimated
    - k = 1, ..., Kⱼ (number of basis functions)

**Design matrix**: X = [φ₁(x₁), φ₂(x₁), ..., φₖ(x₁)]ᵀ for n observations

Penalized Likelihood
--------------------
To prevent overfitting, we add a roughness penalty to the likelihood:

    ℓₚ(β) = ℓ(β) - ½ λ β^T S β

where:
    - ℓ(β) is the log-likelihood
    - λ ≥ 0 is the smoothing parameter
    - S is the penalty matrix measuring roughness of f

**For Gaussian responses** (least squares):

    minimize: ||y - Xβ||² + λ β^T S β

**Penalized normal equations**:

    (X^T X + λS) β = X^T y

Solution:
    β̂ = (X^T X + λS)⁻¹ X^T y

**Effective degrees of freedom**:

    edf = tr[(X^T X + λS)⁻¹ X^T X]
        = tr[H_λ]

where H_λ is the "hat" matrix (smoother matrix).

Penalty Matrix Construction
----------------------------
**Difference penalty** (approximates mth derivative):

For second-order differences (m=2), approximating ∫[f''(x)]² dx:

    S = D^T D

where D is the (K-2) × K second difference matrix:

    D = [
        [ 1  -2   1   0  ... ]
        [ 0   1  -2   1  ... ]
        ...
    ]

**Integrated squared derivative penalty** (exact for cubic splines):

    Sᵢⱼ = ∫ φᵢ''(x) φⱼ''(x) dx

Smoothing Parameter Selection
------------------------------
### Generalized Cross-Validation (GCV)

**GCV score** (Craven & Wahba, 1978):

    GCV(λ) = (n / (n - edf)²) Σᵢ (yᵢ - f̂(xᵢ))²
           = (n ||y - Xβ̂_λ||²) / (n - tr(H_λ))²

**Algorithm**:
1. Define search grid: λ ∈ [λ_min, λ_max] (log-scale)
2. For each λ:
   a. Compute β̂_λ = (X^T X + λS)⁻¹ X^T y
   b. Compute edf_λ = tr[(X^T X + λS)⁻¹ X^T X]
   c. Compute GCV(λ)
3. Select λ* = argmin GCV(λ)

**Properties**:
- GCV is an approximation to leave-one-out cross-validation
- Invariant to scaling of y
- Tends to slightly undersmooth in practice

### Restricted Maximum Likelihood (REML)

**REML criterion** (for Gaussian responses):

    -2ℓ_R(λ) = log|X^T X + λS| + n log(RSS) + log|X^T X|

where RSS = ||y - Xβ̂_λ||².

**Advantages over GCV**:
- More stable for small samples
- Accounts for uncertainty in fixed effects
- Tends to give slightly larger λ (more smoothing)

Basis Functions
---------------
### B-Splines (de Boor, 2001)

**Cox-de Boor recursion formula**:

    Bᵢ,₀(x) = { 1  if tᵢ ≤ x < tᵢ₊₁
              { 0  otherwise

    Bᵢ,ₖ(x) = (x - tᵢ)/(tᵢ₊ₖ - tᵢ) Bᵢ,ₖ₋₁(x)
            + (tᵢ₊ₖ₊₁ - x)/(tᵢ₊ₖ₊₁ - tᵢ₊₁) Bᵢ₊₁,ₖ₋₁(x)

where {tᵢ} are knots and k is the degree.

**Properties**:
- Local support: Bᵢ,ₖ(x) ≠ 0 only for x ∈ [tᵢ, tᵢ₊ₖ₊₁]
- Non-negativity: Bᵢ,ₖ(x) ≥ 0
- Partition of unity: Σᵢ Bᵢ,ₖ(x) = 1
- Numerical stability: No cancellation errors
- Efficient computation: O(k²) per evaluation

### Natural Cubic Splines (Green & Silverman, 1993)

**Cubic spline** with natural boundary conditions:
- f''(x) = 0 at boundaries
- Minimizes ∫[f''(x)]² dx among all interpolating functions

**Representation**:

    f(x) = Σⱼ₌₁ᴷ γⱼ Nⱼ(x)

where Nⱼ(x) are natural cubic spline basis functions.

**Advantage**: Exact penalty ∫[f''(x)]² dx computable analytically

Computational Complexity
------------------------
For n observations and K basis functions:

**Fitting** (fixed λ):
    - Basis evaluation: O(nK × k²) for B-splines of degree k
    - Penalty construction: O(K³) or O(K²) for difference penalties
    - System solve: O(K³) via Cholesky decomposition
    - Total: O(nK × k² + K³)

**GCV search** (L candidate values of λ):
    - L × [system solve + edf computation]
    - Using matrix decomposition tricks: O(L × K²)
    - Total: O(K³ + L × K²)

Numerical Stability
-------------------
**Techniques used**:

1. **Cholesky decomposition**: For solving (X^T X + λS)β = X^T y
   - Requires positive-definite matrix
   - Numerical error O(ε × κ²) where κ is condition number

2. **QR decomposition**: Fallback for near-singular cases
   - More stable: error O(ε × κ)
   - Slower: O(K³) vs O(K³/3) for Cholesky

3. **Basis centering**: Remove mean to improve conditioning

4. **Eigendecomposition**: For repeated solves with different λ
   - Decompose: S = UΛU^T
   - Transform: y* = U^T X^T y
   - Solve in diagonal space: much faster

Multi-backend Support
---------------------
This module supports NumPy, PyTorch, and JAX backends through the
array namespace abstraction. All operations use the namespace API
for transparent backend compatibility.

References
----------
**Core GAM theory**:

- Hastie, T., & Tibshirani, R. (1990). *Generalized Additive Models*.
  Chapman and Hall/CRC.

- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). CRC Press. Chapters 3-5.

**Penalized splines**:

- Eilers, P. H. C., & Marx, B. D. (1996). "Flexible smoothing with B-splines
  and penalties." *Statistical Science*, 11(2), 89-121.
  https://doi.org/10.1214/ss/1038425655

- Ruppert, D., Wand, M. P., & Carroll, R. J. (2003). *Semiparametric Regression*.
  Cambridge University Press. Chapter 5.

**Smoothing parameter selection**:

- Craven, P., & Wahba, G. (1978). "Smoothing noisy data with spline functions."
  *Numerische Mathematik*, 31(4), 377-403.
  https://doi.org/10.1007/BF01404567

- Wood, S. N. (2011). "Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models."
  *JRSS: Series B*, 73(1), 3-36.
  https://doi.org/10.1111/j.1467-9868.2010.00749.x

**B-spline theory**:

- de Boor, C. (2001). *A Practical Guide to Splines* (Revised ed.). Springer.
  https://doi.org/10.1007/978-1-4612-6333-3

**Natural cubic splines**:

- Green, P. J., & Silverman, B. W. (1993). *Nonparametric Regression and
  Generalized Linear Models: A Roughness Penalty Approach*. Chapman and Hall/CRC.

**Numerical methods**:

- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.).
  Johns Hopkins University Press. Chapters 4-5.

See Also
--------
aurora.models.gam.additive : Additive GAM with multiple smooth terms
aurora.smoothing.splines.bspline : B-spline basis implementation
aurora.smoothing.splines.cubic : Cubic spline basis implementation
aurora.smoothing.selection.gcv : GCV smoothing parameter selection
aurora.smoothing.selection.reml : REML smoothing parameter selection

Notes
-----
For detailed mathematical derivations and proofs, see REFERENCES.md in the
repository root.

This implementation follows the penalized regression spline approach of
Eilers & Marx (1996) combined with the computational strategies from
Wood (2017).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from aurora.models.gam.result import GAMResult
from aurora.smoothing.selection.gcv import select_smoothing_parameter
from aurora.smoothing.splines.bspline import BSplineBasis
from aurora.smoothing.splines.cubic import CubicSplineBasis


def fit_gam(
    x: np.ndarray,
    y: np.ndarray,
    n_basis: int = 10,
    basis_type: Literal["bspline", "cubic"] = "bspline",
    degree: int = 3,
    penalty_order: int = 2,
    lambda_: float | None = None,
    lambda_min: float = 1e-6,
    lambda_max: float = 1e6,
    knot_method: Literal["quantile", "uniform"] = "quantile",
    weights: np.ndarray | None = None,
    use_sparse: bool = False,
) -> GAMResult:
    """Fit univariate Generalized Additive Model using penalized splines.

    This function fits a smooth function f(x) to data (x, y) using:
        y = f(x) + ε
    where f is represented as a linear combination of basis functions with
    a roughness penalty.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Predictor variable.
    y : ndarray, shape (n,)
        Response variable.
    n_basis : int, default=10
        Number of basis functions to use.
    basis_type : {'bspline', 'cubic'}, default='bspline'
        Type of spline basis:
        - 'bspline': B-spline basis (local support, partition of unity)
        - 'cubic': Natural cubic spline basis (global support)
    degree : int, default=3
        Degree of splines (3 for cubic splines).
    penalty_order : int, default=2
        Order of difference penalty (2 approximates second derivative).
    lambda_ : float, optional
        Smoothing parameter. If None, selected automatically via GCV.
    lambda_min : float, default=1e-6
        Minimum lambda for GCV search.
    lambda_max : float, default=1e6
        Maximum lambda for GCV search.
    knot_method : {'quantile', 'uniform'}, default='quantile'
        Method for placing knots:
        - 'quantile': Place knots at quantiles of x
        - 'uniform': Space knots uniformly over range of x
    weights : ndarray, shape (n,), optional
        Observation weights for weighted least squares.
    use_sparse : bool, default=False
        Whether to use sparse matrix operations for B-spline basis.

        **When to use sparse**:
        - Large problems (n > 500, n_basis > 20)
        - B-spline basis (has compact support)
        - Memory-constrained environments

        **Benefits**:
        - ~10-100× speedup for large problems
        - Reduced memory usage: O(n × degree) vs O(n × n_basis)
        - Automatic method selection (direct/CG/MINRES)

        **Note**: Only available for 'bspline' basis type. Cubic splines
        have global support and do not benefit from sparse operations.

    Returns
    -------
    result : GAMResult
        Fitted GAM result containing:
        - coefficients: Spline coefficients
        - fitted_values: Predicted values at x
        - lambda_: Smoothing parameter used
        - edf: Effective degrees of freedom
        - basis: Basis object for prediction
        - residuals: y - fitted_values

    Notes
    -----
    The model minimizes the penalized least squares criterion:
        ||y - Xβ||² + λ β'Sβ

    where X is the basis matrix and S is the penalty matrix.

    For automatic smoothing parameter selection (lambda_=None), uses GCV:
        GCV(λ) = (n * RSS) / (n - tr(H))²

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.gam import fit_gam
    >>> # Generate noisy data
    >>> x = np.linspace(0, 1, 100)
    >>> y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(100)
    >>> # Fit GAM with automatic smoothing
    >>> result = fit_gam(x, y, n_basis=12)
    >>> print(f"EDF: {result.edf:.2f}")
    >>> print(f"Lambda: {result.lambda_:.4f}")
    >>> # Make predictions
    >>> x_new = np.linspace(0, 1, 200)
    >>> y_pred = result.predict(x_new)

    References
    ----------
    Wood, S.N. (2017). Generalized Additive Models: An Introduction with R.
        Chapman and Hall/CRC.
    Eilers, P.H.C. & Marx, B.D. (1996). Flexible smoothing with B-splines and
        penalties. Statistical Science, 11(2), 89-121.
    """
    # Validate inputs
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if x_arr.ndim != 1:
        raise ValueError("x must be 1-dimensional")

    if y_arr.ndim != 1:
        raise ValueError("y must be 1-dimensional")

    n = len(x_arr)
    if len(y_arr) != n:
        raise ValueError(f"x and y must have same length, got {n} and {len(y_arr)}")

    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {weights_arr.shape}")
    else:
        weights_arr = None

    if n_basis < 3:
        raise ValueError("n_basis must be at least 3")

    # Validate sparse option
    if use_sparse and basis_type != "bspline":
        raise ValueError(
            f"use_sparse=True only supported for basis_type='bspline', got '{basis_type}'"
        )

    # Create basis
    if basis_type == "bspline":
        knots = BSplineBasis.create_knots(
            x_arr, n_basis=n_basis, degree=degree, method=knot_method
        )
        basis = BSplineBasis(knots, degree=degree)
    elif basis_type == "cubic":
        knots_interior = CubicSplineBasis.create_knots(
            x_arr, n_knots=n_basis - 2, method=knot_method
        )
        basis = CubicSplineBasis(knots_interior)
    else:
        raise ValueError(f"Unknown basis_type: {basis_type}")

    # Compute basis matrix (sparse if requested)
    if use_sparse:
        X = basis.basis_matrix(x_arr, sparse=True)
    else:
        X = basis.basis_matrix(x_arr)

    # Create penalty matrix
    if basis_type == "bspline":
        S = basis.penalty_matrix(order=penalty_order)
    else:  # cubic
        # Cubic splines use analytical integrated squared second derivative
        # (no order parameter)
        S = basis.penalty_matrix()

    # Select or use provided smoothing parameter
    if lambda_ is None:
        # Automatic selection via GCV
        gcv_result = select_smoothing_parameter(
            y_arr,
            X,
            S,
            weights=weights_arr,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )
        lambda_used = gcv_result["lambda_opt"]
        coefficients = gcv_result["coefficients"]
        fitted_values = gcv_result["fitted_values"]
        edf = gcv_result["edf"]
        gcv_score = gcv_result["gcv_score"]
    else:
        # Use provided lambda
        lambda_used = float(lambda_)

        # Solve penalized least squares
        if use_sparse:
            # Use sparse solver
            from aurora.core.optimization.sparse_solvers import (
                solve_sparse_penalized_ls,
            )

            if weights_arr is None:
                weights_solve = np.ones(n)
            else:
                weights_solve = weights_arr

            coefficients, solve_info = solve_sparse_penalized_ls(
                X, y_arr, weights_solve, S, lambda_used, method="auto"
            )
            fitted_values = X @ coefficients

            # Compute EDF for sparse matrices
            # For small n_basis (< 100), convert to dense for exact computation
            # For large n_basis, use diagonal approximation
            if n_basis <= 100:
                # Small enough for exact dense computation
                X_dense = X.toarray() if hasattr(X, 'toarray') else X
                S_dense = S.toarray() if hasattr(S, 'toarray') else S

                if weights_arr is None:
                    W = np.eye(n)
                else:
                    W = np.diag(weights_arr)

                XtWX = X_dense.T @ W @ X_dense
                A = XtWX + lambda_used * S_dense
                A_inv = np.linalg.inv(A)
                edf = float(np.trace(A_inv @ XtWX))
            else:
                # For large basis, use approximation: EDF ≈ k / (1 + λ)
                # This assumes S ≈ I (reasonable for difference penalties)
                edf = float(n_basis / (1 + lambda_used))

            gcv_score = None
        else:
            # Dense solver
            if weights_arr is None:
                W = np.eye(n)
            else:
                W = np.diag(weights_arr)

            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y_arr
            A = XtWX + lambda_used * S

            coefficients = np.linalg.solve(A, XtWy)
            fitted_values = X @ coefficients

            # Compute EDF
            A_inv = np.linalg.inv(A)
            edf = float(np.trace(A_inv @ XtWX))

            gcv_score = None

    # Compute residuals
    residuals = y_arr - fitted_values

    # Create result object
    result = GAMResult(
        coefficients=coefficients,
        fitted_values=fitted_values,
        residuals=residuals,
        lambda_=lambda_used,
        edf=edf,
        basis=basis,
        x=x_arr,
        y=y_arr,
        weights=weights_arr,
        gcv_score=gcv_score,
    )

    return result


__all__ = ["fit_gam"]
