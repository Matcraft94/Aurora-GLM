"""Smoothing Components for Generalized Additive Models.

This module provides smoothing primitives for non-parametric regression
in Generalized Additive Models (GAM) and mixed model extensions.

Mathematical Framework
======================

Smooth Functions in GAMs
------------------------
In a GAM, each predictor xⱼ enters via a smooth function:

    g(E[Y]) = β₀ + f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ)

where fⱼ(·) are unknown smooth functions estimated from data.

Basis Function Representation
------------------------------
Each smooth function is represented using a basis expansion:

    f(x) = Σₖ₌₁ᴷ βₖ φₖ(x) = φ(x)^T β

where:
    - φₖ(x) are basis functions (B-splines, thin plate, etc.)
    - βₖ are coefficients to be estimated
    - K is the number of basis functions

**Design matrix**: For n observations,

    B[i,k] = φₖ(xᵢ)

So: f(x) = Bβ in matrix form.

Roughness Penalties
-------------------
To prevent overfitting, we penalize the integrated squared second derivative:

    J(f) = ∫ [f''(x)]² dx = β^T S β

where S is the **penalty matrix**:

    S[j,k] = ∫ φⱼ''(x) φₖ''(x) dx

**Discrete approximation** (difference penalty):

    S = D^T D

where D is the second difference matrix:

    D = [ 1  -2   1   0  ...]
        [ 0   1  -2   1  ...]
        ...

This approximates the integrated squared second derivative.

Penalized Least Squares
------------------------
The penalized objective function for Gaussian responses:

    min_β ||y - Bβ||² + λ β^T S β

**Solution** (penalized normal equations):

    β̂ = (B^T B + λS)⁻¹ B^T y

**Smoother matrix** (hat matrix):

    H_λ = B(B^T B + λS)⁻¹ B^T

So: ŷ = H_λ y

**Effective degrees of freedom**:

    edf = tr(H_λ) = tr[(B^T B + λS)⁻¹ B^T B]

- edf → K as λ → 0 (interpolation)
- edf → rank(null space of S) as λ → ∞

Basis Types
-----------

**B-Splines** (see aurora.smoothing.splines):
    - Piecewise polynomials with local support
    - Cox-de Boor recursion formula
    - Numerically stable, efficient
    - Requires knot placement

**Thin Plate Splines** (see aurora.smoothing.thinplate):
    - Radial basis: η(r) = r² log(r) for 2D
    - Rotation-invariant penalty
    - Optimal for isotropic smoothness
    - No need to choose knots

**Tensor Products** (see aurora.smoothing.tensor):
    - For multidimensional smoothing: f(x, z)
    - Separable penalty: λ_x J_x + λ_z J_z
    - Additive penalty allows different smoothness by direction

Tensor Product Smooths
-----------------------
For smooth functions of multiple variables f(x₁, x₂, ..., xₘ):

    f(x₁, x₂) = Σᵢ Σⱼ βᵢⱼ φᵢ(x₁) ψⱼ(x₂)

**Kronecker product** basis:

    B = B₁ ⊗ B₂

where B₁, B₂ are marginal basis matrices.

**Penalty** (separable):

    S = λ₁(S₁ ⊗ I₂) + λ₂(I₁ ⊗ S₂)

This allows different smoothness in each direction.

Smoothing Parameter Selection
------------------------------
The smoothing parameter λ controls the bias-variance tradeoff:

    - λ = 0: Interpolation (high variance, low bias)
    - λ → ∞: Linear fit (low variance, high bias)

**GCV (Generalized Cross-Validation)**:

    GCV(λ) = n ||y - ŷ_λ||² / (n - edf_λ)²

Minimizing GCV approximately minimizes prediction error.

**REML (Restricted Maximum Likelihood)**:

    ℓ_R(λ) = -½ [log|B^T B + λS| + n log(σ̂²) + constant]

REML is more stable for small samples and accounts for
uncertainty in the coefficients.

**UBRE (Un-Biased Risk Estimator)**:

    UBRE(λ) = ||y - ŷ_λ||² / n - 2σ² edf_λ / n + σ²

Equivalent to GCV when σ² is unknown.

Identifiability Constraints
----------------------------
To ensure model identifiability, smooth terms are centered:

    Σᵢ f(xᵢ) = 0

This is achieved by:
1. Absorbing the mean into the intercept
2. Using a contrast reparameterization
3. Applying a centering constraint: C^T β = 0

The penalty null space (polynomials) requires special handling.

Computational Complexity
-------------------------
For n observations and K basis functions:

    - Basis evaluation: O(nK) for B-splines
    - Penalty matrix: O(K²) or O(K) for difference penalties
    - Fitting (fixed λ): O(K³) for direct solve
    - EDF computation: O(K³) for trace

For multiple smoothing parameters:
    - Joint optimization: O(L × K³) for L parameters
    - REML gradient/Hessian available for efficient optimization

Available Components
=====================

Thin Plate Splines
------------------
- tps_basis: Compute thin plate spline basis
- tps_penalty: Compute TPS penalty matrix
- fit_tps: Fit a thin plate spline
- select_knots: Automatic knot selection

Tensor Products
---------------
- tensor_product_basis: Kronecker product of marginal bases
- tensor_product_penalty: Separable penalty for tensor products
- fit_tensor_product: Fit a tensor product smooth

See Also
--------
aurora.smoothing.splines : B-spline and cubic spline bases
aurora.smoothing.penalties : Penalty matrix construction
aurora.smoothing.selection : Smoothing parameter selection (GCV, REML)

References
----------
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). CRC Press. Chapters 4-5.

- Eilers, P. H. C., & Marx, B. D. (1996). "Flexible smoothing with B-splines
  and penalties." *Statistical Science*, 11(2), 89-121.

- Green, P. J., & Silverman, B. W. (1993). *Nonparametric Regression and
  Generalized Linear Models: A Roughness Penalty Approach*. Chapman & Hall.

- Wahba, G. (1990). *Spline Models for Observational Data*. SIAM.
"""
from __future__ import annotations

from aurora.smoothing.tensor import (
    fit_tensor_product,
    tensor_product_basis,
    tensor_product_penalty,
)
from aurora.smoothing.thinplate import (
    fit_tps,
    select_knots,
    tps_basis,
    tps_penalty,
)

__all__ = [
    "tensor_product_basis",
    "tensor_product_penalty",
    "fit_tensor_product",
    "tps_basis",
    "tps_penalty",
    "fit_tps",
    "select_knots",
]