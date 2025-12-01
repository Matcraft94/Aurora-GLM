"""Estimation Strategies for Smoothing and Mixed Models.

This module provides various estimation methods for variance components
and smoothing parameters in mixed and additive models.

Mathematical Framework
======================

Estimation in Aurora-GLM involves three main paradigms:

Maximum Likelihood (ML)
-----------------------
ML estimates parameters by maximizing the log-likelihood:

    β̂_ML = argmax_β ℓ(β; y)

where:
    - ℓ(β; y) = Σᵢ log f(yᵢ | β) is the log-likelihood
    - f(y | β) is the density/mass function

For exponential family distributions:

    ℓ(β; y) = Σᵢ [yᵢθᵢ - b(θᵢ)] / a(φ) + c(yᵢ, φ)

**Properties**:
- Consistent: β̂_ML → β₀ as n → ∞
- Asymptotically efficient: achieves Cramér-Rao lower bound
- Asymptotically normal: √n(β̂ - β₀) →ᵈ N(0, I⁻¹(β₀))

**Variance estimation**:

    Var(β̂) ≈ I(β̂)⁻¹ = -(∂²ℓ/∂β∂β^T)⁻¹

Restricted Maximum Likelihood (REML)
-------------------------------------
REML estimates variance components while accounting for fixed effects
uncertainty:

    ℓ_R(θ) = ℓ(θ) - ½ log|X^T V⁻¹(θ) X|

where:
    - θ = (σ², Ψ) are variance parameters
    - V(θ) = σ²I + ZΨZ^T is the marginal covariance
    - The second term is the "profiled" contribution from fixed effects

**For Linear Mixed Models**:

    ℓ_R(σ², Ψ) = -½ [log|V| + log|X^T V⁻¹ X| + (n-p) log σ² + r^T V⁻¹ r / σ²]

where:
    - r = y - Xβ̂ are residuals
    - p = dim(β) is the number of fixed effects
    - Degrees of freedom correction: (n-p) instead of n

**Advantages over ML**:
- Unbiased variance estimates for σ²
- Accounts for uncertainty in β estimation
- Preferred for small to moderate samples

**Disadvantages**:
- Cannot compare models with different fixed effects
- Computationally slightly more expensive

Penalized Likelihood / Smoothing
---------------------------------
For smooth functions f(x), we balance fit against roughness:

    ℓ_p(f; λ) = ℓ(y | f) - λ/2 ∫ [f''(x)]² dx

**Discrete approximation** (penalized regression splines):

    ℓ_p(β; λ) = ℓ(y | Bβ) - λ/2 β^T S β

where:
    - B is the spline basis matrix
    - S is the penalty matrix (second derivative)
    - λ ≥ 0 is the smoothing parameter

**Smoothing parameter selection**:

GCV (Generalized Cross-Validation):
    GCV(λ) = n ||y - ŷ_λ||² / (n - tr(H_λ))²

REML for smoothing:
    ℓ_R(λ) = -½ [log|X^T X + λS| + n log(RSS_λ)]

**Connection to mixed models**:
Penalized splines are equivalent to random effects:

    f(x) = Bβ + Zu,  u ~ N(0, σ_u² I)

with λ = σ²/σ_u² controlling the smoothness.

Laplace Approximation
----------------------
For non-Gaussian mixed models, the marginal likelihood is intractable:

    L(β, θ) = ∫ f(y | β, b) f(b | θ) db

**Laplace approximation**:

    ∫ exp(g(b)) db ≈ (2π)^{q/2} |H|^{-1/2} exp(g(b̂))

where:
    - b̂ = argmax_b g(b) is the mode
    - H = -∂²g/∂b∂b^T is the Hessian at the mode
    - q = dim(b) is the random effect dimension

**Procedure**:
1. For fixed (β, θ), find b̂ = argmax_b log f(y | β, b) + log f(b | θ)
2. Compute Hessian H at b̂
3. Approximate log-likelihood: ℓ(β, θ) ≈ log f(y | β, b̂) + log f(b̂ | θ) - ½ log|H|

**Accuracy**: O(n⁻¹) error for n observations per group

Penalized Quasi-Likelihood (PQL)
---------------------------------
PQL iteratively approximates non-Gaussian responses as Gaussian:

**Working response**:
    z = η + (y - μ) g'(μ)

**Working variance**:
    w = [g'(μ)]⁻² V(μ)⁻¹

**Iterate**:
1. Given current (β, b), compute z and w
2. Fit linear mixed model: z = Xβ + Zb + ε, Var(ε) = w⁻¹
3. Update (β, b) from LMM fit
4. Repeat until convergence

**Advantages**: Fast, leverages LMM machinery
**Disadvantages**: Biased for binary outcomes, no valid likelihood

Submodules
----------
laplace
    Laplace approximation for non-Gaussian GLMMs.
    See: aurora.models.gamm.laplace for current implementation.

ml
    Maximum Likelihood estimation.
    (Planned for future release)

reml
    Restricted Maximum Likelihood estimation.
    See: aurora.models.gamm.estimation for current implementation.

References
----------
- Patterson, H. D., & Thompson, R. (1971). "Recovery of inter-block 
  information when block sizes are unequal." *Biometrika*, 58(3), 545-554.
  
- Harville, D. A. (1977). "Maximum likelihood approaches to variance 
  component estimation and to related problems." *JASA*, 72(358), 320-338.

- Breslow, N. E., & Clayton, D. G. (1993). "Approximate inference in
  generalized linear mixed models." *JASA*, 88(421), 9-25.

- Wood, S. N. (2011). "Fast stable restricted maximum likelihood and 
  marginal likelihood estimation of semiparametric generalized linear
  models." *Journal of the Royal Statistical Society: Series B*, 73(1), 3-36.

Notes
-----
The main estimation functionality is currently implemented in:

- aurora.models.gamm.estimation (REML for GAMM)
- aurora.models.gamm.laplace (Laplace approximation)
- aurora.smoothing.selection (GCV and REML for smoothing parameters)

These will be consolidated here in a future refactoring.
"""