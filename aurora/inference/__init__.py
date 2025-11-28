"""Statistical Inference Utilities for Aurora-GLM.

This module provides tools for hypothesis testing, confidence intervals,
model diagnostics, and robust inference in generalized linear models.

Mathematical Framework
======================

Hypothesis Testing
------------------
For testing H₀: Cβ = d against H₁: Cβ ≠ d:

**Wald Test**:

    W = (Cβ̂ - d)^T [C Var(β̂) C^T]⁻¹ (Cβ̂ - d) ~ χ²_r

where:
    - C is an r × p contrast matrix
    - r = rank(C) is the number of constraints
    - Var(β̂) is the estimated covariance matrix

For a single coefficient H₀: βⱼ = 0:

    z = β̂ⱼ / se(β̂ⱼ) ~ N(0,1)  asymptotically

**Score Test** (Rao's test):

    S = U(β₀)^T I(β₀)⁻¹ U(β₀) ~ χ²_r

where:
    - U(β) = ∂ℓ/∂β is the score function
    - I(β) = E[-∂²ℓ/∂β∂β^T] is the Fisher information
    - Evaluated at the null hypothesis β₀

**Likelihood Ratio Test**:

    LRT = 2[ℓ(β̂₁) - ℓ(β̂₀)] ~ χ²_r

where:
    - β̂₁ is the unrestricted MLE
    - β̂₀ is the restricted MLE under H₀
    - r = difference in number of parameters

Confidence Intervals
--------------------
**Wald-type intervals** (asymptotic normality):

    βⱼ ∈ [β̂ⱼ - z_{α/2} se(β̂ⱼ), β̂ⱼ + z_{α/2} se(β̂ⱼ)]

where z_{α/2} is the (1 - α/2) quantile of N(0,1).

**Profile likelihood intervals** (more accurate):

Based on the likelihood ratio:

    {β: 2[ℓ(β̂) - ℓ(β)] ≤ χ²_{1,α}}

Profile intervals are invariant to reparameterization.

**Bootstrap intervals** (non-parametric):

    - Percentile method: [β̂*_{α/2}, β̂*_{1-α/2}]
    - BCa (bias-corrected accelerated): adjusts for bias and skewness
    - Studentized: uses bootstrap t-statistics

Standard Errors
---------------
**Model-based (Fisher information)**:

    Var(β̂) = I(β̂)⁻¹ = (X^T W X)⁻¹

where W = diag(wᵢ) with wᵢ = [g'(μᵢ)]⁻² / V(μᵢ).

**Robust (sandwich estimator)**:

    Var(β̂)_robust = (X^T W X)⁻¹ (X^T Ω X) (X^T W X)⁻¹

where Ω accounts for heteroscedasticity and clustering.

**HC variants** (heteroscedasticity-consistent):

| Type | Ω diagonal                              | Use case              |
|------|----------------------------------------|----------------------|
| HC0  | r²ᵢ                                    | Large n, homoscedastic|
| HC1  | n/(n-p) × r²ᵢ                         | Finite-sample bias   |
| HC2  | r²ᵢ / (1 - hᵢᵢ)                       | Moderate leverage    |
| HC3  | r²ᵢ / (1 - hᵢᵢ)²                      | High leverage        |

where rᵢ are Pearson residuals and hᵢᵢ are leverage values.

Model Diagnostics
-----------------
**Residuals** for GLMs:

| Type        | Definition                             | Interpretation      |
|-------------|----------------------------------------|---------------------|
| Response    | yᵢ - μ̂ᵢ                               | Raw scale           |
| Pearson     | (yᵢ - μ̂ᵢ) / √V(μ̂ᵢ)                   | Standardized        |
| Deviance    | sign(yᵢ - μ̂ᵢ) × √dᵢ                   | Likelihood scale    |
| Studentized | rᵢ / √(1 - hᵢᵢ)                       | Leave-one-out       |

**Leverage** (influence on fit):

    hᵢᵢ = [H]ᵢᵢ = [X(X^T W X)⁻¹ X^T W]ᵢᵢ

Properties:
    - 0 < hᵢᵢ < 1 (for intercept model)
    - Σᵢ hᵢᵢ = p (number of parameters)
    - High leverage: hᵢᵢ > 2p/n

**Cook's Distance** (influence on coefficients):

    Dᵢ = (β̂ - β̂₍₋ᵢ₎)^T (X^T W X) (β̂ - β̂₍₋ᵢ₎) / (p × s²)

One-step approximation:

    Dᵢ ≈ r²ᵢ × hᵢᵢ / [p × (1 - hᵢᵢ)²]

where rᵢ is the Pearson residual.

Cutoff: Dᵢ > 4/n or Dᵢ > 1 indicates influential point.

**DFBETAS** (influence on each coefficient):

    DFBETAS_{ij} = (β̂ⱼ - β̂ⱼ₍₋ᵢ₎) / se(β̂ⱼ)

Cutoff: |DFBETAS| > 2/√n

**DFFITS** (influence on fitted values):

    DFFITS_i = (ŷᵢ - ŷ₍₋ᵢ₎) / (s × √hᵢᵢ)

Cutoff: |DFFITS| > 2√(p/n)

ANOVA for GLMs
--------------
**Analysis of Deviance** generalizes ANOVA to GLMs:

For comparing nested models M₀ ⊂ M₁:

    F = [(D₀ - D₁) / (df₀ - df₁)] / [D₁ / df₁]

where:
    - D₀, D₁ are deviances under M₀, M₁
    - df₀, df₁ are residual degrees of freedom

Under H₀ (smaller model is correct):
    - F ~ F_{df₀-df₁, df₁} approximately
    - Or use χ² test: (D₀ - D₁) ~ χ²_{df₀-df₁}

**Type I SS** (sequential):
    - Add terms one at a time
    - Order-dependent
    - SS for each term adjusted for preceding terms

**Type III SS** (partial):
    - Each term adjusted for all others
    - Order-independent
    - Used for unbalanced designs

Available Components
=====================

Hypothesis Testing
------------------
- wald_test: Wald test for linear hypotheses

Confidence Intervals
--------------------
- confidence_intervals: Wald-type confidence intervals
- ConfidenceIntervalResult: Structured interval results

Diagnostics
-----------
- glm_diagnostics: Comprehensive GLM diagnostics
- GLMDiagnosticResult: Residuals, leverage, Cook's D

Robust Inference
----------------
- robust_covariance: Sandwich/HC standard errors
- bootstrap_inference: Bootstrap-based inference
- RobustInferenceResult: Results with robust SEs
- HCType: Heteroscedasticity-consistent estimator types

See Also
--------
aurora.inference.anova : ANOVA and model comparison
aurora.inference.intervals : Confidence interval methods
aurora.inference.diagnostics : Model diagnostics
aurora.inference.hypothesis : Hypothesis testing
aurora.inference.robust : Robust standard errors

References
----------
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*
  (2nd ed.). Chapman and Hall/CRC. Chapter 11 (Inference).

- White, H. (1980). "A heteroskedasticity-consistent covariance matrix
  estimator and a direct test for heteroskedasticity." *Econometrica*,
  48(4), 817-838.

- MacKinnon, J. G., & White, H. (1985). "Some heteroskedasticity-consistent
  covariance matrix estimators with improved finite sample properties."
  *Journal of Econometrics*, 29(3), 305-325.

- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall/CRC.

- Cook, R. D. (1977). "Detection of influential observation in linear
  regression." *Technometrics*, 19(1), 15-18.
"""

from .diagnostics import GLMDiagnosticResult, glm_diagnostics
from .hypothesis import wald_test
from .intervals import ConfidenceIntervalResult, confidence_intervals
from .robust import (
    HCType,
    RobustInferenceResult,
    bootstrap_inference,
    robust_covariance,
)

__all__ = [
    "confidence_intervals",
    "ConfidenceIntervalResult",
    "glm_diagnostics",
    "GLMDiagnosticResult",
    "wald_test",
    "robust_covariance",
    "bootstrap_inference",
    "RobustInferenceResult",
    "HCType",
]