"""High-Level Modeling Interface for Aurora-GLM.

This module provides the main API for fitting Generalized Linear Models (GLM),
Generalized Additive Models (GAM), and Generalized Additive Mixed Models (GAMM).

Model Hierarchy
===============

Generalized Linear Models (GLM)
-------------------------------
GLMs extend linear regression to non-Gaussian responses:

    g(E[Y]) = Xβ

where:
    - g(·) is a link function
    - E[Y] = μ follows an exponential family distribution
    - X is the design matrix
    - β are regression coefficients

**Components**:
1. Random component: Y ~ Exponential Family (Gaussian, Poisson, Binomial, Gamma)
2. Systematic component: Linear predictor η = Xβ
3. Link function: g(μ) = η

**Fitting**: Iteratively Reweighted Least Squares (IRLS)
See `aurora.models.glm.fitting` for algorithm details.

Generalized Additive Models (GAM)
---------------------------------
GAMs extend GLMs with smooth non-parametric effects:

    g(E[Y]) = β₀ + f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ)

where fⱼ(·) are smooth functions represented using basis expansions:

    fⱼ(x) = Σₖ βⱼₖ φⱼₖ(x)

**Penalized likelihood**:

    ℓₚ(β) = ℓ(β) - ½ Σⱼ λⱼ βⱼᵀSⱼβⱼ

where:
    - λⱼ are smoothing parameters
    - Sⱼ are penalty matrices (roughness)

**Smoothing parameter selection**:
    - GCV (Generalized Cross-Validation)
    - REML (Restricted Maximum Likelihood)

See `aurora.models.gam.fitting` for algorithm details.

Generalized Additive Mixed Models (GAMM)
-----------------------------------------
GAMMs combine smooth functions with random effects:

    g(E[Y|b]) = Xβ + Σⱼ fⱼ(xⱼ) + Zb

where:
    - Xβ: Parametric fixed effects
    - fⱼ(xⱼ): Smooth functions (non-parametric)
    - Zb: Random effects, b ~ N(0, Ψ)

**For Gaussian responses**: Reduces to Linear Mixed Model (LMM)

**For non-Gaussian responses**: 
    - PQL (Penalized Quasi-Likelihood)
    - Laplace approximation
    - Adaptive Gauss-Hermite quadrature

See `aurora.models.gamm.fitting` for algorithm details.

Quick Start Examples
====================

GLM (Logistic Regression)
-------------------------
    >>> from aurora.models import fit_glm
    >>> result = fit_glm(X, y, family='binomial', link='logit')
    >>> print(result.coefficients)
    >>> print(result.summary())

GAM (Smooth Effects)
--------------------
    >>> from aurora.models import fit_gam
    >>> result = fit_gam(X, y, n_splines=10, family='gaussian')
    >>> print(result.smooth_terms)
    >>> result.plot_smooth()

GAMM (Random Effects)
---------------------
    >>> from aurora.models import fit_gamm
    >>> result = fit_gamm(
    ...     X, y, groups,
    ...     family='gaussian',
    ...     random_effects={'intercept': True}
    ... )
    >>> print(result.variance_components)
    >>> print(result.random_effects)

Result Objects
==============

All model-fitting functions return result objects with:

**Common attributes**:
    - coefficients: Estimated parameters
    - fitted_values: Predicted E[Y|X]
    - residuals: y - fitted_values
    - deviance: Goodness-of-fit measure
    - log_likelihood: Log-likelihood at MLE
    - aic, bic: Information criteria

**GLM-specific**:
    - standard_errors: SE of coefficients
    - z_values, p_values: Hypothesis tests

**GAM-specific**:
    - smooth_terms: Information about smooth functions
    - edf: Effective degrees of freedom per smooth
    - gcv_score: GCV criterion value

**GAMM-specific**:
    - variance_components: σ², Ψ estimates
    - random_effects: BLUPs of random effects
    - conditional_modes: Mode of b|y

Model Comparison
================

**Nested models**:
    >>> from aurora.inference import likelihood_ratio_test
    >>> lrt = likelihood_ratio_test(model_reduced, model_full)
    >>> print(f"LRT statistic: {lrt.statistic}, p-value: {lrt.p_value}")

**Information criteria**:
    >>> print(f"AIC: {result.aic}, BIC: {result.bic}")
    >>> # Lower is better

**ANOVA**:
    >>> from aurora.inference import anova_glm
    >>> anova_glm(result).summary()

Multi-Backend Support
=====================

All models support NumPy, PyTorch, and JAX backends:

    >>> result = fit_glm(X, y, family='gaussian', backend='torch', device='cuda')

For PyTorch and JAX, automatic differentiation can be used for
gradient-based optimization and uncertainty quantification.

Available Functions
===================

GLM
---
- fit_glm: Fit a Generalized Linear Model
- predict_glm: Generate predictions from fitted GLM

GAM
---
- fit_gam: Fit a Generalized Additive Model

GAMM
----
- fit_gamm: Fit a Generalized Additive Mixed Model
- fit_gamm_with_smooth: Fit GAMM with explicit smooth terms
- predict_from_gamm: Generate predictions from fitted GAMM

Result Classes
--------------
- ModelResult: Base class for all model results
- GLMResult: Results from GLM fitting

See Also
--------
aurora.models.glm : GLM implementation details
aurora.models.gam : GAM implementation details
aurora.models.gamm : GAMM implementation details
aurora.distributions : Distribution families and links
aurora.smoothing : Smoothing primitives

References
----------
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*
  (2nd ed.). Chapman and Hall/CRC.

- Hastie, T., & Tibshirani, R. (1990). *Generalized Additive Models*.
  Chapman and Hall/CRC.

- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). CRC Press.

- Pinheiro, J. C., & Bates, D. M. (2000). *Mixed-Effects Models in S and S-PLUS*.
  Springer.
"""
from __future__ import annotations

from .base import GLMResult, ModelResult
from .gam import fit_gam
from .gamm import fit_gamm, fit_gamm_with_smooth, predict_from_gamm
from .glm import fit_glm, predict_glm

__all__ = [
    "ModelResult",
    "GLMResult",
    "fit_glm",
    "predict_glm",
    "fit_gam",
    "fit_gamm",
    "fit_gamm_with_smooth",
    "predict_from_gamm",
]
