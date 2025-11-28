r"""Probability Distribution Families and Link Functions for Aurora-GLM.

This module provides the mathematical foundation for Generalized Linear Models:
exponential family distributions and their associated link functions.

Mathematical Framework
======================

Exponential Family Distributions
--------------------------------
A distribution belongs to the exponential family if its density/mass can
be written in canonical form:

    f(y; θ, φ) = exp{[yθ - b(θ)] / a(φ) + c(y, φ)}

where:
    - y ∈ Y is the response (random variable)
    - θ is the **canonical (natural) parameter**
    - φ > 0 is the **dispersion parameter**
    - b(θ) is the **cumulant function** (log-partition function)
    - a(φ) is typically φ/w for prior weight w
    - c(y, φ) is the normalizing constant

**Key Properties**:

1. **Mean**: μ = E[Y] = b'(θ) = ∂b/∂θ
2. **Variance**: Var(Y) = a(φ) b''(θ) = a(φ) V(μ)
3. **Variance function**: V(μ) = b''(θ) characterizes the family

**Maximum Likelihood**:
The score equation is:

    ∂ℓ/∂θ = (y - μ) / a(φ) = 0

so μ̂ = ȳ for the saturated model.

Standard Families
-----------------

**Gaussian** (Normal):
    - Support: Y ∈ ℝ
    - Canonical form: θ = μ, φ = σ²
    - Cumulant: b(θ) = θ²/2
    - Variance function: V(μ) = 1 (constant)
    - Canonical link: identity g(μ) = μ

**Poisson** (Count data):
    - Support: Y ∈ {0, 1, 2, ...}
    - Canonical form: θ = log(μ), φ = 1 (fixed)
    - Cumulant: b(θ) = exp(θ) = μ
    - Variance function: V(μ) = μ (mean = variance)
    - Canonical link: log g(μ) = log(μ)

**Binomial** (Binary/proportion data):
    - Support: Y ∈ {0, 1, ..., n}
    - Canonical form: θ = log[μ/(n-μ)], φ = 1 (fixed)
    - Cumulant: b(θ) = n log(1 + exp(θ))
    - Variance function: V(μ) = μ(n-μ)/n
    - Canonical link: logit g(μ) = log(μ/(1-μ))

**Gamma** (Positive continuous):
    - Support: Y ∈ (0, ∞)
    - Canonical form: θ = -1/μ, φ = 1/α (shape⁻¹)
    - Cumulant: b(θ) = -log(-θ)
    - Variance function: V(μ) = μ² (CV constant)
    - Canonical link: inverse g(μ) = 1/μ

Heavy-Tailed Distributions
---------------------------

**Student's t** (Robust regression):
    - Heavier tails than Gaussian
    - Parameterized by degrees of freedom ν
    - As ν → ∞, approaches Gaussian
    - Robust to outliers

**Negative Binomial** (Overdispersed counts):
    - For count data with Var(Y) > E[Y]
    - Variance function: V(μ) = μ + μ²/θ
    - θ controls overdispersion (θ → ∞ gives Poisson)

**Tweedie** (Zero-inflated continuous):
    - For semi-continuous data with exact zeros
    - Power variance: V(μ) = μ^p for 1 < p < 2
    - p = 1: Poisson limit
    - p = 2: Gamma limit

Link Functions
--------------
A link function g(·) relates the mean μ to the linear predictor η:

    η = g(μ) = X^T β

**Requirements**:
    - Monotonic and differentiable
    - Maps mean space to ℝ
    - Inverse μ = g⁻¹(η) must exist

**Standard Links**:

| Link          | g(μ)            | g⁻¹(η)           | Domain         |
|---------------|-----------------|------------------|----------------|
| Identity      | μ               | η                | ℝ              |
| Log           | log(μ)          | exp(η)           | (0, ∞)         |
| Logit         | log(μ/(1-μ))    | exp(η)/(1+exp(η))| (0, 1)         |
| Probit        | Φ⁻¹(μ)          | Φ(η)             | (0, 1)         |
| Complementary log-log | log(-log(1-μ)) | 1-exp(-exp(η)) | (0, 1)   |
| Inverse       | 1/μ             | 1/η              | (0, ∞) or ℝ\{0}|
| Sqrt          | √μ              | η²               | [0, ∞)         |

**Canonical Links**:
When g(μ) = θ (canonical parameter), the link is called canonical.
Benefits:
    - Simplified Fisher information
    - IRLS = Fisher scoring exactly
    - Sufficient statistics are linear in data

Deviance
--------
The deviance measures goodness-of-fit:

    D(y; μ) = 2 [ℓ(y; y) - ℓ(μ; y)]

where ℓ(y; y) is the saturated model log-likelihood.

**Family-specific formulas**:

| Family   | Deviance dᵢ = D(yᵢ; μᵢ)                        |
|----------|------------------------------------------------|
| Gaussian | (yᵢ - μᵢ)²                                    |
| Poisson  | 2[yᵢ log(yᵢ/μᵢ) - (yᵢ - μᵢ)]                 |
| Binomial | 2[yᵢ log(yᵢ/μᵢ) + (nᵢ-yᵢ) log((nᵢ-yᵢ)/(nᵢ-μᵢ))]|
| Gamma    | 2[-log(yᵢ/μᵢ) + (yᵢ - μᵢ)/μᵢ]                |

**Properties**:
    - D ≥ 0 with equality iff μ = y
    - Asymptotically D ~ χ²_{n-p} under correct model
    - Used for model comparison via likelihood ratio tests

Multi-Backend Support
---------------------
All distributions support NumPy, PyTorch, and JAX backends through
the array namespace abstraction. Use the `namespace()` function from
`aurora.distributions._utils` to detect and work with arrays.

Available Components
====================

Base Classes
------------
- Family: Abstract base for distribution families
- LinkFunction: Abstract base for link functions

Distribution Families
---------------------
- GaussianFamily: Normal distribution
- PoissonFamily: Poisson distribution
- BinomialFamily: Binomial distribution
- GammaFamily: Gamma distribution

Link Functions
--------------
- IdentityLink: g(μ) = μ
- LogLink: g(μ) = log(μ)
- LogitLink: g(μ) = log(μ/(1-μ))
- InverseLink: g(μ) = 1/μ

See Also
--------
aurora.distributions.families : Complete family implementations
aurora.distributions.links : All link function implementations
aurora.distributions.base : Mathematical framework documentation

References
----------
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*
  (2nd ed.). Chapman and Hall/CRC. Chapters 2 (Exponential Family) and 4 (Link).

- Jørgensen, B. (1997). *The Theory of Dispersion Models*. Chapman & Hall.

- Dunn, P. K., & Smyth, G. K. (2018). *Generalized Linear Models with
  Examples in R*. Springer. Chapters 3-4.
"""
from __future__ import annotations

from .base import Family, LinkFunction
from .families.binomial import BinomialFamily
from .families.gamma import GammaFamily
from .families.gaussian import GaussianFamily
from .families.poisson import PoissonFamily
from .links import IdentityLink, InverseLink, LogLink, LogitLink

__all__ = [
	"Family",
	"LinkFunction",
	"GaussianFamily",
	"BinomialFamily",
	"PoissonFamily",
	"GammaFamily",
	"IdentityLink",
	"LogLink",
	"LogitLink",
	"InverseLink",
]
