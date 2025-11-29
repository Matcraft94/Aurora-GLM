"""Abstract base classes for GLM distribution families and link functions.

Mathematical Framework
----------------------
This module defines the core abstractions for exponential family distributions
and link functions used in Generalized Linear Models (GLM).

Exponential Family of Distributions
------------------------------------
A distribution belongs to the exponential family if its density/mass function
can be written as:

    f(y; θ, φ) = exp{[y θ - b(θ)] / a(φ) + c(y, φ)}

where:
    - y is the response variable
    - θ is the canonical (natural) parameter
    - φ is the dispersion parameter
    - b(θ) is the cumulant function
    - a(φ) is a known function (typically a(φ) = φ/w for weight w)
    - c(y, φ) is a normalization term

**Properties**:

1. **Mean-variance relationship**:
   - E[Y] = μ = b'(θ)
   - Var(Y) = a(φ) b''(θ) = a(φ) V(μ)

   where V(μ) is the variance function (family-specific).

2. **Log-likelihood**:
   ℓ(θ, φ; y) = [y θ - b(θ)] / a(φ) + c(y, φ)

3. **Score function**:
   ∂ℓ/∂θ = [y - μ] / a(φ)

Link Functions
--------------
A link function g(·) relates the mean μ to the linear predictor η:

    g(μ) = η = X^T β

**Requirements**:
- Monotonic: g is strictly increasing or decreasing
- Differentiable: g'(μ) exists and is continuous
- Domain: g maps (support of μ) → ℝ

**Canonical link**:
The canonical link is g(μ) = θ, i.e., when the linear predictor equals
the natural parameter. This leads to simplifications:
- Fisher information matrix is particularly simple
- IRLS = Fisher scoring

**Common links**:

| Family    | Canonical Link | Other Links           |
|-----------|----------------|-----------------------|
| Gaussian  | Identity       | Log, Inverse          |
| Poisson   | Log            | Identity, Sqrt        |
| Binomial  | Logit          | Probit, CLogLog       |
| Gamma     | Inverse        | Log, Identity         |

Variance Functions
------------------
The variance function V(μ) characterizes the mean-variance relationship:

    Var(Y) = φ V(μ)

where φ is the dispersion parameter.

**Standard families**:

| Family    | Support      | V(μ)         | φ                |
|-----------|--------------|--------------|------------------|
| Gaussian  | (-∞, ∞)      | 1            | σ²               |
| Poisson   | {0,1,2,...}  | μ            | 1 (fixed)        |
| Binomial  | {0,1,...,n}  | μ(n-μ)/n     | 1 (fixed)        |
| Gamma     | (0, ∞)       | μ²           | shape^{-1}       |

Deviance
--------
The deviance measures goodness-of-fit:

    D(y; μ) = 2 φ [ℓ(y; y) - ℓ(μ; y)]

where ℓ(y; y) is the saturated model log-likelihood.

**Properties**:
- D ≥ 0 with equality iff μ = y
- Asymptotically χ² distributed under H₀: correct model
- Used for model comparison via likelihood ratio tests

Initialization
--------------
Starting values μ⁽⁰⁾ for IRLS must satisfy:
1. μ⁽⁰⁾ ∈ support of the distribution
2. V(μ⁽⁰⁾) > 0 (non-degenerate)
3. Avoid boundary of parameter space

**Common strategies**:
- Gaussian: μ⁽⁰⁾ = y
- Poisson: μ⁽⁰⁾ = y + 0.1 (avoid zeros)
- Binomial: μ⁽⁰⁾ = (y + 0.5) / (n + 1) (avoid 0 and 1)
- Gamma: μ⁽⁰⁾ = y (already positive)

Implementation Notes
--------------------
**Multi-backend support**:
All implementations must work with NumPy, PyTorch, and JAX via the
array namespace abstraction. Use the `namespace()` function to get
the appropriate backend.

**Numerical stability**:
- Avoid log(0), exp(large), division by zero
- Use log-sum-exp trick where appropriate
- Clip μ to valid range with small epsilon margin

**Type annotations**:
- Array: Generic array type (works with all backends)
- Scalar: Single numerical value (float or int)

Extending with Custom Families
-------------------------------
To add a new distribution family:

1. Inherit from `Family`:
   ```python
   from aurora.distributions.base import Family, LinkFunction

   class MyFamily(Family):
       def log_likelihood(self, y, mu, **params):
           # Implement log p(y | μ)
           ...

       def deviance(self, y, mu, **params):
           # Implement 2[ℓ(y; y) - ℓ(μ; y)]
           ...

       def variance(self, mu, **params):
           # Implement V(μ)
           ...

       def initialize(self, y):
           # Return starting μ⁽⁰⁾
           ...

       @property
       def default_link(self):
           # Return canonical LinkFunction
           return MyLink()
   ```

2. Implement the corresponding link function:
   ```python
   class MyLink(LinkFunction):
       def link(self, mu):
           return ...  # g(μ)

       def inverse(self, eta):
           return ...  # g⁻¹(η)

       def derivative(self, mu):
           return ...  # dg/dμ
   ```

3. Register in the family registry (optional):
   ```python
   _FAMILY_REGISTRY['myfamily'] = MyFamily
   ```

References
----------
**Exponential family theory**:

- Barndorff-Nielsen, O. (2014). *Information and Exponential Families in
  Statistical Theory*. John Wiley & Sons.

- Brown, L. D. (1986). *Fundamentals of Statistical Exponential Families*.
  Institute of Mathematical Statistics.

**GLM families**:

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.).
  Chapman and Hall/CRC. Chapter 2: Model Specification.
  https://doi.org/10.1007/978-1-4899-3242-6

- Dobson, A. J., & Barnett, A. G. (2018). *An Introduction to Generalized
  Linear Models* (4th ed.). CRC Press.

**Variance functions and quasi-likelihood**:

- Wedderburn, R. W. M. (1974). "Quasi-likelihood functions, generalized linear
  models, and the Gauss-Newton method." *Biometrika*, 61(3), 439-447.
  https://doi.org/10.1093/biomet/61.3.439

**Deviance and model selection**:

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
  Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.).
  Springer. https://doi.org/10.1007/978-0-387-84858-7 (Chapter 7)

See Also
--------
aurora.distributions.families : Concrete family implementations
aurora.distributions.links : Link function implementations
aurora.models.glm.fitting : GLM fitting using families

Notes
-----
For detailed mathematical derivations, see REFERENCES.md in the repository root.

The exponential family framework unifies many common distributions and provides
a consistent theory for inference (MLE, Fisher information, asymptotic normality).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import Array, Scalar


class LinkFunction(ABC):
    """Abstract base class for link functions."""

    @abstractmethod
    def link(self, mu: Array) -> Array:
        """Apply the link function ``g(mu)``."""

    @abstractmethod
    def inverse(self, eta: Array) -> Array:
        """Apply the inverse link ``g^{-1}(eta)``."""

    @abstractmethod
    def derivative(self, mu: Array) -> Array:
        """Return the derivative ``dg/dmu`` evaluated at ``mu``."""


class Family(ABC):
    """Abstract base class for probability distribution families."""

    @abstractmethod
    def log_likelihood(self, y: Array, mu: Array, **params) -> Scalar:
        """Return the log-likelihood of observations ``y`` given mean ``mu``."""

    @abstractmethod
    def deviance(self, y: Array, mu: Array, **params) -> Scalar:
        """Return the deviance contribution for observations ``y`` and mean ``mu``."""

    @abstractmethod
    def variance(self, mu: Array, **params) -> Array:
        """Return the variance function evaluated at ``mu``."""

    @abstractmethod
    def initialize(self, y: Array) -> Array:
        """Return starting values for the mean parameter ``mu`` given data ``y``."""

    @property
    @abstractmethod
    def default_link(self) -> LinkFunction:
        """Return the canonical link for this family."""


__all__ = ["Family", "LinkFunction"]
