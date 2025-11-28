# Phase 5: Extended Features - Design Document

This document outlines the design and implementation plan for Phase 5 features in Aurora-GLM.

---

## Overview

Phase 5 extends the GAMM implementation with advanced features:

1. **Smooth terms in non-Gaussian GAMM** (5.1)
2. **Laplace approximation for GLMM** (5.2)
3. **Additional covariance structures** (5.3)
4. **GAMM visualization tools** (5.4)

---

## 5.1 Smooth Terms in Non-Gaussian GAMM

### Objective

Enable smooth functions in PQL-estimated GAMMs:

```python
fit_gamm(
    formula='count ~ s(time) + s(temperature) + (1 | subject)',
    data=df,
    family='poisson'
)
```

### Mathematical Framework

**Model specification**:

```
g(E[Y_ij | b_i]) = f_1(x_{ij1}) + f_2(x_{ij2}) + ... + Z_ij^T b_i
```

where:
- g(·) is the link function
- f_k(·) are smooth functions represented as spline bases
- b_i ~ N(0, Ψ) are random effects

**Representation**:

Each smooth term is represented using basis expansion:

```
f_k(x) = Σ_m β_km B_km(x)
```

Combined model:

```
η_ij = X_para^T β_para + X_smooth^T β_smooth + Z_ij^T b_i
```

where:
- X_para: parametric design matrix
- X_smooth: smooth term basis matrix (concatenated for all smooths)
- β_smooth: smooth coefficients
- Z: random effects design matrix

### PQL with Smooth Terms

**Modified PQL algorithm**:

**Step 1**: Update (β_para, β_smooth, b) given Ψ and λ

Inner loop:
1. Compute linear predictor:
   ```
   η = X_para β_para + X_smooth β_smooth + Z b
   ```

2. Compute working response and weights (as before)

3. Solve penalized weighted mixed model equations:
   ```
   [X_para^T W X_para          X_para^T W X_smooth      X_para^T W Z    ] [β_para  ]   [X_para^T W z]
   [X_smooth^T W X_para  X_smooth^T W X_smooth + λS  X_smooth^T W Z    ] [β_smooth] = [X_smooth^T W z]
   [Z^T W X_para              Z^T W X_smooth           Z^T W Z + Ψ^{-1}] [b       ]   [Z^T W z]
   ```

   where S = block-diagonal penalty matrix for all smooth terms

**Step 2**: Update λ (smoothing parameters)

Options:
- GCV on working response
- REML on pseudo-data (Wood, 2011)
- Performance iteration (alternate between β/b and λ)

**Step 3**: Update Ψ (variance components) - as before

### Implementation Plan

#### Phase 5.1.1: Foundation (Week 1)

**File**: `aurora/models/gamm/pql_smooth.py`

```python
def fit_pql_with_smooth(
    X_parametric: np.ndarray,
    X_smooth_dict: dict[str, np.ndarray],
    Z: np.ndarray,
    Z_info: list[dict],
    y: np.ndarray,
    family: str,
    S_smooth_dict: dict[str, np.ndarray],
    lambda_smooth: dict[str, float] | None = None,
    covariance: str = 'unstructured',
    maxiter_outer: int = 20,
    maxiter_inner: int = 10,
    tol_outer: float = 1e-4,
    tol_inner: float = 1e-6,
) -> GAMMResult:
    """Fit non-Gaussian GAMM with smooth terms using PQL.

    Parameters
    ----------
    X_parametric : ndarray (n, p_para)
        Parametric fixed effects design matrix
    X_smooth_dict : dict[str, ndarray]
        Smooth term basis matrices {term_name: X_smooth}
    Z : ndarray (n, q)
        Random effects design matrix
    Z_info : list[dict]
        Random effects metadata
    y : ndarray (n,)
        Response vector
    family : str
        Distribution family
    S_smooth_dict : dict[str, ndarray]
        Penalty matrices for each smooth term
    lambda_smooth : dict[str, float], optional
        Smoothing parameters. If None, use GCV.
    covariance : str
        Covariance structure for random effects
    ...

    Returns
    -------
    result : GAMMResult
        Fitted model with smooth coefficients
    """
```

**Key functions**:

1. `_construct_augmented_system()`: Build the full mixed model equations
2. `_solve_penalized_mixed_equations()`: Solve with smooth penalties
3. `_update_smoothing_parameters()`: GCV or REML for λ
4. `_compute_edf_smooth()`: Effective degrees of freedom for smooth terms

#### Phase 5.1.2: Integration with Formula Parser (Week 1)

Extend `fit_gamm()` to handle smooth terms in formula:

```python
# In aurora/models/gamm/interface.py

def fit_gamm(...):
    # Parse formula
    spec = parse_formula(formula)

    # Build smooth bases
    X_smooth_dict = {}
    S_smooth_dict = {}
    for smooth_term in spec.smooth_terms:
        basis = create_basis(smooth_term, data)
        X_smooth_dict[smooth_term.name] = basis.design_matrix
        S_smooth_dict[smooth_term.name] = basis.penalty_matrix

    # Dispatch to appropriate fitting function
    if family == 'gaussian':
        # Use existing fit_gamm_gaussian
        ...
    else:
        # Use new fit_pql_with_smooth
        result = fit_pql_with_smooth(
            X_parametric=X_para,
            X_smooth_dict=X_smooth_dict,
            Z=Z,
            Z_info=Z_info,
            y=y,
            family=family,
            S_smooth_dict=S_smooth_dict,
            ...
        )
```

#### Phase 5.1.3: Smoothing Parameter Selection (Week 2)

**GCV for non-Gaussian GAMM**:

```python
def gcv_pql(
    X_para: np.ndarray,
    X_smooth: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    family: str,
    S: np.ndarray,
    lambda_candidates: np.ndarray,
    Psi: np.ndarray,
) -> float:
    """Select λ via GCV on PQL working response.

    Algorithm:
    1. For each λ candidate:
       a. Fit PQL with fixed λ (inner loop only)
       b. Compute EDF from smoother matrix
       c. Compute GCV score on working response
    2. Return λ with minimum GCV
    """
    gcv_scores = []
    for lambda_ in lambda_candidates:
        # Fit with fixed lambda (simplified PQL)
        beta, b, z_working, W = _pql_inner_loop(...)

        # Compute smoother matrix
        H = _compute_smoother_matrix(X_smooth, W, S, lambda_)
        edf = np.trace(H)

        # GCV score
        residuals = z_working - (X_para @ beta + X_smooth @ beta_smooth)
        gcv = (n / (n - edf)**2) * np.sum(W * residuals**2)
        gcv_scores.append(gcv)

    return lambda_candidates[np.argmin(gcv_scores)]
```

**REML for non-Gaussian GAMM** (more accurate):

Based on Wood (2011), use Laplace approximation of REML:

```
-2 ℓ_R(λ) ≈ log|H| + log|S + λI| + ...
```

where H is the Hessian of the penalized log-likelihood.

#### Phase 5.1.4: Testing (Week 2)

**Test cases**:

1. **Poisson GAMM with 1 smooth term**:
   ```python
   # Example: Species counts over time with random site effects
   fit_gamm(
       formula='count ~ s(time, n_basis=10) + (1 | site)',
       data=df,
       family='poisson'
   )
   ```

2. **Binomial GAMM with multiple smooth terms**:
   ```python
   # Example: Disease probability vs age and temperature
   fit_gamm(
       formula='disease ~ s(age) + s(temperature) + (1 | clinic)',
       data=df,
       family='binomial'
   )
   ```

3. **Gamma GAMM with smooth and parametric terms**:
   ```python
   fit_gamm(
       formula='response ~ s(x1) + x2 + x3 + (1 + x2 | group)',
       data=df,
       family='gamma'
   )
   ```

**Validation**:
- Compare against R mgcv::gamm() for Gaussian family
- Check convergence properties
- Verify smoothing parameter selection
- Test edge cases (very small/large λ)

---

## 5.2 Laplace Approximation for GLMM

### Objective

Implement Laplace approximation as a more accurate alternative to PQL for non-Gaussian GAMM.

### Mathematical Framework

**Marginal likelihood approximation**:

The marginal log-likelihood is:

```
ℓ(β, Ψ) = log ∫ p(y | b, β) p(b | Ψ) db
```

**Laplace approximation** (Breslow & Lin, 1995):

```
ℓ(β, Ψ) ≈ log p(y | b̂, β) + log p(b̂ | Ψ) - ½ log|H(b̂)|
```

where:
- b̂ = argmax_b [log p(y | b, β) + log p(b | Ψ)]
- H(b) is the Hessian of -[log p(y | b, β) + log p(b | Ψ)]

**Advantages over PQL**:
- More accurate for small cluster sizes
- Less biased for binary data
- Better performance with large random effect variance

**Disadvantages**:
- Computationally more expensive
- Requires Hessian computation and inversion

### Algorithm

**Laplace GLMM estimation**:

1. Initialize β^(0), Ψ^(0)

2. For t = 1, 2, ... until convergence:

   a. **Update b̂** given (β, Ψ):
      Maximize h(b) = log p(y | b, β) + log p(b | Ψ)

      Using Newton-Raphson:
      ```
      b^{k+1} = b^k - H(b^k)^{-1} ∇h(b^k)
      ```

   b. **Compute Hessian**:
      ```
      H(b̂) = -∇²_b [log p(y | b, β) + log p(b | Ψ)]|_{b=b̂}
      ```

   c. **Update (β, Ψ)** via Newton-Raphson on approximate likelihood:
      ```
      ℓ_L(β, Ψ) = log p(y | b̂, β) + log p(b̂ | Ψ) - ½ log|H(b̂)|
      ```

### Implementation Plan

**File**: `aurora/models/gamm/laplace.py`

```python
def fit_laplace_glmm(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    family: str,
    psi_init: np.ndarray | None = None,
    maxiter_outer: int = 50,
    maxiter_inner: int = 20,
    tol_outer: float = 1e-6,
    tol_inner: float = 1e-8,
) -> LaplacleResult:
    """Fit GLMM using Laplace approximation.

    More accurate than PQL but computationally expensive.
    Recommended for:
    - Small cluster sizes (n_i < 10)
    - Binary outcomes with rare events
    - High-precision inference
    """
```

**Key challenges**:
- Efficient Hessian computation (use automatic differentiation?)
- Numerical stability of log|H|
- Convergence diagnostics

---

## 5.3 Additional Covariance Structures

### Objective

Support more flexible covariance structures for longitudinal/spatial data.

### Structures to Implement

#### 5.3.1 AR(1) - Autoregressive Order 1

**Model**:
```
Cov(b_it, b_is) = σ² ρ^{|t-s|}
```

**Use case**: Longitudinal data with temporal correlation

**Implementation**:
```python
class AR1Covariance(CovarianceStructure):
    def __init__(self, rho: float, sigma2: float):
        self.rho = rho    # Correlation parameter ∈ (-1, 1)
        self.sigma2 = sigma2

    def matrix(self, n_timepoints: int) -> np.ndarray:
        """Construct n×n AR(1) covariance matrix."""
        i, j = np.ogrid[:n_timepoints, :n_timepoints]
        return self.sigma2 * self.rho ** np.abs(i - j)

    def inverse(self, n_timepoints: int) -> np.ndarray:
        """Efficient inverse for AR(1) (tridiagonal)."""
        # Use banded matrix solver
        ...
```

#### 5.3.2 Compound Symmetry (Exchangeable)

**Model**:
```
Cov(b_it, b_is) = { σ²       if t = s
                  { σ² ρ     if t ≠ s
```

**Use case**: Clustered data with constant correlation

**Implementation**: Already partially supported via `'identity'` covariance

#### 5.3.3 Exponential Spatial

**Model**:
```
Cov(b_i, b_j) = σ² exp(-d_ij / φ)
```

where d_ij is the distance between locations i and j.

**Use case**: Spatial data (geostatistics)

### Integration

Extend `RandomEffect` class:

```python
re = RandomEffect(
    grouping='subject',
    variables=(1,),  # time variable
    covariance='ar1',
    covariance_params={'rho': 0.7}  # Initial value
)
```

---

## 5.4 GAMM Visualization Tools

### Objective

Provide diagnostic and interpretive plots for GAMM results.

### Plots to Implement

#### 5.4.1 Caterpillar Plots (Random Effect Intervals)

```python
def plot_caterpillar(
    result: GAMMResult,
    level: float = 0.95,
    sort_by: str = 'magnitude',
) -> Figure:
    """Plot random effect estimates with confidence intervals.

    Shows:
    - Point estimates for each random effect
    - Confidence intervals
    - Zero reference line
    - Sorted by magnitude/group
    """
```

**Example**:
```
           Random Intercepts (95% CI)

Group 1    •--------
Group 2         •-----
Group 3    •---------
Group 4             •------
...
       -1.0  -0.5   0.0   0.5   1.0
```

#### 5.4.2 Q-Q Plots for Random Effects

```python
def plot_qq_random(result: GAMMResult) -> Figure:
    """Q-Q plot to check normality of random effects.

    Diagnostic for assumption b_i ~ N(0, Ψ).
    """
```

#### 5.4.3 Partial Effect Plots (Smooth Terms)

```python
def plot_smooth_effect(
    result: GAMMResult,
    term_name: str,
    level: float = 0.95,
) -> Figure:
    """Plot estimated smooth function with confidence band.

    Shows:
    - Estimated smooth f(x)
    - Pointwise confidence intervals
    - Rug plot of data density
    - Partial residuals (optional)
    """
```

#### 5.4.4 Residual Diagnostics

```python
def plot_diagnostics(result: GAMMResult) -> Figure:
    """Create 2x2 diagnostic plot panel.

    Includes:
    - Residuals vs fitted
    - Q-Q plot of residuals
    - Scale-location plot
    - Residuals vs leverage (Cook's distance)
    """
```

**File**: `aurora/models/gamm/plotting.py`

---

## Implementation Timeline

### Week 1-2: Smooth Terms in PQL
- [ ] Implement `fit_pql_with_smooth()`
- [ ] Integrate with formula parser
- [ ] Add GCV selection for smooth parameters
- [ ] Write tests (Poisson, Binomial, Gamma)

### Week 3-4: Laplace Approximation
- [ ] Implement core Laplace algorithm
- [ ] Add automatic differentiation for Hessian
- [ ] Compare accuracy vs PQL
- [ ] Documentation and examples

### Week 5: Covariance Structures
- [ ] Implement AR(1) covariance
- [ ] Implement compound symmetry
- [ ] Extend RandomEffect interface
- [ ] Add tests for longitudinal data

### Week 6: Visualization
- [ ] Caterpillar plots
- [ ] Q-Q plots
- [ ] Smooth effect plots
- [ ] Diagnostic panel
- [ ] Gallery of examples

---

## Success Criteria

**Phase 5.1 (Smooth GAMM)**:
- ✓ Formula `y ~ s(x) + (1|g)` works with Poisson/Binomial
- ✓ Smoothing parameters selected automatically
- ✓ Results comparable to R mgcv (when available)
- ✓ 90%+ test coverage

**Phase 5.2 (Laplace)**:
- ✓ More accurate than PQL for small clusters
- ✓ Converges reliably
- ✓ Documented when to use vs PQL

**Phase 5.3 (Covariance)**:
- ✓ AR(1) works for longitudinal data
- ✓ Spatial correlation works for geostatistics
- ✓ Easy to specify in formula/interface

**Phase 5.4 (Visualization)**:
- ✓ Publication-quality plots
- ✓ Follows R mgcv style
- ✓ Customizable and extensible

---

## References

**Smooth terms in GLMM**:
- Wood, S. N. (2017). *GAMs: An Introduction with R* (2nd ed.). Chapter 6.
- Lin, X., & Zhang, D. (1999). "Inference in generalized additive mixed models by using smoothing splines." *JRSS B*, 61(2), 381-400.

**Laplace approximation**:
- Breslow, N. E., & Lin, X. (1995). "Bias correction in generalised linear mixed models with a single component of dispersion." *Biometrika*, 82(1), 81-91.
- Rue, H., Martino, S., & Chopin, N. (2009). "Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations." *JRSS B*, 71(2), 319-392.

**Covariance structures**:
- Pinheiro, J. C., & Bates, D. M. (2000). *Mixed-Effects Models in S and S-PLUS*. Springer. Chapter 5.
- Diggle, P., Heagerty, P., Liang, K. Y., & Zeger, S. (2002). *Analysis of Longitudinal Data* (2nd ed.). Oxford.

---

*Last updated: 2025-01-26*
*Aurora-GLM Version: 0.5.0-dev*

---

## 5.5 Heavy-Tailed Distributions for Robust Modeling

### Objective

Support robust statistical modeling with heavy-tailed distributions that accommodate outliers and extreme values better than standard exponential family distributions.

### Motivation

**When to use heavy-tailed distributions**:

1. **Outliers present**: Data contains extreme values that are legitimate (not errors)
2. **Robustness required**: Model should not be overly influenced by unusual observations
3. **Thick tails empirical**: Q-Q plots or residual analysis show heavier tails than Gaussian/Poisson
4. **Overdispersion**: Count data with variance >> mean (Negative Binomial)
5. **Zero-inflation + continuous**: Insurance claims, actuarial data (Tweedie)

### Distributions to Implement

#### 5.5.1 Student's t Distribution

**Density**:
```
f(y; μ, σ, ν) = Γ((ν+1)/2) / [√(νπ)σ Γ(ν/2)] × [1 + (y-μ)²/(νσ²)]^{-(ν+1)/2}
```

where:
- μ is the location parameter
- σ² is the scale parameter  
- ν > 0 is degrees of freedom (shape parameter)

**Properties**:
- E[Y] = μ for ν > 1
- Var(Y) = νσ²/(ν-2) for ν > 2
- ν → ∞: converges to Gaussian
- ν small (ν ≈ 3-5): very heavy tails
- ν = 1: Cauchy distribution (infinite variance!)

**Use cases**:
- Financial returns (fat tails)
- Robust regression (resistant to outliers)
- Contaminated normal data

**Link functions**:
- Canonical: Identity (location parameter)
- Alternative: Log (for positive responses)

**Implementation**:

```python
class StudentTFamily(Family):
    """Student's t distribution family for robust GLM.
    
    Parameters
    ----------
    df : float
        Degrees of freedom (ν). Fixed or estimated via profile likelihood.
        Common choices:
        - df=3: Very heavy tails
        - df=5: Moderately heavy tails
        - df=10: Nearly Gaussian
        - df='estimate': Estimate from data (computationally expensive)
    
    Notes
    -----
    For GLM/GAMM, we typically fix df rather than estimate it.
    Estimation requires iterative profiling or Bayesian methods.
    """
    
    def __init__(self, df: float = 5.0):
        self.df = df
        self._link = IdentityLink()
    
    def variance(self, mu: Array, **params) -> Array:
        """Variance function: σ² × ν/(ν-2)."""
        if self.df <= 2:
            raise ValueError("df must be > 2 for finite variance")
        return np.ones_like(mu) * self.df / (self.df - 2)
    
    def log_likelihood(self, y: Array, mu: Array, **params) -> Scalar:
        """Log-likelihood for t-distribution."""
        # Scale parameter (estimated or provided)
        sigma = params.get('scale', 1.0)
        
        from scipy.special import gammaln
        xp = namespace(y, mu)
        
        log_const = gammaln((self.df + 1)/2) - gammaln(self.df/2) - \
                    0.5*xp.log(self.df * xp.pi * sigma**2)
        
        log_kernel = -(self.df + 1)/2 * xp.log(1 + (y - mu)**2 / (self.df * sigma**2))
        
        return xp.sum(log_const + log_kernel)
    
    def deviance(self, y: Array, mu: Array, **params) -> Scalar:
        """Deviance for t-distribution."""
        return -2 * (self.log_likelihood(y, mu, **params) - 
                     self.log_likelihood(y, y, **params))
    
    def initialize(self, y: Array) -> Array:
        """Initialize with sample mean (robust to outliers for ν > 2)."""
        xp = namespace(y)
        return xp.full_like(y, xp.mean(y))
    
    @property
    def default_link(self):
        return self._link
```

**IRLS for t-GLM**:

The t-distribution is not in the exponential family, so IRLS requires modification.
Use **iteratively reweighted robust regression**:

1. Start with OLS estimates
2. For iteration t:
   - Compute residuals: $r_i = y_i - \mu_i^{(t)}$
   - Compute weights: $w_i = (\nu + 1) / (\nu + r_i^2/\sigma^2)$ (downweight outliers)
   - Update via weighted LS: $\beta^{(t+1)} = (X^T W X)^{-1} X^T W y$
   - Update scale: $\sigma^2 = \text{median}(|r_i|^2) / 0.454$ (robust)

#### 5.5.2 Negative Binomial Distribution

**Mass function** (NB2 parameterization):
```
P(Y = y) = Γ(y + θ) / [Γ(θ) y!] × (θ/(θ+μ))^θ × (μ/(θ+μ))^y
```

where:
- μ is the mean
- θ > 0 is the dispersion parameter (shape)

**Properties**:
- E[Y] = μ
- Var(Y) = μ + μ²/θ (overdispersion!)
- θ → ∞: converges to Poisson
- θ small: high overdispersion

**Variance function**:
```
V(μ) = μ + μ²/θ
```

**Use cases**:
- Overdispersed count data
- RNA-seq analysis (bioinformatics)
- Crash frequency (transportation safety)
- Disease counts with heterogeneity

**Link function**:
- Canonical: Log
- Alternative: Sqrt, Identity

**Implementation**:

```python
class NegativeBinomialFamily(Family):
    """Negative Binomial distribution for overdispersed counts.
    
    Parameters
    ----------
    theta : float or 'estimate'
        Dispersion parameter. If 'estimate', use iterative estimation.
        Larger theta → closer to Poisson.
        
    Notes
    -----
    The NB2 parameterization is used: Var(Y) = μ + μ²/θ
    
    Estimation:
    - Fix θ: Fast, use if θ known from prior data
    - Estimate θ: Slower, iterate between GLM for β and method-of-moments for θ
    """
    
    def __init__(self, theta: float | str = 1.0):
        self.theta = theta
        self._link = LogLink()
    
    def variance(self, mu: Array, **params) -> Array:
        """Variance: μ + μ²/θ."""
        theta = params.get('theta', self.theta)
        if isinstance(theta, str):
            raise ValueError("theta must be estimated before calling variance")
        
        xp = namespace(mu)
        return mu + mu**2 / theta
    
    def log_likelihood(self, y: Array, mu: Array, **params) -> Scalar:
        """Log-likelihood for NB2."""
        theta = params.get('theta', self.theta)
        xp = namespace(y, mu)
        
        from scipy.special import gammaln
        
        log_lik = (gammaln(y + theta) - gammaln(theta) - gammaln(y + 1) +
                   theta * xp.log(theta / (theta + mu)) +
                   y * xp.log(mu / (theta + mu)))
        
        return xp.sum(log_lik)
    
    def deviance(self, y: Array, mu: Array, **params) -> Scalar:
        """Deviance for NB."""
        theta = params.get('theta', self.theta)
        xp = namespace(y, mu)
        
        # Unit deviance
        d = 2 * (y * xp.log(y / mu) - (y + theta) * xp.log((y + theta) / (mu + theta)))
        
        # Handle y=0 case
        d = xp.where(y == 0, 2 * theta * xp.log(1 + mu/theta), d)
        
        return xp.sum(d)
    
    def initialize(self, y: Array) -> Array:
        """Initialize with sample mean + small constant."""
        xp = namespace(y)
        return xp.mean(y) + 0.1
    
    @property
    def default_link(self):
        return self._link
```

**Theta estimation**:

Method-of-moments estimator:
```
θ̂ = μ̂² / (s² - μ̂)
```

where μ̂ is sample mean and s² is sample variance.

Iterative refinement via score equations (more accurate).

#### 5.5.3 Tweedie Distribution

**Density** (no closed form, but has exponential family structure):

The Tweedie distribution with variance power p has:
```
Var(Y) = φ μ^p
```

where 1 < p < 2 (compound Poisson-Gamma).

**Special cases**:
- p = 0: Gaussian
- p = 1: Poisson  
- p = 2: Gamma
- 1 < p < 2: Compound Poisson-Gamma (continuous with point mass at zero)

**Properties**:
- Support: Y ≥ 0 with P(Y = 0) > 0 (zero-inflation built-in!)
- E[Y] = μ
- Var(Y) = φ μ^p

**Use cases**:
- Insurance claims (many zeros, continuous positive values)
- Rainfall amount (dry days vs rainy days)
- Healthcare costs (many zeros, skewed positives)
- Ecology (species abundance with absences)

**Link function**:
- Log (standard)
- Identity
- Power links

**Implementation**:

```python
class TweedieFamily(Family):
    """Tweedie distribution for zero-inflated positive data.
    
    Parameters
    ----------
    power : float
        Variance power parameter p ∈ (1, 2).
        Common choices:
        - p=1.5: Balanced between Poisson and Gamma
        - p=1.7: Heavily right-skewed claims
        
    Notes
    -----
    The Tweedie is a compound Poisson-Gamma distribution.
    It naturally handles exact zeros (dry periods, no claims).
    
    Estimation of p can be done via profile likelihood,
    but is often fixed based on domain knowledge.
    """
    
    def __init__(self, power: float = 1.5):
        if not (1 < power < 2):
            raise ValueError("Tweedie power must be in (1, 2)")
        self.power = power
        self._link = LogLink()
    
    def variance(self, mu: Array, **params) -> Array:
        """Variance: φ μ^p."""
        xp = namespace(mu)
        return mu**self.power
    
    def deviance(self, y: Array, mu: Array, **params) -> Scalar:
        """Deviance for Tweedie (Jørgensen, 1987)."""
        xp = namespace(y, mu)
        p = self.power
        
        # Handle y=0 case separately
        d_zero = 2 * mu**(2-p) / (2-p)
        d_pos = 2 * (y**(2-p) / ((1-p)*(2-p)) - 
                     y * mu**(1-p) / (1-p) + 
                     mu**(2-p) / (2-p))
        
        d = xp.where(y == 0, d_zero, d_pos)
        return xp.sum(d)
    
    def log_likelihood(self, y: Array, mu: Array, **params) -> Scalar:
        """Approximate log-likelihood (no closed form)."""
        # Use deviance relationship: -2ℓ = D + constant
        return -0.5 * self.deviance(y, mu, **params)
    
    def initialize(self, y: Array) -> Array:
        """Initialize with positive mean."""
        xp = namespace(y)
        y_pos = xp.where(y > 0, y, xp.nan)
        mu_init = xp.nanmean(y_pos) if xp.any(y > 0) else 0.1
        return xp.full_like(y, mu_init)
    
    @property
    def default_link(self):
        return self._link
```

### Integration with GLM/GAMM

**File structure**:
```
aurora/distributions/families/
├── student_t.py       # Student's t family
├── negative_binomial.py  # Negative Binomial family  
├── tweedie.py         # Tweedie family
└── __init__.py        # Export all families
```

**Usage examples**:

```python
# Robust regression with t-distribution
result = fit_glm(
    X, y,
    family='student_t',
    family_params={'df': 5},
    link='identity'
)

# Overdispersed counts with Negative Binomial
result = fit_glm(
    X, y,
    family='negativebinomial',
    family_params={'theta': 2.0},  # or 'estimate'
    link='log'
)

# Insurance claims with Tweedie
result = fit_glm(
    X, y,
    family='tweedie',
    family_params={'power': 1.6},
    link='log'
)

# GAMM with heavy-tailed distributions
result = fit_gamm(
    formula='claims ~ s(age) + s(vehicle_age) + (1 | region)',
    data=df,
    family='tweedie',
    family_params={'power': 1.7}
)
```

### Implementation Timeline

**Week 1: Student's t Distribution**
- [ ] Implement StudentTFamily class
- [ ] Add robust IRLS algorithm
- [ ] Test with simulated outlier data
- [ ] Compare robustness vs Gaussian

**Week 2: Negative Binomial Distribution**
- [ ] Implement NegativeBinomialFamily class
- [ ] Add theta estimation (method-of-moments + ML)
- [ ] Test with overdispersed count data
- [ ] Validate against R MASS::glm.nb()

**Week 3: Tweedie Distribution**
- [ ] Implement TweedieFamily class
- [ ] Add power parameter estimation (optional)
- [ ] Test with insurance/rainfall data
- [ ] Validate against R statmod::tweedie

**Week 4: Integration & Documentation**
- [ ] Integrate with fit_glm() and fit_gamm()
- [ ] Add to family registry
- [ ] Write comprehensive examples
- [ ] Performance benchmarks

### Testing Strategy

**Unit tests**:
- Log-likelihood computation accuracy
- Deviance correctness
- Variance function
- Initialization stability

**Integration tests**:
- GLM with heavy-tailed families
- GAMM with heavy-tailed families
- Convergence under various scenarios

**Validation**:
- Compare against R implementations
- Simulate data with known parameters
- Real-world datasets with outliers/overdispersion

### Success Criteria

- ✓ All three families implemented and tested
- ✓ Robust to outliers (t) vs Gaussian
- ✓ Handles overdispersion (NB) better than Poisson
- ✓ Correctly models zero-inflation (Tweedie)
- ✓ Results match R within tolerance
- ✓ Documented use cases and examples

### References

**Student's t Distribution**:
- Lange, K. L., Little, R. J., & Taylor, J. M. (1989). "Robust statistical modeling using the t distribution." *Journal of the American Statistical Association*, 84(408), 881-896.

**Negative Binomial Regression**:
- Hilbe, J. M. (2011). *Negative Binomial Regression* (2nd ed.). Cambridge University Press.
- Ver Hoef, J. M., & Boveng, P. L. (2007). "Quasi-Poisson vs. negative binomial regression." *Environmetrics*, 18(3), 255-259.

**Tweedie Distribution**:
- Jørgensen, B. (1987). "Exponential dispersion models." *Journal of the Royal Statistical Society: Series B*, 49(2), 127-162.
- Smyth, G. K., & Jørgensen, B. (2002). "Fitting Tweedie's compound Poisson model to insurance claims data." *ASTIN Bulletin*, 32(1), 143-157.
- Dunn, P. K., & Smyth, G. K. (2005). "Series evaluation of Tweedie exponential dispersion model densities." *Statistics and Computing*, 15(4), 267-280.

**Robust GLM**:
- Cantoni, E., & Ronchetti, E. (2001). "Robust inference for generalized linear models." *Journal of the American Statistical Association*, 96(455), 1022-1030.

