# Phase 4: GAMM Implementation Design

## Overview

This document outlines the design and implementation plan for Generalized Additive Mixed Models (GAMM) in Aurora-GLM. GAMMs extend GAMs by adding random effects to model correlation structures in hierarchical/longitudinal data.

## Mathematical Foundation

### GAMM Model

A GAMM has the form:

```
g(E[Y_ij]) = X_ij β + f₁(z₁_ij) + f₂(z₂_ij) + ... + Z_ij b_i + ε_ij
```

Where:
- `g()` is the link function (from GLM)
- `X_ij β` are fixed effects (parametric terms)
- `fₖ(z_ij)` are smooth functions (from GAM)
- `Z_ij b_i` are random effects for group `i`
- `b_i ~ N(0, Ψ)` are random effect coefficients
- `ε_ij` is the residual error

### Random Effects Structure

Random effects model correlation within groups:

```
b_i = [b₀ᵢ, b₁ᵢ, ..., bₚᵢ]ᵀ ~ N(0, Ψ)
```

Where `Ψ` is the variance-covariance matrix:

```
Ψ = [σ²₀      ρσ₀σ₁  ...
     ρσ₀σ₁    σ²₁    ...
     ...      ...    ...]
```

Common structures:
- **Random intercept**: `(1 | group)` - only σ²₀
- **Random slope**: `(x | group)` - σ²₀, σ²₁, and ρ
- **Nested**: `(1 | group1/group2)` - hierarchical nesting
- **Crossed**: `(1 | group1) + (1 | group2)` - independent groupings

### Estimation Methods

#### 1. REML (Restricted Maximum Likelihood)

REML maximizes the likelihood of the data after "projecting out" fixed effects:

```
l_REML(θ, Ψ) = -½[log|V| + (y - Xβ̂)'V⁻¹(y - Xβ̂) + log|X'V⁻¹X|]
```

Where:
- `V = ZΨZ' + R` (total covariance)
- `θ` are smoothing parameters
- `Ψ` are variance components
- `R` is residual variance

**Advantages**:
- Unbiased variance component estimates
- Standard for mixed models
- Accounts for degrees of freedom used by fixed effects

#### 2. Laplace Approximation

For non-Gaussian responses, uses Laplace approximation to the likelihood:

```
l(β, θ, Ψ) ≈ l(β, b̂, θ, Ψ) - ½log|H|
```

Where `H` is the Hessian of the joint log-likelihood with respect to `b`.

**Advantages**:
- Works for all GLM families
- Fast convergence
- Good approximation for large groups

#### 3. Penalized Quasi-Likelihood (PQL)

Iterative algorithm similar to IRLS but includes random effects:

```
1. Initialize b = 0
2. Compute working response: z = η + (y - μ)/g'(μ)
3. Solve penalized least squares:
   (β̂, b̂) = argmin[(z - Xβ - Zb)'W(z - Xβ - Zb) + b'Ψ⁻¹b]
4. Update variance components
5. Repeat until convergence
```

## Implementation Plan

### Phase 4.1: Random Effects Structures

#### 4.1.1 Random Effect Specification

```python
@dataclass
class RandomEffect:
    """Specification of a random effect term.

    Attributes
    ----------
    variables : tuple of (str or int)
        Variables with random effects (slopes)
    grouping : str or int
        Grouping variable name or column index
    include_intercept : bool, default=True
        Whether to include random intercept
    covariance : {'unstructured', 'diagonal', 'identity'}, default='unstructured'
        Covariance structure for random effects
    """
    variables: tuple[str | int, ...] = ()
    grouping: str | int = None
    include_intercept: bool = True
    covariance: str = 'unstructured'

    def __post_init__(self):
        if self.grouping is None:
            raise ValueError("grouping must be specified")

        # Validate covariance structure
        valid_cov = {'unstructured', 'diagonal', 'identity'}
        if self.covariance not in valid_cov:
            raise ValueError(f"covariance must be one of {valid_cov}")
```

#### 4.1.2 Design Matrices for Random Effects

```python
def construct_Z_matrix(
    X: np.ndarray,
    random_effects: list[RandomEffect],
    data: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Construct random effects design matrix Z.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Fixed effects design matrix
    random_effects : list of RandomEffect
        Random effect specifications
    data : DataFrame, optional
        Original data with grouping variables

    Returns
    -------
    Z : ndarray, shape (n, q)
        Random effects design matrix
    Z_blocks : list of ndarray
        List of Z matrices per random effect term

    Notes
    -----
    For random intercept + slope on x1 grouped by subject:
        Z_i = [1, x1_i] for each observation in subject i

    Full Z is block diagonal with blocks for each group.
    """
    pass
```

### Phase 4.2: Variance Components Estimation

#### 4.2.1 REML for Variance Components

```python
def reml_variance_components(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    groups: np.ndarray,
    initial_psi: np.ndarray | None = None,
) -> dict:
    """Estimate variance components via REML.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response variable
    X : ndarray, shape (n, p)
        Fixed effects design matrix
    Z : ndarray, shape (n, q)
        Random effects design matrix
    groups : ndarray, shape (n,)
        Group indicators
    initial_psi : ndarray, optional
        Initial variance-covariance matrix

    Returns
    -------
    result : dict
        - 'psi': Estimated variance-covariance matrix
        - 'sigma2': Residual variance
        - 'log_likelihood': REML log-likelihood
        - 'converged': Boolean convergence status

    Algorithm
    ---------
    1. Parameterize Ψ using Cholesky: Ψ = LL'
    2. Optimize l_REML(L) using scipy.optimize
    3. Return Ψ = LL'

    Notes
    -----
    Uses Fisher scoring or Nelder-Mead optimization.
    Log scale for variance parameters ensures positivity.
    """
    pass
```

#### 4.2.2 Variance Structures

```python
class CovarianceStructure:
    """Base class for random effect covariance structures."""

    def n_parameters(self, n_effects: int) -> int:
        """Number of free parameters."""
        raise NotImplementedError

    def construct_psi(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Construct covariance matrix from parameters."""
        raise NotImplementedError

    def extract_params(self, psi: np.ndarray) -> np.ndarray:
        """Extract parameters from covariance matrix."""
        raise NotImplementedError


class UnstructuredCovariance(CovarianceStructure):
    """Unstructured covariance (full matrix).

    For q random effects: q(q+1)/2 parameters.
    """

    def n_parameters(self, n_effects: int) -> int:
        return n_effects * (n_effects + 1) // 2

    def construct_psi(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Construct via Cholesky: Ψ = LL'."""
        L = np.zeros((n_effects, n_effects))
        # Fill lower triangle
        idx = 0
        for i in range(n_effects):
            for j in range(i + 1):
                L[i, j] = params[idx]
                idx += 1
        return L @ L.T


class DiagonalCovariance(CovarianceStructure):
    """Diagonal covariance (independent random effects).

    For q random effects: q parameters.
    """

    def n_parameters(self, n_effects: int) -> int:
        return n_effects

    def construct_psi(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Construct diagonal matrix with variances."""
        return np.diag(params ** 2)  # Square to ensure positivity


class IdentityCovariance(CovarianceStructure):
    """Identity covariance (equal variances, no correlation).

    For q random effects: 1 parameter (σ²).
    """

    def n_parameters(self, n_effects: int) -> int:
        return 1

    def construct_psi(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Construct σ²I."""
        return (params[0] ** 2) * np.eye(n_effects)
```

### Phase 4.3: GAMM Fitting Algorithm

#### 4.3.1 Penalized Iteratively Reweighted Least Squares (P-IRLS) for GAMM

```python
def fit_gamm(
    X: np.ndarray,
    y: np.ndarray,
    smooth_terms: list[SmoothTerm] | None = None,
    random_effects: list[RandomEffect] | None = None,
    family: str = 'gaussian',
    link: str | None = None,
    method: str = 'REML',
    max_iter: int = 100,
    tol: float = 1e-6,
) -> 'GAMMResult':
    """Fit a Generalized Additive Mixed Model.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix for parametric terms
    y : ndarray, shape (n,)
        Response variable
    smooth_terms : list of SmoothTerm, optional
        Smooth term specifications (from Phase 3)
    random_effects : list of RandomEffect, optional
        Random effect specifications
    family : str, default='gaussian'
        Distribution family
    link : str, optional
        Link function (uses default if None)
    method : {'REML', 'ML', 'Laplace'}, default='REML'
        Estimation method for variance components
    max_iter : int, default=100
        Maximum iterations
    tol : float, default=1e-6
        Convergence tolerance

    Returns
    -------
    GAMMResult
        Fitted model with:
        - fixed_effects: β̂
        - smooth_coefficients: smooth term coefficients
        - random_effects: b̂ (BLUPs)
        - variance_components: Ψ̂
        - residual_variance: σ̂²

    Algorithm (PQL)
    ---------------
    1. Initialize: β = 0, b = 0, Ψ = I
    2. Outer loop (variance components):
        a. Inner loop (coefficients):
            i. Compute working response z and weights W
            ii. Solve penalized mixed model equations:
               [X'WX + λS    X'WZ ] [β]   [X'Wz]
               [Z'WX         Z'WZ + Ψ⁻¹] [b] = [Z'Wz]
            iii. Update η = Xβ + Zb, μ = g⁻¹(η)
        b. Update Ψ via REML
        c. Update smoothing parameters λ via REML
    3. Compute standard errors, diagnostics

    Notes
    -----
    For Gaussian family with identity link, reduces to LMM solved exactly.
    For other families, uses iterative PQL or Laplace approximation.
    """
    pass
```

#### 4.3.2 Mixed Model Equations

The key computational step is solving the augmented system:

```python
def solve_mixed_model_equations(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    S_smooth: np.ndarray,
    Psi_inv: np.ndarray,
    lambda_: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve penalized mixed model equations.

    Solves:
        [X'WX + λS    X'WZ      ] [β]   [X'Wy]
        [Z'WX         Z'WZ + Ψ⁻¹] [b] = [Z'Wy]

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Fixed effects design
    Z : ndarray, shape (n, q)
        Random effects design
    y : ndarray, shape (n,)
        Working response
    W : ndarray, shape (n, n)
        Weights (diagonal)
    S_smooth : ndarray, shape (p, p)
        Smoothing penalty
    Psi_inv : ndarray, shape (q, q)
        Inverse of variance-covariance matrix
    lambda_ : float
        Smoothing parameter

    Returns
    -------
    beta : ndarray, shape (p,)
        Fixed effect coefficients
    b : ndarray, shape (q,)
        Random effect coefficients (BLUPs)

    Notes
    -----
    Uses Cholesky decomposition for efficiency.
    Can exploit sparsity for large datasets.
    """
    # Construct coefficient matrix
    p = X.shape[1]
    q = Z.shape[1]

    # Build system
    XtWX = X.T @ (W @ X) + lambda_ * S_smooth
    XtWZ = X.T @ (W @ Z)
    ZtWZ = Z.T @ (W @ Z) + Psi_inv

    A = np.block([
        [XtWX, XtWZ],
        [XtWZ.T, ZtWZ]
    ])

    Xty = X.T @ (W @ y)
    Zty = Z.T @ (W @ y)
    b_rhs = np.concatenate([Xty, Zty])

    # Solve
    coef = np.linalg.solve(A, b_rhs)

    beta = coef[:p]
    b = coef[p:]

    return beta, b
```

### Phase 4.4: GAMMResult Class

```python
@dataclass
class GAMMResult:
    """Result of GAMM fitting.

    Attributes
    ----------
    fixed_effects : ndarray
        Estimated fixed effect coefficients β̂
    smooth_coefficients : dict
        Smooth term coefficients per term
    random_effects : dict
        Estimated random effects b̂ per group
    variance_components : dict
        Variance-covariance matrices Ψ̂ per random effect term
    residual_variance : float
        Estimated residual variance σ̂²
    smoothing_parameters : dict
        Estimated smoothing parameters λ̂ per smooth term
    edf_fixed : float
        Effective degrees of freedom for fixed effects
    edf_smooth : dict
        Effective degrees of freedom per smooth term
    log_likelihood : float
        Log-likelihood (or REML log-likelihood)
    aic : float
        Akaike Information Criterion
    bic : float
        Bayesian Information Criterion
    converged : bool
        Whether fitting converged
    n_iter : int
        Number of iterations
    """
    fixed_effects: np.ndarray
    smooth_coefficients: dict[str, np.ndarray]
    random_effects: dict[str, np.ndarray]
    variance_components: dict[str, np.ndarray]
    residual_variance: float
    smoothing_parameters: dict[str, float]
    edf_fixed: float
    edf_smooth: dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    n_iter: int

    # Store fitting details
    _X: np.ndarray = None
    _Z: np.ndarray = None
    _y: np.ndarray = None
    _fitted_values: np.ndarray = None
    _groups: np.ndarray = None

    def predict(
        self,
        X_new: np.ndarray,
        Z_new: np.ndarray | None = None,
        groups_new: np.ndarray | None = None,
        type: str = 'response',
        include_random: bool = False,
    ) -> np.ndarray:
        """Make predictions on new data.

        Parameters
        ----------
        X_new : ndarray
            New fixed effects design matrix
        Z_new : ndarray, optional
            New random effects design matrix
        groups_new : ndarray, optional
            Group indicators for new data
        type : {'link', 'response'}, default='response'
            Prediction scale
        include_random : bool, default=False
            Whether to include random effects in predictions
            (requires Z_new and groups_new)

        Returns
        -------
        predictions : ndarray
            Predicted values

        Notes
        -----
        If include_random=False, predictions are population-level (η = Xβ + smooth).
        If include_random=True, predictions include group-specific effects (η = Xβ + Zb + smooth).
        For new groups not in training data, b̂ = 0 (population prediction).
        """
        pass

    def summary(self, *, detailed: bool = True) -> str:
        """Print comprehensive summary.

        Includes:
        - Fixed effects table (like GLM summary)
        - Smooth terms table (like GAM summary)
        - Random effects variance components
        - Model fit statistics
        """
        pass

    def plot_random_effects(
        self,
        random_effect: str,
        **kwargs,
    ) -> 'Figure':
        """Plot random effect distributions.

        Creates:
        - Histogram/density of random intercepts
        - Caterpillar plot with confidence intervals
        - Q-Q plot for normality check
        """
        pass
```

### Phase 4.5: Formula Extensions for Random Effects

Extend formula parser to support lme4-style syntax:

```python
# Current GAM syntax (Phase 3)
"y ~ s(x1, k=10) + s(x2) + x3"

# Extended GAMM syntax (Phase 4)
"y ~ s(x1, k=10) + s(x2) + x3 + (1 | subject)"           # Random intercept
"y ~ s(x1) + x3 + (1 + x3 | subject)"                    # Random intercept + slope
"y ~ s(x1) + (1 | subject) + (1 | clinic)"               # Crossed random effects
"y ~ s(x1) + (1 | clinic/subject)"                       # Nested random effects
"y ~ s(x1, by=treatment) + (1 + time | subject)"         # Smooth-by + random slope
```

Parser modifications needed:

```python
def parse_random_effects(formula: str) -> list[RandomEffect]:
    """Parse random effects from formula.

    Syntax: (variables | grouping)

    Examples:
    - "(1 | subject)" → RandomEffect(variables=(), grouping='subject', include_intercept=True)
    - "(x1 | subject)" → RandomEffect(variables=('x1',), grouping='subject', include_intercept=False)
    - "(1 + x1 | subject)" → RandomEffect(variables=('x1',), grouping='subject', include_intercept=True)
    - "(x1 + x2 | subject)" → RandomEffect(variables=('x1', 'x2'), grouping='subject')
    """
    pass
```

## Implementation Phases

### Milestone 1: Random Effects Structures (Week 1-2)
- [ ] Implement `RandomEffect` dataclass with validation
- [ ] Implement covariance structures (Unstructured, Diagonal, Identity)
- [ ] Implement `construct_Z_matrix()` for design matrices
- [ ] Unit tests for design matrix construction
- [ ] Tests for nested and crossed structures

### Milestone 2: Variance Component Estimation (Week 3-4)
- [ ] Implement REML estimation for variance components
- [ ] Implement variance structure parameterizations
- [ ] Optimize REML objective via scipy.optimize
- [ ] Unit tests for REML convergence
- [ ] Validate against lme4 on simple examples

### Milestone 3: GAMM Fitting for Gaussian (Week 5-6)
- [ ] Implement `solve_mixed_model_equations()`
- [ ] Implement `fit_gamm()` for Gaussian family (LMM)
- [ ] Integrate with Phase 3 smooth terms
- [ ] Implement GAMMResult class
- [ ] Tests on Gaussian data with random intercepts
- [ ] Validate against R's gamm() or mgcv::gamm()

### Milestone 4: Extension to GLM Families (Week 7-8)
- [ ] Implement PQL algorithm for non-Gaussian families
- [ ] Implement Laplace approximation
- [ ] Tests on Poisson and Binomial with random effects
- [ ] Convergence diagnostics for GLMM

### Milestone 5: Formula Parser Extensions (Week 9)
- [ ] Extend `parse_formula()` to handle random effects
- [ ] Implement `parse_random_effects()`
- [ ] Support nested: `(1 | a/b)`
- [ ] Support crossed: `(1 | a) + (1 | b)`
- [ ] Integration tests with formula-based fitting

### Milestone 6: Visualization and Documentation (Week 10)
- [ ] Implement `plot_random_effects()`
- [ ] Caterpillar plots for random effects
- [ ] Q-Q plots for normality checks
- [ ] Tutorial notebook: Longitudinal data example
- [ ] Tutorial notebook: Hierarchical data example
- [ ] Validate all examples against mgcv

## Design Decisions

### 1. Sparse Matrices

For large datasets with many groups:
- Use `scipy.sparse` for Z matrices
- Block-diagonal structure for Ψ
- Exploit sparsity in mixed model equations

### 2. Computational Efficiency

**Strategies**:
- Cholesky decomposition for positive definite systems
- Update formulas to avoid redundant computation
- Cache Ψ⁻¹ and its Cholesky factor
- Vectorize across groups where possible

**Performance targets**:
- 1000 observations, 100 groups: <5 seconds
- 10000 observations, 1000 groups: <30 seconds

### 3. Multi-Backend Support

Random effects computations should work with NumPy, PyTorch, and JAX:

```python
from aurora.distributions._utils import namespace

def construct_Z_matrix(X, random_effects, groups):
    xp = namespace(X)
    # Use xp.* for all operations
    return xp.array(...)
```

### 4. Formula Compatibility

Syntax should be familiar to R users:

```python
# R lme4:
lmer(y ~ x1 + (1 + x2 | subject), data=df)

# Aurora equivalent:
fit_gamm(
    formula="y ~ x1 + (1 + x2 | subject)",
    data=df,
    method='REML'
)
```

## Testing Strategy

### Unit Tests

- Random effect design matrix construction
- Variance component estimation (REML objective)
- Covariance structure parameterizations
- Mixed model equation solver
- Each component independently testable

### Integration Tests

- Complete GAMM fit on synthetic data
- Compare with R's lme4 on simple random intercept models
- Compare with mgcv::gamm on GAM + random effects
- Check convergence and stability

### Validation Tests

- Reproduce mgcv examples from Wood (2017)
- Sleepstudy dataset (lme4 example)
- Rail dataset (nested random effects)
- Performance benchmarks vs mgcv

## Risks and Mitigations

### Risk 1: Numerical Instability

**Issue**: Near-singular variance-covariance matrices
**Mitigation**:
- Use Cholesky parameterization: Ψ = LL'
- Add ridge penalty if needed
- Check condition numbers

### Risk 2: Slow Convergence

**Issue**: PQL may converge slowly for GLMMs
**Mitigation**:
- Implement Laplace approximation as alternative
- Adaptive step sizes
- Warm starts from previous fits

### Risk 3: Complex Formula Parsing

**Issue**: Nested/crossed syntax is complex
**Mitigation**:
- Start with simple cases
- Extensive testing of parser
- Clear error messages

### Risk 4: Memory for Large Datasets

**Issue**: Full Z matrix may be large
**Mitigation**:
- Use sparse matrices
- Block computation where possible
- Chunk-wise processing for huge datasets

## Success Criteria

Phase 4 is complete when:

- [ ] Random effects structures implemented and tested
- [ ] REML variance component estimation works
- [ ] fit_gamm() handles Gaussian family
- [ ] fit_gamm() handles Poisson/Binomial families (PQL or Laplace)
- [ ] Formula parser supports random effects syntax
- [ ] Results match lme4/mgcv within tolerance (coef ≈ 1e-4, variance components ≈ 1e-3)
- [ ] Visualization of random effects works
- [ ] Tutorial notebooks demonstrate usage
- [ ] Test coverage ≥ 85%

## References

### Theory

- Pinheiro, J.C. & Bates, D.M. (2000). *Mixed-Effects Models in S and S-PLUS*. Springer.
- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapter 6: GAMMs.
- McCulloch, C.E. & Searle, S.R. (2001). *Generalized, Linear, and Mixed Models*. Wiley.

### Implementation References

- **R lme4**: Linear mixed-effects models
- **R mgcv::gamm**: GAM with random effects (wraps lme)
- **R nlme**: Flexible mixed-effects framework
- **Python statsmodels.MixedLM**: Basic LMM (limited functionality)

## Estimated Effort

- **Week 1-2**: Random effects structures (15-20 hours)
- **Week 3-4**: REML estimation (20-25 hours)
- **Week 5-6**: GAMM fitting (Gaussian) (20-25 hours)
- **Week 7-8**: Extension to GLMMs (25-30 hours)
- **Week 9**: Formula parser (10-15 hours)
- **Week 10**: Visualization and docs (15-20 hours)

**Total**: 105-135 hours (~6-8 weeks with parallel development)

## Next Steps

1. Create `aurora/models/gamm/` directory
2. Implement `RandomEffect` dataclass in `random_effects.py`
3. Implement covariance structures in `covariance.py`
4. Create test suite in `tests/test_models/test_gamm/`
5. Start with simple random intercept model validation
