# Phase 3: GAM Implementation Design

## Overview

This document outlines the design and implementation plan for Generalized Additive Models (GAM) in Aurora-GLM. GAMs extend GLMs by allowing smooth, non-parametric functions of predictors instead of linear terms.

## Mathematical Foundation

### GAM Model

A GAM has the form:

```
g(E[Y]) = β₀ + f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ) + ε
```

Where:
- `g()` is the link function (same as GLM)
- `fⱼ(xⱼ)` are smooth functions represented as spline basis expansions
- `Y` follows an exponential family distribution

### Penalized Regression Splines

Each smooth term is represented as:

```
fⱼ(x) = Σᵢ βⱼᵢ bⱼᵢ(x)
```

Where `bⱼᵢ(x)` are basis functions (B-splines, cubic splines, thin-plate splines, etc.)

To avoid overfitting, we add a roughness penalty:

```
J(f) = ∫ [f''(x)]² dx
```

The penalized likelihood becomes:

```
l_p(β) = l(β) - λ/2 β'Sβ
```

Where:
- `l(β)` is the log-likelihood
- `S` is the penalty matrix
- `λ` is the smoothing parameter (controls roughness)

## Implementation Plan

### Phase 3.1: Spline Basis Functions

#### 3.1.1 Cubic Splines (`aurora/smoothing/splines/cubic.py`)

```python
class CubicSplineBasis:
    """Natural cubic spline basis functions.

    Features:
    - Knot placement (uniform, quantile-based)
    - Boundary constraints (natural splines)
    - Basis matrix B(x) computation
    - Derivative computation for penalty matrix
    """

    def __init__(self, knots, degree=3):
        self.knots = knots
        self.degree = degree

    def basis_matrix(self, x):
        """Compute basis matrix B where B[i,j] = b_j(x_i)"""
        pass

    def penalty_matrix(self):
        """Compute penalty matrix S = ∫ b''(x) b''(x)' dx"""
        pass
```

#### 3.1.2 B-Splines (`aurora/smoothing/splines/bspline.py`)

```python
class BSplineBasis:
    """B-spline basis functions.

    Advantages:
    - Local support (computational efficiency)
    - Stable computation via de Boor algorithm
    - Flexible degree control
    """

    def __init__(self, knots, degree=3, cyclic=False):
        self.knots = knots
        self.degree = degree
        self.cyclic = cyclic  # For periodic data

    def basis_matrix(self, x):
        """Compute B-spline basis matrix using de Boor recursion"""
        pass
```

#### 3.1.3 Thin-Plate Splines (`aurora/smoothing/splines/tps.py`)

```python
class ThinPlateBasis:
    """Thin-plate spline basis for multi-dimensional smoothing.

    Features:
    - Automatically adapts to data density
    - Rotation-invariant penalty
    - Works in arbitrary dimensions
    """

    def __init__(self, knots, dimension=1):
        self.knots = knots
        self.dimension = dimension
```

### Phase 3.2: Penalty Matrices (`aurora/smoothing/penalties/`)

#### 3.2.1 Difference Penalties

```python
def difference_penalty(k, order=2):
    """Create penalty based on kth order differences.

    For order=2: penalizes (β_i - 2β_{i+1} + β_{i+2})²
    Approximates integrated squared second derivative.
    """
    D = np.diff(np.eye(k), n=order, axis=0)
    return D.T @ D
```

#### 3.2.2 Integrated Squared Derivative

```python
def integrated_derivative_penalty(basis, order=2):
    """Compute ∫ [f^(m)(x)]² dx analytically.

    More accurate than difference penalties.
    Requires closed-form integration of basis derivatives.
    """
    pass
```

### Phase 3.3: Smoothing Parameter Selection

#### 3.3.1 GCV (Generalized Cross-Validation)

```python
def gcv_score(y, mu, lambda_, hat_matrix):
    """Compute GCV score for smoothing parameter selection.

    GCV = n * RSS / (n - tr(H))²

    Where H is the hat matrix (influence matrix).
    Minimizing GCV selects optimal λ.
    """
    n = len(y)
    rss = np.sum((y - mu) ** 2)
    df = np.trace(hat_matrix)
    return n * rss / (n - df) ** 2
```

#### 3.3.2 REML (Restricted Maximum Likelihood)

```python
def reml_score(y, mu, lambda_, precision_matrix):
    """Compute REML criterion.

    Often better than GCV, especially with multiple smooths.
    Treats smoothing parameters as variance components.
    """
    pass
```

#### 3.3.3 AIC (Akaike Information Criterion)

```python
def aic_smooth(log_likelihood, df):
    """Penalized likelihood with effective degrees of freedom."""
    return -2 * log_likelihood + 2 * df
```

### Phase 3.4: GAM Fitting (`aurora/models/gam/fitting.py`)

#### Core Algorithm: Penalized IRLS (P-IRLS)

```python
def fit_gam(X, y, smooth_terms, family='gaussian', method='GCV'):
    """Fit GAM using penalized IRLS.

    Algorithm:
    1. Initialize β, λ
    2. Iterate until convergence:
       a. Compute working response z and weights w (GLM part)
       b. Solve penalized weighted least squares:
          β = (B'WB + λS)⁻¹ B'Wz
       c. Update λ using selected criterion (GCV/REML)
    3. Compute standard errors, diagnostics

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Design matrix (for parametric terms)
    y : array-like, shape (n_samples,)
        Response variable
    smooth_terms : list of SmoothTerm
        Smooth terms specification
    family : str
        Distribution family
    method : str
        Method for λ selection ('GCV', 'REML', 'AIC')

    Returns
    -------
    GAMResult
        Fitted model with coefficients, smooths, diagnostics
    """
    pass
```

#### Data Structure for Smooth Terms

```python
@dataclass
class SmoothTerm:
    """Specification of a smooth term s(x, ...)"""
    variable: str | int
    basis: str = 'tp'  # 'tp', 'cr', 'bs', 'ps'
    k: int = 10  # Number of basis functions
    by: str | None = None  # Interaction variable
    penalty: bool = True
    lambda_: float | None = None  # Fixed λ or None for auto
```

### Phase 3.5: GAM Result and Visualization

```python
class GAMResult:
    """Result of GAM fitting.

    Attributes
    ----------
    coefficients_ : dict
        Parametric term coefficients
    smooth_coefficients_ : dict
        Smooth term coefficients
    smooths_ : dict
        Fitted smooth functions
    edf_ : dict
        Effective degrees of freedom per term
    lambda_ : dict
        Smoothing parameters
    """

    def predict(self, X_new, type='response'):
        """Predict on new data."""
        pass

    def plot_smooth(self, term, partial_residuals=True):
        """Visualize smooth function with confidence bands."""
        pass

    def summary(self):
        """R-style summary with parametric and smooth terms."""
        pass
```

## Implementation Phases

### Milestone 1: Basic Cubic Splines (Week 1)
- [ ] Implement CubicSplineBasis
- [ ] Implement difference penalty
- [ ] Unit tests for basis evaluation
- [ ] Validate against R mgcv for simple case

### Milestone 2: B-Splines and Penalties (Week 2)
- [ ] Implement BSplineBasis
- [ ] Implement integrated derivative penalty
- [ ] Add support for cyclic splines
- [ ] Unit tests

### Milestone 3: Smoothing Parameter Selection (Week 3)
- [ ] Implement GCV
- [ ] Implement REML
- [ ] Optimize λ using scipy.optimize
- [ ] Tests comparing with mgcv

### Milestone 4: GAM Fitting (Week 4)
- [ ] Implement P-IRLS algorithm
- [ ] Handle multiple smooth terms
- [ ] Compute effective degrees of freedom
- [ ] Basic GAMResult class

### Milestone 5: Visualization and Documentation (Week 5)
- [ ] plot_smooth() with confidence bands
- [ ] summary() method
- [ ] Tutorial notebook
- [ ] Validation against mgcv

## Design Decisions

### Multi-Backend Support

All spline operations should work with NumPy, PyTorch, and JAX:

```python
from aurora.distributions._utils import namespace, as_namespace_array

def basis_matrix(self, x):
    xp = namespace(x)
    # Use xp.* for all operations
    return xp.array(...)
```

### Modularity

Each component should be independently testable:
- Spline bases compute B and S matrices
- Penalties are separate from bases
- Smoothing parameter selection is decoupled from fitting
- GAM fitting orchestrates components

### Performance

- Cache basis matrices when possible
- Use sparse matrices for large k
- Leverage backend-specific optimizations (JAX jit, PyTorch autograd)

### R Compatibility

API should feel familiar to mgcv users:

```python
# R: gam(y ~ s(x1) + s(x2, bs='cr', k=15) + x3)
# Aurora:
result = fit_gam(
    data=df,
    formula="y ~ s(x1) + s(x2, bs='cr', k=15) + x3",
    family='gaussian'
)
```

## Testing Strategy

### Unit Tests
- Each basis function against known values
- Penalty matrices against analytical solutions
- GCV/REML scores on synthetic data

### Integration Tests
- Complete GAM fit on simple datasets
- Compare with mgcv on standard test cases
- Check convergence and stability

### Validation Tests
- Reproduce mgcv results on real datasets
- Performance benchmarks vs mgcv
- Multi-backend consistency

## Risks and Mitigations

### Risk 1: Numerical Stability
**Mitigation**: Use QR decomposition, regularization, careful scaling

### Risk 2: Slow Convergence
**Mitigation**: Good initialization, adaptive step sizes, early stopping

### Risk 3: Complex API
**Mitigation**: Start simple, iterate based on usage, document extensively

### Risk 4: Memory for Large k
**Mitigation**: Sparse matrices, chunked computation, low-rank approximations

## References

- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.)
- Wood, S.N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *JRSS-B*, 73(1), 3-36.
- Eilers, P.H.C. & Marx, B.D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89-121.

## Success Criteria

Phase 3 is complete when:
- [ ] Cubic and B-splines implemented and tested
- [ ] GCV smoothing parameter selection works
- [ ] fit_gam() handles single smooth term
- [ ] Results match mgcv within tolerance (coef ≈ 1e-4)
- [ ] Visualization works (plot_smooth)
- [ ] Tutorial notebook demonstrates usage
- [ ] Test coverage ≥ 85%
