# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Aurora-GLM** is a modular, extensible, high-performance Python framework for advanced statistical modeling focused on Generalized Linear Models (GLM), Generalized Additive Models (GAM), and Generalized Additive Mixed Models (GAMM).

- **Package name**: `aurora-glm` (PyPI and repository)
- **Import name**: `import aurora` (Python convention: short names for imports)
- **Repository**: https://github.com/Matcraft94/Aurora-GLM
- **Author**: Lucy E. Arias (@Matcraft94)
- **Version**: 0.2.0-dev
- **Status**: Phase 2 in progress (~80% complete - GLM fitting, inference, and diagnostics implemented)
- **Python**: 3.10+
- **Philosophy**: Functional and modular design (NOT heavy object-oriented)

## Project Vision

### Main Objective

Create a statistical modeling framework in Python that is:

1. **Scientifically rigorous**: Correct implementations validated against R (mgcv) and statsmodels
2. **High performance**: Competitive or superior to existing alternatives, with GPU support
3. **Extensible**: Users can add custom distributions, link functions, and algorithms
4. **Multi-backend**: Transparent support for NumPy, PyTorch, and JAX
5. **Modular and functional**: Clean design favoring composition over heavy inheritance

### Use Cases

- Academic research (ecology, epidemiology, social sciences)
- Pharmaceutical industry (clinical trials)
- Financial analysis (credit scoring, risk modeling)
- Machine learning with solid statistical foundations

## Core Architecture Principles

### Design Philosophy
- **Paradigm**: Functional and modular over heavy OOP
- **Extensibility**: Every component must be replaceable/extensible by users
- **Composability**: Small components that combine to create complex functionality
- **Immutability**: Prefer immutable data structures where possible
- **Type hints**: Exhaustive use of Python type annotations with Protocols

### Key Architectural Patterns

#### 1. Array Namespace Abstraction (`aurora/distributions/_utils.py`)

**CRITICAL PATTERN**: The codebase supports both NumPy and PyTorch arrays transparently through a namespace pattern:

```python
from aurora.distributions._utils import namespace, as_namespace_array

def my_function(x, y):
    # Detect which array library is being used
    xp = namespace(x, y)  # Returns np or torch

    # Convert values to the appropriate array type
    x_arr = as_namespace_array(x, xp, like=y)

    # Use namespace-specific operations
    if xp is torch:
        return torch.sum(x_arr)
    return np.sum(x_arr)
```

**Key utilities in `_utils.py`**:
- `namespace(*values)`: Detects NumPy vs PyTorch from input arrays
- `as_namespace_array(value, xp, like=None)`: Converts to appropriate array type
- `is_torch(value)`: Checks if value is a torch tensor
- `ones_like(value)`: Creates ones array in the correct namespace
- `clip_probability(prob, xp, eps)`: Clips probabilities with namespace-aware ops
- `log_factorial(value, xp)`: Log factorial using correct backend
- `log_gamma(value, xp)`: Log gamma using correct backend

**Why this pattern?** Distribution families (Gaussian, Poisson, etc.) work with both NumPy and PyTorch arrays without explicitly using the Backend abstraction. This allows mathematical code to be backend-agnostic at the implementation level.

**IMPORTANT**: This pattern MUST be maintained in all new distribution and link implementations.

#### 2. Backend Registry Pattern (`aurora/core/backends/__init__.py`)

Lazy-loading backend system with plugin support:

```python
from aurora.core.backends import get_backend, register_backend

# Built-in backends are loaded on first use
backend = get_backend("jax")  # Loads JAX backend lazily
backend = get_backend("pytorch")  # Loads PyTorch backend lazily

# Custom backends can be registered
def my_backend_factory():
    return MyCustomBackend()

register_backend("custom", my_backend_factory)
```

**Design details**:
- Backends implement a `Backend` Protocol with: `array()`, `as_numpy()`, `grad()`, `jit()`, `device_put()`
- Built-in backends (jax, pytorch) are registered lazily to avoid import errors
- Each backend module must expose `create_backend()` function
- Registry stored in `_BACKENDS` dict

#### 3. Protocol-Based Type System (`aurora/core/types.py`)

Extensive use of structural typing for flexibility:

```python
from aurora.core.types import Array, Scalar, LossFunction
from aurora.core.types import Distribution, Link, Optimizer

# Protocols define interfaces without inheritance
class MyLink(Link):  # Structural typing via Protocol
    def link(self, mu: Array) -> Array: ...
    def inverse(self, eta: Array) -> Array: ...
    def derivative(self, mu: Array) -> Array: ...
```

**Key type definitions**:
- `Array`: Union of np.ndarray, JAX array, torch.Tensor
- `Scalar`: Union of int, float, complex
- `LossFunction`: Callable returning Scalar
- `OptimizationCallback`: Callable taking (iteration, params, loss)
- Protocols: `Distribution`, `Link`, `Optimizer`, `OptimizationResult`, `ArrayLike`

#### 4. Abstract Base Classes with Protocols

Distributions and optimizers use ABCs for implementation, Protocols for typing:

- **Abstract classes** in `aurora/distributions/base.py`: `Family`, `LinkFunction`
- **Abstract classes** in `aurora/core/optimization/base.py`: `Optimizer`
- **Protocols** in `aurora/core/types.py` for structural typing
- This allows both inheritance-based and duck-typed implementations

## Current Implementation Status

### Phase 1: Core Numerical Foundation - COMPLETED ✅ (100%)

**Backend System** (100% complete):
```
aurora/core/backends/
├── __init__.py           ✅ Backend registry
├── jax_backend.py        ✅ JAX implementation
└── pytorch_backend.py    ✅ PyTorch implementation
```

**Features**:
- ✅ Backend abstraction with unified protocol
- ✅ JAX support (JIT, grad, vmap, device_put)
- ✅ PyTorch support with full compatibility
- ✅ Custom backend registration
- ✅ Automatic array detection (NumPy/PyTorch/JAX)

**Type System** (100% complete):
```
aurora/core/types.py      ✅ Protocols and type aliases
```

**Optimization Algorithms** (100% complete):
```
aurora/core/optimization/
├── __init__.py           ✅ Unified optimize() interface
├── result.py             ✅ OptimizationResult dataclass
├── newton.py             ✅ Newton-Raphson
├── irls.py               ✅ IRLS (Iteratively Reweighted Least Squares)
└── lbfgs.py              ✅ L-BFGS with line search
```

**Features**:
- ✅ Newton-Raphson with automatic Hessian
- ✅ IRLS for GLM
- ✅ L-BFGS with two-loop recursion
- ✅ Line search (Armijo backtracking)
- ✅ Callbacks for monitoring
- ✅ Robust convergence

**Distribution Families** (80% complete - 4 of 10 planned):
```
aurora/distributions/families/
├── __init__.py           ✅ Exports
├── gaussian.py           ✅ Normal/Gaussian
├── poisson.py            ✅ Poisson
├── binomial.py           ✅ Binomial
└── gamma.py              ✅ Gamma
```

**Implemented distributions**:
- ✅ Gaussian (Normal) - complete
- ✅ Poisson - complete
- ✅ Binomial - complete
- ✅ Gamma - complete

**Pending distributions** (Phase 5):
- ⏳ Inverse Gaussian
- ⏳ Negative Binomial
- ⏳ Beta
- ⏳ Tweedie
- ⏳ Exponential
- ⏳ Multinomial

**Link Functions** (70% complete - 5 of 8 planned):
```
aurora/distributions/links/
├── __init__.py           ✅ Exports and get_link() registry
└── common.py             ✅ All implemented links
```

**Implemented links**:
- ✅ Identity: `g(μ) = μ`
- ✅ Log: `g(μ) = log(μ)`
- ✅ Logit: `g(μ) = log(μ/(1-μ))`
- ✅ Inverse: `g(μ) = 1/μ`
- ✅ CLogLog: `g(μ) = log(-log(1-μ))`

**Pending links** (Phase 5):
- ⏳ Probit: `g(μ) = Φ⁻¹(μ)`
- ⏳ Square root: `g(μ) = √μ`
- ⏳ Power: `g(μ) = μᵖ`

**Array Namespace Utilities** (100% complete):
```
aurora/distributions/_utils.py  ✅ Multi-backend support
```

### Phase 2: Basic GLM - IN PROGRESS 🚧 (80% complete)

**Objective**: Implement functional GLM models with IRLS fitting, prediction, inference, diagnostics, and evaluation metrics.

**Detailed Phase 2 Goals**:
- IRLS fitting with robust convergence
- Prediction with confidence/prediction intervals
- Basic inference (confidence intervals, p-values via Wald approximation)
- Model diagnostics (residuals, leverage, Cook's distance)
- Evaluation metrics (deviance, AIC, BIC, pseudo R²)
- Validation tools (cross-validation, scoring functions)

**Current structure**:
```
aurora/models/
├── base/
│   ├── __init__.py           ✅ Exports
│   └── result.py             ✅ GLMResult with predict()
├── glm/
│   ├── __init__.py           ✅ Exports fit_glm
│   └── fitting.py            ✅ Complete IRLS implementation (317 lines)
├── gam/                      ⏳ Phase 3
└── gamm/                     ⏳ Phase 4
```

**Implemented Components**:

#### 1. ✅ GLM Fitting Function (`aurora/models/glm/fitting.py`) - COMPLETE

**Status**: Fully implemented with 317 lines of code, including:
- Complete IRLS algorithm
- Multi-backend support (NumPy/PyTorch)
- Weights and offset support
- Deviance, AIC, BIC computation
- Null deviance calculation
- Family and link registry

**Original specification**:

**Main function signature**:
```python
def fit_glm(
    X: Array,
    y: Array,
    *,
    family: str | Family = "gaussian",
    link: str | LinkFunction | None = None,
    weights: Array | None = None,
    offset: Array | None = None,
    backend: str = "jax",
    max_iter: int = 25,
    tol: float = 1e-8,
    fit_intercept: bool = True,
) -> GLMResult:
    """
    Fit a Generalized Linear Model using IRLS.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Design matrix
    y : array-like, shape (n_samples,)
        Response variable
    family : str or Family
        Distribution family ('gaussian', 'poisson', 'binomial', 'gamma')
    link : str or LinkFunction, optional
        Link function. If None, uses canonical link for family
    weights : array-like, optional
        Observation weights
    offset : array-like, optional
        Offset term
    backend : str
        Backend to use ('jax', 'pytorch', 'numpy')
    max_iter : int
        Maximum IRLS iterations
    tol : float
        Convergence tolerance
    fit_intercept : bool
        Whether to fit intercept

    Returns
    -------
    GLMResult
        Fitted model result with parameters, predictions, inference
    """
```

**IRLS Algorithm Steps**:
1. Initialize μ using `family.initialize(y)`
2. For each iteration:
   - Calculate η = link(μ)
   - Calculate working response: z = η + (y - μ) * link.derivative(μ)
   - Calculate weights: w = 1 / (link.derivative(μ)² * family.variance(μ))
   - Solve: β = (X'WX)⁻¹ X'Wz
   - Update: η = Xβ, μ = link.inverse(η)
   - Check convergence

**Convergence criterion**:
```python
# Convergence based on deviance change
dev_change = abs(deviance_new - deviance_old) / (abs(deviance_old) + 0.1)
converged = dev_change < tol
```

#### 2. ✅ GLMResult Class (`aurora/models/base/result.py`) - MOSTLY COMPLETE

**Status**: Implemented with 142 lines, includes:
- ✅ Fitted coefficients and intercept
- ✅ Fitted values (mu, eta)
- ✅ Model statistics (deviance, null_deviance, AIC, BIC)
- ✅ Convergence info
- ✅ `predict()` method with 'response' and 'link' types
- 🚧 Lazy-computed inference (std_errors, p_values, coef_cov) - placeholders only
- 🚧 Confidence/prediction intervals - implemented in `aurora/inference/intervals/`
- ❌ `summary()` method - not yet implemented
- ❌ `plot_diagnostics()` - not yet implemented

#### 3. ✅ Inference Module (`aurora/inference/`) - MOSTLY IMPLEMENTED

**Current structure**:
```
aurora/inference/
├── diagnostics/
│   ├── __init__.py           ✅ Exports
│   └── glm.py                ✅ Residuals, Cook's distance, leverage
├── intervals/
│   ├── __init__.py           ✅ Exports
│   └── confidence.py         ✅ Confidence intervals
├── hypothesis/
│   ├── __init__.py           ✅ Exports
│   └── wald.py               ✅ Wald tests
└── anova/
    └── __init__.py           🚧 ANOVA (placeholder)
```

**Implemented functions**:
- ✅ `glm_diagnostics()`: Returns GLMDiagnosticResult with residuals (response, Pearson, deviance, working), leverage, Cook's distance
- ✅ Confidence intervals (in intervals module)
- ✅ Wald tests (in hypothesis module)

**Still to implement**:
- ⏳ `likelihood_ratio_test()`: Compare nested models
- ⏳ ANOVA for GLMs

#### 4. ✅ Validation Metrics (`aurora/validation/`) - IMPLEMENTED

**Current structure**:
```
aurora/validation/
├── metrics/
│   ├── __init__.py           ✅ Exports
│   ├── glm.py                ✅ GLM-specific metrics
│   ├── regression.py         ✅ Regression metrics
│   └── classification.py     ✅ Classification metrics
└── cross_val/
    ├── __init__.py           ✅ Exports
    ├── split.py              ✅ KFold splitter
    └── evaluate.py           ✅ Cross-validation evaluation
```

**Implemented metrics**:
- ✅ Deviance (computed during IRLS in fit_glm)
- ✅ AIC (computed in fit_glm)
- ✅ BIC (computed in fit_glm)
- ✅ `pseudo_r2()`: McFadden and deviance-based pseudo R²
- ✅ KFold cross-validation splitter
- ✅ Cross-validation evaluation utilities

**Still to implement**:
- ⏳ Cox-Snell and Nagelkerke pseudo R²
- ⏳ `concordance_index()`: C-statistic for binary outcomes

### Phase 2 Progress Checklist

**Week 1 (Completed)**:
- [x] Implement `fit_glm()` with robust IRLS and synthetic data tests
- [x] Complete `GLMResult` with lazy properties and prediction method
- [x] Calculate covariance matrix, standard errors, and p-values

**Week 2 (In Progress)**:
- [x] Basic residuals and initial influence measures
- [x] Evaluation metrics (regression, classification, pseudo R²)
- [x] Generic cross-validation (`KFold`, `cross_val_score`)
- [ ] Advanced metrics (specific deviance, concordance index)
- [ ] Integration with benchmarks and external validation

**Week 3 (Pending)**:
- [ ] Documented examples and notebooks
- [ ] Visualizations and diagnostic plots
- [ ] Documentation and release 0.2.0 preparation

### Representative Test Coverage

The following test suites cover the implemented functionality:
```bash
pytest tests/test_inference/test_confidence_intervals.py
pytest tests/test_inference/test_hypothesis.py
pytest tests/test_inference/test_diagnostics.py
pytest tests/test_validation/test_metrics.py
pytest tests/test_validation/test_classification_metrics.py
pytest tests/test_validation/test_cross_val.py
pytest tests/test_validation/  # Run all validation tests
```

### Phase 3: GAM (Splines and Smoothing) - PLANNED 📋 (Not started)

**Timeline**: 6-8 weeks

**Objective**: Implement Generalized Additive Models with:
- Spline basis functions (cubic, B-splines, P-splines, thin plate)
- Penalization and smoothing parameter selection (GCV, REML, AIC)
- R-style formula parser (patsy-like)
- Visualization of smooth terms
- Integration with existing GLM infrastructure

**Components**:
```
aurora/smoothing/
├── splines/
│   ├── cubic.py              # Cubic splines
│   ├── bsplines.py           # B-splines
│   ├── psplines.py           # P-splines (penalized B-splines)
│   ├── thinplate.py          # Thin plate splines
│   └── tensor.py             # Tensor product splines
├── penalties/
│   ├── ridge.py              # Ridge penalty
│   └── difference.py         # Difference penalty
└── selection/
    ├── gcv.py                # Generalized Cross-Validation
    ├── reml.py               # Restricted Maximum Likelihood
    └── aic.py                # AIC-based selection

aurora/models/gam/
├── fitting.py                # fit_gam()
├── formula.py                # Formula parser (patsy-like)
└── result.py                 # GAMResult
```

**Target API**:
```python
from aurora.models.gam import fit_gam

# R-style formula
result = fit_gam(
    formula="y ~ s(x1, bs='tp', k=10) + s(x2, bs='cr') + x3 + x4",
    data=df,
    family='gaussian',
    method='REML'
)

# Visualize smooth terms
result.plot_smooth('s(x1)')
result.summary()
```

### Phase 4: GAMM (Random Effects) - PLANNED 📋 (Not started)

**Timeline**: 6-8 weeks

**Objective**: Add random effects to GAM:
- Random intercepts and slopes
- Crossed and nested random effects
- Hierarchical multilevel models
- REML/ML/Laplace estimation
- Covariance structures (AR, compound symmetry, custom)

**Components**:
```
aurora/models/gamm/
├── fitting.py                # fit_gamm()
├── random_effects.py         # Random effects structures
└── result.py                 # GAMMResult

aurora/estimation/
├── reml/
│   └── reml.py               # REML estimation
├── ml/
│   └── ml.py                 # Maximum Likelihood
└── laplace/
    └── laplace.py            # Laplace approximation
```

**Target API**:
```python
from aurora.models.gamm import fit_gamm

# Mixed model with random effects
result = fit_gamm(
    formula="""
        y ~ s(time, by=treatment, k=10) +
            s(age, bs='cr') +
            (1 + time | subject) +
            (1 | clinic)
    """,
    data=df,
    family='gamma',
    link='log',
    method='REML'
)
```

### Phase 5: Extended Features - PLANNED 📋 (Not started)

**Timeline**: Ongoing

**Additional Distribution Families**:
- Inverse Gaussian (for positively skewed continuous data)
- Negative Binomial (for overdispersed count data)
- Beta (for proportions and rates in (0,1))
- Tweedie (for insurance claims, zero-inflated positive continuous)
- Exponential (for survival/time-to-event data)
- Multinomial (for multi-class categorical outcomes)

**Additional Link Functions**:
- Probit: `g(μ) = Φ⁻¹(μ)` (for binary data, alternative to logit)
- Square root: `g(μ) = √μ` (for count data with variance proportional to mean)
- Power: `g(μ) = μᵖ` (generalized power links)

**Advanced Features**:
- Comprehensive visualization suite
- Performance optimizations (Cython for critical loops, sparse matrices)
- Advanced diagnostics and influence measures
- Model selection tools (stepwise, LASSO, elastic net)
- Robust standard errors (sandwich estimators, bootstrap)
- Handling of missing data (imputation, deletion strategies)

## Current Usage (Phase 2 - Implemented)

### Fitting a GLM

The `fit_glm()` function is fully implemented and ready to use:

```python
import numpy as np
from aurora.models.glm import fit_glm

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = np.random.poisson(np.exp(X[:, 0] * 0.5 + 0.1))

# Fit a Poisson GLM with log link (canonical)
result = fit_glm(
    X, y,
    family='poisson',
    link='log',  # or None for canonical link
    fit_intercept=True,
    max_iter=25,
    tol=1e-8
)

# Access results
print(f"Coefficients: {result.coef_}")
print(f"Intercept: {result.intercept_}")
print(f"Deviance: {result.deviance_}")
print(f"AIC: {result.aic_}")
print(f"BIC: {result.bic_}")
print(f"Converged: {result.converged_}")
print(f"Iterations: {result.n_iter_}")

# Make predictions
X_new = np.random.randn(10, 3)
predictions = result.predict(X_new, type='response')  # or type='link'
```

### Using Diagnostics

```python
from aurora.inference.diagnostics import glm_diagnostics

# Compute diagnostics
diag = glm_diagnostics(result)

print(f"Response residuals: {diag.response_residuals}")
print(f"Pearson residuals: {diag.pearson_residuals}")
print(f"Deviance residuals: {diag.deviance_residuals}")
print(f"Leverage: {diag.leverage}")
print(f"Cook's distance: {diag.cooks_distance}")
```

### Computing Metrics

```python
from aurora.validation.metrics import pseudo_r2

# Compute pseudo R²
r2_mcfadden = pseudo_r2(result, method='mcfadden')
r2_deviance = pseudo_r2(result, method='deviance')

print(f"McFadden pseudo R²: {r2_mcfadden}")
print(f"Deviance pseudo R²: {r2_deviance}")
```

### Cross-Validation

```python
from aurora.validation.cross_val import KFold
from aurora.models.glm import fit_glm

# Create KFold splitter
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
scores = []
for train_idx, val_idx in kfold.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Fit on training set
    result = fit_glm(X_train, y_train, family='poisson')

    # Predict on validation set
    y_pred = result.predict(X_val, type='response')

    # Compute score (example: mean squared error)
    score = np.mean((y_val - y_pred) ** 2)
    scores.append(score)

print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

## Development Commands

### Package Installation

**IMPORTANT**: The package currently has no `pyproject.toml` or `setup.py`, so installation is done in development mode by adding the repository root to `PYTHONPATH`:

```bash
# On Windows (Git Bash/MSYS):
export PYTHONPATH="${PYTHONPATH}:/j/Aurora-GLM"

# Or add to your .bashrc/.zshrc:
echo 'export PYTHONPATH="${PYTHONPATH}:/j/Aurora-GLM"' >> ~/.bashrc

# On Linux/macOS:
export PYTHONPATH="${PYTHONPATH}:/path/to/Aurora-GLM"
```

**Dependencies** (install manually):
```bash
pip install numpy pytest pytest-cov ruff mypy

# Optional backends:
pip install torch  # PyTorch support
pip install jax jaxlib  # JAX support (CPU)
```

**TODO**: Add `pyproject.toml` with proper package metadata and dependencies.

### Testing

```bash
# Run all tests
pytest

# Run specific test directory
pytest tests/test_distributions/

# Run specific test file
pytest tests/test_distributions/test_links.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=aurora --cov-report=html

# Run specific test function
pytest tests/test_distributions/test_links.py::test_identity_link_roundtrip

# Run tests for a specific backend (if PyTorch installed)
pytest tests/test_core/test_backends/test_pytorch_backend.py
```

**Test structure**:
- Tests use `@pytest.mark.parametrize("xp", _namespaces())` to test both NumPy and PyTorch
- PyTorch tests are skipped automatically if PyTorch is not installed
- Fixtures in `tests/conftest.py` provide backend instances

### Code Quality

```bash
# Format code
ruff format aurora/ tests/

# Lint code
ruff check aurora/ tests/

# Type checking
mypy aurora/
```

### Code Style

The project follows modern Python conventions:
- Type hints on all public functions
- Docstrings in NumPy/Google format
- `from __future__ import annotations` for forward references
- Use of Protocol for structural typing

## Success Criteria for Phase 2

### Functionality (Must Have)
- [x] `fit_glm()` works with all implemented families (Gaussian, Poisson, Binomial, Gamma)
- [x] Prediction works correctly (`predict()` method implemented)
- [x] Basic inference implemented (diagnostics, residuals, intervals, Wald tests)
- [ ] Confidence intervals fully integrated into GLMResult
- [ ] P-values computed and validated
- [x] Residuals implemented (response, Pearson, deviance, working)
- [ ] `summary()` method for GLMResult

### Validation (Must Have)
- [ ] Results match statsmodels (within tolerance 1e-6 for coefficients)
- [ ] Results match R glm() (within tolerance)
- [x] Tests pass with NumPy backend
- [ ] Tests pass with PyTorch backend (if installed)
- [ ] Tests pass with JAX backend (if installed)
- [ ] Coverage >90%

### Performance (Should Have)
- [ ] Comparable or faster than statsmodels
- [ ] No memory leaks
- [ ] Scalable to 100K+ observations

### Documentation (Must Have)
- [ ] All docstrings complete
- [ ] At least 3 working examples with real datasets
- [ ] README updated with current status
- [ ] Basic tutorial notebook
- [x] CLAUDE.md updated (this file)

## Key Implementation Patterns

### Writing Distribution Families

When implementing a new distribution family in `aurora/distributions/families/`:

```python
from aurora.distributions.base import Family, LinkFunction
from aurora.distributions._utils import namespace, as_namespace_array

class MyFamily(Family):
    def __init__(self, link: LinkFunction | None = None) -> None:
        self._link = link or MyDefaultLink()

    def log_likelihood(self, y, mu, **params):
        # 1. Detect namespace
        xp = namespace(y, mu)

        # 2. Convert to arrays
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = as_namespace_array(mu, xp, like=y_arr)

        # 3. Implement logic using xp operations
        return compute_log_likelihood(y_arr, mu_arr, xp)

    def deviance(self, y, mu, **params):
        xp = namespace(y, mu)
        # ... similar pattern

    def variance(self, mu, **params):
        xp = namespace(mu)
        # ... similar pattern

    def initialize(self, y):
        xp = namespace(y)
        return as_namespace_array(y, xp, like=y)

    @property
    def default_link(self) -> LinkFunction:
        return self._link
```

### Writing Link Functions

When implementing a new link function in `aurora/distributions/links/`:

```python
from aurora.distributions.base import LinkFunction
from aurora.distributions._utils import namespace, as_namespace_array

class MyLink(LinkFunction):
    def link(self, mu: Array) -> Array:
        xp = namespace(mu)
        mu_arr = as_namespace_array(mu, xp, like=mu)
        # Implement g(μ)
        return xp.some_function(mu_arr)

    def inverse(self, eta: Array) -> Array:
        xp = namespace(eta)
        # Implement g^(-1)(η)
        return xp.inverse_function(eta)

    def derivative(self, mu: Array) -> Array:
        xp = namespace(mu)
        # Implement dg/dμ
        return xp.derivative_function(mu)
```

### Writing Tests for Multi-Backend Code

Tests should verify both NumPy and PyTorch compatibility:

```python
import pytest
import numpy as np

try:
    import torch
except ImportError:
    torch = None

def _namespaces():
    libs = [np]
    if torch is not None:
        libs.append(torch)
    return tuple(libs)

@pytest.mark.parametrize("xp", _namespaces())
def test_my_function(xp):
    # Create test data in appropriate namespace
    if xp is np:
        data = np.array([1.0, 2.0, 3.0])
    else:
        data = torch.tensor([1.0, 2.0, 3.0])

    # Test function
    result = my_function(data)

    # Assertions using appropriate comparison
    # ...
```

### Testing Strategy

**Multi-Backend Testing**:
- Use `@pytest.mark.parametrize` to test NumPy, PyTorch, and JAX backends
- Automatically skip backends that aren't installed (see `tests/conftest.py`)
- Ensure numerical results are consistent across backends (within tolerance)

**Validation Testing**:
```python
def test_glm_matches_statsmodels():
    """GLM results should match statsmodels."""
    # Create test data
    X = np.random.randn(100, 3)
    y = np.random.poisson(np.exp(X @ [0.5, -0.3, 0.2]))

    # Fit with Aurora
    aurora_result = fit_glm(X, y, family='poisson')

    # Fit with statsmodels
    import statsmodels.api as sm
    statsmodels_result = sm.GLM(y, sm.add_constant(X),
                                 family=sm.families.Poisson()).fit()

    # Compare coefficients (tolerance 1e-6)
    np.testing.assert_allclose(
        aurora_result.coef_,
        statsmodels_result.params[1:],
        rtol=1e-6
    )

    # Compare std errors (tolerance 1e-5)
    np.testing.assert_allclose(
        aurora_result.std_errors_,
        statsmodels_result.bse[1:],
        rtol=1e-5
    )

    # Compare p-values (tolerance 1e-4)
    np.testing.assert_allclose(
        aurora_result.p_values_,
        statsmodels_result.pvalues[1:],
        rtol=1e-4
    )
```

**Test Organization**:
```
tests/
├── conftest.py                    # Shared fixtures (backend instances, etc.)
├── test_core/
│   ├── test_backends/            # Backend-specific tests
│   └── test_optimization/        # Optimizer tests
├── test_distributions/
│   ├── test_families.py          # Distribution family tests
│   └── test_links.py             # Link function tests
├── test_models/
│   ├── test_glm_fitting.py       # GLM fitting tests
│   └── test_glm_result.py        # GLMResult tests
├── test_inference/
│   ├── test_confidence_intervals.py
│   ├── test_hypothesis.py
│   └── test_diagnostics.py
├── test_validation/
│   ├── test_metrics.py
│   ├── test_classification_metrics.py
│   └── test_cross_val.py
└── test_integration/
    ├── test_vs_statsmodels.py    # Validation against statsmodels
    └── test_vs_r.py               # Validation against R (if available)
```

**Coverage Goals**:
- Overall: >90%
- Critical modules (fitting, inference): >95%
- Edge cases and error handling: Well-covered
- Multi-backend compatibility: All code paths tested

## Module Structure

```
aurora/
├── __init__.py                    # Package exports (backends only currently)
├── core/
│   ├── backends/                  # ✅ Backend abstraction (JAX, PyTorch)
│   │   ├── __init__.py           # Registry, get_backend(), register_backend()
│   │   ├── jax_backend.py        # JAX implementation
│   │   └── pytorch_backend.py    # PyTorch implementation
│   ├── autodiff/                  # 🚧 Auto-differentiation helpers (empty)
│   ├── optimization/              # ✅ Optimization algorithms
│   │   ├── base.py               # Abstract Optimizer class
│   │   ├── newton.py             # Newton-Raphson
│   │   ├── irls.py               # IRLS
│   │   ├── lbfgs.py              # L-BFGS
│   │   └── result.py             # OptimizationResult container
│   ├── linalg/                    # 🚧 Matrix operations (empty)
│   └── types.py                   # ✅ Type definitions and Protocols
├── distributions/
│   ├── base.py                    # ✅ Abstract Family and LinkFunction
│   ├── _utils.py                  # ✅ Array namespace utilities (CRITICAL!)
│   ├── families/                  # ✅ Distribution families
│   │   ├── gaussian.py           # Gaussian/Normal
│   │   ├── poisson.py            # Poisson
│   │   ├── binomial.py           # Binomial
│   │   └── gamma.py              # Gamma
│   ├── links/                     # ✅ Link functions
│   │   ├── common.py             # Identity, Log, Logit, Inverse, CLogLog
│   │   └── __init__.py           # Link registry and get_link()
│   └── custom/                    # 🚧 User-defined distributions (empty)
├── models/                        # 🚧 GLM/GAM/GAMM (in progress)
│   ├── base/                      # 🚧 Base model result
│   ├── glm/                       # 🚧 GLM fitting (to implement)
│   ├── gam/                       # ⏳ GAM (planned)
│   └── gamm/                      # ⏳ GAMM (planned)
├── smoothing/                     # ⏳ Splines (planned)
├── inference/                     # ✅ Diagnostics, intervals, hypothesis tests
│   ├── diagnostics/              # ✅ Residuals, Cook's distance, leverage
│   ├── intervals/                # ✅ Confidence intervals
│   ├── hypothesis/               # ✅ Wald tests
│   └── anova/                    # 🚧 ANOVA (placeholder)
├── estimation/                    # ⏳ REML/ML/Laplace (planned)
├── validation/                    # ✅ Metrics and cross-validation
│   ├── metrics/                  # ✅ GLM, regression, classification metrics
│   └── cross_val/                # ✅ KFold splitter and evaluation
├── visualization/                 # ⏳ Plotting (planned)
├── io/                            # ⏳ Data I/O (planned)
└── utils/                         # ✅ Custom exceptions
    └── exceptions.py              # BackendNotAvailableError, etc.
```

## Important Implementation Notes

1. **Always use the namespace pattern** when implementing distributions or links that need to work with both NumPy and PyTorch
2. **Type hints are mandatory** - use Protocols from `aurora.core.types` for flexibility
3. **Docstrings should include mathematical formulas** in LaTeX format when relevant (`:math:`f(x) = x^2``)
4. **Tests must cover both NumPy and PyTorch** using parametrization (see existing tests)
5. **The Backend abstraction is separate from the namespace pattern** - backends are for high-level orchestration, namespace utilities are for implementation-level array operations
6. **Validate against statsmodels and R**: All GLM implementations must match reference implementations within reasonable tolerance
7. **Follow NumPy docstring style** with Parameters, Returns, Examples, Notes, References sections

## Key Architectural Insights

### Two-Level Array Abstraction

The codebase has **two separate but complementary** array abstractions:

1. **Namespace pattern** (`aurora/distributions/_utils.py`):
   - Used at **implementation level** (distributions, links)
   - Detects NumPy vs PyTorch automatically via `namespace()` function
   - Allows math code to be backend-agnostic
   - Example: `xp = namespace(y, mu)` then use `xp.sum()`, `xp.log()`, etc.

2. **Backend abstraction** (`aurora/core/backends/`):
   - Used at **orchestration level** (optimization, high-level algorithms)
   - Provides JIT compilation, automatic differentiation, device management
   - Explicitly selected: `backend = get_backend("jax")`
   - Example: `backend.grad(loss_fn)`, `backend.jit(compute_fn)`

**Why both?** The namespace pattern keeps distribution code simple and doesn't require JIT/grad machinery. The backend abstraction provides advanced features when needed (e.g., in optimization).

### GLM Fitting Architecture

The GLM implementation follows this flow:

1. **`fit_glm()`** (`aurora/models/glm/fitting.py`):
   - Entry point, validates inputs, selects family/link
   - Implements IRLS algorithm directly (no Backend needed)
   - Returns `GLMResult`

2. **`GLMResult`** (`aurora/models/base/result.py`):
   - Stores fitted model parameters and metadata
   - Provides `predict()` method
   - **Does NOT** compute inference automatically (lazy evaluation)
   - Stores references to original data for diagnostics

3. **Inference functions** (`aurora/inference/`):
   - Separate functions that take `GLMResult` as input
   - `glm_diagnostics(result)` → computes residuals, leverage, Cook's D
   - `confidence_intervals(result)` → computes intervals (in intervals module)
   - `wald_test(result)` → hypothesis tests (in hypothesis module)

**Design rationale**: Separating inference from fitting allows:
- Users to skip expensive computations if not needed
- Different inference methods without bloating GLMResult
- Better testability and modularity

### Family and Link Registry

Both families and links use string-based registries:

```python
# In aurora/models/glm/fitting.py
_FAMILY_REGISTRY = {
    "gaussian": GaussianFamily,
    "poisson": PoissonFamily,
    "binomial": BinomialFamily,
    "gamma": GammaFamily,
}

_LINK_REGISTRY = {
    "identity": IdentityLink,
    "log": LogLink,
    "logit": LogitLink,
    "inverse": InverseLink,
    "cloglog": CLogLogLink,
}
```

Users can pass either:
- String: `family='poisson'` → looks up in registry
- Instance: `family=PoissonFamily()` → uses directly
- Custom class: `family=MyCustomFamily()` → full extensibility

This pattern enables both convenience and extensibility.

### Performance Considerations

**Current Optimizations**:
- ✅ Backend abstraction allows JIT compilation (with JAX)
- ✅ Vectorized operations throughout (no Python loops in hot paths)
- ✅ Efficient memory usage with in-place operations where safe
- ✅ BLAS-free matrix operations (`_matvec`) for namespace compatibility

**Planned Optimizations**:
- Cython for critical loops (if profiling shows benefit)
- Sparse matrix support for large-scale datasets
- Chunking/batching for datasets that don't fit in memory
- GPU acceleration via JAX backend for large models
- Parallel cross-validation with multiple backends

**Performance Goals**:
- **GLM fitting**: Competitive with statsmodels, ideally 1-2x faster
- **GAM fitting**: Within 2x of R's mgcv package
- **Scalability**: Handle 1M+ observations efficiently on standard hardware
- **GPU acceleration**: Efficient utilization when available (JAX backend)

**Benchmarking Strategy**:
1. Compare against statsmodels GLM on various datasets (small, medium, large)
2. Compare against R's `glm()` and `mgcv::gam()` for accuracy and speed
3. Profile critical code paths to identify bottlenecks
4. Memory profiling to prevent leaks and excessive allocation
5. Publish reproducible benchmarks in documentation

## Documentation Standards

**NumPy-style docstrings**:
```python
def function(param1: Type1, param2: Type2) -> ReturnType:
    """
    Short one-line summary.

    Longer description explaining the function's purpose,
    behavior, and any important notes. Can include LaTeX
    for mathematical notation: :math:`f(x) = x^2`.

    Parameters
    ----------
    param1 : Type1
        Description of param1
    param2 : Type2
        Description of param2

    Returns
    -------
    ReturnType
        Description of return value

    Raises
    ------
    ValueError
        When this error occurs

    Examples
    --------
    >>> result = function(arg1, arg2)
    >>> result
    expected_output

    Notes
    -----
    Additional implementation or mathematical details.

    References
    ----------
    .. [1] Author, "Title", Journal, Year.
    """
```

## Key References

### Theory
- McCullagh & Nelder (1989) - "Generalized Linear Models" 2nd Ed.
- Wood (2017) - "Generalized Additive Models: An Introduction with R" 2nd Ed.
- Hastie & Tibshirani (1990) - "Generalized Additive Models"

### Reference Implementations
- **R glm()**: Base stats package
- **R mgcv**: GAM implementation by Simon Wood
- **statsmodels.genmod**: Python GLM implementation
- **scikit-learn**: For API design patterns

### Technical Resources
- JAX documentation: https://jax.readthedocs.io
- PyTorch documentation: https://pytorch.org/docs
- Array API standard: https://data-apis.org/array-api

## Immediate Next Steps

### Priority 1: Complete Phase 2 GLM Implementation (This Week)

**Status**: Core fitting, inference, and validation are implemented. Focus on integration and validation.

**Day 1-2: GLMResult Enhancement**
- [ ] Wire up inference methods to GLMResult (std_errors, p_values, coef_cov from placeholders)
- [ ] Integrate `glm_diagnostics()` into GLMResult as properties/methods
- [ ] Add `summary()` method to GLMResult with formatted output
- [ ] Add convenience methods for residuals access

**Day 3-4: Validation and Testing**
- [ ] Tests comparing with statsmodels for all families (Gaussian, Poisson, Binomial, Gamma)
- [ ] Tests comparing with R's `glm()` for all families
- [ ] Edge case handling tests (perfect separation, zero counts, etc.)
- [ ] Performance benchmarks against statsmodels

**Day 5: Documentation**
- [ ] Complete all docstrings in GLM modules
- [ ] Create at least one example notebook with real dataset
- [ ] Update README with current Phase 2 status

### Priority 2: Advanced Features (Next 2 Weeks)

**Week 2: Extended Inference and Metrics**
- [ ] Extend `wald_test()` to support multivariate contrasts and chi-square tests
- [ ] Implement likelihood ratio tests (`likelihood_ratio_test()`) for nested models
- [ ] Add Cox-Snell and Nagelkerke pseudo R²
- [ ] Implement concordance index for binary outcomes
- [ ] Add additional diagnostics (DFBETAs, studentized residuals)

**Week 3: Visualization and Examples**
- [ ] Documented examples with real datasets:
  - Poisson regression (count data)
  - Logistic regression (binomial)
  - Gamma regression (positive continuous)
  - Examples with weights and offsets
- [ ] Basic diagnostic plots:
  - Residual plots
  - QQ plots
  - Influence plots
  - Partial residual plots
- [ ] Jupyter notebooks with complete workflows
- [ ] Tutorial documentation

### Priority 3: Infrastructure and Release Preparation

1. **Package setup**
   - [ ] Create `pyproject.toml` with proper metadata and dependencies
   - [ ] Define optional dependencies ([dev], [test], [torch], [jax])
   - [ ] Make package installable via `pip install aurora-glm`
   - [ ] Set up package versioning

2. **Testing and CI/CD**
   - [ ] Achieve >90% test coverage
   - [ ] Set up GitHub Actions for CI
   - [ ] Add pre-commit hooks (ruff, mypy)
   - [ ] Add automated testing for multiple Python versions (3.10, 3.11, 3.12)

3. **Documentation**
   - [ ] Set up documentation site (Sphinx/MkDocs)
   - [ ] API reference documentation
   - [ ] User guide with examples
   - [ ] Prepare release notes for v0.2.0

## Important Notes for Development

### Working with Git

**Current uncommitted changes** (as of last session):
- `aurora/validation/cross_val/__init__.py` - Modified
- `aurora/validation/cross_val/evaluate.py` - Modified
- `aurora/validation/cross_val/split.py` - Modified
- `tests/test_validation/test_cross_val.py` - Modified

These files are currently staged/modified. Review changes before committing.

### PYTHONPATH Setup

Since there's no `pyproject.toml`, you **must** add the repository to `PYTHONPATH`:

```bash
# Check if PYTHONPATH is set correctly:
python -c "import aurora; print(aurora.__file__)"

# Should output: J:\Aurora-GLM\aurora\__init__.py (or similar)
```

If import fails, set PYTHONPATH as shown in "Package Installation" section above.

## Common Gotchas

1. **Don't mix Backend and namespace patterns**: Use backends for high-level control flow, namespace utilities for implementations
2. **PyTorch device handling**: `as_namespace_array()` handles device placement automatically
3. **Type annotations**: Use `from __future__ import annotations` to avoid circular imports
4. **Optional dependencies**: Wrap PyTorch imports in try/except blocks
5. **Link function derivatives**: Can be computed manually or via auto-diff (when available)
6. **IRLS convergence**: Monitor deviance changes, not just parameter changes
7. **Numerical stability**: Check for singular matrices, overflow/underflow in link functions
8. **Import errors**: If you get `ModuleNotFoundError: No module named 'aurora'`, check that `PYTHONPATH` includes the repository root

## Development Workflow

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/Matcraft94/Aurora-GLM.git
cd Aurora-GLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Set PYTHONPATH (since no pyproject.toml yet)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Install dependencies
pip install numpy pytest pytest-cov ruff mypy

# Optional: Install backends
pip install torch  # PyTorch backend
pip install jax jaxlib  # JAX backend
```

### Standard Development Workflow

```bash
# Create feature branch
git checkout -b feature/implement-new-feature

# Make changes and test frequently
pytest tests/test_specific_module.py -v

# Run full test suite before committing
pytest

# Check code quality
ruff format aurora/ tests/
ruff check aurora/ tests/
mypy aurora/

# Commit with descriptive message
git add .
git commit -m "feat(module): add new feature with tests"

# Push and create PR
git push origin feature/implement-new-feature
```

### Contribution Priorities

**HIGH PRIORITY** (needed for Phase 2 completion):
1. Complete GLMResult integration with inference
2. Add `summary()` method to GLMResult
3. Validation tests against statsmodels and R
4. Documentation and examples
5. Create `pyproject.toml` for proper package installation

**MEDIUM PRIORITY** (nice to have for Phase 2):
1. Additional distributions (Inverse Gaussian, Negative Binomial)
2. Additional link functions (Probit, Square root)
3. Performance benchmarks
4. More comprehensive examples

**LOW PRIORITY** (future phases):
1. GAM implementation (Phase 3)
2. GAMM implementation (Phase 4)
3. Advanced visualization

## Commit Checklist

Before each commit, verify:

- [ ] Code formatted with `ruff format aurora/ tests/`
- [ ] Passes `ruff check aurora/ tests/` without errors
- [ ] Passes `mypy aurora/` without errors (or with acceptable warnings)
- [ ] Relevant tests added for new functionality
- [ ] All tests pass (`pytest`)
- [ ] Docstrings complete (NumPy style)
- [ ] Type hints present on all public functions
- [ ] Examples in docstrings work and are tested
- [ ] No debugging print statements or commented code
- [ ] CHANGELOG.md updated (if applicable)
