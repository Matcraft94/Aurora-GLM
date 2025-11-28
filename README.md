# Aurora-GLM

**Aurora-GLM** is a modular, extensible, and high-performance Python framework for statistical modeling, focusing on Generalized Linear Models (GLM), Generalized Additive Models (GAM), and Generalized Additive Mixed Models (GAMM).

> ✅ **Development Status**: Phase 5 IN PROGRESS (75%). Full GAM/GAMM implementation complete. Non-Gaussian GAMM (Poisson, Binomial) with PQL estimation available!

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/Matcraft94/Aurora-GLM)

## Project Identity

- **Package name**: `aurora-glm`
- **Python import**: `import aurora`
- **Repository**: [github.com/Matcraft94/Aurora-GLM](https://github.com/Matcraft94/Aurora-GLM)
- **Author**: Lucy E. Arias ([@Matcraft94](https://github.com/Matcraft94))
- **Version**: 0.5.0
- **Status**: Phase 5 IN PROGRESS (75%) - Non-Gaussian GAMM (Poisson, Binomial), PQL estimation, nested/crossed random effects
- **Python**: 3.10+
- **Tagline**: *Illuminating complex data with modern generalized linear modeling tools*

## Vision and Goals

Aurora-GLM aims to be:

1. **Scientifically rigorous**: Correct implementations validated against R (mgcv) and statsmodels
2. **High performance**: Competitive or superior to existing alternatives, with GPU support
3. **Extensible**: Users can add custom distributions, link functions, and algorithms
4. **Multi-backend**: Transparent support for NumPy, PyTorch, and JAX
5. **Modular and functional**: Clean design favoring composition over complex inheritance

### Use Cases

- **Academic research**: Ecology, epidemiology, social sciences
- **Pharmaceutical industry**: Clinical trials analysis
- **Financial analysis**: Credit scoring, risk modeling
- **Machine learning**: Statistical foundations with modern tools

## Current Implementation Status

### Phase 1: Core Numerical Foundation - COMPLETED ✅ (100%)

**Backend Infrastructure**:
- ✅ Backend abstraction layer (JAX, PyTorch)
- ✅ Type system with comprehensive Protocols
- ✅ Array namespace utilities for transparent NumPy/PyTorch compatibility

**Optimization Algorithms**:
- ✅ Newton-Raphson with automatic Hessian
- ✅ IRLS (Iteratively Reweighted Least Squares) for GLM
- ✅ L-BFGS with line search and two-loop recursion
- ✅ Callbacks for monitoring
- ✅ Robust convergence checking

**Distribution Families** (4/10 planned):
- ✅ Gaussian (Normal)
- ✅ Poisson
- ✅ Binomial
- ✅ Gamma

**Link Functions** (5/8 planned):
- ✅ Identity: `g(μ) = μ`
- ✅ Log: `g(μ) = log(μ)`
- ✅ Logit: `g(μ) = log(μ/(1-μ))`
- ✅ Inverse: `g(μ) = 1/μ`
- ✅ CLogLog: `g(μ) = log(-log(1-μ))`

### Phase 2: Basic GLM - COMPLETED ✅ (100%)

**Implemented**:
- ✅ IRLS-based `fit_glm()` with multi-backend support (NumPy/PyTorch), weights, and offsets
- ✅ `GLMResult` with predictions, metrics (deviance, AIC, BIC, null deviance), and lazy inference
- ✅ `GLMResult.summary()` - R-style formatted tables with coef, std err, z-scores, p-values, significance codes
- ✅ `GLMResult.plot_diagnostics()` - 4 standard diagnostic plots (residuals, Q-Q, scale-location, leverage)
- ✅ Confidence intervals integrated in `predict(interval='confidence')` with delta method
- ✅ P-values and standard errors via Wald approximation (lazy computation)
- ✅ Wald hypothesis tests for single and multi-constraint hypotheses (chi-square)
- ✅ Comprehensive diagnostics: response, Pearson, deviance, working, and studentized residuals
- ✅ Influence measures: leverage, Cook's distance, DFBETAs
- ✅ Validation metrics: MSE, MAE, RMSE, pseudo R², accuracy, log-loss, Brier score, concordance index (C-index)
- ✅ Cross-validation: `KFold`, `StratifiedKFold`, and `cross_val_score` with aggregated results
- ✅ Validation against statsmodels (max |Δcoef| ≈ 4e-06) and R glm() (max |Δcoef| ≈ 5e-05)
- ✅ 119 tests passing (84% coverage) across inference, diagnostics, validation, and fitting
- ✅ Demo notebooks: Poisson regression and logistic regression with visualizations


### Phase 3: GAM (Splines and Smoothing) - COMPLETED ✅ (100%)

**Implemented** (229 new tests):
- ✅ **B-spline basis functions**: Cox-de Boor recursion, local support, partition of unity (17 tests)
- ✅ **Natural cubic spline basis**: Truncated power basis with analytical penalties (16 tests)
- ✅ **Penalty matrices**: Difference penalties, weighted penalties, ridge penalties, combinations (20 tests)
- ✅ **GCV smoothing selection**: Automatic λ selection via Generalized Cross-Validation (15 tests)
- ✅ **REML smoothing selection**: Restricted Maximum Likelihood for better multi-term selection (20 tests)
- ✅ **Univariate GAM fitting**: `fit_gam()` with automatic smoothing, predictions, summaries (20 tests)
- ✅ **Multivariate additive GAMs**: `fit_additive_gam()` with multiple smooth and parametric terms (15 tests)
- ✅ **R-style formula parser**: `y ~ s(x1, bs='tp') + s(x2) + x3` syntax with comprehensive validation (12 tests)
- ✅ **Formula-based fitting**: `fit_gam_formula()` for high-level API (5 tests)
- ✅ **Visualization**: `plot_smooth()` and `plot_all_smooths()` with confidence bands (18 tests)
- ✅ **Tensor product smooths**: `te(x1, x2)` for multidimensional interactions (13 tests)
- ✅ **Thin plate splines**: Multidimensional smoothing with radial basis functions (24 tests)
- ✅ **Term specifications**: SmoothTerm, ParametricTerm, TensorTerm dataclasses (14 tests)

> Full design documentation in `aurora/smoothing/DESIGN.md`. Total: **348 tests passing** (up from 119 in Phase 2).

### Phase 4: GAMM (Random Effects) - COMPLETED ✅ (100%)

**Implemented** (150+ tests):
- ✅ **Random effects infrastructure**: RandomEffect specification with intercepts and slopes
- ✅ **Design matrix construction**: Z matrix builder with block-diagonal structure
- ✅ **REML estimation**: Variance component estimation via restricted maximum likelihood
- ✅ **Mixed model equations solver**: Augmented system solver for β and random effects b
- ✅ **Gaussian GAMM fitting**: `fit_gamm_gaussian()` with smooth terms and random effects
- ✅ **High-level interface**: `fit_gamm()` and `fit_gamm_with_smooth()` with pandas support
- ✅ **Predictions**: Population-level and conditional predictions with `predict_from_gamm()`
- ✅ **Covariance structures**: Unstructured, diagonal, and identity parameterizations with Cholesky
- ✅ **Non-Gaussian families**: Poisson, Binomial with PQL (Penalized Quasi-Likelihood) estimation
- ✅ **Formula parser extensions**: lme4-style syntax `(1 + x | group)` supported
- ✅ **Nested and crossed random effects**: Full support for complex random effect structures
- ✅ **Visualization**: Caterpillar plots, Q-Q plots for random effects
- ✅ **Comprehensive example**: Longitudinal data analysis (sleep study) with visualizations

### Phase 5: Extended Features - IN PROGRESS 🚧 (75%)

**Implemented in Phase 5**:
- ✅ **Multi-backend stability**: JAX float32 tolerance handling, overflow protection
- ✅ **Numerical robustness**: LogLink clamping, PQL NaN/Inf protection
- ✅ **Unified result hierarchy**: LinearModelResult, MixedModelResultBase
- ✅ **I/O module**: CSV reading, result save/load, coefficient export
- ✅ **Validation decorators**: @validate_array, @validate_positive, @validate_probability
- ✅ **Sensitivity analysis**: Cook's distance, leverage, influence diagnostics
- ✅ **ANOVA module**: Type I/II/III SS, likelihood ratio tests
- ✅ **Helper functions**: summary(), plot(), compare() unified API

**Remaining for Phase 5**:
- 📋 Additional distributions (Inverse Gaussian, Negative Binomial, Beta, Tweedie)
- 📋 Additional link functions (Probit, Square root, Power)
- 📋 Performance optimizations (sparse matrices, Cython for critical paths)
- 📋 AR1 and compound symmetry covariance structures

## Quick Start - GLM API (Phase 2 - AVAILABLE NOW!)

### Basic Poisson Regression

```python
import numpy as np
from aurora.models.glm import fit_glm

# Generate sample count data
np.random.seed(42)
X = np.random.randn(200, 2)
y = np.random.poisson(np.exp(X[:, 0] * 0.5 - 0.3))

# Fit a Poisson GLM with log link
result = fit_glm(X, y, family='poisson', link='log')

# Print R-style summary with coefficients, std errors, p-values
print(result.summary())
```

**Output:**
```
================================================================================
                    Generalized Linear Model Results
================================================================================
Family:                  Poisson           Link Function:         Log
No. Observations:            200           Df Residuals:          197
Df Model:                      2           Pseudo R-squared:    0.123
Converged:                   Yes           No. Iterations:          5
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
intercept       -0.2987     0.0712     -4.197      0.000     -0.4382     -0.1593  ***
X0               0.5124     0.0718      7.137      0.000      0.3717      0.6531  ***
X1              -0.0234     0.0706     -0.331      0.741     -0.1617      0.1150
================================================================================
Deviance:          212.54                  Null Deviance:       240.32
AIC:               465.43                  BIC:               475.52
================================================================================
Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

### Predictions and Diagnostics

```python
# Make predictions
X_new = np.random.randn(10, 2)
predictions = result.predict(X_new, type='response')

# Confidence intervals for predictions
ci_lower, ci_upper = result.predict(X_new, interval='confidence', level=0.95)

# Generate diagnostic plots (residuals, Q-Q, scale-location, leverage)
result.plot_diagnostics()
```

### Logistic Regression for Classification

```python
# Generate binary classification data
X = np.random.randn(300, 2)
probabilities = 1 / (1 + np.exp(-(X @ np.array([1.2, -0.9]) + 0.3)))
y = np.random.binomial(1, probabilities)

# Fit logistic regression
result = fit_glm(X, y, family='binomial', link='logit')

# View summary
print(result.summary())

# Predict probabilities
y_prob = result.predict(X, type='response')

# Classification metrics
from aurora.validation.metrics import accuracy_score, concordance_index

y_pred = (y_prob >= 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
print(f"C-index (AUROC): {concordance_index(y, y_prob):.3f}")
```

### Multi-Backend Support

```python
import torch

# PyTorch tensors work transparently
X_torch = torch.randn(100, 2)
y_torch = torch.poisson(torch.exp(X_torch[:, 0] * 0.5))

# Same API, PyTorch backend
result = fit_glm(X_torch, y_torch, family='poisson')
print(result.summary())
```

### Cross-Validation

```python
from aurora.validation.cross_val import cross_val_score, KFold

# Evaluate model with 5-fold cross-validation
scores = cross_val_score(
    X, y,
    family='poisson',
    cv=KFold(n_splits=5),
    metrics=['deviance', 'pseudo_r2']
)

print(f"Mean deviance: {scores['deviance'].mean():.2f}")
print(f"Mean pseudo R²: {scores['pseudo_r2'].mean():.3f}")
```

For complete examples with visualizations, see:
- `examples/01_poisson_regression.ipynb` - Count data regression
- `examples/02_logistic_regression.ipynb` - Binary classification with ROC curves

## GAM API (Phase 3 - AVAILABLE NOW!)

### Basic Univariate GAM

```python
from aurora.models.gam import fit_gam
import numpy as np

# Generate noisy data with non-linear relationship
np.random.seed(42)
x = np.linspace(0, 1, 100)
y_true = np.sin(2 * np.pi * x)
y = y_true + 0.1 * np.random.randn(100)

# Fit GAM with automatic smoothing parameter selection (GCV)
result = fit_gam(x, y, n_basis=12, basis_type='bspline')

# Print model summary
print(result.summary())
# Shows: lambda, EDF, R², residual diagnostics

# Make predictions at new points
x_new = np.linspace(0, 1, 200)
y_pred = result.predict(x_new)
```

**Output:**
```
============================================================
Generalized Additive Model (GAM) - Fitted Summary
============================================================

Model Information:
  Basis type:          BSplineBasis
  Number of basis:     12
  Observations:        100

Smoothing:
  Lambda:              8.806605e-02
  Effective DoF:       9.69
  GCV score:           3.662963e-02

Fit Statistics:
  Residual sum sq:     2.9872
  R-squared:           0.9421
  Residual std:        0.1727

Residuals:
  Min:                 -0.7174
  Q1:                  -0.0823
  Median:              0.0123
  Q3:                  0.0957
  Max:                 0.6197
============================================================
```

### Advanced GAM Options

```python
# Specify smoothing parameter manually
result = fit_gam(x, y, n_basis=15, lambda_=0.1)

# Use cubic splines instead of B-splines
result = fit_gam(x, y, n_basis=10, basis_type='cubic')

# Weighted observations
weights = np.random.uniform(0.5, 1.5, size=len(x))
result = fit_gam(x, y, n_basis=12, weights=weights)

# Different knot placement methods
result = fit_gam(x, y, n_basis=12, knot_method='uniform')  # or 'quantile'
```

### Multivariate Additive GAM

```python
from aurora.models.gam import fit_additive_gam, SmoothTerm, ParametricTerm
import numpy as np

# Generate data with multiple nonlinear relationships
np.random.seed(42)
n = 200
X = np.random.randn(n, 3)
y = (np.sin(2 * X[:, 0]) +         # Nonlinear effect of x1
     np.cos(X[:, 1]) +              # Nonlinear effect of x2
     2 * X[:, 2] +                  # Linear effect of x3
     0.1 * np.random.randn(n))

# Fit additive GAM: y = intercept + f1(x1) + f2(x2) + β*x3
result = fit_additive_gam(
    X, y,
    smooth_terms=[
        SmoothTerm(variable=0, n_basis=12, basis_type='bspline'),
        SmoothTerm(variable=1, n_basis=12, basis_type='cubic')
    ],
    parametric_terms=[
        ParametricTerm(variable=2)  # Linear term
    ]
)

# Print comprehensive summary
print(result.summary())
# Shows: parametric coefficients, smooth term details (λ, EDF), fit stats

# Make predictions
X_new = np.random.randn(50, 3)
y_pred = result.predict(X_new)
```

### Visualizing Smooth Terms

```python
from aurora.models.gam import plot_smooth, plot_all_smooths

# Plot single smooth term with confidence bands
fig = plot_smooth(
    result,
    term='s(0)',                    # or term=0 for first variable
    confidence_level=0.95,
    partial_residuals=True
)

# Plot all smooth terms in a grid
fig = plot_all_smooths(
    result,
    ncols=2,
    confidence_level=0.95
)

# Customize plot
fig = plot_smooth(
    result,
    term=0,
    title="Effect of X1 on Response",
    xlabel="X1",
    ylabel="f(X1)",
    n_points=200
)
```

### Formula-Based API (Now Available!)

```python
from aurora.models.gam import fit_gam_formula
import pandas as pd

# R-style formula with smooth terms
result = fit_gam_formula(
    formula="y ~ s(x1, k=10) + s(x2, bs='cubic') + x3",
    data=df,
    method='REML'
)

# Tensor product interactions
result = fit_gam_formula(
    formula="y ~ te(x1, x2) + s(x3)",
    data=df,
    method='GCV'
)

# Print comprehensive summary
print(result.summary())

# Visualize all smooth terms
from aurora.models.gam import plot_all_smooths
plot_all_smooths(result)
```

## GAMM API (Phase 4 - AVAILABLE NOW for Gaussian!)

### Basic Random Intercept Model

```python
from aurora.models import fit_gamm
from aurora.models.gamm import RandomEffect
import numpy as np

# Generate longitudinal data
np.random.seed(42)
n_subjects, n_per_subject = 10, 15
n = n_subjects * n_per_subject
subject_id = np.repeat(np.arange(n_subjects), n_per_subject)
time = np.tile(np.arange(n_per_subject), n_subjects)

# Design matrix
X = np.column_stack([np.ones(n), time])

# Random intercepts (subject-specific baseline)
b_subj = np.random.randn(n_subjects) * 0.8
y = 2.0 + 0.5 * time + b_subj[subject_id] + np.random.randn(n) * 0.3

# Fit GAMM with random intercept
re = RandomEffect(grouping='subject')
result = fit_gamm(
    y=y,
    X=X,
    random_effects=[re],
    groups_data={'subject': subject_id},
    covariance='identity'
)

# View results
print(f"Fixed effects (β): {result.beta_parametric}")
print(f"Variance components (Ψ): {result.variance_components}")
print(f"Residual variance (σ²): {result.residual_variance}")
print(f"AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")
```

### Random Intercept + Slope Model

```python
# Random intercept + slope on 'time' (variable index 1)
re_slope = RandomEffect(
    grouping='subject',
    variables=(1,),  # Random slope for 'time'
    include_intercept=True
)

# Fit model with unstructured covariance (allows correlation)
result = fit_gamm(
    y=y,
    X=X,
    random_effects=[re_slope],
    groups_data={'subject': subject_id},
    covariance='unstructured'
)

# Extract variance-covariance matrix
psi = result.variance_components
print(f"Var(intercept): {psi[0, 0]:.3f}")
print(f"Cov(intercept, slope): {psi[0, 1]:.3f}")
print(f"Var(slope): {psi[1, 1]:.3f}")

# Correlation between random intercept and slope
corr = psi[0, 1] / np.sqrt(psi[0, 0] * psi[1, 1])
print(f"Correlation: {corr:.3f}")
```

### Making Predictions

```python
from aurora.models import predict_from_gamm

# Population-level predictions (for new, unobserved subjects)
X_new = np.column_stack([np.ones(20), np.arange(20)])
pred_pop = predict_from_gamm(result, X_new, include_random=False)

# Conditional predictions (for existing subject 0)
groups_new = np.zeros(20, dtype=int)  # All for subject 0
pred_cond = predict_from_gamm(
    result,
    X_new,
    groups_new=groups_new,
    include_random=True
)

print(f"Population prediction at time=10: {pred_pop[10]:.2f}")
print(f"Conditional prediction (subject 0) at time=10: {pred_cond[10]:.2f}")
```

### Pandas Support

```python
import pandas as pd

# Work with DataFrames
df = pd.DataFrame({
    'y': y,
    'time': time,
    'subject': subject_id,
    'treatment': np.random.choice(['A', 'B'], n)
})

# Fit model with DataFrame inputs
re = RandomEffect(grouping='subject')
result = fit_gamm(
    y=df['y'],
    X=df[['time']],
    random_effects=[re],
    groups_data=df[['subject']],
    covariance='identity'
)
```

### Complete Longitudinal Example

For a complete example with model comparison, diagnostics, and visualization, see:
- `examples/gamm_example.py` - Simulated sleep study with random intercepts and slopes

**Coming soon** (Non-Gaussian families with PQL/Laplace):
```python
# Future: Poisson GAMM for count data
result = fit_gamm(
    y=counts,
    X=X,
    random_effects=[re],
    groups_data={'subject': subject_id},
    family='poisson',  # Not yet implemented
    link='log'
)
```

## Current Usage (Low-Level Components)

While high-level GLM fitting is not yet available, you can use the foundational components:

### Using Distribution Families

```python
from aurora.distributions.families import GaussianFamily, PoissonFamily
from aurora.distributions.links import LogLink
import numpy as np

# Create a Poisson family with log link
poisson = PoissonFamily(link=LogLink())

# Generate data
y = np.array([1, 2, 3, 4, 5])
mu = np.array([1.5, 2.0, 2.8, 4.2, 5.1])

# Compute log-likelihood
log_lik = poisson.log_likelihood(y, mu)

# Compute deviance
dev = poisson.deviance(y, mu)

# Variance function
var = poisson.variance(mu)
```

### Multi-Backend Support

The same code works seamlessly with PyTorch tensors:

```python
import torch

# PyTorch tensors work transparently
y_torch = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
mu_torch = torch.tensor([1.5, 2.0, 2.8, 4.2, 5.1])

# Same API, different backend
log_lik_torch = poisson.log_likelihood(y_torch, mu_torch)
dev_torch = poisson.deviance(y_torch, mu_torch)
```

### Using Backend Abstraction

```python
from aurora.core.backends import get_backend

# Get JAX backend
jax_backend = get_backend("jax")
x = jax_backend.array([1, 2, 3])
grad_fn = jax_backend.grad(my_loss_function)

# Get PyTorch backend
torch_backend = get_backend("pytorch")
```

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/Matcraft94/Aurora-GLM.git
cd Aurora-GLM

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Optional: Install PyTorch for multi-backend support
pip install torch

# Optional: Install JAX for GPU support
pip install jax jaxlib
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aurora --cov-report=html

# Run specific test file
pytest tests/test_distributions/test_links.py

# Run specific test
pytest tests/test_distributions/test_links.py::test_identity_link_roundtrip

# Run with verbose output
pytest -v
```

### External Validation

Compare Aurora fits against statsmodels using the benchmarking harness:

```bash
PYTHONPATH=. python benchmarks/run_glm_checks.py --replicates 3 --output benchmarks/results/glm_vs_statsmodels.json
```

By default the script benchmarks Gaussian (identity), Poisson (log), Binomial (logit) and Gamma (log). Append `--gamma-inverse` to include the numerically fragile Gamma+inverse combination.

**Latest Statsmodels comparison (replicates=3):**
- max |delta_coef| ≈ `4.07e-06`
- max delta_deviance ≈ `5.69e-09`
- max mean |delta_mu| ≈ `2.07e-05`

### Code Quality

```bash
# Format code
ruff format aurora/ tests/

# Lint code
ruff check aurora/ tests/

# Type checking
mypy aurora/
```

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/implement-glm-fitting

# Make changes and test
pytest
pytest --cov=aurora --cov-report=html

# Format and lint
ruff format aurora/ tests/
ruff check aurora/ tests/
mypy aurora/

# Commit with descriptive message
git commit -m "feat(glm): implement IRLS fitting algorithm"

# Push and create PR
git push origin feature/implement-glm-fitting
```

## Contributing

Contributions are welcome! This project is in active development with many opportunities to help:

### HIGH PRIORITY (needed for Phase 2)

1. **Core GLM implementation**
   - Implement `fit_glm()` with IRLS algorithm
   - Complete `GLMResult` with inference (std errors, p-values)
   - Implement residuals and diagnostics
   - Validate against statsmodels and R

2. **Testing and validation**
   - Tests comparing with statsmodels
   - Tests comparing with R's `glm()`
   - Performance benchmarks
   - Edge case handling

3. **Documentation**
   - Examples with real datasets
   - Tutorial notebooks
   - API documentation

### MEDIUM PRIORITY (nice to have)

1. **Additional distributions**
   - Inverse Gaussian
   - Negative Binomial
   - Beta
   - Tweedie

2. **Additional link functions**
   - Probit
   - Square root
   - Power

3. **Performance optimizations**
   - Benchmarking suite
   - Cython for critical loops (if needed)
   - Sparse matrix support

### LOW PRIORITY (future phases)

1. GAM implementation (Phase 3)
2. GAMM implementation (Phase 4)
3. Advanced visualization

### Contribution Guidelines

Please ensure:
- **Type hints** on all public functions
- **Tests** covering both NumPy and PyTorch backends
- **Docstrings** in NumPy/Google format with examples
- **Code formatting** with `ruff format`
- **Multi-backend support** using the namespace pattern

## Design Principles

### Array Namespace Pattern

Aurora-GLM uses a namespace abstraction to support multiple array libraries transparently:

```python
from aurora.distributions._utils import namespace, as_namespace_array

def my_function(x, y):
    # Automatically detect NumPy or PyTorch
    xp = namespace(x, y)

    # Convert to appropriate array type
    x_arr = as_namespace_array(x, xp, like=y)

    # Use namespace-specific operations
    return xp.sum(x_arr)
```

This pattern allows distribution families and link functions to work seamlessly with any array library without code changes.

### Extensibility

Users can create custom distributions and link functions:

```python
from aurora.distributions.base import Family, LinkFunction

class MyDistribution(Family):
    def log_likelihood(self, y, mu, **params):
        # Implementation using namespace pattern
        xp = namespace(y, mu)
        # ...

    def deviance(self, y, mu, **params): ...
    def variance(self, mu, **params): ...
    def initialize(self, y): ...

    @property
    def default_link(self):
        return MyLink()

class MyLink(LinkFunction):
    def link(self, mu): ...
    def inverse(self, eta): ...
    def derivative(self, mu): ...
```

## Project Structure

```
aurora/
├── core/
│   ├── backends/         # Backend abstraction (JAX, PyTorch)
│   ├── optimization/     # Optimization algorithms
│   └── types.py          # Type definitions and Protocols
├── distributions/
│   ├── families/         # Distribution families
│   ├── links/            # Link functions
│   └── _utils.py         # Array namespace utilities
├── models/               # GLM/GAM/GAMM (in progress)
├── smoothing/            # Splines and penalties (planned)
├── inference/            # Hypothesis testing (to implement)
├── estimation/           # REML/ML/Laplace (planned)
├── validation/           # Metrics and cross-validation (to implement)
└── visualization/        # Plotting utilities (planned)
```

## Roadmap

### Short Term (3 months)
- [x] Core infrastructure (backends, types, optimization)
- [x] Distribution families (Gaussian, Poisson, Binomial, Gamma)
- [x] Link functions (Identity, Log, Logit, Inverse, CLogLog)
- [ ] GLM fitting with IRLS
- [ ] Inference (confidence intervals, p-values)
- [ ] Diagnostics (residuals, Cook's distance)
- [ ] Validation against statsmodels and R
- [ ] Basic examples and documentation

### Medium Term (6 months)
- [ ] GAM with spline basis functions
- [ ] Formula parser for R-style formulas
- [ ] Smoothing parameter selection (GCV, REML)
- [ ] Visualization of smooth terms
- [ ] Performance benchmarks published
- [ ] 10+ GitHub stars
- [ ] External users reporting issues

### Long Term (12 months)
- [ ] GAMM with random effects
- [ ] REML/ML/Laplace estimation
- [ ] 100+ GitHub stars
- [ ] 1000+ PyPI downloads/month
- [ ] Research paper or conference presentation
- [ ] 5+ active contributors

## Success Metrics

### Phase 2 Success Criteria

**Functionality** (Must Have):
- [ ] `fit_glm()` works with all implemented families
- [ ] Predictions are correct
- [ ] Confidence intervals match reference implementations
- [ ] P-values match reference implementations
- [ ] Residuals (deviance, Pearson) implemented

**Validation** (Must Have):
- [ ] Results match statsmodels within tolerance (1e-6 for coefficients)
- [ ] Results match R's `glm()` within tolerance
- [ ] Tests pass with NumPy, PyTorch, and JAX backends
- [ ] Test coverage >90%

**Performance** (Should Have):
- [ ] Comparable or faster than statsmodels
- [ ] No memory leaks
- [ ] Scalable to 100K+ observations

**Documentation** (Must Have):
- [ ] All docstrings complete
- [ ] At least 3 working examples
- [ ] README updated with usage examples
- [ ] Basic tutorial notebook

## Performance Goals

- **GLM**: Competitive with statsmodels, ideally faster
- **GAM**: Within 2x of R's mgcv package
- **Scalability**: Handle 1M+ observations efficiently
- **GPU acceleration**: Efficient utilization when available

## References and Mathematical Foundations

Aurora-GLM is built on rigorous statistical foundations with comprehensive mathematical documentation. For detailed mathematical formulations, proofs, and derivations, see **[REFERENCES.md](REFERENCES.md)**.

### Core Statistical Theory

**Generalized Linear Models (GLM)**:
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman and Hall/CRC.
- Nelder, J. A., & Wedderburn, R. W. M. (1972). "Generalized linear models." *JRSS: Series A*, 135(3), 370-384.

**Generalized Additive Models (GAM)**:
- Hastie, T., & Tibshirani, R. (1990). *Generalized Additive Models*. Chapman and Hall/CRC.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). CRC Press.
- Wood, S. N. (2011). "Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models." *JRSS: Series B*, 73(1), 3-36.

**Generalized Additive Mixed Models (GAMM)**:
- Breslow, N. E., & Clayton, D. G. (1993). "Approximate inference in generalized linear mixed models." *JASA*, 88(421), 9-25.
- Lin, X., & Breslow, N. E. (1996). "Bias correction in generalized linear mixed models with multiple components of dispersion." *JASA*, 91(435), 1007-1016.
- Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). "Fitting linear mixed-effects models using lme4." *Journal of Statistical Software*, 67(1), 1-48.

**Smoothing and Splines**:
- de Boor, C. (2001). *A Practical Guide to Splines* (Revised ed.). Springer.
- Eilers, P. H. C., & Marx, B. D. (1996). "Flexible smoothing with B-splines and penalties." *Statistical Science*, 11(2), 89-121.
- Craven, P., & Wahba, G. (1978). "Smoothing noisy data with spline functions." *Numerische Mathematik*, 31(4), 377-403.

### Numerical Methods

**Optimization**:
- Green, P. J. (1984). "Iteratively reweighted least squares for maximum likelihood estimation." *JRSS: Series B*, 46(2), 149-192.
- Liu, D. C., & Nocedal, J. (1989). "On the limited memory BFGS method for large scale optimization." *Mathematical Programming*, 45(1-3), 503-528.

**Numerical Stability**:
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.

### Reference Implementations (Validation)

- **R glm()**: Base stats package (GLM reference)
- **R mgcv**: GAM/GAMM by Simon Wood (gold standard for smoothing)
- **R lme4**: Mixed models by Bates et al. (GLMM reference)
- **statsmodels**: Python statistical modeling library
- **scikit-learn**: API design patterns and conventions

### Technical Resources

- **JAX**: Composable transformations - https://jax.readthedocs.io
- **PyTorch**: Automatic differentiation - https://pytorch.org/docs
- **Array API Standard**: Cross-library compatibility - https://data-apis.org/array-api

### Complete Bibliography

For a comprehensive list of mathematical foundations, algorithms, and validation references, including detailed equations and derivations, please see **[REFERENCES.md](REFERENCES.md)**.

## License

*(Add license information here - e.g., MIT)*

## Citation

*(Add citation information when published)*

## Acknowledgments

Aurora-GLM draws inspiration from:
- R's **mgcv** package by Simon Wood
- Python's **statsmodels** library
- The JAX ecosystem for modern array programming

Special thanks to the open-source community for providing excellent tools and libraries.

---

**Status**: 🚧 Phase 5 IN PROGRESS (75%): Extended features, multi-backend stability, comprehensive test coverage
**Tests**: 1021 passing, 14 skipped (comprehensive coverage across GLM, GAM, GAMM)
**Version**: 0.5.0
**Python**: 3.10+
**Maintained by**: Lucy E. Arias ([@Matcraft94](https://github.com/Matcraft94))

*Illuminating complex data with modern generalized linear and additive modeling tools.*
