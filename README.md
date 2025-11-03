# Aurora-GLM

**Aurora-GLM** is a modular, extensible, and high-performance Python framework for statistical modeling, focusing on Generalized Linear Models (GLM), Generalized Additive Models (GAM), and Generalized Additive Mixed Models (GAMM).

> ✅ **Development Status**: Phase 3 IN PROGRESS (~50% complete). GAM core functionality implemented: spline bases (B-spline, cubic), penalties, GCV smoothing selection, and univariate fitting. Contributions and feedback welcome!

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/Matcraft94/Aurora-GLM)

## Project Identity

- **Package name**: `aurora-glm`
- **Python import**: `import aurora`
- **Repository**: [github.com/Matcraft94/Aurora-GLM](https://github.com/Matcraft94/Aurora-GLM)
- **Author**: Lucy E. Arias ([@Matcraft94](https://github.com/Matcraft94))
- **Version**: 0.3.0-dev
- **Status**: Phase 3 IN PROGRESS (~50%) - GAM splines, penalties, GCV selection, and univariate fitting implemented
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


### Phase 3: GAM (Splines and Smoothing) - IN PROGRESS 🔨 (~50% complete)

**Implemented** (87 new tests):
- ✅ **B-spline basis functions**: Cox-de Boor recursion, local support, partition of unity (17 tests)
- ✅ **Natural cubic spline basis**: Truncated power basis with analytical penalties (16 tests)
- ✅ **Penalty matrices**: Difference penalties, weighted penalties, ridge penalties, combinations (20 tests)
- ✅ **GCV smoothing selection**: Automatic λ selection via Generalized Cross-Validation (15 tests)
- ✅ **Univariate GAM fitting**: `fit_gam()` with automatic smoothing, predictions, summaries (20 tests)
- ✅ **GAMResult**: Comprehensive result object with predict(), summary(), EDF tracking

**Remaining features**:
- 📋 Multivariate GAMs (additive models with multiple smooth terms)
- 📋 REML/ML smoothing parameter selection (alternative to GCV)
- 📋 Tensor product smooths for interactions
- 📋 R-style formula parser (`y ~ s(x1, bs='tp') + s(x2)`)
- 📋 Visualization of smooth terms

> Full design available in `aurora/smoothing/DESIGN.md` (incremental plan with identified risks).

### Phase 4: GAMM (Random Effects) - PLANNED 📋

**Planned features**:
- 📋 Random effects (intercepts, slopes, crossed, nested)
- 📋 REML/ML/Laplace estimation
- 📋 Covariance structures (AR, compound symmetry, custom)
- 📋 Hierarchical multilevel models

### Phase 5: Extended Features - PLANNED 📋

**Additional distributions**:
- 📋 Inverse Gaussian
- 📋 Negative Binomial
- 📋 Beta
- 📋 Tweedie
- 📋 Exponential
- 📋 Multinomial

**Additional link functions**:
- 📋 Probit: `g(μ) = Φ⁻¹(μ)`
- 📋 Square root: `g(μ) = √μ`
- 📋 Power: `g(μ) = μᵖ`

**Advanced features**:
- 📋 Comprehensive diagnostics and visualization
- 📋 Validation against R's mgcv and statsmodels
- 📋 Performance optimizations (Cython, sparse matrices)

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

### Planned Multivariate GAM API (Coming Soon)

```python
# R-style formula with smooth terms (not yet implemented)
result = fit_gam(
    formula="y ~ s(x1, bs='tp', k=10) + s(x2, bs='cr') + x3",
    data=df,
    family='gaussian',
    method='REML'
)

# Visualize smooth terms
result.plot_smooth('s(x1)')
```

### Planned GAMM API (Phase 4)

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

## References

### Theory
- McCullagh, P. & Nelder, J.A. (1989). *Generalized Linear Models* (2nd ed.)
- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.)
- Hastie, T. & Tibshirani, R. (1990). *Generalized Additive Models*

### Reference Implementations
- **R glm()**: Base stats package
- **R mgcv**: GAM implementation by Simon Wood
- **statsmodels.genmod**: Python GLM implementation
- **scikit-learn**: API design patterns

### Technical Resources
- JAX: https://jax.readthedocs.io
- PyTorch: https://pytorch.org/docs
- Array API Standard: https://data-apis.org/array-api

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

**Status**: 🔨 Phase 3 IN PROGRESS (~50% complete): GAM splines, penalties, GCV, univariate fitting
**Tests**: 206 passing, 2 skipped (88 new GAM tests added)
**Version**: 0.3.0-dev
**Python**: 3.10+
**Maintained by**: Lucy E. Arias ([@Matcraft94](https://github.com/Matcraft94))

*Illuminating complex data with modern generalized linear and additive modeling tools.*
