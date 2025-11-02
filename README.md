# Aurora-GLM

**Aurora-GLM** is a modular, extensible, and high-performance Python framework for statistical modeling, focusing on Generalized Linear Models (GLM), Generalized Additive Models (GAM), and Generalized Additive Mixed Models (GAMM).

> ⚠️ **Development Status**: Phase 2 in progress. Core infrastructure is complete, GLM fitting functions are currently being implemented. Contributions and feedback are welcome!

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/Matcraft94/Aurora-GLM)

## Project Identity

- **Package name**: `aurora-glm`
- **Python import**: `import aurora`
- **Repository**: [github.com/Matcraft94/Aurora-GLM](https://github.com/Matcraft94/Aurora-GLM)
- **Author**: Lucy E. Arias ([@Matcraft94](https://github.com/Matcraft94))
- **Version**: 0.2.0-dev
- **Status**: Phase 2 - GLM fitting implemented (~60% complete, inference pending)
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

### Phase 2: Basic GLM - IN PROGRESS 🚧 (60%)

**Implemented**:
- ✅ GLM model fitting with IRLS (317 lines)
- ✅ `fit_glm()` function with multi-backend support
- ✅ `GLMResult` class with `predict()` method
- ✅ Evaluation metrics (deviance, AIC, BIC, null deviance)
- ✅ Support for weights and offsets

**Still to implement**:
- 🚧 Inference (std errors, p-values, confidence intervals)
- 🚧 Model diagnostics (residuals, Cook's distance, leverage)
- 🚧 Additional metrics (pseudo R², concordance index)
- 🚧 Validation against statsmodels and R
- 🚧 `summary()` and `plot_diagnostics()` methods

**Timeline**: Expected completion in 2 weeks

### Phase 3: GAM (Splines and Smoothing) - PLANNED 📋

**Timeline**: 6-8 weeks after Phase 2

**Planned features**:
- 📋 Spline basis functions (cubic, B-splines, P-splines, thin plate, tensor product)
- 📋 Penalization and smoothing parameter selection (GCV, REML, AIC)
- 📋 R-style formula parser (`y ~ s(x1, bs='tp') + s(x2)`)
- 📋 Visualization of smooth terms

### Phase 4: GAMM (Random Effects) - PLANNED 📋

**Timeline**: 6-8 weeks after Phase 3

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

## Planned API (Phase 2 - Not Yet Available)

> ⚠️ **Note**: The API below shows the planned design. GLM fitting is currently being implemented.

```python
import numpy as np
from aurora.models.glm import fit_glm

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = np.random.poisson(np.exp(X[:, 0] * 0.5))

# Fit a Poisson GLM with log link
result = fit_glm(
    X, y,
    family='poisson',
    link='log',
    backend='jax'
)

# Model summary
print(result.summary())

# Make predictions
X_new = np.random.randn(10, 3)
predictions = result.predict(X_new, type='response')

# Confidence intervals
ci_lower, ci_upper = result.predict(X_new, interval='confidence', level=0.95)

# Diagnostic plots
result.plot_diagnostics()
```

### Planned GAM API (Phase 3)

```python
from aurora.models.gam import fit_gam

# R-style formula with smooth terms
result = fit_gam(
    formula="y ~ s(x1, bs='tp', k=10) + s(x2, bs='cr') + x3",
    data=df,
    family='gaussian',
    method='REML'
)

# Visualize smooth terms
result.plot_smooth('s(x1)')
result.summary()
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

See [CLAUDE.md](CLAUDE.md) for detailed implementation patterns and architectural guidelines.

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

**Status**: 🚧 Phase 2 in development (GLM fitting complete, inference pending)
**Version**: 0.2.0-dev
**Python**: 3.10+
**Maintained by**: Lucy E. Arias ([@Matcraft94](https://github.com/Matcraft94))

*Illuminating complex data with modern generalized linear modeling tools.*
