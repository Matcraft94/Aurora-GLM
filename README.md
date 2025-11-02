# Aurora-GLM

**Aurora-GLM** is a modular, extensible, and high-performance Python framework for statistical modeling, focusing on Generalized Linear Models (GLM), Generalized Additive Models (GAM), and Generalized Additive Mixed Models (GAMM).

> ⚠️ **Early Development**: This project is in active development. Core infrastructure is in place, but GLM fitting functions are not yet implemented. Contributions and feedback are welcome!

## Project Identity

- **Package name**: `aurora-glm`
- **Python import**: `import aurora`
- **Repository**: [aurora-glm](https://github.com/yourusername/aurora-glm) *(update with actual URL)*
- **PyPI install**: `pip install aurora-glm` *(planned)*
- **Tagline**: *Illuminating complex data with modern generalized linear modeling tools.*

## Design Philosophy

Aurora-GLM embraces modern Python best practices while prioritizing scientific rigor and performance:

- **Functional and modular**: Prefer composition over complex object hierarchies
- **Extensible by design**: Every major component can be replaced or extended by users
- **Type-safe**: Exhaustive use of Python type hints with Protocol-based interfaces
- **Multi-backend**: Transparent support for NumPy, PyTorch, and JAX arrays
- **Performance-oriented**: Designed for JIT compilation, GPU acceleration, and scalability

## Current Implementation Status

### ✅ Implemented

**Core Infrastructure**:
- ✅ Backend abstraction layer (JAX, PyTorch)
- ✅ Type system with comprehensive Protocols
- ✅ Array namespace utilities for transparent NumPy/PyTorch compatibility

**Optimization Algorithms**:
- ✅ Newton-Raphson
- ✅ IRLS (Iteratively Reweighted Least Squares)
- ✅ L-BFGS

**Distribution Families**:
- ✅ Gaussian (Normal)
- ✅ Poisson
- ✅ Binomial
- ✅ Gamma

**Link Functions**:
- ✅ Identity: g(μ) = μ
- ✅ Log: g(μ) = log(μ)
- ✅ Logit: g(μ) = log(μ/(1-μ))
- ✅ Inverse: g(μ) = 1/μ
- ✅ CLogLog: g(μ) = log(-log(1-μ))

### 🚧 In Progress / Planned

**Phase 2: Basic GLM** (In Progress)
- 🚧 GLM model fitting with IRLS
- 🚧 Prediction and inference
- 🚧 Model diagnostics
- 🚧 Evaluation metrics

**Phase 3: GAM** (Planned)
- 📋 Spline basis functions (cubic, B-splines, P-splines, thin plate)
- 📋 Penalization and smoothing parameter selection (GCV, REML)
- 📋 R-style formula parser
- 📋 Visualization of smooth terms

**Phase 4: GAMM** (Planned)
- 📋 Random effects (intercepts, slopes, crossed, nested)
- 📋 REML/ML estimation
- 📋 Covariance structures (AR, compound symmetry, custom)

**Phase 5: Extended Features** (Planned)
- 📋 Additional distributions (Inverse Gaussian, Negative Binomial, Beta, Tweedie)
- 📋 Additional link functions (Probit, Square root, Power)
- 📋 Comprehensive diagnostics and visualization
- 📋 Validation against R's mgcv and statsmodels

## Technology Stack

### Numerical Backends

- **JAX** (primary): Auto-differentiation, JIT compilation, GPU/TPU support
- **PyTorch** (optional): Flexible backend selection
- **NumPy**: Always available as fallback

Backend selection is transparent to users through an abstraction layer that supports pluggable custom backends.

### Data Handling (Planned)

- **Polars**: High-performance data manipulation (primary)
- **Pandas**: Broad ecosystem compatibility (with PyArrow backend)
- Internal computations use backend-native arrays (JAX, PyTorch, NumPy)

### External Compatibility (Planned)

- **scikit-learn**: Compatible API for metrics and pipelines
- **statsmodels**: Interoperability and validation
- **R/mgcv**: Numerical parity checks for GAM/GAMM

## Quick Start (Planned API)

> ⚠️ Note: The API below is the planned design. GLM fitting is not yet implemented.

```python
import numpy as np
from aurora.models.glm import fit_glm, predict_glm

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = np.random.poisson(np.exp(X[:, 0] * 0.5))

# Fit a Poisson GLM with log link
result = fit_glm(X, y, family='poisson', link='log', backend='jax')

# Make predictions
X_new = np.random.randn(10, 3)
predictions = predict_glm(result, X_new)

# GAM with smooth terms (planned)
from aurora.models.gam import fit_gam

result = fit_gam(
    formula="y ~ s(x1, bs='tp') + s(x2, bs='cr')",
    data=df,
    family='gaussian',
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

The same code works with PyTorch tensors:

```python
import torch

# PyTorch tensors work transparently
y_torch = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
mu_torch = torch.tensor([1.5, 2.0, 2.8, 4.2, 5.1], dtype=torch.float32)

# Same API, different backend
log_lik_torch = poisson.log_likelihood(y_torch, mu_torch)
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
git clone https://github.com/yourusername/aurora-glm.git
cd aurora-glm

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
```

### Project Structure

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
├── models/               # GLM/GAM/GAMM (planned)
├── smoothing/            # Splines and penalties (planned)
├── inference/            # Hypothesis testing (planned)
├── estimation/           # REML/ML/Laplace (planned)
├── validation/           # Metrics and cross-validation (planned)
└── visualization/        # Plotting utilities (planned)
```

## Contributing

Contributions are welcome! This project is in early stages and there are many opportunities to help:

- **Core GLM implementation**: Help build the GLM fitting functions
- **Additional distributions**: Implement Inverse Gaussian, Negative Binomial, Beta, Tweedie
- **Additional link functions**: Implement Probit, Square root, Power links
- **Spline algorithms**: Implement various spline basis functions
- **Testing and validation**: Compare results against R and statsmodels
- **Documentation**: Write examples and tutorials

Please ensure:
- All code has type hints
- Tests cover both NumPy and PyTorch backends
- Docstrings follow NumPy/Google format
- Code follows the namespace pattern for multi-backend support

## Design Principles

### Array Namespace Pattern

Aurora-GLM uses a namespace abstraction pattern to support multiple array libraries transparently:

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

This allows distribution families and link functions to work seamlessly with NumPy arrays, PyTorch tensors, or JAX arrays without code changes.

### Extensibility

Users can create custom distributions and link functions:

```python
from aurora.distributions.base import Family, LinkFunction

class MyDistribution(Family):
    def log_likelihood(self, y, mu, **params):
        # Implementation using namespace pattern
        pass

    def deviance(self, y, mu, **params):
        pass

    def variance(self, mu, **params):
        pass

    def initialize(self, y):
        pass

    @property
    def default_link(self):
        return MyLink()

class MyLink(LinkFunction):
    def link(self, mu):
        pass

    def inverse(self, eta):
        pass

    def derivative(self, mu):
        pass
```

## Roadmap

**Q1 2025**: GLM implementation with IRLS fitting, basic inference, and diagnostics

**Q2 2025**: Spline basis functions, GAM fitting with penalization

**Q3 2025**: Random effects, GAMM implementation, REML estimation

**Q4 2025**: Comprehensive validation, documentation, performance optimization

## Performance Goals

- **GLM**: Competitive with statsmodels, ideally faster
- **GAM**: Within 2x of R's mgcv package
- **Scalability**: Handle 1M+ observations efficiently
- **GPU acceleration**: Efficient utilization when available

## License

*(Add license information here)*

## Citation

*(Add citation information when published)*

## Acknowledgments

Aurora-GLM draws inspiration from:
- R's **mgcv** package by Simon Wood
- Python's **statsmodels** library
- The JAX ecosystem for array programming

---

**Status**: 🚧 Under active development | **Version**: 0.1.0-dev | **Python**: 3.10+

*Illuminating complex data with modern generalized linear modeling tools.*
