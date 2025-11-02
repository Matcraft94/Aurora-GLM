# Aurora-GLM

Aurora-GLM is a modular, extensible, and high-performance framework under active
development for statistical modeling in Python. The project centers on
Generalized Linear Models (GLM) with planned support for Generalized Additive
Models (GAM) and Generalized Additive Mixed Models (GAMM).

## Project Identity

- **Package name**: `aurora-glm`
- **Python import**: `import aurora`
- **Repository**: `aurora-glm`
- **PyPI install**: `pip install aurora-glm` (planned)
- **Tagline**: *Illuminating complex data with modern generalized linear
  modeling tools.*

## Design Philosophy

- Prefer functional, modular composition instead of heavyweight OOP.
- Encourage extensibility: every major component is replaceable by the user.
- Embrace composability through small, independent building blocks.
- Default to immutable data structures wherever feasible.
- Use exhaustive Python type hints for clarity and documentation.

## Planned Technology Stack

### Numerical Backends

- **Primary**: JAX (auto-differentiation, JIT, GPU/TPU support).
- **Optional**: PyTorch (flexible backend selection via `backend="pytorch"`).
- Operations abstracted behind backend registry for seamless switching.

### Data Handling

- **Preferred**: Polars for fast data ingestion and manipulation.
- **Fallback**: Pandas (with PyArrow backend) for broad compatibility.
- Internal computation uses `jax.numpy.ndarray` (or backend equivalents).

### Performance Acceleration

- Selective use of Cython for critical loops or dense linear algebra.
- Optional C++ extensions for advanced spline algorithms.
- Benchmark suite to identify hotspots before low-level optimization.

### External Compatibility

- scikit-learn-compatible API surfaces for metrics and tooling.
- Interoperability with statsmodels for validation.
- Numerical parity checks with R's `mgcv` for GAM/GAMM components.

## Module Layout

```
aurora/
├── __init__.py
├── core/
│   ├── backends/
│   ├── autodiff/
│   ├── optimization/
│   └── linalg/
├── distributions/
│   ├── families/
│   ├── links/
│   └── custom/
├── models/
│   ├── glm/
│   ├── gam/
│   ├── gamm/
│   └── base/
├── smoothing/
│   ├── splines/
│   ├── penalties/
│   └── selection/
├── inference/
│   ├── hypothesis/
│   ├── anova/
│   ├── intervals/
│   └── diagnostics/
├── estimation/
│   ├── reml/
│   ├── ml/
│   └── laplace/
├── validation/
│   ├── metrics/
│   ├── cross_val/
│   └── sensitivity/
├── visualization/
│   ├── model_plots/
│   ├── residuals/
│   └── predictions/
├── io/
│   ├── readers/
│   ├── writers/
│   └── converters/
└── utils/
	 ├── exceptions/
	 └── validation/
```

## Core Components (Planned)

### Backends

```python
from aurora.core.backends import get_backend

backend = get_backend("jax")
x = backend.array([1, 2, 3])
grad_fn = backend.grad(loss_function)
```

- Unified interface for linear algebra, differentiation, and device transfers.
- Automatic backend registration; users can plug in custom backends.
- JIT compilation support where available.

### Optimization

Algorithms to implement:

- Newton-Raphson, IRLS, BFGS/L-BFGS, conjugate gradient, trust-region.
- Proximal gradient, Adam/RMSprop, Nelder-Mead.
- Configurable convergence criteria, warm starts, and rich callbacks.

### Distributions & Links

- Built-in families: Gaussian, Binomial, Poisson, Gamma, Inverse Gaussian,
  Negative Binomial, Beta, Tweedie, Exponential, Multinomial.
- Standard link functions: identity, log, logit, probit, inverse,
  square-root, power.
- Extensible protocol for user-defined distributions/link functions.

### Model APIs

- `fit_glm` for generalized linear models with weights, offsets, and missing
  data handling.
- `fit_gam` supporting varied spline bases, penalization, and formula parsing.
- `fit_gamm` with hierarchical random effects and covariance structures.
- Predictive helpers, summaries, and diagnostic hooks.

### Smoothing & Splines

- Basis types: cubic, natural cubic, B-splines, P-splines, thin plate, tensor
  product, adaptive, monotonic, cyclic.
- Selection methods: ridge, CV/GCV, REML, AIC/BIC, Mallows' Cp.

### Inference & Validation

- Hypothesis testing: Wald, likelihood-ratio, score, F, t, chi-squared,
  permutation tests.
- ANOVA (Type I/II/III), deviance analysis, smooth term tests.
- Metrics: deviance, AIC/BIC/EBIC, GCV, concordance, Brier, calibration.
- Cross-validation and sensitivity analysis utilities.

### Visualization

- Partial effects, smooth components, residual diagnostics, Q-Q plots,
  leverage/Cook's distance, ACF/PACF of residuals.
- Prediction intervals, contour/heatmap representations for interactions.
- Matplotlib/Seaborn-based plotting with theming hooks.

## Development Roadmap

1. **Core Numerical Foundations (Weeks 1-3)**
	- Backend abstraction (JAX/PyTorch) and autodiff helpers.
	- Initial optimization suite (Newton, IRLS, L-BFGS).
	- Linear algebra utilities, testing, and benchmarking harness.

2. **Foundational GLM Support (Weeks 4-7)**
	- Canonical families and link functions.
	- IRLS solver, inference basics, and evaluation metrics.

3. **GAM Functionality (Weeks 8-13)**
	- Spline implementations, penalization strategies, REML/GCV selection.
	- Formula parser and visualization of smooth terms.

4. **GAMM & Mixed Effects (Weeks 14-19)**
	- Random effect structures, REML/ML/Laplace estimation.
	- Covariance matrices (independent, compound symmetry, AR/ARMA, custom).

5. **Extensibility & Ecosystem (Weeks 20-23)**
	- Plugin systems for custom distributions, links, loss functions, optimizers.
	- Validation against R/Stata implementations.

## Quality & Testing Targets

- Unit, integration, property-based, and performance tests (pytest + Hypothesis).
- Coverage > 90% with continuous benchmarking.
- Numerical stability safeguards and informative error handling.

## Documentation & UX Goals

- Comprehensive docstrings (NumPy/Google style) with LaTeX formulas.
- Sphinx-based API reference and narrative guides.
- Example notebooks showcasing GLM, GAM, and GAMM usage.
- Clear error messages with actionable guidance.

## Branding Notes

- Color palette inspired by auroras: greens (#00FF87), blues (#00D4FF),
  accented with dark purples (#2D1B69) and light highlights (#B4FFA4).
- Consistent naming across code, documentation, and publications.

---

Aurora-GLM aims to provide a scientifically rigorous, performant, and pleasant
experience for practitioners and researchers working with generalized linear
modeling techniques.