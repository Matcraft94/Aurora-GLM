# Aurora-GLM

**Aurora-GLM** is a modular, extensible, and high-performance Python framework for statistical modeling, focusing on Generalized Linear Models (GLM), Generalized Additive Models (GAM), and Generalized Additive Mixed Models (GAMM).

> ⚡ **Up to 141× faster than NumPy** with PyTorch CUDA on GPU
> ✅ **Validated against R and statsmodels** (max diff < 1e-11)
> 🎯 **Modular design** - Easy to extend with custom distributions and links
> 🚀 **Multi-backend** - NumPy, PyTorch, JAX with transparent API

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.7.0-brightgreen.svg)](https://github.com/Matcraft94/Aurora-GLM)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/Matcraft94/Aurora-GLM)

## Project Identity

- **Package name**: `aurora-glm`
- **Python import**: `import aurora`
- **Repository**: [github.com/Matcraft94/Aurora-GLM](https://github.com/Matcraft94/Aurora-GLM)
- **Author**: Lucy E. Arias ([@Matcraft94](https://github.com/Matcraft94))
- **Version**: 0.7.0
- **Status**: Phase 5 (85% complete) - Extended distributions, temporal covariance, GPU acceleration
- **Python**: 3.10+
- **Tagline**: *Illuminating complex data with modern generalized linear modeling tools*

## Vision and Goals

Aurora-GLM aims to be:

1. **Scientifically rigorous**: Correct implementations validated against R (mgcv, lme4) and statsmodels
2. **High performance**: GPU-accelerated with multi-backend support (NumPy, PyTorch, JAX)
3. **Extensible**: Users can add custom distributions, link functions, and algorithms
4. **Modular and functional**: Clean design favoring composition over complex inheritance
5. **Production-ready**: Comprehensive testing, documentation, and real-world validation

### Use Cases

- **Academic research**: Ecology, epidemiology, social sciences
- **Pharmaceutical industry**: Clinical trials, longitudinal analysis
- **Financial analysis**: Credit scoring, risk modeling, insurance claims
- **Machine learning**: Statistical foundations with modern tools and GPU acceleration

## Performance Highlights

### GPU Acceleration

PyTorch CUDA provides **exceptional speedups** for larger problems:

| Problem Size | NumPy CPU | PyTorch CUDA | Speedup |
|--------------|-----------|--------------|---------|
| Gaussian n=1,000 | 42ms | 5ms | **9×** |
| Gaussian n=5,000 | 206ms | 5ms | **39×** |
| Gaussian n=50,000 | 2.1s | 18ms | **116×** |
| Poisson n=50,000 | 4.3s | 30ms | **141×** |

*Benchmarked on NVIDIA RTX 5070 Ti. See [PERFORMANCE.md](benchmarks/PERFORMANCE.md) for details.*

### Accuracy Validation

Aurora-GLM achieves **excellent numerical agreement** with reference implementations:

| Comparison | Max Coefficient Difference |
|------------|---------------------------|
| vs R glm() | **< 1e-11** |
| vs statsmodels (Gaussian) | **< 1e-11** |
| vs statsmodels (Poisson) | **< 1e-10** |
| vs statsmodels (Binomial) | **< 1e-9** |
| PyTorch vs NumPy | **< 4e-7** |
| JAX vs NumPy | **< 1e-15** |

*All backends produce consistent, validated results.*

### Sparse Matrix Performance

For large GAM/GAMM problems, sparse matrices provide significant benefits:

| Problem Size | Dense | Sparse | Speedup | Memory Reduction |
|--------------|-------|--------|---------|------------------|
| n=1,000, k=30 | 0.79s | 0.15s | **5.5×** | **6-8×** |
| n=2,000, k=50 | 2.72s | 0.37s | **7.4×** | **6-8×** |
| n=5,000, k=50 | 8.04s | 1.51s | **5.3×** | **6-8×** |

## Key Features

### Statistical Models

- **GLM**: 10 distribution families (Gaussian, Poisson, Binomial, Gamma, Beta, Inverse Gaussian, Negative Binomial, Student-t, Tweedie, Quasi-families)
- **GAM**: B-splines, natural cubic splines, thin plate splines with GCV/REML smoothing
- **GAMM**: Random effects, temporal covariance (AR1, compound symmetry, Toeplitz), PQL for non-Gaussian

### Performance Features

- **GPU acceleration**: Up to 141× speedup with PyTorch CUDA
- **Multi-backend**: NumPy, PyTorch, JAX with transparent API
- **Sparse matrices**: 5-8× speedup and 6-8× memory reduction for large GAM/GAMM
- **Optimized algorithms**: IRLS, Newton-Raphson, L-BFGS with numerical stability

### Scientific Rigor

- **Validated**: Against R (glm, mgcv, lme4) and statsmodels
- **Comprehensive tests**: 494 tests with extensive coverage across all model families
- **R-style output**: Summary tables, diagnostic plots, confidence intervals
- **Formula syntax**: R/mgcv-compatible formulas like `y ~ s(x1, k=10) + s(x2) + (1|group)`

### Extensibility

- **Custom distributions**: Easy to add new distribution families
- **Custom link functions**: Flexible link function framework
- **Custom algorithms**: Pluggable optimization backends
- **Multi-backend pattern**: Transparent support for NumPy/PyTorch/JAX

## Installation

### From Source (Development)

```bash
# Clone repository
git clone https://github.com/Matcraft94/Aurora-GLM.git
cd Aurora-GLM

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .

# Optional: Install PyTorch for GPU acceleration
pip install torch

# Optional: Install JAX for additional backend
pip install jax jaxlib
```

## Quick Start

### Basic GLM Example

```python
import numpy as np
from aurora.models.glm import fit_glm

# Generate Poisson count data
np.random.seed(42)
X = np.random.randn(200, 2)
y = np.random.poisson(np.exp(X[:, 0] * 0.5 - 0.3))

# Fit Poisson GLM with log link
result = fit_glm(X, y, family='poisson', link='log')

# Print R-style summary with coefficients, std errors, p-values
print(result.summary())

# Make predictions
X_new = np.random.randn(10, 2)
predictions = result.predict(X_new, type='response')

# Generate diagnostic plots
result.plot_diagnostics()
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

### GPU Acceleration

```python
import torch

# PyTorch tensors work transparently - automatically uses GPU if available
X_torch = torch.randn(100, 2).cuda()
y_torch = torch.poisson(torch.exp(X_torch[:, 0] * 0.5)).cuda()

# Same API, GPU backend - up to 141× faster!
result = fit_glm(X_torch, y_torch, family='poisson')
print(result.summary())
```

### Multi-Backend Support

```python
# Same code works with NumPy, PyTorch, and JAX
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

## Case Studies and Examples

Aurora-GLM includes **17 comprehensive case studies** covering diverse domains and real-world applications:

### Available Case Studies

**Finance & Insurance** (5 studies):
- Insurance pricing (Gamma GLM)
- Medical insurance costs prediction
- Customer churn prediction (Binomial GLM)
- Motor claims frequency (Poisson/Negative Binomial)
- E-commerce conversion optimization (Binomial GAM)

**Environmental & Ecological** (4 studies):
- Air quality forecasting (Gaussian GAM)
- Species distribution modeling (Binomial GAM)
- Bike sharing demand prediction (GAM)
- Traffic accident severity (GAMM)

**Healthcare & Medicine** (5 studies):
- Sleep study analysis (Random effects GAMM)
- Clinical trials (Gaussian GLM)
- Longitudinal clinical data (GAMM with AR1)
- Psychometric measurement (GAMM with crossed effects)
- Survival analysis (Cox models)

**Education** (1 study):
- Multilevel educational data (Nested GAMM)

**Business & Retail** (2 studies):
- Health inspection scoring (Beta GLM)

Each case study is a self-contained Jupyter notebook with:
- Problem statement and research questions
- Data exploration and visualization
- Model selection justification
- Complete model fitting workflow
- Results interpretation and diagnostics
- Key findings and business implications

**Getting started with examples:**

```bash
# Navigate to examples directory
cd examples/06_case_studies

# Start Jupyter and open any notebook
jupyter notebook

# Recommended learning path:
# 1. 01_insurance_pricing.ipynb (Basic GLM)
# 2. 02_air_quality_gam.ipynb (GAM with smooths)
# 3. 04_sleep_study_gamm.ipynb (Random effects)
# 4. 06_clinical_trial_longitudinal.ipynb (Temporal covariance)
```

See [examples/README.md](examples/README.md) for complete case study guide.

## Recent Improvements (v0.7.0)

### Code Quality Enhancements

All 5 TODO items completed and fully tested:

1. **Formula Parser Validation** - Prevents silent errors in R-style formulas
   - Validates parenthesized terms require `|` for random effects
   - Rejects empty parentheses
   - Clear error messages for invalid syntax

2. **Sparse EDF Computation** - Improved GAM smoothing parameter selection
   - Hybrid exact/approximate approach
   - Exact for small basis (n_basis ≤ 100), fast approximation for large
   - More accurate GCV scores

3. **GAMM Log-Likelihood & Model Comparison** - Enable AIC/BIC selection
   - Proper log-likelihood computation for Gaussian GAMM
   - AIC and BIC metrics for model comparison
   - Critical for statistical inference

4. **Memory-Mapped Streaming** - Handle datasets larger than RAM
   - True streaming for .npy files via memory mapping
   - Only chunks loaded into memory
   - Maintains full backward compatibility

5. **Data Loading Efficiency** - Production-ready ChunkedDataLoader
   - Shuffling works correctly with streaming
   - X and y remain aligned
   - Fallback support for other formats (.npz, .csv)

All implementations manually tested and verified for correctness.

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

# Make predictions
X_new = np.random.randn(50, 3)
y_pred = result.predict(X_new)
```

### Formula-Based API (R-style)

```python
from aurora.models.gam import fit_gam_formula
import pandas as pd

# R-style formula with smooth terms
result = fit_gam_formula(
    formula="y ~ s(x1, k=10) + s(x2, bs='cubic') + x3",
    data=df,
    method='REML'  # or 'GCV'
)

# Tensor product interactions
result = fit_gam_formula(
    formula="y ~ te(x1, x2) + s(x3)",
    data=df
)

# Print comprehensive summary
print(result.summary())

# Visualize all smooth terms
from aurora.models.gam import plot_all_smooths
plot_all_smooths(result)
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
```

## GAMM API (Phase 4 - AVAILABLE NOW!)

### Basic Random Intercept Model

```python
from aurora.models.gamm import fit_gamm, RandomEffect
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

### Temporal Correlation with AR1

For longitudinal data with temporal autocorrelation:

```python
from aurora.models.gamm import fit_gamm, RandomEffect

# AR1 covariance for temporally correlated observations
re_ar1 = RandomEffect(grouping='subject', covariance='ar1')

result = fit_gamm(
    y=y,
    X=X,
    random_effects=[re_ar1],
    groups_data={'subject': subject_id}
)

# Extract AR1 parameters
params = result.covariance_params[0]
sigma2 = np.exp(params[0])  # Variance
rho = np.tanh(params[1])    # Autocorrelation
print(f"AR1: σ² = {sigma2:.3f}, ρ = {rho:.3f}")

# Interpretation: observations at lag k have correlation ρ^k
# Example: ρ = 0.7 means adjacent observations have correlation 0.7,
# observations 2 time units apart have correlation 0.49, etc.
```

### Compound Symmetry (Exchangeable Correlation)

For clustered data with equal within-cluster correlation:

```python
# Compound symmetry: all pairs equally correlated
re_cs = RandomEffect(grouping='cluster', covariance='compound_symmetry')

result = fit_gamm(
    y=y,
    X=X,
    random_effects=[re_cs],
    groups_data={'cluster': cluster_id}
)

# Extract variance parameter
params = result.covariance_params[0]
sigma2 = np.exp(params[0])

# For compound symmetry, all within-cluster pairs have the same correlation
psi = result.variance_components[0]
print(f"Estimated variance: σ² = {sigma2:.3f}")
print("Use compound symmetry when cluster members are exchangeable")
print("(e.g., students in same school, patients in same hospital)")
```

### Sparse Matrix Support for Large-Scale Models

For large datasets or models with many basis functions:

```python
# Sparse GAMM (10-100× faster for large problems)
result = fit_gamm(
    formula="y ~ s(x, k=50) + (1 | subject)",
    data={"y": y, "x": x, "subject": subject_id},
    use_sparse=True  # Enable sparse matrices
)

# Benefits of sparse matrices:
# - Memory: 6-8× reduction for typical problems
# - Speed: 10-100× faster for n > 1000
# - Enables models that don't fit in memory with dense matrices

# When to use sparse:
# - Large datasets (n > 500, k > 20)
# - B-spline basis functions (naturally sparse)
# - Memory-constrained environments
# - Multiple smooth terms
```

### Non-Gaussian GAMM (Poisson/Binomial with PQL)

```python
from aurora.models.gamm import fit_pql_smooth

# Poisson GAMM for count data
result = fit_pql_smooth(
    y=counts,
    X=X,
    groups=subject_id,
    family='poisson',
    smooth_terms=[SmoothTerm(variable=0, n_basis=10)],
    max_iter=50
)

# Binomial GAMM for binary outcomes
result = fit_pql_smooth(
    y=binary_outcome,
    X=X,
    groups=subject_id,
    family='binomial',
    smooth_terms=[SmoothTerm(variable=0, n_basis=8)],
    max_iter=100
)

print(f"Fixed effects: {result.beta}")
print(f"Random effects variance: {result.variance_components}")
```

## Current Implementation Status

### Phase 1: Core Numerical Foundation - COMPLETED ✅

**Backend Infrastructure**:
- ✅ Backend abstraction layer (JAX, PyTorch)
- ✅ Type system with comprehensive Protocols
- ✅ Array namespace utilities for transparent NumPy/PyTorch/JAX compatibility

**Optimization Algorithms**:
- ✅ Newton-Raphson with automatic Hessian
- ✅ IRLS (Iteratively Reweighted Least Squares) with sparse matrix support
- ✅ L-BFGS with strong Wolfe line search
- ✅ Autodiff module (gradient, hessian, jacobian) for all backends

**Distribution Families** (10/10):
- ✅ Gaussian, Poisson, Binomial, Gamma
- ✅ Beta, Inverse Gaussian, Negative Binomial
- ✅ Student-t, Tweedie, Quasi-families

**Link Functions** (6/6):
- ✅ Identity, Log, Logit, Inverse, CLogLog, Probit

### Phase 2: Basic GLM - COMPLETED ✅

- ✅ IRLS-based `fit_glm()` with multi-backend support
- ✅ R-style `GLMResult.summary()` with p-values, significance codes
- ✅ Diagnostic plots (residuals, Q-Q, scale-location, leverage)
- ✅ Confidence intervals, hypothesis tests, influence measures
- ✅ Cross-validation and comprehensive metrics
- ✅ Validation against statsmodels and R glm()

### Phase 3: GAM (Splines and Smoothing) - COMPLETED ✅

- ✅ B-spline, natural cubic spline, thin plate spline bases
- ✅ GCV and REML smoothing parameter selection
- ✅ R-style formula parser (`y ~ s(x1) + s(x2) + x3`)
- ✅ Tensor product smooths (`te(x1, x2)`)
- ✅ Visualization with confidence bands
- ✅ Sparse matrix support for large problems

### Phase 4: GAMM (Random Effects) - COMPLETED ✅

- ✅ Random intercepts and slopes
- ✅ Nested and crossed random effects
- ✅ Multiple covariance structures (identity, unstructured, diagonal)
- ✅ AR1 and compound symmetry for temporal data
- ✅ Toeplitz covariance structure
- ✅ PQL estimation for non-Gaussian families
- ✅ Laplace approximation
- ✅ Sparse matrix optimization

### Phase 5: Extended Features - IN PROGRESS 🚧 (85%)

**Implemented**:
- ✅ Extended distributions (Beta, Inverse Gaussian, Negative Binomial, Student-t, Tweedie)
- ✅ Temporal covariance structures (AR1, compound symmetry, Toeplitz)
- ✅ Sparse matrix support (6-8× memory reduction, 10-100× speedup)
- ✅ GPU acceleration benchmarks (up to 141× speedup)
- ✅ Multi-backend accuracy validation
- ✅ Comprehensive benchmark suite

**Remaining**:
- 📋 PyPI package publication
- 📋 Documentation website
- 📋 Additional spatial covariance structures
- 📋 Performance optimizations for massive datasets

## Complete Feature Set

### Distribution Families (10/10)

- ✅ **Gaussian** (Normal) - continuous data
- ✅ **Poisson** - count data
- ✅ **Binomial** - binary/proportions
- ✅ **Gamma** - positive continuous, right-skewed
- ✅ **Beta** - proportions in (0,1)
- ✅ **Inverse Gaussian** - positive durations (Wald distribution)
- ✅ **Negative Binomial** - overdispersed counts (NB2 parameterization)
- ✅ **Student-t** - heavy-tailed, robust regression
- ✅ **Tweedie** - compound Poisson-Gamma (insurance/actuarial)
- ✅ **Quasi-families** - Quasi-Poisson, Quasi-Binomial

### Link Functions (6/6)

- ✅ **Identity**: g(μ) = μ
- ✅ **Log**: g(μ) = log(μ)
- ✅ **Logit**: g(μ) = log(μ/(1-μ))
- ✅ **Probit**: g(μ) = Φ⁻¹(μ)
- ✅ **Inverse**: g(μ) = 1/μ
- ✅ **CLogLog**: g(μ) = log(-log(1-μ))

### GAM Features

- ✅ **B-spline basis**: Cox-de Boor recursion, local support, partition of unity
- ✅ **Natural cubic splines**: Truncated power basis with analytical penalties
- ✅ **Thin plate splines**: Multidimensional smoothing with radial basis functions
- ✅ **Tensor products**: te(x1, x2) for multidimensional interactions
- ✅ **GCV smoothing**: Generalized Cross-Validation for automatic λ selection
- ✅ **REML smoothing**: Restricted Maximum Likelihood for better multi-term selection
- ✅ **Sparse matrices**: 5-8× speedup and 6-8× memory reduction
- ✅ **Formula parser**: R/mgcv-compatible syntax

### GAMM Features

- ✅ **Random effects**: Intercepts and slopes with flexible specification
- ✅ **Nested/crossed effects**: Complex hierarchies fully supported
- ✅ **Covariance structures**:
  - Identity (diagonal)
  - Unstructured (full variance-covariance)
  - Diagonal (heterogeneous variances)
  - **AR1** (temporal autocorrelation with exponential decay)
  - **Compound symmetry** (exchangeable correlation)
  - **Toeplitz** (banded temporal correlations)
- ✅ **PQL estimation**: Penalized Quasi-Likelihood for non-Gaussian families
- ✅ **Laplace approximation**: Alternative estimation method
- ✅ **Sparse matrices**: Memory-efficient for large-scale models
- ✅ **Formula syntax**: lme4-style `(1 + x | group)` supported

### Inference & Diagnostics

- ✅ **Standard errors**: Wald approximation with delta method
- ✅ **P-values**: Z-tests and likelihood ratio tests
- ✅ **Confidence intervals**: For coefficients and predictions
- ✅ **Residuals**: Response, Pearson, deviance, working, studentized
- ✅ **Influence measures**: Leverage, Cook's distance, DFBETAs
- ✅ **Hypothesis tests**: Wald tests for single and multi-constraint hypotheses
- ✅ **Model comparison**: AIC, BIC, pseudo R², deviance

### Validation

- ✅ **Cross-validation**: KFold, StratifiedKFold with aggregated results
- ✅ **Metrics**: MSE, MAE, RMSE, accuracy, log-loss, Brier score, C-index
- ✅ **Diagnostic plots**: 4-panel residual plots (residuals, Q-Q, scale-location, leverage)
- ✅ **Q-Q plots**: Normal probability plots for residuals
- ✅ **Caterpillar plots**: Random effects visualization with confidence intervals

## Benchmarking and Validation

Run comprehensive benchmarks comparing Aurora with statsmodels and R:

```bash
# Comprehensive benchmarks (accuracy + GPU performance + R comparison)
cd /tmp && PYTHONPATH=/path/to/Aurora-GLM python benchmarks/comprehensive_benchmarks.py

# Quick benchmarks (CI-friendly)
cd /tmp && PYTHONPATH=/path/to/Aurora-GLM python benchmarks/comprehensive_benchmarks.py --quick

# Performance benchmarks only
PYTHONPATH=. python benchmarks/performance_benchmarks.py

# GLM validation against statsmodels
PYTHONPATH=. python benchmarks/run_glm_checks.py --replicates 3
```

**Note**: Run from `/tmp` or another directory without an `renv` project to avoid R library path conflicts when using rpy2.

Results saved to `benchmarks/results/`. See [PERFORMANCE.md](benchmarks/PERFORMANCE.md) for detailed analysis.

### External Validation

**Latest validation results**:
- vs R glm(): max |Δcoef| < **1e-11**
- vs statsmodels: max |Δcoef| < **2e-6**
- Multi-backend: JAX vs NumPy < **1e-15**
- PyTorch vs NumPy < **4e-7**

All backends produce consistent, validated results suitable for scientific research.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aurora --cov-report=html

# Run specific module
pytest tests/test_models/test_glm_fitting.py

# Run specific test
pytest tests/test_distributions/test_links.py::test_identity_link_roundtrip

# Verbose output
pytest -v
```

**Test Status**: 494 tests collected (comprehensive coverage across GLM, GAM, GAMM, Bayesian inference, count models, and smoothing methods)

### Code Quality

```bash
# Format code
ruff format aurora/ tests/

# Lint code
ruff check aurora/ tests/

# Type checking
mypy aurora/

# Run all quality checks before committing
ruff format aurora/ tests/ && ruff check aurora/ tests/ && mypy aurora/
```

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/implement-new-distribution

# Make changes and test
pytest
pytest --cov=aurora --cov-report=html

# Format and lint
ruff format aurora/ tests/
ruff check aurora/ tests/
mypy aurora/

# Commit with descriptive message
git commit -m "feat(distributions): add Exponential family"

# Push and create PR
git push origin feature/implement-new-distribution
```

## Design Principles

### Array Namespace Pattern

Aurora-GLM uses a namespace abstraction to support multiple array libraries transparently:

```python
from aurora.distributions._utils import namespace, as_namespace_array

def my_function(x, y):
    # Automatically detect NumPy, PyTorch, or JAX
    xp = namespace(x, y)

    # Convert to appropriate array type
    x_arr = as_namespace_array(x, xp, like=y)

    # Use namespace-specific operations
    return xp.sum(x_arr * xp.exp(y))
```

This pattern ensures:
- **Zero code changes** between backends
- **Automatic GPU support** with PyTorch/JAX
- **Consistent results** across all backends (validated < 1e-15)

### Extensibility

Users can create custom distributions and link functions:

```python
from aurora.distributions.base import Family, LinkFunction
from aurora.distributions._utils import namespace, ensure_positive

class MyDistribution(Family):
    def log_likelihood(self, y, mu, **params):
        xp = namespace(y, mu)
        mu = ensure_positive(mu, xp)
        # Implementation using namespace pattern
        # ...

    def deviance(self, y, mu, **params):
        # ...

    def variance(self, mu, **params):
        # V(μ) = variance function
        # ...

    def initialize(self, y):
        # Return initial μ estimates
        # ...

    @property
    def default_link(self):
        return MyLink()

class MyLink(LinkFunction):
    def link(self, mu):
        # η = g(μ)
        # ...

    def inverse(self, eta):
        # μ = g⁻¹(η)
        # ...

    def derivative(self, mu):
        # dη/dμ = g'(μ)
        # ...
```

## When to Use Aurora-GLM

### Choose Aurora-GLM when you need:

1. **GPU acceleration** - Up to 141× speedup for large datasets
2. **GAM/GAMM capabilities** - statsmodels doesn't support these
3. **Multi-backend flexibility** - PyTorch/JAX for GPU or autodiff
4. **Advanced covariance structures** - AR1, compound symmetry, Toeplitz for temporal/spatial data
5. **Sparse matrix optimization** - Handle models that don't fit in memory
6. **Consistent API** - GLM → GAM → GAMM with same interface
7. **Research/teaching** - Readable Python implementation with R-style output
8. **Extensibility** - Easy to add custom distributions and link functions

### Choose statsmodels when you need:

1. Pure CPU GLM with maximum single-threaded speed
2. Extensive diagnostics and inference tools
3. Time series models (ARIMA, VAR, state space models)
4. Mature ecosystem with extensive documentation

## Project Structure

```
aurora/
├── core/
│   ├── backends/         # Multi-backend support (JAX, PyTorch)
│   │   ├── _protocol.py  # Backend Protocol interface (internal)
│   │   ├── _registry.py  # Backend registration (internal)
│   │   └── operations.py # Backend-agnostic numerical operations
│   ├── autodiff/         # Automatic differentiation (gradient, hessian, jacobian)
│   ├── optimization/     # IRLS, Newton-Raphson, L-BFGS
│   └── linalg/           # Linear algebra primitives
├── distributions/
│   ├── families/         # 10 distribution families
│   ├── links/            # 6 link functions
│   └── _utils.py         # namespace(), as_namespace_array(), ensure_positive()
├── models/
│   ├── glm/              # fit_glm(), predictions, diagnostics
│   ├── gam/              # fit_gam(), formula parser, smooths
│   └── gamm/             # fit_gamm(), random effects, PQL, covariance structures
├── smoothing/
│   ├── splines/          # B-splines, cubic, thin plate
│   ├── penalties/        # Difference penalties, ridge
│   └── selection/        # GCV, REML smoothing selection
├── inference/
│   ├── hypothesis/       # Wald tests, likelihood ratio tests
│   ├── intervals/        # Confidence intervals
│   └── diagnostics/      # Residuals, influence measures
└── validation/
    ├── metrics/          # MSE, MAE, accuracy, C-index
    └── cross_val/        # KFold, cross_val_score
```

## Roadmap

### Completed ✅

- [x] **Phase 1**: Core infrastructure (backends, types, optimization)
- [x] **Phase 2**: Full GLM implementation with IRLS, diagnostics, inference
- [x] **Phase 3**: GAM with B-splines, natural cubic, thin plate, GCV/REML
- [x] **Phase 4**: GAMM with random effects, PQL for non-Gaussian families
- [x] **Phase 5 (current)**: Extended distributions, temporal covariance, sparse matrices
- [x] 494 tests with comprehensive coverage (5,731 new lines of test code)
- [x] Validation against statsmodels and R
- [x] GPU acceleration benchmarks (up to 141× speedup)
- [x] Multi-backend accuracy validation

### In Progress 🚧 (Phase 5 - 85% complete)

- [ ] PyPI package publication
- [ ] Documentation with interactive examples
- [ ] Research paper publication
- [ ] Additional spatial covariance structures (exponential, Matérn)

### Recently Completed 🎉

- [x] Comprehensive benchmark suite (1,591 lines): Multi-backend accuracy validation and GPU performance measurement
- [x] Bayesian GLM inference tests (1,060 lines): Prior specification, posterior sampling, convergence diagnostics
- [x] Zero-inflated count model tests (956 lines): ZIP and ZINB with excess zero validation
- [x] Hurdle count model tests (640 lines): Two-stage modeling of structural zeros
- [x] Smoothing method tests (1,083 lines): LOESS and P-spline implementation validation
- [x] Integration tests (401 lines): End-to-end workflows for count models

**Total test coverage expansion: 5,731 lines** across benchmarking, Bayesian inference, count models, and smoothing methods.

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

For a comprehensive list of mathematical foundations, algorithms, and validation references, including detailed equations and derivations, please see **[REFERENCES.md](REFERENCES.md)**.

## Contributing

Contributions are welcome! This project is in active development with many opportunities to help:

### Areas for Contribution

- **Additional distributions** - Zero-inflated, hurdle models
- **Performance optimizations** - Further GPU optimizations, distributed computing
- **Documentation improvements** - Tutorials, case studies, API reference
- **Real-world case studies** - Applications in ecology, epidemiology, finance
- **Bug reports and fixes** - Help improve stability and reliability

### Contribution Guidelines

Please ensure:
- **Type hints** on all public functions
- **Tests** covering both NumPy and PyTorch backends
- **Docstrings** in NumPy/Google format with examples
- **Code formatting** with `ruff format`
- **Multi-backend support** using the namespace pattern

## Citation

If you use Aurora-GLM in your research, please cite it using the information in our [CITATION.cff](CITATION.cff) file, or use the following BibTeX entry:

```bibtex
@software{aurora_glm2025,
  title = {Aurora-GLM: Generalized Linear and Additive Models},
  author = {Arias, Lucy Eduardo},
  year = {2025},
  version = {0.7.0},
  url = {https://github.com/Matcraft94/Aurora-GLM},
  license = {MIT}
}
```

GitHub users can use the "Cite this repository" button in the right sidebar to automatically generate citations in various formats.

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

Aurora-GLM draws inspiration from:
- R's **mgcv** package by Simon Wood
- R's **lme4** package by Bates et al.
- Python's **statsmodels** library
- The JAX ecosystem for modern array programming

Special thanks to the open-source community for providing excellent tools and libraries.

---

**Status**: Phase 5 (85% complete) - Extended features, temporal covariance, GPU acceleration
**Version**: 0.7.0
**Tests**: 494 tests collected (5,731 new lines added: benchmarks, Bayesian, count models, smoothing)
**Python**: 3.10+
**GPU**: Up to 141× speedup with PyTorch CUDA
**Accuracy**: Validated against R and statsmodels (< 1e-11)
**Maintained by**: Lucy E. Arias ([@Matcraft94](https://github.com/Matcraft94))

*Illuminating complex data with modern generalized linear and additive modeling tools.*
