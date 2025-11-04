# Changelog

All notable changes to Aurora-GLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

### Phase 2 Completion - Full GLM Implementation

This release marks the completion of Phase 2, delivering a production-ready Generalized Linear Model implementation with comprehensive inference, diagnostics, and validation.

### Added

#### Core Features
- **GLMResult.summary()** - R-style formatted summary tables with:
  - Coefficient estimates, standard errors, z-scores, and p-values
  - 95% confidence intervals
  - Significance codes (***,**,*,.)
  - Model fit statistics (AIC, BIC, deviance, pseudo R²)
  - Convergence status and iteration count

- **GLMResult.plot_diagnostics()** - Four standard diagnostic plots:
  - Residuals vs Fitted with smooth trend line
  - Normal Q-Q plot with reference line
  - Scale-Location plot
  - Residuals vs Leverage with Cook's distance heatmap
  - Optional scipy integration for smooth lines
  - Highlights influential points

#### Metrics & Validation
- **Concordance Index (C-index)** - Binary classification discriminative ability:
  - Equivalent to AUROC
  - Sample weight support
  - Handles tied predictions correctly

- **R Validation Infrastructure**:
  - `benchmarks/run_r_checks.R` - Validates against R's glm()
  - `benchmarks/compare_with_r.py` - Python wrapper for R comparison
  - Validates all families: Gaussian, Poisson, Binomial, Gamma
  - Results: max |Δcoef| ≈ 5e-05, excellent agreement

#### Documentation & Examples
- **Jupyter Notebooks**:
  - `examples/01_poisson_regression.ipynb` - Count data regression tutorial
  - `examples/02_logistic_regression.ipynb` - Binary classification with ROC curves
  - Both include visualizations, metrics, and best practices

- **Validation Utilities**:
  - `aurora.utils.validation.ensure_positive()` - Validate positive parameters
  - `aurora.utils.validation.ensure_non_empty()` - Validate non-empty sequences

#### Testing
- **Edge Case Tests** (`test_glm_edge_cases.py`):
  - Invalid family/link error handling
  - Singular design matrices
  - Zero max_iter behavior
  - Family-specific data requirements
  - 1D input reshaping
  - Models without intercept
  - Perfect separation handling
  - Diagnostics caching

- **Validation Utility Tests** (`test_utils/test_validation.py`):
  - Full coverage for ensure_positive() and ensure_non_empty()
  - Error message validation

### Changed
- **README.md** - Complete rewrite:
  - Phase 2 status updated: IN PROGRESS → COMPLETED ✅
  - Added "Quick Start" section with working code examples
  - Included example summary() output
  - Added usage examples for all major features
  - Updated test count: 86 → 119 tests
  - Updated coverage: 83% → 84%

- **Contexto_Proyecto_Objetivos_desarollo.md**:
  - Detailed backlog for remaining Phase 2 tasks
  - Accurate component completion status
  - Implementation timeline and specifications

### Fixed
- **GLMResult.summary()** - Fixed UnboundLocalError with quantile variable scope
- **concordance_index()** - Added sample_weight parameter support (was missing from initial implementation)

### Infrastructure
- **Coverage Measurement**:
  - Installed pytest-cov
  - Current coverage: 84% (366/2219 lines uncovered)
  - Target: 90% (deferred to future iteration)

- **R Environment Setup**:
  - Configured renv for R package management
  - Installed jsonlite for R-Python data exchange
  - Created .Renviron and .Rprofile for local R environment

### Validation Results

#### Against statsmodels (20 comparisons):
- max |Δcoef| ≈ 4.1e-06
- max delta_deviance ≈ 5.7e-09
- max mean |Δμ| ≈ 2.1e-05
- **All tests passing** ✅

#### Against R glm() (20 comparisons):
- max |Δcoef| ≈ 5.1e-05
- max delta_deviance ≈ 0.014
- mean |Δμ| ≈ 1.8e-04
- **All tests passing** ✅

### Statistics
- **Tests**: 119 passing, 1 skipped (up from 86 in development)
- **Coverage**: 84% (2219 statements, 366 uncovered)
- **Commits**: 9 major commits in this release
- **Files Changed**: 25+ files across codebase

### Breaking Changes
None - This is the first feature-complete release.

### Dependencies
- matplotlib >= 3.10.7 (for plot_diagnostics)
- scipy (optional, for smooth trend lines in diagnostics)
- R >= 4.3.0 (optional, for validation scripts)
- jsonlite R package (optional, for R validation)

### Migration Guide
No migration needed - first production release.

### Known Issues
- AIC values differ from R by constant offset (different formulation, mathematically equivalent)
- Coverage at 84%, short of 90% target (deferred to v0.2.1)
- Some edge case validations could be more stringent

### Contributors
- Lucy E. Arias ([@Matcraft94](https://github.com/Matcraft94))

---

## [0.1.0]

### Added
- Initial release with Phase 1 complete
- Backend abstraction layer (NumPy, PyTorch, JAX)
- Type system with comprehensive Protocols
- Optimization algorithms (Newton-Raphson, IRLS, L-BFGS)
- Distribution families (Gaussian, Poisson, Binomial, Gamma)
- Link functions (Identity, Log, Logit, Inverse, CLogLog)
- Basic `fit_glm()` implementation
- GLMResult with predictions and basic metrics
- 86 tests passing

---

## [0.3.0] - 2025-11-04

### Phase 3 Completion - Full GAM Implementation

This release marks the completion of Phase 3, delivering production-ready Generalized Additive Models with comprehensive smoothing functionality, formula parsing, and advanced visualization.

### Added

#### Spline Basis Functions (Phase 3.1)
- **B-spline basis** (`aurora/smoothing/bspline.py`):
  - Cox-de Boor recursive computation
  - Local support for computational efficiency
  - Partition of unity property
  - Multiple knot placement strategies (uniform, quantile)
  - 17 comprehensive unit tests

- **Natural cubic spline basis** (`aurora/smoothing/cubic.py`):
  - Truncated power basis with analytical penalties
  - Natural boundary conditions (f''=0 at boundaries)
  - Efficient penalty matrix computation
  - 16 comprehensive unit tests

#### Penalty Matrices (Phase 3.2)
- **Difference penalties** (`aurora/smoothing/difference.py`):
  - Second-order difference penalties (default)
  - Ridge penalties for regularization
  - Weighted penalty combinations
  - Block-diagonal penalty construction for multiple terms
  - 20 comprehensive unit tests

#### Smoothing Parameter Selection (Phase 3.3)
- **GCV selection** (`aurora/smoothing/gcv.py`):
  - Generalized Cross-Validation criterion
  - Automatic λ optimization via scipy.optimize
  - Golden section and Brent methods
  - Effective degrees of freedom tracking
  - 15 comprehensive unit tests

- **REML selection** (`aurora/smoothing/reml.py`):
  - Restricted Maximum Likelihood criterion
  - Better for multiple smooth terms than GCV
  - Log scale optimization for numerical stability
  - Per-term λ optimization
  - Multi-term simultaneous optimization
  - 20 comprehensive unit tests

#### GAM Fitting (Phase 3.4)
- **Univariate GAM** (`aurora/models/gam/fitting.py`):
  - Penalized least squares fitting
  - Automatic smoothing parameter selection
  - Support for observation weights
  - Comprehensive GAMResult with predictions and summaries
  - 20 comprehensive unit tests

- **Multivariate additive GAM** (`aurora/models/gam/additive.py`):
  - Multiple smooth terms: f(y) = β₀ + f₁(x₁) + f₂(x₂) + ...
  - Mixed smooth and parametric terms
  - Block-diagonal penalty structure
  - Per-term λ and EDF tracking
  - AdditiveGAMResult with comprehensive summaries
  - 15 comprehensive unit tests

#### Formula Parser (Phase 3.5)
- **R-style formula syntax** (`aurora/models/gam/formula.py`):
  - Parse formulas: `y ~ s(x1, bs='tp') + s(x2) + x3`
  - Support for smooth terms with basis specification
  - Parametric terms
  - Tensor product terms: `te(x1, x2)`
  - Comprehensive validation and error messages
  - 12 comprehensive unit tests

- **High-level API** (`aurora/models/gam/additive.py`):
  - `fit_gam_formula(formula, data, ...)` - Fit from R-style formula
  - Automatic term construction from parsed formula
  - Data frame or array support
  - 5 integration tests

#### Visualization (Phase 3.6)
- **Smooth term plotting** (`aurora/models/gam/plotting.py`):
  - `plot_smooth()` - Plot individual smooth terms
  - Confidence bands (Bayesian credible intervals)
  - Partial residuals overlay
  - Rug plots for data distribution
  - Customizable appearance (colors, styles, labels)
  - 8 comprehensive unit tests

- **Multi-term visualization**:
  - `plot_all_smooths()` - Grid of all smooth terms
  - Consistent styling across panels
  - Optional partial residuals per term
  - 10 comprehensive unit tests

#### Advanced Smoothing (Phase 3.7)
- **Tensor product smooths** (`aurora/smoothing/tensor.py`):
  - Multidimensional interactions: te(x1, x2)
  - Kronecker product basis construction
  - Dual penalty structure (separate λ per dimension)
  - Efficient computation via outer products
  - 13 comprehensive unit tests

- **Thin plate splines** (`aurora/smoothing/thinplate.py`):
  - Multidimensional smoothing without tensor structure
  - Radial basis functions varying by dimension:
    - d=1: η(r) = r³
    - d=2: η(r) = r²log(r)
    - d=3: η(r) = r
  - Polynomial null space (unpenalized)
  - Efficient knot selection (uniform, random)
  - 24 comprehensive unit tests

#### Term Specifications
- **SmoothTerm** dataclass - Specify univariate smooth terms
- **ParametricTerm** dataclass - Specify linear terms
- **TensorTerm** dataclass - Specify tensor product interactions
- Comprehensive validation with informative error messages
- 14 unit tests for term specifications

### Changed
- **README.md** - Major update:
  - Phase 3 status: IN PROGRESS → COMPLETED ✅
  - Added "GAM API" section with comprehensive examples
  - Univariate GAM usage examples
  - Multivariate additive GAM examples
  - Formula syntax examples
  - Visualization examples
  - Updated test count: 119 → 348 tests

- **Contexto_Proyecto_Objetivos_desarollo.md**:
  - Phase 3 status updated to 100% complete
  - Detailed component breakdown
  - Implementation statistics
  - Updated roadmap for Phase 4

- **CLAUDE.md**:
  - Updated Phase 3 status to complete
  - Added smoothing module documentation
  - Added formula parser documentation
  - Updated testing patterns for GAM

### Statistics
- **Tests**: 348 passing, 2 skipped (up from 119 in Phase 2)
- **New tests**: 229 new tests added for Phase 3
- **Lines of code**: ~3,600 new lines across 14 new modules
- **Commits**: 5 major commits in this phase
- **Files**: 25+ new files across smoothing, models/gam, and tests

### Validation Results

#### GAM Functional Tests
- Univariate GAM: R² > 0.90 on smooth test functions
- Multivariate GAM: Correct term recovery on additive data
- GCV selection: λ convergence within expected range
- REML selection: Improved MSE over GCV in multi-term models
- Tensor products: 20%+ R² improvement on interaction data
- Thin plate splines: Exact interpolation at λ=0

### Breaking Changes
None - All Phase 2 APIs remain unchanged.

### Dependencies
- scipy >= 1.10.0 (for optimization and special functions)
- matplotlib >= 3.10.7 (for visualization)
- All existing dependencies from Phase 2

### Known Issues
- TPS penalty matrices can be nearly singular with many knots (numerical issue)
- REML optimization can produce invalid values for extreme λ ranges (caught with warnings)
- Formula parser does not yet support all mgcv syntax (e.g., `by=` interactions)

### Contributors
- Lucy E. Arias ([@Matcraft94](https://github.com/Matcraft94))

---

## Roadmap

### [0.3.1] - Planned
- Additional basis types (P-splines, cyclic splines)
- Extended formula syntax (by= interactions, offset=)
- Performance optimizations for large datasets
- Extended documentation and tutorials

### [0.4.0] - Planned - Phase 4: GAMM
- Random effects (intercepts, slopes, crossed, nested)
- REML/ML/Laplace estimation for variance components
- Hierarchical multilevel models
- Covariance structures (AR, compound symmetry)

### [0.5.0] - Planned - Phase 5: Extended Features
- Additional distributions (Inverse Gaussian, Negative Binomial, Beta, Tweedie)
- Additional link functions (Probit, Square root, Power)
- Sparse matrix support for large-scale problems
- GPU acceleration via JAX backend

---

**Full Changelog**: https://github.com/Matcraft94/Aurora-GLM/compare/v0.2.0...v0.3.0
