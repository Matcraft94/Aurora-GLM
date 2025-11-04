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

## [0.4.0] - 2025-11-04

### Phase 4 Completion - GAMM with Random Effects

This release completes Generalized Additive Mixed Models (GAMM) with random effects for hierarchical and longitudinal data, including non-Gaussian families, formula parsing, and comprehensive visualization.

### Added

#### Random Effects Infrastructure (Milestone 1)
- **RandomEffect specification** (`aurora/models/gamm/random_effects.py`):
  - Random intercepts: `RandomEffect(grouping='subject')`
  - Random slopes: `RandomEffect(grouping='subject', variables=(1,))`
  - Include/exclude intercept control
  - Covariance structure specification (unstructured, diagonal, identity)
  - 11 comprehensive unit tests

- **Design matrix construction** (`aurora/models/gamm/design.py`):
  - Z matrix builder for random effects
  - Block-diagonal structure for multiple groups
  - Support for random intercepts and slopes
  - Indicator matrix construction
  - 12 comprehensive unit tests

#### REML Estimation (Milestone 2)
- **Variance component estimation** (`aurora/models/gamm/estimation.py`):
  - Restricted Maximum Likelihood (REML) for Ψ and σ²
  - Three covariance parameterizations:
    - **Unstructured**: Full covariance matrix via Cholesky (Ψ = LL')
    - **Diagonal**: Independent random effects
    - **Identity**: Equal variances, no correlation
  - Efficient V matrix computation (V = ZΨZ' + σ²I)
  - BLUPs (Best Linear Unbiased Predictors) for random effects
  - REMLResult with convergence diagnostics
  - 19 comprehensive unit tests

#### GAMM Fitting (Milestone 3)
- **Gaussian GAMM fitting** (`aurora/models/gamm/fitting.py`):
  - `fit_gamm_gaussian()` - Mixed model with smooth terms
  - Integrates GAM smoothing with random effects
  - Mixed model equations solver: [X'X + λS, X'Z; Z'X, Z'Z + Ψ⁻¹]
  - Effective degrees of freedom (EDF) accounting for penalties
  - GAMMResult dataclass with comprehensive results
  - Predictions: population-level and conditional
  - 16 comprehensive unit tests

- **Mixed model equations** (`aurora/models/gamm/fitting.py`):
  - Augmented system solver for β and random effects b
  - Cholesky decomposition with ridge regularization fallback
  - Block matrix inversion for efficiency
  - 16 unit tests

- **EDF computation**:
  - Accounts for both smoothing and random effect penalties
  - Correct AIC/BIC calculation for mixed models
  - Hat matrix trace via Cholesky solve

#### High-Level Interface
- **User-friendly API** (`aurora/models/gamm/interface.py`):
  - `fit_gamm()` - Simple interface with pandas support
  - `fit_gamm_with_smooth()` - Advanced interface with smooth terms
  - `predict_from_gamm()` - Population and conditional predictions
  - Automatic Z matrix construction from RandomEffect specs
  - Input validation and error messages
  - 14 integration tests

- **Pandas integration**:
  - Accept DataFrame for X, y, groups_data
  - Automatic Series/DataFrame conversion to numpy
  - Column name preservation where possible

#### Non-Gaussian Families (Milestone 4)
- **PQL estimation** (`aurora/models/gamm/pql.py`):
  - Penalized Quasi-Likelihood for GLMMs
  - Iterative weighted least squares with random effects
  - Support for Poisson, Binomial, Gaussian families
  - Working response and weights computation per family
  - Automatic variance component updates
  - PQLResult with convergence diagnostics
  - 14 comprehensive unit tests

- **Laplace approximation** (`aurora/models/gamm/laplace.py`):
  - More accurate alternative to PQL
  - Conditional mode optimization via L-BFGS-B
  - Hessian-based Laplace correction
  - Support for all GLM families
  - Smoothing penalty integration
  - LaplaceResult with detailed diagnostics
  - Full coverage via PQL tests (shared test structure)

#### Formula Parser Extensions (Milestone 5)
- **Random effects syntax** (`aurora/models/gam/formula.py`):
  - Extended parser for lme4-style random effects:
    - Random intercepts: `(1 | group)`
    - Random slopes: `(1 + x | group)`
    - Slope without intercept: `(x | group)`
    - Multiple slopes: `(1 + x + y | group)`
    - Crossed effects: `(1 | a) + (1 | b)`
    - Nested effects: `(1 | a/b)` → creates multiple REs
  - Intelligent parenthesis handling for complex formulas
  - Automatic variable mapping to column indices
  - 28 comprehensive unit tests

- **Enhanced fit_gamm() interface** (`aurora/models/gamm/interface.py`):
  - **Formula mode** (recommended): `fit_gamm(formula="y ~ x + (1 | subject)", data=df)`
  - **Matrix mode** (advanced): Original API unchanged for backward compatibility
  - Automatic design matrix construction from DataFrames
  - Variable name → column index mapping for random slopes
  - Comprehensive validation with informative errors
  - 17 integration tests

#### Visualization (Milestone 6)
- **Random effects plotting** (`aurora/models/gamm/plotting.py`):
  - `plot_caterpillar()` - Random effects with confidence intervals:
    - Sorted by magnitude
    - Reference line at zero
    - Customizable confidence levels
    - Support for multiple grouping variables

  - `plot_random_effects_qq()` - Normality check:
    - Q-Q plots for random effects
    - Standardized effects vs theoretical quantiles
    - Separate plots for intercepts/slopes

  - `plot_random_effects_density()` - Distribution visualization:
    - Histogram of random effects
    - Overlaid theoretical normal distribution
    - Visual fit assessment

  - `plot_diagnostics()` - Model diagnostics:
    - Residuals vs fitted values
    - Fitted vs observed values
    - Q-Q plot of residuals
    - Scale-location plot (homoscedasticity check)

  - `plot_random_effects_summary()` - Comprehensive 2x2 grid:
    - Caterpillar, Q-Q, density, and residuals plots
    - Single function for complete diagnostics

  - 31 comprehensive unit tests

#### Examples and Documentation
- **Comprehensive example** (`examples/gamm_example.py`):
  - Simulated longitudinal sleep study data
  - Random intercept model fitting
  - Random intercept + slope model fitting
  - Model comparison (AIC, BIC)
  - BLUPs extraction and interpretation
  - Population-level predictions
  - Conditional predictions
  - 4-panel diagnostic visualization

- **Documentation**:
  - Complete docstrings with examples
  - Design document in `aurora/models/gamm/DESIGN.md`
  - README updated with GAMM API examples
  - Formula syntax documentation with multiple examples

### Statistics
- **Tests**: 547+ passing, 2 skipped (up from 348 in Phase 3)
- **New tests**: 199 new GAMM tests added across all 6 milestones
- **Lines of code**: ~4,500 new lines across 10 new modules
- **Commits**: 6 major implementation commits
- **Files**: 18 new files across models/gamm, models/gam (formula extensions), and tests
- **Functions**: 35+ new exported functions

### Validation Results

#### GAMM Functional Tests
- Random intercept models: Variance component recovery within 20% of truth
- Random slope models: Covariance matrix recovery with <30% error
- Correlation estimation: Within 0.1 of true correlation for n_groups≥10
- Predictions: Population vs conditional differ as expected
- REML convergence: >95% convergence rate across test cases
- BLUPs: Shrinkage toward zero for small groups

#### Numerical Stability
- Cholesky parameterization ensures positive definiteness
- Ridge regularization (λ=1e-6) prevents singular matrices
- Log-scale optimization for REML avoids numerical issues

### Breaking Changes
None - All Phase 2 and Phase 3 APIs remain unchanged.

### Dependencies
- scipy >= 1.10.0 (for optimization)
- All existing dependencies from Phase 3

### Known Issues
- AR1 and compound symmetry covariance structures not yet implemented (planned for v0.4.1)
- Z matrix reconstruction in predictions could be more elegant
- PQL approximation can be inaccurate for sparse binary data (use Laplace instead)
- Smooth terms in formula mode not yet supported (use fit_gamm_with_smooth)

### Contributors
- Lucy E. Arias ([@Matcraft94](https://github.com/Matcraft94))

---

## Roadmap

### [0.4.0] - Complete ✅ - Phase 4: GAMM
**Status: 100% complete** (All 6 milestones done)
- ✅ Random effects (intercepts, slopes) for Gaussian family
- ✅ REML estimation for variance components
- ✅ Covariance structures (unstructured, diagonal, identity)
- ✅ Non-Gaussian families (Poisson, Binomial) with PQL/Laplace
- ✅ Formula parser integration: `(1 + x | group)` syntax
- ✅ Crossed and nested random effects
- ✅ Visualization (caterpillar plots, Q-Q plots, density, diagnostics)

### [0.4.1] - Planned
- Additional basis types (P-splines, cyclic splines)
- Extended formula syntax (by= interactions, offset=)
- Performance optimizations for large datasets
- Extended documentation and tutorials

### [0.5.0] - Planned - Phase 5: Extended Features
- Additional distributions (Inverse Gaussian, Negative Binomial, Beta, Tweedie)
- Additional link functions (Probit, Square root, Power)
- Sparse matrix support for large-scale problems
- GPU acceleration via JAX backend

---

**Full Changelog**: https://github.com/Matcraft94/Aurora-GLM/compare/v0.2.0...v0.3.0
