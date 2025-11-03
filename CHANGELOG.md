# Changelog

All notable changes to Aurora-GLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-02

### 🎉 Phase 2 Completion - Full GLM Implementation

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
- Claude Code (AI pair programmer)

---

## [0.1.0] - 2025-10-15

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

## Roadmap

### [0.2.1] - Planned
- Increase test coverage to ≥90%
- Additional edge case handling
- Performance optimizations
- Extended documentation

### [0.3.0] - Planned - Phase 3: GAM
- Spline basis functions
- Smoothing parameter selection
- R-style formula parser
- Visualization of smooth terms

### [0.4.0] - Planned - Phase 4: GAMM
- Random effects
- REML/ML/Laplace estimation
- Hierarchical models

---

**Full Changelog**: https://github.com/Matcraft94/Aurora-GLM/compare/v0.1.0...v0.2.0
