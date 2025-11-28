# Aurora-GLM Improvement Plan

**Version**: 0.5.0-dev → 0.6.0  
**Created**: 2024-12-19  
**Status**: In Progress

---

## Executive Summary

This plan outlines a comprehensive improvement initiative for Aurora-GLM, focusing on three main areas in order of priority:

1. **Usability Improvements (B)** - Enhanced API, unified Results, convenience helpers
2. **Feature Completion (C)** - Implement empty modules (IO, ANOVA, Sensitivity, Linalg)
3. **Documentation (D)** - Mathematical documentation with equations and references

---

## Decision Summary

| Category | Decision |
|----------|----------|
| **Priority Order** | B → C → D (API → Features → Docs) |
| **Result Classes** | Unified inheritance hierarchy |
| **Convenience Helpers** | `summary()`, `plot()`, `compare()` |
| **Input Validation** | Automatic decorators on fit_* functions |
| **IO Module** | Full (csv, dataframe, save/load, R compatibility) |
| **ANOVA Module** | Full (glm, gam, Type I/II/III) |
| **Sensitivity Module** | Full (influence, case deletion, perturbation) |
| **Linalg Module** | Full (safe_solve, safe_cholesky, woodbury) |
| **Doc Style** | Keep current mixed style |
| **Math Documentation** | Complete with equations and references |
| **Examples** | Only in main functions (fit_glm, fit_gam, fit_gamm) |
| **Testing** | High coverage for all new functionality |
| **Backward Compatibility** | Breaking changes allowed for improvements |
| **Commits** | Small and frequent |

---

## Phase B: Usability Improvements

### B1: Unified Result Hierarchy

**Objective**: Create a consistent inheritance hierarchy for all model results.

```
BaseResult (ABC)
├── coefficients, fitted_values, converged, diagnostics
├── summary() → dict
├── predict(X) → array
│
├── LinearModelResult
│   ├── coef_, intercept_, residuals_
│   │
│   └── GLMResult
│       ├── family, link, deviance_, aic_, bic_
│       ├── std_errors_, p_values_, coef_cov_
│       │
│       ├── GAMResult
│       │   ├── smooth_terms, edf_, gcv_score_
│       │   └── plot_smooth(), get_term_contribution()
│       │
│       └── MixedModelResult
│           ├── fixed_effects_, random_effects_
│           ├── variance_components_, log_likelihood_
│           ├── GAMMResult
│           ├── PQLResult
│           └── LaplaceResult
```

**Files to modify**:
- `aurora/models/base/result.py` - Create BaseResult, update GLMResult
- `aurora/models/gam/result.py` - Update GAMResult to inherit from GLMResult
- `aurora/models/gamm/fitting.py` - Update GAMMResult
- `aurora/models/gamm/pql.py` - Update PQLResult
- `aurora/models/gamm/laplace.py` - Update LaplaceResult

**Breaking Changes**:
- Some attribute names may change for consistency
- Result classes will have new base class

---

### B2: Universal Summary Function

**Objective**: Single `aurora.summary(result)` that works with any model type.

```python
from aurora import summary

# Works with any result type
summary(glm_result)   # → GLM summary table
summary(gam_result)   # → GAM summary with smooth terms
summary(gamm_result)  # → GAMM summary with random effects
```

**Features**:
- Coefficient table with std errors, z-values, p-values
- Model fit statistics (deviance, AIC, BIC, R²)
- For GAM: effective degrees of freedom per smooth term
- For GAMM: variance components, random effects summary

---

### B3: Universal Plot Function

**Objective**: Single `aurora.plot(result)` with automatic plot type selection.

```python
from aurora import plot

plot(glm_result)              # → Diagnostic plots (residuals, Q-Q)
plot(gam_result)              # → Smooth term plots
plot(gamm_result)             # → Random effects + diagnostics
plot(result, kind="residuals") # → Specific plot type
```

**Plot Types**:
- `"diagnostics"` - 2x2 diagnostic panel (default for GLM)
- `"smooth"` - Smooth terms with CI (default for GAM)
- `"random_effects"` - Caterpillar plots (default for GAMM)
- `"residuals"` - Residual vs fitted
- `"qq"` - Q-Q plot of residuals

---

### B4: Model Comparison Function

**Objective**: Easy comparison of nested models.

```python
from aurora import compare

comparison = compare(model1, model2, model3)
# Returns comparison table with:
# - Deviance, AIC, BIC for each model
# - Likelihood ratio tests for nested models
# - Delta AIC/BIC
```

---

### B5: Input Validation Decorators

**Objective**: Automatic input validation with clear error messages.

```python
@validate_inputs(
    X="array_2d",
    y="array_1d", 
    family="family_or_str",
    weights="positive_array_or_none"
)
def fit_glm(X, y, family="gaussian", weights=None, ...):
    ...
```

**Validation Types**:
- `array_1d`, `array_2d` - Shape validation
- `positive_array` - All values > 0
- `probability_array` - All values in [0, 1]
- `family_or_str` - Family instance or valid string
- `link_or_none` - LinkFunction or None

---

## Phase C: Feature Completion

### C1: IO Module (`aurora/io/`)

**Files to create**:
- `aurora/io/readers.py` - Data loading utilities
- `aurora/io/writers.py` - Export utilities  
- `aurora/io/serialization.py` - Model save/load
- `aurora/io/converters.py` - Format conversions

**Functions**:

```python
# Readers
read_csv(path, y_col=None, X_cols=None, **kwargs) → dict
read_stata(path, ...) → dict
read_r_data(path, ...) → dict

# Writers
to_dataframe(result) → pd.DataFrame
to_latex(result, caption=None) → str
to_r_formula(terms) → str

# Serialization
save_model(result, path, format="pickle")
load_model(path) → Result

# Converters
from_statsmodels(sm_result) → GLMResult
from_sklearn(sk_model) → Result
to_statsmodels(result) → sm.GLMResults
```

---

### C2: ANOVA Module (`aurora/inference/anova/`)

**Files to create**:
- `aurora/inference/anova/glm.py` - GLM ANOVA
- `aurora/inference/anova/gam.py` - GAM smooth term tests
- `aurora/inference/anova/tables.py` - ANOVA table formatting

**Functions**:

```python
# GLM ANOVA
anova_glm(*models, test="LRT") → ANOVAResult
# Compare nested GLM models using likelihood ratio test

# GAM ANOVA  
anova_gam(result, test="F") → ANOVAResult
# Test significance of each smooth term

# Type specification
anova_glm(model, type="III")  # Type III SS

# ANOVA Result contains:
# - df, deviance, F/chi2, p-value per term
# - Residual deviance and df
```

**Mathematical Framework**:
- Likelihood Ratio Test: $\Lambda = -2(\ell_0 - \ell_1) \sim \chi^2_{df}$
- F-test for smooth terms using EDF
- Type I (sequential), II (marginal), III (partial) SS

---

### C3: Sensitivity Module (`aurora/validation/sensitivity/`)

**Files to create**:
- `aurora/validation/sensitivity/influence.py` - Influence measures
- `aurora/validation/sensitivity/deletion.py` - Case deletion diagnostics
- `aurora/validation/sensitivity/perturbation.py` - Perturbation analysis

**Functions**:

```python
# Influence measures
influence_measures(result) → InfluenceResult
# Returns: leverage, Cook's D, DFBETAS, DFFITS, COVRATIO

# Case deletion
case_deletion_diagnostics(result) → CaseDeletionResult
# Leave-one-out influence on coefficients and fitted values

# Perturbation
perturbation_analysis(result, perturbation="response", n_sim=100)
# Sensitivity to small perturbations in data
```

**Mathematical Details**:
- Hat matrix: $H = X(X'WX)^{-1}X'W$
- Cook's Distance: $D_i = \frac{r_i^2}{p \cdot \hat{\phi}} \cdot \frac{h_{ii}}{(1-h_{ii})^2}$
- DFBETAS: $\text{DFBETAS}_{j,i} = \frac{\hat{\beta}_j - \hat{\beta}_{j(-i)}}{SE(\hat{\beta}_j)}$

---

### C4: Linalg Module (`aurora/core/linalg/`)

**Files to create**:
- `aurora/core/linalg/solve.py` - Robust linear solvers
- `aurora/core/linalg/decomposition.py` - Matrix decompositions
- `aurora/core/linalg/updates.py` - Efficient matrix updates

**Functions**:

```python
# Safe solving with fallbacks
safe_solve(A, b, method="auto") → x
# Tries: Cholesky → LU → QR → SVD with regularization

# Robust Cholesky
safe_cholesky(A, reg=1e-10) → L
# Adds regularization if not positive definite

# Woodbury identity for efficient updates
woodbury_update(A_inv, U, C, V) → (A + UCV')^{-1}
# (A + UCV')^{-1} = A^{-1} - A^{-1}U(C^{-1} + V'A^{-1}U)^{-1}V'A^{-1}

# Efficient rank-1 updates
rank1_update(L, x, sign=1) → L'
# Cholesky update: L'L' = LL' ± xx'
```

---

## Phase D: Documentation

### D1: Mathematical Docstrings

**Objective**: Add complete mathematical documentation to core functions.

**Template**:
```python
def fit_glm(X, y, family="gaussian", link=None, ...):
    """Fit a Generalized Linear Model using IRLS.
    
    Mathematical Framework
    ----------------------
    GLMs model the relationship between response Y and predictors X:
    
    .. math::
        g(\\mu_i) = \\eta_i = X_i^T \\beta
    
    where g(·) is the link function and μ = E[Y].
    
    The IRLS algorithm iteratively solves:
    
    .. math::
        \\beta^{(t+1)} = (X^T W^{(t)} X)^{-1} X^T W^{(t)} z^{(t)}
    
    Parameters
    ----------
    ...
    
    References
    ----------
    .. [1] McCullagh & Nelder (1989). Generalized Linear Models, 2nd ed.
    .. [2] Nelder & Wedderburn (1972). JRSS-A, 135(3), 370-384.
    """
```

**Functions to document**:
- `fit_glm` - IRLS, exponential family, deviance
- `fit_gam` - Penalized regression splines, GCV/REML
- `fit_gamm` - Mixed model equations, REML estimation
- All Family classes - Variance functions, deviance formulas

---

### D2: Executable Examples

**Objective**: Add runnable examples to main functions.

```python
def fit_glm(X, y, family="gaussian", ...):
    """...
    
    Examples
    --------
    Fit a simple linear regression:
    
    >>> import numpy as np
    >>> from aurora import fit_glm
    >>> X = np.random.randn(100, 2)
    >>> y = X @ [1.5, -0.5] + np.random.randn(100) * 0.1
    >>> result = fit_glm(X, y)
    >>> result.converged_
    True
    >>> np.allclose(result.coef_, [1.5, -0.5], atol=0.1)
    True
    
    Fit a Poisson regression for count data:
    
    >>> counts = np.random.poisson(np.exp(X @ [0.5, 0.3]))
    >>> result = fit_glm(X, counts, family="poisson")
    >>> result.family
    PoissonFamily()
    """
```

---

## Implementation Timeline

| Task | Description | Status |
|------|-------------|--------|
| B1 | Unified Result Hierarchy | ⬜ Not Started |
| B2 | Universal Summary Function | ⬜ Not Started |
| B3 | Universal Plot Function | ⬜ Not Started |
| B4 | Model Comparison Function | ⬜ Not Started |
| B5 | Validation Decorators | ⬜ Not Started |
| C1 | IO Module | ⬜ Not Started |
| C2 | ANOVA Module | ⬜ Not Started |
| C3 | Sensitivity Module | ⬜ Not Started |
| C4 | Linalg Module | ⬜ Not Started |
| D1 | Mathematical Docstrings | ⬜ Not Started |
| D2 | Executable Examples | ⬜ Not Started |

---

## Success Criteria

- [ ] All new code has >90% test coverage
- [ ] All public functions have complete docstrings
- [ ] `import aurora; help(aurora)` shows clear API
- [ ] Breaking changes documented in CHANGELOG
- [ ] All examples in docstrings are executable

---

*Plan approved and implementation started: 2024-12-19*
