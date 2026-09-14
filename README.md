# Aurora-GLM

**Aurora-GLM** is a Python framework for statistical modeling with
Generalized Linear Models (GLM), Generalized Additive Models (GAM), and
Generalized Additive Mixed Models (GAMM). It provides R-style formula
syntax, R-style model summaries, and a multi-backend numerical core
(NumPy by default, with optional PyTorch and JAX backends).

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/aurora-glm.svg)](https://pypi.org/project/aurora-glm/)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/Matcraft94/Aurora-GLM)
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)](https://github.com/Matcraft94/Aurora-GLM)

- **Package name**: `aurora-glm` · **Import**: `import aurora`
- **Repository**: [github.com/Matcraft94/Aurora-GLM](https://github.com/Matcraft94/Aurora-GLM)
- **Python**: ≥ 3.10 · **License**: MIT

## Installation

From PyPI:

```bash
pip install aurora-glm          # Core (NumPy backend)
pip install aurora-glm[torch]   # + PyTorch backend (CPU/GPU)
pip install aurora-glm[jax]     # + JAX backend
pip install aurora-glm[all]     # All optional dependencies
```

From source (development):

```bash
git clone https://github.com/Matcraft94/Aurora-GLM.git
cd Aurora-GLM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"         # Core + pytest, ruff, mypy
```

## Quickstart

### GLM: Poisson regression

```python
import numpy as np
from aurora.models.glm import fit_glm

# Generate Poisson count data
np.random.seed(42)
X = np.random.randn(200, 2)
y = np.random.poisson(np.exp(X[:, 0] * 0.5 - 0.3))

# Fit Poisson GLM with log link
result = fit_glm(X, y, family='poisson', link='log')

# R-style summary with coefficients, std errors, p-values
print(result.summary())
```

**Output:**

```
==============================================================================
                       Generalized Linear Model Results                       
==============================================================================
Family:                   Poisson                    Link function:  Log
No. Observations:         200                        Df Residuals:   197
Df Model:                 2                         
Converged:                Yes                        No. Iterations: 5
==============================================================================
                   coef    std err          z      P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
   intercept    -0.3858     0.0904     -4.268      0.000    -0.5630    -0.2086 ***
          X0     0.5291     0.0856      6.184      0.000     0.3614     0.6968 ***
          X1    -0.0410     0.0818     -0.502      0.616    -0.2013     0.1192 
==============================================================================
Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
==============================================================================
Deviance:                               208.10 Null Deviance:           248.20
AIC:                                    443.43 BIC:                     453.33
Pseudo R-squared:                       0.1616
==============================================================================
```

`fit_glm` supports observation `weights` (propagated to deviance,
log-likelihood, AIC/BIC following R conventions) and `offset`. For the
Gaussian, Gamma and Inverse Gaussian families the dispersion
φ̂ = deviance/(n − rank) is estimated and standard errors are scaled
accordingly (`result.dispersion_`, `result.rank_`,
`result.condition_number_`). The IRLS solver uses step-halving and warns
on non-convergence, complete separation, and ill-conditioning (κ > 1e8).
Binomial responses are proportions in [0, 1] — pass `weights=n_trials`
for grouped data (raw counts raise a `ValueError` with a remedial
message).

### GAM: R-style formula interface

```python
import numpy as np
from aurora.models.gam import fit_gam_formula

np.random.seed(42)
n = 200
data = {
    "x1": np.random.uniform(0, 1, n),
    "x2": np.random.uniform(0, 1, n),
}
data["y"] = np.sin(2 * np.pi * data["x1"]) + data["x2"] ** 2 + 0.1 * np.random.randn(n)

# Smooth terms with automatic smoothing-parameter selection (GCV or REML)
result = fit_gam_formula("y ~ s(x1, k=12) + s(x2)", data)
print(result.summary())
```

**Output:**

```
======================================================================
Additive Generalized Additive Model (GAM) - Fitted Summary
======================================================================

Model Structure:
  Smooth terms:        2
  Parametric terms:    0
  Observations:        200
  Total EDF:           15.81

Smooth Terms:
  s(0):
    Basis:             bspline
    n_basis:           12
    Lambda:            2.362942e-01
    EDF:               8.06
  s(1):
    Basis:             bspline
    n_basis:           10
    Lambda:            2.362942e-01
    EDF:               6.74

Fit Statistics:
  Residual sum sq:     1.7191
  R-squared:           0.9862
  Residual std:        0.0927
  GCV score:           1.013389e-02

Residuals:
  Min:                 -0.2636
  Q1:                  -0.0652
  Median:              -0.0081
  Q3:                  0.0651
  Max:                 0.2785

======================================================================
```

Lower-level entry points: `fit_gam` (single smooth) and
`fit_additive_gam` (explicit `SmoothTerm`/`ParametricTerm` lists).

### GAMM: random effects

```python
import numpy as np
import pandas as pd
from aurora.models.gamm import fit_gamm

np.random.seed(42)
n_groups, n_per = 10, 20
n = n_groups * n_per
subject = np.repeat(np.arange(n_groups), n_per)
x = np.random.randn(n)
b = np.random.randn(n_groups) * 0.8

data = pd.DataFrame({
    "y": 2.0 + 0.5 * x + b[subject] + np.random.randn(n) * 0.3,
    "x": x,
    "subject": subject,
})

# lme4-style formula: random intercept per subject
result = fit_gamm(formula="y ~ x + (1 | subject)", data=data, covariance="identity")

print(f"Fixed effects (beta): {result.beta_parametric}")
print(f"Variance components:  {result.variance_components}")
print(f"Residual variance:    {result.residual_variance:.4f}")
print(f"AIC: {result.aic:.2f}   BIC: {result.bic:.2f}")
```

**Output:**

```
Fixed effects (beta): [2.50379065 0.50315389]
Variance components:  [array([[1.17554844]])]
Residual variance:    0.0814
AIC: 134.69   BIC: 147.88
```

Gaussian models are fitted by REML via Henderson's mixed-model
equations; non-Gaussian families use PQL (`fit_pql`) or a Laplace
approximation (`fit_laplace`). Nested `(1 | a/b)` and crossed
`(1 | a) + (1 | b)` effects are supported.

## Feature overview

**Distribution families** (`aurora.distributions.families`):

| Family | Use case | Canonical link |
|--------|----------|----------------|
| Gaussian | Continuous outcomes | Identity |
| Poisson | Count data | Log |
| Binomial | Binary / proportions | Logit |
| Gamma | Positive continuous | Inverse |
| Inverse Gaussian | Positive durations | InverseSquare |
| Negative Binomial | Overdispersed counts (NB2) | Log |
| Beta | Proportions in (0, 1) | Logit |
| Student-t | Heavy-tailed, robust regression (NumPy backend only) | Identity |
| Tweedie | Compound Poisson–Gamma (insurance) | Log |

**Link functions** (9): Identity, Log, Logit, Probit, CLogLog, Inverse,
InverseSquare, Sqrt, Power.

**GAM / smoothing**: B-spline and natural cubic spline bases in the
fitting API (P-spline and thin-plate bases in `aurora.smoothing`);
difference penalties; smoothing-parameter selection by GCV or REML
(Wood 2011); sum-to-zero identifiability constraints; confidence bands
from the Bayesian covariance Vp with σ̂² = RSS/(n − edf).

**Inference**: Wald tests and Wald confidence intervals (the only CI
method), ANOVA via partial Wald or deviance-based likelihood-ratio
tests, Self & Liang (1987) boundary correction for LRT on variance
components, sandwich covariance estimators HC0–HC4, non-parametric
bootstrap (percentile intervals, with `n_failed` accounting), residuals
and influence measures (leverage, Cook's distance, DFBETAS).

**GAMM**: random intercepts and slopes; nested and crossed effects;
covariance structures: identity, diagonal, unstructured, AR(1),
compound symmetry, Toeplitz, exponential spatial, Matérn; estimation by
REML (Gaussian), PQL or Laplace approximation (non-Gaussian); sparse
matrix support (`use_sparse=True`) for large problems.

**Multi-backend**: NumPy (default), PyTorch, JAX. Backend detection is
automatic from input array types; cross-backend agreement is enforced
by automated tests (see below). PyTorch CUDA provides large speedups
on big problems (up to ~140× in our benchmarks on an RTX 5070 Ti; see
[benchmarks/PERFORMANCE.md](benchmarks/PERFORMANCE.md)). Note: the
optional backends are exercised mainly through the core numerical
paths — some higher-level features (e.g. Student-t) are NumPy-only.

## Validation

Numerical accuracy is enforced by automated tests, not just claimed:

| Claim | Executable test |
|-------|-----------------|
| GLM coefficients, log-likelihood, AIC/BIC, deviance, fitted values match statsmodels (< 1e-8) | `tests/test_models/test_glm_vs_statsmodels.py` |
| GLM coefficients, deviance, fitted values match R `glm()` (< 1e-6) | `tests/test_models/test_glm_vs_r.py` |
| Observation/frequency weights match statsmodels | `tests/test_models/test_glm_weights.py` |
| Grouped (n-trial) binomial matches statsmodels | `tests/test_models/test_glm_grouped_binomial.py` |
| Sandwich (HC) standard errors match statsmodels | `tests/test_inference/test_robust_vs_statsmodels.py` |
| ANOVA / likelihood-ratio tests match statsmodels | `tests/test_inference/test_anova.py` |
| Multi-backend consistency (NumPy / PyTorch / JAX) | `tests/test_backends/` |

The R comparison runs in CI (`.github/workflows/tests.yml`,
`r-validation` job, Python 3.10/3.12 test matrix + lint) and skips
locally when `Rscript` is unavailable. See
[docs/VALIDATION.md](docs/VALIDATION.md) for the methodology, known
convention differences, and the full claim-to-test mapping.

## Known limitations

- **PQL is not bias-corrected**: variance-component estimates for
  non-Gaussian GAMMs are uncorrected (cf. Breslow & Lin 1995) and should
  be treated as approximate. `fit_gamm`/`fit_pql` do not accept `offset`
  or prior `weights`.
- **Confidence intervals are Wald-only**: profile-likelihood and
  bootstrap BCa intervals are not implemented (`bootstrap_inference`
  provides percentile intervals).
- **No tensor product smooths**: `te(x1, x2)` is not supported by the
  formula parser or the public fitting functions.
- **Tweedie AIC is quasi-likelihood based** and is not comparable across
  different power parameters `p` or dispersion values.
- **Student-t family is NumPy-only.**
- **Automated R validation currently covers GLM** (`glm()`); systematic
  test-backed comparisons against `mgcv`/`lme4` are planned (see
  `docs/VALIDATION.md`).

## Documentation and examples

- **User guide and API reference** (Sphinx): `docs/` — build with
  `cd docs && python -m sphinx -b html . _build/html`
- **Validation methodology**: [docs/VALIDATION.md](docs/VALIDATION.md)
- **Case studies**: 16 Jupyter notebooks in
  [`examples/06_case_studies/`](examples/06_case_studies/) covering GLM,
  GAM and GAMM applications (insurance pricing, air quality, sleep
  study, clinical trials, …). See [examples/README.md](examples/README.md).

## Testing

```bash
pytest                       # Full suite (3,379 tests collected)
pytest -m "not slow"         # Skip slow tests
pytest tests/test_models/    # Subset
ruff check . && ruff format .   # Lint / format
mypy aurora                  # Type check
```

Test markers: `slow`, `gpu`, `integration`.

## Citation

If you use Aurora-GLM in your research, please cite it using the
information in [CITATION.cff](CITATION.cff), or the BibTeX entry:

```bibtex
@software{aurora_glm2025,
  title = {Aurora-GLM: Generalized Linear and Additive Models},
  author = {Arias, Lucy Eduardo},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/Matcraft94/Aurora-GLM},
  license = {MIT}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Authors

Maintained by Lucy Eduardo Arias ([@Matcraft94](https://github.com/Matcraft94))
— ORCID: [0009-0003-1905-7138](https://orcid.org/0009-0003-1905-7138).
Aurora-GLM draws inspiration from R's **mgcv** and **lme4** packages and
from Python's **statsmodels**.
