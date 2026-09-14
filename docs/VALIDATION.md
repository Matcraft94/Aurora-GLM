# Validation Methodology

**Aurora-GLM v1.0.0** — Reproducible numerical validation against reference implementations.

This document describes the validation methodology used to guarantee that
Aurora-GLM produces results identical to the gold-standard implementations
(R `glm()`, `mgcv::gam()`, `lme4::lmer()`, and statsmodels). It also documents
known convention differences and how to reproduce each validation locally.

---

## 1. Reference Implementations

| Software | Version | Role |
|----------|---------|------|
| **R** `glm()` | R >= 4.3 | GLM gold standard (Nelder & Wedderburn 1972) |
| **R** `mgcv` | >= 1.9 | GAM gold standard (Wood 2017) |
| **R** `lme4` | >= 1.4 | LMM/GLMM gold standard (Bates et al. 2015) |
| **statsmodels** | >= 0.14 | Python GLM reference (Seabold & Perktold 2010) |
| **scipy** | >= 1.10 | Numerical primitives |

---

## 2. Validation Mechanisms

Aurora-GLM is validated through three layers, listed in increasing order of
authority:

### Layer 1: Internal unit tests (3,377 tests collected)

Located in `tests/`. Verify algorithmic correctness against closed-form
solutions, analytical gradients, and synthetic data with known ground truth.
Run with:

```bash
pytest tests/
```

### Layer 2: Cross-validation against statsmodels (v1.0.0)

Located in `tests/test_models/test_glm_vs_statsmodels.py`. Validates
log-likelihood, AIC, BIC, coefficients, deviance, and fitted values across
Gaussian, Poisson, Binomial, and Gamma families. Tolerances:

| Quantity | Tolerance | Justification |
|----------|-----------|---------------|
| Coefficients | rtol = 1e-6 | Algorithms agree to floating point |
| log-likelihood | atol = 1e-6 | Full formula with normalizing constants |
| AIC | atol = 1e-6 | Derived directly from log-likelihood |
| BIC | atol = 1e-6 | Derived directly from log-likelihood |
| Deviance | rtol = 1e-6 | Closed-form per family |
| Fitted values | rtol = 1e-6 | Same algorithm on same data |

Run with:

```bash
pytest tests/test_models/test_glm_vs_statsmodels.py -v
```

### Layer 3: Cross-validation against R (v1.0.0)

Located in `tests/test_models/test_glm_vs_r.py`. Subprocess-based test that
runs `benchmarks/run_r_checks.R` (synthetic-data GLM fits in R) and compares
against Aurora-GLM. **Auto-skips when R/Rscript is not installed.**

Run locally:

```bash
# Install R first: https://www.r-project.org/
Rscript -e 'install.packages(c("jsonlite", "mgcv", "lme4"))'
pytest tests/test_models/test_glm_vs_r.py -v
```

---

## 3. Validated Families

Each family's log-likelihood now includes **all normalizing constants**
(matching R/statsmodels conventions). Prior to v1.0.0, the log-likelihood
omitted constants — this was a bug for absolute AIC/BIC reporting.

### 3.1 Poisson

**R formula** (`dpois(y, lambda, log=TRUE)`):

    ell(y; mu) = y * log(mu) - mu - log(y!)

**Aurora implementation** (`aurora/distributions/families/poisson.py`):

```python
def log_likelihood(self, y, mu, **params):
    return (y * log(mu) - mu - lgamma(y + 1)).sum()
```

**Validation result (representative, n=300):**

```
Aurora log_lik  = -262.781488
statsmodels llf = -262.781488
diff            = 0.00e+00
```

### 3.2 Binomial (Bernoulli and n-trial)

**R formula** (`dbinom(y, n, mu/n, log=TRUE)`):

    ell(y; mu, n) = log C(n, y) + y * log(mu/n) + (n - y) * log(1 - mu/n)

with `log C(n, y) = log Gamma(n+1) - log Gamma(y+1) - log Gamma(n-y+1)`.

For Bernoulli (`n = 1`), the binomial coefficient term is identically zero.

**Validation result (Bernoulli, n=400):**

```
Aurora log_lik  = -112.045796
statsmodels llf = -112.045796
diff            = 2.84e-14
```

### 3.3 Gaussian (with MLE variance)

**R formula** (`dnorm(y, mu, sigma, log=TRUE)` with sigma^2 = RSS / n):

    ell(y; mu, sigma^2) = -0.5 * (RSS / sigma^2 + n * log(2 * pi * sigma^2))

When `variance` is not passed explicitly, Aurora estimates `sigma_hat^2 =
RSS / n` (MLE, matching statsmodels).

**Validation result (n=200):**

```
Aurora log_lik  = -210.157970
statsmodels llf = -210.157970
diff            = 2.84e-14
```

### 3.4 Gamma (R/mgcv convention)

Aurora uses the standard **shape parameterization** (alpha = shape):

    ell(y; mu, alpha) = alpha * log(alpha/mu) - log Gamma(alpha) + (alpha - 1) * log(y) - alpha * y / mu

This matches **R `dgamma(y, shape=alpha, rate=alpha/mu, log=TRUE)`** exactly.

> **Convention note**: statsmodels uses a non-standard `scale` parameter
> (where `scale = 1/shape`) for its Gamma log-likelihood. Aurora's
> `result.log_likelihood_` for Gamma may differ from
> `statsmodels.GLM(...).fit().llf` by a small per-observation constant.
> Use the **R** formula (above) as the authoritative reference. Aurora's
> Gamma AIC/BIC are valid for model comparison within Aurora and against R.

> **Dispersion-estimator note (AIC vs R)**: Aurora plugs the moment
> estimator `phi = D/(n - rank)` into the Gamma log-likelihood
> (statsmodels' `llf` convention), while R's `logLik.glm`/`AIC` for Gamma
> uses the ML dispersion `phi = D/n` (`Gamma()$aic` convention) and counts
> the dispersion as an extra parameter (`k + 1`). Consequently Aurora's
> `result.aic_` and R's `AIC()` differ by a small, phi-dependent amount —
> not a constant. The R-validation harness
> (`benchmarks/compare_with_r.py`) recomputes Aurora's Gamma log-likelihood
> under R's `phi = D/n` before comparing AIC, so the 1e-6 gate tests the
> likelihood agreement itself. Verified: all 20 family × replicate
> comparisons match R to ≤ 1.4e-13.

### 3.5 Other families

- **Negative Binomial, Inverse Gaussian, Beta, Student-t, Tweedie**: each
  family's `log_likelihood()` includes normalizing constants per the
  standard distribution formula. Cross-validation tests against R for these
  families are planned for v1.1.

---

## 4. AIC/BIC Convention

Aurora-GLM uses the standard definitions:

    AIC = -2 * ell + 2 * k
    BIC = -2 * ell + k * log(n)

where `ell` is the total log-likelihood (with all normalizing constants) and
`k` is the number of regression coefficients (including intercept).

> Prior to v1.0.0, Aurora used `deviance / 2 + k` for log-likelihood, which
> omitted family-specific constants. **This was a bug** (Phase 1.1 fix).
> Researchers comparing pre-v1.0.0 AIC values against R/statsmodels should
> expect discrepancies of order `n * mean(log(y!))` for Poisson, or similar
> constant offsets for other families.

---

## 5. IRLS Convergence Criterion

The production GLM fit path (`fit_glm` in `aurora/models/glm/fitting.py`)
checks convergence on the **relative deviance change**, following the R
`glm.fit` convention:

    |D^(t+1) - D^(t)| / (|D^(t)| + 0.1) < tol

Additional safeguards in the same loop:

- **Step-halving**: a step that increases the deviance (or yields a
  non-finite deviance) is halved up to 25 times; if no acceptable step is
  found, iteration stops with ``converged_=False`` rather than accepting a
  worse fit.
- **Solver**: the weighted least-squares step uses LAPACK Cholesky on
  X'WX, falling back to pivoted least squares (``numpy.linalg.lstsq``) when
  X'WX is not positive definite. Numerical rank deficiency is reported
  through ``GLMResult.rank_`` with a ``RuntimeWarning``.
- **Conditioning**: the condition number of X'WX is monitored each
  iteration and a ``RuntimeWarning`` is emitted above κ = 1e8.

The low-level `aurora/core/optimization/irls.py` routine (used by other
algorithms, including the sparse path) instead uses the relative
parameter-change criterion `||delta|| / (||beta|| + 1e-12) < tol`, aligned
with the scipy.optimize convention.

---

## 6. Multi-Backend Consistency

All numerical code must produce identical results (to floating-point
precision) across NumPy, PyTorch, and JAX backends. The following cross-backend
tolerances are enforced in `tests/`:

| Backend pair | rtol | atol | Notes |
|--------------|------|------|-------|
| NumPy <-> PyTorch | 4e-7 | 1e-10 | Different BLAS implementations |
| NumPy <-> JAX | 1e-15 | 1e-15 | JAX uses float64 when configured |
| PyTorch CPU <-> CUDA | 1e-6 | 1e-8 | Floating-point order non-determinism |

JAX float64 must be enabled explicitly. The test fixture `backend` in
`tests/conftest.py` sets `jax.config.update("jax_enable_x64", True)`
automatically.

---

## 7. Reproducing the Validation

### Quick check (5 minutes)

```bash
# Create venv and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run internal tests + statsmodels validation
pytest tests/test_models/test_glm_vs_statsmodels.py -v
```

### Full validation (15 minutes)

```bash
# Include JAX backend tests
pip install -e ".[dev,jax]"
pytest tests/test_models/ tests/test_distributions/ tests/test_core/ -v
```

### R cross-validation (requires R installed)

```bash
# Install R packages once
Rscript -e 'install.packages(c("jsonlite", "mgcv", "lme4"))'

# Run R-comparison test (auto-skips if R missing)
pytest tests/test_models/test_glm_vs_r.py -v

# Or run the underlying benchmark directly
python benchmarks/compare_with_r.py --output /tmp/r_validation.json
cat /tmp/r_validation.json
```

---

## 8. Known Convention Differences

| Topic | Aurora | statsmodels | R |
|-------|--------|-------------|---|
| GLM log-likelihood | Full (with constants) | Full | Full |
| Gamma parameterization | Shape alpha | Scale phi = 1/alpha | Shape alpha |
| AIC formula | -2*ell + 2*k (mean params only) | -2*ell + 2*k | -2*ell + 2*k (+ dispersion for Gaussian-like families, so R's AIC is higher by 2) |
| IRLS convergence (`fit_glm`) | Relative deviance change |ΔD|/(|D|+0.1) (R glm.fit style) | Relative parameter change | Relative deviance change |
| Step-halving | Yes (deviance-based, up to 25 halvings) | Yes | Yes |
| Solver | Cholesky on X'WX, pivoted lstsq fallback, rank reported via `rank_` | QR/pinv | pivoted QR (dqrls) |
| Condition number warning | Yes, at kappa > 1e8 | No | No |
| Boundary correction (LRT) | Yes (Self & Liang 1987) | No | Yes (in lme4) |
| PQL bias correction | No (uncorrected PQL; documented limitation, cf. Breslow & Lin 1995) | No | No |

---

## 9. Validated Claims → Executable Tests

Every public numerical claim is backed by an automated, executable test:

| Claim | Executable test |
|-------|-----------------|
| GLM coefficients, log-likelihood, AIC/BIC, deviance, fitted values match statsmodels (< 1e-8) | `tests/test_models/test_glm_vs_statsmodels.py` |
| GLM coefficients, deviance, fitted values match R `glm()` (< 1e-6; AIC matches after the documented +2 dispersion offset for Gaussian-like families) | `tests/test_models/test_glm_vs_r.py` (skips without Rscript; runs in the CI `r-validation` job) |
| Observation/frequency weights match statsmodels | `tests/test_models/test_glm_weights.py` |
| Grouped (n-trial) binomial matches statsmodels | `tests/test_models/test_glm_grouped_binomial.py` |
| Heteroscedasticity-consistent (sandwich HC) standard errors match statsmodels | `tests/test_inference/test_robust_vs_statsmodels.py` |
| ANOVA / likelihood-ratio tests match statsmodels | `tests/test_inference/test_anova.py` |
| Multi-backend consistency (NumPy / PyTorch / JAX) | `tests/test_backends/` |
| GAM confidence bands use σ̂² = RSS/(n − edf) with the Bayesian covariance Vp | `tests/test_models/test_gam_plotting.py::test_plot_smooth_ci_uses_edf_corrected_scale` |

---

## 10. Test Result Summary (v1.0.0)

Run with: `pytest tests/test_models/test_glm_vs_statsmodels.py -v`

```
TestGaussianVsStatsmodels::test_coefficients_match            PASSED
TestGaussianVsStatsmodels::test_log_likelihood_match          PASSED
TestGaussianVsStatsmodels::test_aic_bic_match                 PASSED
TestGaussianVsStatsmodels::test_deviance_matches              PASSED
TestGaussianVsStatsmodels::test_fitted_values_match           PASSED
TestPoissonVsStatsmodels::test_coefficients_match             PASSED
TestPoissonVsStatsmodels::test_log_likelihood_match           PASSED
TestPoissonVsStatsmodels::test_aic_bic_match                  PASSED
TestPoissonVsStatsmodels::test_deviance_matches               PASSED
TestBinomialVsStatsmodels::test_coefficients_match            PASSED
TestBinomialVsStatsmodels::test_log_likelihood_match          PASSED
TestBinomialVsStatsmodels::test_aic_bic_match                 PASSED
TestGammaAgainstRFormula::test_log_likelihood_matches_r_formula  PASSED
TestGammaAgainstRFormula::test_coefficients_match_statsmodels   PASSED
TestModelSelectionConsistency::test_poisson_model_selection_picks_same_model  PASSED

15 passed in 0.46s
```

---

## 11. Citing This Validation

If you use Aurora-GLM in published research, cite:

```bibtex
@misc{auroraglm2026,
  author = {Arias, Lucy Eduardo},
  title  = {Aurora-GLM: Generalized Linear and Additive Models},
  year   = {2026},
  version = {1.0.0},
  url    = {https://github.com/Matcraft94/Aurora-GLM}
}
```

See `CITATION.cff` for the canonical citation metadata.

---

## References

1. Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized Linear Models.
   *Journal of the Royal Statistical Society. Series A*, 135(3), 370-384.
2. Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
   (2nd ed.). Chapman and Hall/CRC.
3. Bates, D., Machler, M., Bolker, B., & Walker, S. (2015). Fitting Linear
   Mixed-Effects Models Using lme4. *Journal of Statistical Software*, 67(1).
4. Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and
   statistical modeling with python. *SciPy Proceedings*.
5. Self, S. G., & Liang, K.-Y. (1987). Asymptotic properties of maximum
   likelihood estimators and likelihood ratio tests under nonstandard
   conditions. *Journal of the American Statistical Association*, 82(398).
6. Breslow, N. E., & Lin, X. (1995). Bias correction in generalised linear
   mixed models with a single component of disperson. *Biometrika*, 82(1).
