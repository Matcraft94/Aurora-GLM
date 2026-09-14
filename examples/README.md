# Aurora-GLM Examples and Case Studies

Collection of executable Jupyter notebooks demonstrating Aurora-GLM across
GLM, GAM, and GAMM workflows. Every notebook runs end-to-end against the
installed package, and every model mentioned in the text is actually fitted.

**Aurora-GLM Version**: 1.0.0
**Total Case Studies**: 16
**Last Updated**: August 2026

---

## Overview

The case studies in `06_case_studies/` follow a consistent structure:
research questions, data description, EDA, mathematical specification
(family, link, likelihood), model fitting, residual diagnostics,
interpretation on the response scale, and conclusions with limitations and
references. Notebooks are ordered as a learning path: GLM first, then GAM,
then GAMM — real datasets before synthetic ones.

### Running Notebooks

```bash
pip install -e .          # from the repository root
pip install jupyter
jupyter notebook examples/06_case_studies/
```

### Suggested learning path

1. **01** Insurance Pricing (Gamma GLM) — families, links, weights
2. **05** Air Quality (GAM) — smooth terms and GCV
3. **11** Sleep Study (GAMM) — random intercepts and slopes
4. **12** Longitudinal Clinical Trial (GAMM) — covariance structures

---

## The 16 Case Studies

### GLM foundations

**01. Insurance Pricing — Gamma GLM** (`01_insurance_gamma_glm.ipynb`)
- Gamma family with log vs identity link comparison; age × BMI interaction
- Real dataset (`insurance.csv`); weights and offset conventions
- Key result: smokers pay ~4.5× more; identity link preferred by AIC

**02. French Motor Claims — Poisson & Negative Binomial** (`02_french_motor_poisson_nb.ipynb`)
- Poisson GLM with exposure offset, then Negative Binomial with
  ML-estimated θ (Lawless 1987) once overdispersion is detected (φ̂ ≈ 2.8)
- Real dataset (freMTPL2freq, 678,013 policies; sampled to 100k in-notebook)
- Key result: NB improves AIC by ~128 points over Poisson

**03. Telco Customer Churn — Binomial GLM** (`03_telco_churn_binomial_glm.ipynb`)
- Logistic regression with odds ratios, ROC and calibration analysis
- Real dataset (`telco_churn.csv`)
- Key result: AUC ≈ 0.83; contract type and tenure dominate churn risk

**04. Restaurant Health Scores — Beta GLM** (`04_restaurant_health_beta_glm.ipynb`)
- Beta family for bounded (0, 1) scores; spline regression for the
  non-linear age effect (not a penalized GAM — the notebook explains why:
  the additive-GAM fitting API is Gaussian-only)
- Synthetic dataset
- Key result: inspection frequency inversely related to violations

### GAM

**05. Air Quality — Gaussian GAM** (`05_air_quality_gam.ipynb`)
- `fit_additive_gam` with GCV smoothing-parameter selection and
  identifiability constraints
- Real dataset (`airquality.csv`)
- Key result: R² 0.74 vs 0.61 linear; clearly non-linear pollutant effects

**06. Bike-Sharing Demand — GAM** (`06_bike_sharing_gam.ipynb`)
- Additive GAM on log(count + 1) with smooths for temperature, humidity
  and time-of-day; honest discussion of the quasi-Poisson approximation
- Real dataset (`bike_sharing_hourly.csv`)

**07. Wind Power Forecasting — Gamma GLM → GAM** (`07_wind_power_gam.ipynb`)
- Gamma GLM baseline vs Gaussian GAM on log(power); power-curve analysis
- Synthetic dataset (17,520 hourly records)
- Key result: R² 0.70 → 0.87 with smooth terms

**08. E-Commerce Conversion — Binomial GAMM with smooths** (`08_ecommerce_conversion_gam.ipynb`)
- `fit_gamm` with `s()` terms (PQL) for price, duration and hour effects;
  random intercept per day-of-week; partial-dependence visualizations
- Synthetic dataset
- Key result: smooth terms improve Brier score and log-loss over the GLM

**09. Species Distribution — Poisson GLM** (`09_species_distribution_glm.ipynb`)
- Multi-model Poisson regression with rate ratios and nested-model
  comparison; zero-inflation discussed as a limitation
- Synthetic dataset

**10. US Traffic Accidents — Binomial GLM at scale** (`10_us_accidents_binomial_glm.ipynb`)
- Large-n binary severity classification with class-imbalance caveats
  (sensitivity at the default threshold)
- Synthetic dataset (150k records)
- Key result: AUC ≈ 0.79; night and fog are the strongest risk factors

### GAMM

**11. Sleep Study — random intercepts and slopes** (`11_sleep_study_gamm.ipynb`)
- `(1 + days | subject)` on the real sleepstudy dataset; marginal vs
  conditional R²; BLUP diagnostics
- Real dataset (`sleepstudy.csv`)

**12. Longitudinal Clinical Trial — covariance structures** (`12_clinical_trial_longitudinal_gamm.ipynb`)
- Identity vs AR(1) covariance comparison on a balanced subsample;
  boundary estimate (ρ̂ → 1) explained; effect sizes (Cohen's d)
- Synthetic dataset

**13. Clinical Trial — multilevel LMM and the binary-outcome caveat** (`13_clinical_trial_multilevel.ipynb`)
- Patients nested in clinics, treatment crossed with clinic; linear
  probability model with random effects vs PQL attempt under separation
  (documented, with the library's separation warning shown)
- Synthetic dataset

**14. Psychometric Crossed Effects** (`14_psychometric_crossed_gamm.ipynb`)
- Crossed random effects for subjects and items (3,200 observations)
- Synthetic dataset

**15. Educational Multilevel (PISA UK)** (`15_educational_multilevel_gamm.ipynb`)
- Eight models from null to random slopes on real PISA UK data; variance
  partitioning; notes on the absence of tensor-product smooths
- Real dataset (`pisaUK.csv`)

**16. Breast Cancer Outcomes — Binomial GAMM (PQL)** (`16_breast_cancer_binomial_gamm.ipynb`)
- Bernoulli + logit fitted by Penalized Quasi-Likelihood (Breslow &
  Clayton 1993); random intercept per hospital; odds ratios; latent-scale
  ICC; honest PQL limitations (attenuation with few clusters)
- Synthetic dataset (30 hospitals × 50 patients)
- Key result: between-hospital ICC ≈ 11%; ER+ raises survival odds

---

## Data Files

Datasets live in `06_case_studies/data/` (referenced as `data/<name>.csv`).
Notebooks that use them download or fall back to the local cache:

- `insurance.csv` — medical insurance costs (01; also in `examples/data/`)
- `freMTPL2freq.csv` — French motor claim frequencies (02). **Not tracked
  in git** (36 MB); the notebook downloads it from OpenML
  (`openml.org/data/get_csv/20649148/freMTPL2freq.csv`) when missing
- `airquality.csv` (05), `bike_sharing_hourly.csv` (06),
  `telco_churn.csv` (03), `sleepstudy.csv` (11), `pisaUK.csv` (15)

Cases 04, 07, 08, 09, 10, 12, 13, 14, 16 generate synthetic data inline
with fixed seeds.

---

## Model Coverage Matrix

| | GLM | GAM | GAMM |
|---|---|---|---|
| Gaussian | — | 05, 06, 07 | 11, 12, 13, 14, 15 |
| Binomial | 03, 10 | 08 (via PQL) | 16 |
| Poisson / NB | 02, 09 | — | — |
| Gamma | 01 | 07 (baseline) | — |
| Beta | 04 | — | — |

---

## Contributing

To add a new case study:

1. Follow the common structure (RQs → data → EDA → specification → fit →
   diagnostics → interpretation → conclusions with limitations/references)
2. Every model mentioned in the text must be fitted in the notebook
3. Execute the full notebook and keep clean outputs
4. Update this README and `06_case_studies/README.md`

---

## Citation

```bibtex
@software{aurora_glm2025,
  title = {Aurora-GLM: Generalized Linear and Additive Models},
  author = {Arias, Lucy Eduardo},
  year = {2025},
  url = {https://github.com/Matcraft94/Aurora-GLM},
  version = {1.0.0}
}
```

**Questions?** Open an issue at https://github.com/Matcraft94/Aurora-GLM/issues
