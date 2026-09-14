# Aurora-GLM Case Studies

Sixteen executable case studies covering GLM, GAM, and GAMM workflows.
Every notebook runs end-to-end against Aurora-GLM 1.0.0, and every model
described in a notebook is actually fitted in it.

**Version**: 1.0.0 · **Case studies**: 16 · **Last updated**: August 2026

---

## Quick Start

```bash
pip install -e .          # from the repository root
pip install jupyter
jupyter notebook          # then open examples/06_case_studies/
```

The notebooks are numbered as a learning path: **GLM (01–04)** →
**GAM (05–10)** → **GAMM (11–16)**. New to the library? Start with 01,
then 05, then 11.

---

## The Case Studies

### GLM foundations

#### 01. Insurance Pricing — Gamma GLM
`01_insurance_gamma_glm.ipynb` · real data (`insurance.csv`)

Medical insurance charges with a Gamma GLM. Compares log vs identity
links, tests the age × BMI interaction, and interprets rate ratios for
risk segmentation.

- **Families/links**: Gamma, log and identity
- **Key result**: smokers pay ~4.5× more; identity link preferred by AIC

#### 02. French Motor Claims — Poisson & Negative Binomial
`02_french_motor_poisson_nb.ipynb` · real data (freMTPL2freq, OpenML)

Claim frequency modeling with an exposure offset. Detects overdispersion
(φ̂ ≈ 2.8), estimates the NB dispersion parameter θ by ML (Lawless 1987)
in a two-step `glm.nb`-style flow, and compares Poisson vs NB.

- **Families/links**: Poisson, Negative Binomial; log link; offset
- **Key result**: NB improves AIC by ~128 points

#### 03. Telco Customer Churn — Binomial GLM
`03_telco_churn_binomial_glm.ipynb` · real data (`telco_churn.csv`)

Logistic regression for churn with odds ratios, ROC analysis and
calibration diagnostics.

- **Families/links**: Binomial, logit
- **Key result**: AUC ≈ 0.83; contract type and tenure dominate

#### 04. Restaurant Health Scores — Beta GLM
`04_restaurant_health_beta_glm.ipynb` · synthetic data

Beta regression for bounded (0, 1) inspection scores, with spline
regression for the non-linear age effect. The notebook explains why this
is a GLM with basis expansion and not a penalized GAM (the additive-GAM
fitting API is Gaussian-only).

- **Families/links**: Beta, logit
- **Key result**: inspection frequency inversely related to violations

### GAM

#### 05. Air Quality — Gaussian GAM
`05_air_quality_gam.ipynb` · real data (`airquality.csv`)

Additive model fitted with `fit_additive_gam` and GCV smoothing-parameter
selection, with sum-to-zero identifiability constraints.

- **Key result**: R² 0.74 vs 0.61 for the linear baseline

#### 06. Bike-Sharing Demand — GAM
`06_bike_sharing_gam.ipynb` · real data (`bike_sharing_hourly.csv`)

Hourly rental counts via an additive GAM on log(count + 1) with smooths
for temperature, humidity and time-of-day, including an honest treatment
of the quasi-Poisson approximation.

#### 07. Wind Power Forecasting — Gamma GLM → GAM
`07_wind_power_gam.ipynb` · synthetic data (17,520 hourly records)

Gamma GLM baseline against a Gaussian GAM on log(power), with a
power-curve analysis against the theoretical curve.

- **Key result**: R² 0.70 → 0.87 with smooth terms

#### 08. E-Commerce Conversion — Binomial GAMM with smooths
`08_ecommerce_conversion_gam.ipynb` · synthetic data

Conversion modeling with `fit_gamm` formula syntax: smooth terms for
price, session duration and hour (PQL), plus a random intercept per
day-of-week. Partial-dependence visualizations for the smooth effects.

- **Key result**: smooth terms improve Brier score and log-loss

#### 09. Species Distribution — Poisson GLM
`09_species_distribution_glm.ipynb` · synthetic data

Multi-model Poisson regression for abundance data: rate ratios,
nested-model comparison, and zero-inflation discussed as a limitation.

#### 10. US Traffic Accidents — Binomial GLM at scale
`10_us_accidents_binomial_glm.ipynb` · synthetic data (150k records)

Binary severity classification at scale with class-imbalance caveats
(sensitivity at the default 0.5 threshold) and calibration checks.

- **Key result**: AUC ≈ 0.79; night and fog are the strongest factors

### GAMM

#### 11. Sleep Study — Random Intercepts and Slopes
`11_sleep_study_gamm.ipynb` · real data (`sleepstudy.csv`)

The classic reaction-time experiment: `(1 + days | subject)`, marginal vs
conditional R², BLUP extraction and diagnostics.

#### 12. Longitudinal Clinical Trial — Covariance Structures
`12_clinical_trial_longitudinal_gamm.ipynb` · synthetic data

Repeated measures with an identity vs AR(1) covariance comparison on a
balanced subsample. The boundary estimate (ρ̂ → 1) is explained, and
effect sizes (Cohen's d) are computed from the fitted model.

#### 13. Clinical Trial — Multilevel LMM and the Binary-Outcome Caveat
`13_clinical_trial_multilevel.ipynb` · synthetic data

Patients nested in clinics with treatment crossed with clinic. Fitted as
a linear probability model with random effects; the notebook then shows
what happens with PQL under separation (rare outcome) and discusses when
the LPM is the pragmatic choice.

#### 14. Psychometric Crossed Effects
`14_psychometric_crossed_gamm.ipynb` · synthetic data (3,200 obs)

Crossed random effects for subjects and items — the large-sparse
mixed-model case.

#### 15. Educational Multilevel (PISA UK)
`15_educational_multilevel_gamm.ipynb` · real data (`pisaUK.csv`)

Eight models from the null model to random slopes on real PISA UK data:
variance partitioning, ICC, random-slope interpretation. Documents the
absence of tensor-product smooths (`te()`) in the current API.

#### 16. Breast Cancer Outcomes — Binomial GAMM (PQL)
`16_breast_cancer_binomial_gamm.ipynb` · synthetic data (30 × 50)

Multi-center binary outcome fitted by Penalized Quasi-Likelihood
(Breslow & Clayton 1993): odds ratios, latent-scale ICC, and an explicit
discussion of PQL attenuation with few clusters.

- **Key result**: between-hospital ICC ≈ 11%; ER+ raises survival odds

---

## Data

Datasets live in `data/`. `freMTPL2freq.csv` (case 02) is **not tracked
in git** (36 MB) — the notebook downloads it from OpenML
(`openml.org/data/get_csv/20649148/freMTPL2freq.csv`) when missing. Cases
04, 07, 08, 09, 10, 12, 13, 14, and 16 generate synthetic data inline
with fixed seeds.

---

## Common Structure

All notebooks follow the same template:

1. Overview and research questions
2. Setup and data (source, size, license)
3. Exploratory data analysis
4. Mathematical specification (family, link, likelihood, penalties)
5. Baseline fit → main model
6. Residual diagnostics
7. Interpretation on the response scale
8. Optional multi-backend benchmark (skipped gracefully without torch)
9. Conclusions per research question, limitations, references

---

## Repository

https://github.com/Matcraft94/Aurora-GLM — issues and contributions
welcome.
