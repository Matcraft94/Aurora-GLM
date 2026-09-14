# Data Sources

Datasets used by the case-study notebooks. All are small, public, and
redistributed here for reproducibility so the notebooks run offline after a
fresh clone. The 36 MB `freMTPL2freq.csv` (case 02) is intentionally **not
tracked** — the notebook downloads it from OpenML when missing.

| File | Source | License / Terms |
|---|---|---|
| `airquality.csv` | R base `datasets::airquality` — New York air quality measurements (Chambers et al., 1983) | Public domain (R datasets) |
| `sleepstudy.csv` | `lme4::sleepstudy` — Belenky et al. (2003), sleep deprivation reaction times | GPL ≥ 2 (as part of lme4); citation: Bates et al. (2015), JSS |
| `insurance.csv` | Medical insurance charges, via *Machine Learning with R* datasets repo (github.com/stedy/Machine-Learning-with-R-datasets) | Public domain |
| `pisaUK.csv` | PISA UK subsample (reading score, school-level) — used in multilevel modeling examples | OECD PISA data terms |
| `telco_churn.csv` | IBM Telco Customer Churn sample (Kaggle) | Sample data, redistributable |
| `bike_sharing_hourly.csv` | UCI Bike Sharing Dataset (Fanaee-T & Gama, 2014) | CC BY 4.0 |

When adding a dataset: keep it small (< ~2 MB), add a row here, and
whitelist the file in the root `.gitignore` (CSV/TSV are ignored by default).
