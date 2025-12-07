# Aurora-GLM Examples and Case Studies

Complete collection of production-ready examples demonstrating Aurora-GLM capabilities for statistical modeling across diverse domains.

**Aurora-GLM Version**: 0.6.1
**Total Case Studies**: 17
**Last Updated**: December 2024

---

## Overview

The examples directory contains comprehensive Jupyter notebooks and datasets for learning Aurora-GLM through realistic, domain-specific applications. Each case study follows a consistent 8-part structure:

1. **Problem Statement** - Research questions and domain context
2. **Data Exploration** - Summary statistics and visualizations
3. **Model Selection** - Justification for chosen family/link functions
4. **Model Specification** - Formula syntax and parameter setup
5. **Model Fitting** - Convergence diagnostics and numerical details
6. **Results & Interpretation** - Coefficient estimates with practical meaning
7. **Model Diagnostics** - Residual analysis and goodness-of-fit
8. **Conclusions** - Key findings and business implications

---

## Getting Started

### Installation

```bash
# Install Aurora-GLM from source
pip install -e /path/to/Aurora-GLM

# Or from PyPI (when available)
pip install aurora-glm

# Install Jupyter for running notebooks
pip install jupyter jupyterlab
```

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to examples/06_case_studies and open any notebook
# Example: 01_insurance_pricing.ipynb
```

### Quick Start Path

If you're new to Aurora-GLM, follow this learning sequence:

1. **Case 01**: Insurance Pricing (Gamma GLM)
   - Classic regression with GLM
   - Introduction to model families and link functions

2. **Case 02**: Air Quality (GAM)
   - Smooth terms and additive models
   - Non-linear relationship discovery

3. **Case 04**: Sleep Study (GAMM)
   - Random effects for clustered data
   - Mixed models with temporal structure

4. **Case 06**: Clinical Trial Longitudinal (GAMM with AR1)
   - Temporal covariance structures
   - Longitudinal data analysis

---

## Case Studies by Domain

### Finance & Insurance (5 cases)

**01. Insurance Pricing with Gamma GLM**
- Model: Gamma family, log link
- Focus: Cost prediction for medical insurance
- Techniques: GLM, log link, risk segmentation
- Key Result: Smokers pay 3-4× more than non-smokers

**09. French Motor Claims (Poisson/Negative Binomial)**
- Model: Poisson GLM, Negative Binomial GLM
- Focus: Insurance claim frequency modeling
- Techniques: Overdispersion detection, count data
- Key Result: Negative Binomial reduces AIC by 200+ points

**10. Medical Insurance Costs (Gamma GLM)**
- Model: Gamma GLM with interactions
- Focus: Healthcare cost prediction
- Techniques: Feature engineering, interaction terms
- Key Result: Age × BMI interaction significant

**12. Telco Customer Churn (Binomial GLM)**
- Model: Binomial GLM with logit link
- Focus: Customer retention and churn prediction
- Techniques: Logistic regression, odds ratios, ROC analysis
- Key Result: Month-to-month customers churn 5× more

**16. E-Commerce Conversion Rate Optimization**
- Model: Binomial GAM with logit link
- Focus: Dynamic conversion rate modeling
- Techniques: Smooth terms for price effect, additive model
- Key Result: Non-linear relationship between price and conversion

---

### Environmental & Ecological (4 cases)

**02. Air Quality Assessment (GAM)**
- Model: Gaussian GAM
- Focus: PM2.5 prediction from meteorological variables
- Techniques: Multiple smooth terms, non-parametric regression
- Key Result: Non-linear relationships between weather and pollution

**03. Species Distribution Modeling**
- Model: Binomial GAM (presence/absence)
- Focus: Spatial prediction of species presence
- Techniques: Thin plate splines, spatial smoothing
- Key Result: Habitat predictors identified via smooth terms

**11. Bike Sharing Demand (Gaussian GAM)**
- Model: Gaussian GAM with multiple smooths
- Focus: Time-series prediction of bike share usage
- Techniques: Temporal patterns, additive decomposition
- Key Result: Strong day-of-week and temperature effects

**15. US Traffic Accidents Severity (Ordinal GAMM)**
- Model: Mixed models for ordinal outcomes
- Focus: Road safety and accident severity factors
- Techniques: Random intercepts by location
- Key Result: Location and weather interact strongly

---

### Healthcare & Medicine (5 cases)

**04. Sleep Study with Random Effects (GAMM)**
- Model: Linear mixed model (GAMM)
- Focus: Sleep efficiency across individuals
- Techniques: Random intercepts by subject, random slopes
- Key Result: Significant individual variation in response

**05. Clinical Trial (Gaussian GLM)**
- Model: Gaussian GLM with group structure
- Focus: Treatment effect estimation
- Techniques: Contrast coding, confidence intervals
- Key Result: Treatment shows significant improvement

**06. Clinical Trial Longitudinal (GAMM with AR1)**
- Model: GAMM with AR1 temporal covariance
- Focus: Repeated measurements over time
- Techniques: Autoregressive covariance structure
- Key Result: AR1 improves fit vs independence assumption

**07. Psychometric Measurement (GAMM, crossed effects)**
- Model: Linear mixed model with crossed random effects
- Focus: Item response theory and measurement
- Techniques: Crossed random effects (items × subjects)
- Key Result: Item and subject effects orthogonal

**13. Breast Cancer Survival (Survival analysis)**
- Model: Cox proportional hazards (Gaussian approximation)
- Focus: Time-to-event prediction
- Techniques: Censoring handling, survival curves
- Key Result: Tumor characteristics strong predictors

---

### Education (1 case)

**08. Educational Multilevel Data (GAMM, nested effects)**
- Model: Nested random effects GAMM
- Focus: Student achievement across schools and districts
- Techniques: Nested random effects (students within schools within districts)
- Key Result: District explains 30% of variance

---

### Business & Retail (2 cases)

**17. Restaurant Health Inspection (Beta GLM)**
- Model: Beta GLM for [0,1] bounded outcomes
- Focus: Restaurant health scores and violations
- Techniques: Beta family for proportions, feature importance
- Key Result: Inspection frequency inversely related to violations

**Data**: Various industry datasets (see details below)

---

## Data Files

The `data/` directory contains supporting datasets:

- `medical_insurance.csv` - Insurance costs (Case 01, 10)
- `air_quality.csv` - Air pollution measurements (Case 02)
- `species_distribution.csv` - Species presence/habitat (Case 03)
- `sleep_study.csv` - Sleep efficiency across subjects (Case 04)
- `clinical_trial.csv` - Treatment response data (Case 05, 06)
- `psychometric_data.csv` - Item response data (Case 07)
- `education_multilevel.csv` - Student achievement (Case 08)
- `french_motor_claims.csv` - Insurance claims (Case 09)
- `telco_churn.csv` - Customer churn data (Case 12)
- `bike_sharing.csv` - Bike rental demand (Case 11)
- `breast_cancer.csv` - Survival data (Case 13)
- `wind_power.csv` - Renewable energy (Case 14)
- `traffic_accidents.csv` - Road safety (Case 15)
- `ecommerce_conversions.csv` - Conversion rates (Case 16)
- `restaurant_health.csv` - Health inspection scores (Case 17)

---

## Model Types by Case

### GLM (Generalized Linear Models)

- **Case 01**: Gamma GLM (insurance costs)
- **Case 05**: Gaussian GLM (clinical trial)
- **Case 09**: Poisson, Negative Binomial (count data)
- **Case 10**: Gamma GLM (medical costs)
- **Case 12**: Binomial GLM (logistic regression)
- **Case 17**: Beta GLM (proportion outcomes)

### GAM (Generalized Additive Models)

- **Case 02**: Gaussian GAM (air quality)
- **Case 03**: Binomial GAM (species distribution)
- **Case 11**: Gaussian GAM (bike sharing)
- **Case 16**: Binomial GAM (e-commerce conversion)

### GAMM (Generalized Additive Mixed Models)

- **Case 04**: Linear mixed model (sleep study)
- **Case 06**: GAMM with AR1 covariance (clinical longitudinal)
- **Case 07**: GAMM with crossed effects (psychometric)
- **Case 08**: GAMM with nested effects (education)
- **Case 13**: Cox model approximation (survival)
- **Case 15**: Ordinal GAMM (accident severity)

---

## Key Features Demonstrated

### Model Families
- Gaussian, Binomial, Poisson, Gamma, Negative Binomial, Beta

### Link Functions
- Identity, Log, Logit, Probit, Log-log, Inverse

### Smooth Terms
- B-splines, Natural cubic splines, Thin plate splines, Tensor products

### Random Effects
- Random intercepts, Random slopes, Nested effects, Crossed effects

### Covariance Structures
- Identity, Unstructured, Diagonal, AR1, Compound symmetry, Exponential, Matérn, Toeplitz

### Backend Support
- NumPy, PyTorch, JAX (all notebooks demonstrate multi-backend usage)

---

## Workflow Template

Each notebook follows this workflow:

```python
# 1. Load and explore data
import pandas as pd
data = pd.read_csv('data/example.csv')

# 2. Create formula specification
formula = "y ~ s(x1) + x2 + (1 | group)"

# 3. Fit model
from aurora.models.gamm import fit_gamm
result = fit_gamm(y=data['y'], X=data[['x1', 'x2']],
                  random_effects=[RandomEffect(grouping='group')],
                  groups_data={'group': data['group']})

# 4. Interpret results
print(result.summary())
result.plot_smooth_terms()

# 5. Diagnostics
result.plot_residuals()
result.plot_qq()
```

---

## Contributing

To add a new case study:

1. Create a Jupyter notebook following the 8-part structure
2. Include publication-quality visualizations
3. Document all assumptions and limitations
4. Compare alternative models when appropriate
5. Add to this README with brief description
6. Test across NumPy, PyTorch, JAX backends

---

## Troubleshooting

### Jupyter kernel not found
```bash
python -m ipykernel install --user
```

### Import errors for Aurora-GLM
```bash
# Verify installation
python -c "import aurora; print(aurora.__version__)"

# Reinstall if needed
pip install -e /path/to/Aurora-GLM
```

### GPU/backend issues
- PyTorch: `pip install torch`
- JAX: `pip install jax jaxlib`
- NumPy (default): included with pandas

### Slow notebook execution
- Reduce data size for exploration
- Use `use_sparse=True` for large datasets
- Consider desktop GPU for PyTorch/JAX backend

---

## References

### Aurora-GLM Documentation
- GitHub: https://github.com/anthropics/aurora-glm
- Main module: `import aurora`

### Background Reading
- GLM Theory: McCullagh & Nelder (1989)
- GAM Theory: Wood (2017) "Generalized Additive Models: An Introduction with R"
- GAMM Theory: Bates et al. (2015) lme4 paper
- Splines: de Boor (2001) "A Practical Guide to Splines"

---

## Version History

**0.6.1** (December 2024)
- All 17 case studies implemented with 8-part structure
- Enhanced sparse matrix support (6-8× memory reduction)
- Temporal covariance structures (AR1, compound symmetry, etc.)
- Multi-backend validation (NumPy, PyTorch, JAX)
- Comprehensive diagnostics and plotting

---

## Citation

If you use these examples in your research, please cite Aurora-GLM:

```bibtex
@software{aurora_glm2025,
  title = {Aurora-GLM: Generalized Linear and Additive Models},
  author = {Arias, Lucy E.},
  year = {2025},
  url = {https://github.com/anthropics/aurora-glm},
  version = {0.6.1}
}
```

---

**Questions?** Open an issue on GitHub or consult the case study READMEs for domain-specific guidance.

**Last Updated**: December 6, 2024
**Maintainer**: Lucy E. Arias
