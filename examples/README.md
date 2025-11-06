# Aurora-GLM Examples

Comprehensive collection of Jupyter notebooks demonstrating Aurora-GLM capabilities from beginner to advanced topics.

## 📚 Structure

```
examples/
├── 00_quickstart/          # ⭐ Start here! (3 notebooks, ~20 min total)
├── 01_regression/          # Linear models and extensions (3 notebooks)
├── 02_classification/      # Binary and multi-class (3 notebooks)
├── 03_count_data/          # Poisson, NB, zero-inflation (3 notebooks)
├── 04_longitudinal/        # Repeated measures, GAMM (3 notebooks)
├── 05_advanced_topics/     # Advanced techniques (4 notebooks)
├── 06_case_studies/        # Real-world applications (5 notebooks)
├── utils/                  # Data loading utilities
└── data/                   # Cached datasets (auto-downloaded)
```

**Total:** 24 notebooks covering GLM, GAM, and GAMM

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to examples directory
cd examples/

# Install dependencies
pip install -r ../requirements.txt

# Launch Jupyter
jupyter notebook
```

### Your First Model (5 minutes)

Start with [`00_quickstart/01_first_glm.ipynb`](00_quickstart/01_first_glm.ipynb):
- Generate synthetic data
- Fit Gaussian GLM (linear regression)
- Interpret coefficients
- Make predictions
- Visualize results

---

## 📖 Learning Path

### 🌟 Beginner (Start Here!)

**00_quickstart/** - Essential introductions
1. **01_first_glm.ipynb** (5 min) - Your first linear model
2. **02_first_gam.ipynb** (7 min) - Add smooth non-linear terms
3. **03_first_gamm.ipynb** (10 min) - Handle hierarchical data

**Outcome:** Understand GLM → GAM → GAMM progression

---

### ⭐⭐ Intermediate

**01_regression/** - Deep dive into continuous outcomes
1. **01_linear_gaussian_glm.ipynb** (20 min) - Complete linear regression workflow
   - Diagnostics, inference, model comparison
2. **02_weighted_regression.ipynb** (15 min) - Handle heteroscedasticity
   - WLS, variance modeling
3. **03_polynomial_vs_gam.ipynb** (18 min) - Compare polynomial and smooth approaches
   - Model selection, flexibility vs stability

**02_classification/** - Binary and multi-class outcomes
1. **01_logistic_regression.ipynb** (20 min) - Binary classification
   - ROC curves, confusion matrices, calibration
2. **02_multinomial_classification.ipynb** (15 min) - Multi-class problems
   - One-vs-rest, softmax
3. **03_gam_classification.ipynb** (15 min) - Non-linear decision boundaries
   - Smooth classification surfaces

**03_count_data/** - Modeling counts (0, 1, 2, ...)
1. **01_poisson_regression.ipynb** (15 min) - Basic count models
   - Rate models, offset, overdispersion checks
2. **02_negative_binomial.ipynb** (12 min) - Handle overdispersion
   - Quasi-Poisson, NB GLM
3. **03_zero_inflated.ipynb** (12 min) - Excess zeros
   - Two-part models, hurdle models

---

### ⭐⭐⭐ Advanced

**04_longitudinal/** - Repeated measurements
1. **01_repeated_measures.ipynb** (15 min) - Random intercepts
   - Within-subject correlation
2. **02_random_slopes.ipynb** (10 min) - Subject-specific trends
   - Growth curves
3. **03_nested_random_effects.ipynb** (10 min) - Multi-level structure
   - Students in classes in schools

**05_advanced_topics/** - Specialized techniques
1. **01_tensor_smooths.ipynb** (12 min) - 2D smooth surfaces
   - Interaction smooths
2. **02_robust_estimation.ipynb** (10 min) - Handle outliers
   - Robust standard errors
3. **03_variable_selection.ipynb** (12 min) - Choose predictors
   - AIC/BIC, stepwise
4. **04_interaction_terms.ipynb** (12 min) - Effect modification
   - Stratification, moderation

**06_case_studies/** - Real-world applications
1. **01_insurance_pricing.ipynb** (20 min) - Gamma GLM for insurance claims
   - Real data: medical costs
2. **02_air_quality_gam.ipynb** (18 min) - Environmental modeling
   - Real data: NYC air quality 1973
3. **03_species_distribution.ipynb** (15 min) - Ecological counts
   - Synthetic: species abundance
4. **04_sleep_study_gamm.ipynb** (18 min) - Classic longitudinal study
   - Real data: sleep deprivation (lme4)
5. **05_clinical_trial.ipynb** (18 min) - Binary nested outcomes
   - Synthetic: patients within clinics

---

## 📊 By Model Type

### GLM (Generalized Linear Models)
- **Gaussian**: `01_regression/01_linear_gaussian_glm.ipynb`
- **Binomial**: `02_classification/01_logistic_regression.ipynb`
- **Poisson**: `03_count_data/01_poisson_regression.ipynb`
- **Gamma**: `06_case_studies/01_insurance_pricing.ipynb`

### GAM (Generalized Additive Models)
- **Basics**: `00_quickstart/02_first_gam.ipynb`
- **vs Polynomial**: `01_regression/03_polynomial_vs_gam.ipynb`
- **Classification**: `02_classification/03_gam_classification.ipynb`
- **Real data**: `06_case_studies/02_air_quality_gam.ipynb`

### GAMM (Generalized Additive Mixed Models)
- **Basics**: `00_quickstart/03_first_gamm.ipynb`
- **Repeated measures**: `04_longitudinal/01_repeated_measures.ipynb`
- **Random slopes**: `04_longitudinal/02_random_slopes.ipynb`
- **Real data**: `06_case_studies/04_sleep_study_gamm.ipynb`

---

## 🎯 By Application

### Business Analytics
- Customer churn: `02_classification/01_logistic_regression.ipynb`
- Insurance pricing: `06_case_studies/01_insurance_pricing.ipynb`
- Website visits: `03_count_data/01_poisson_regression.ipynb`

### Healthcare
- Clinical trials: `06_case_studies/05_clinical_trial.ipynb`
- Longitudinal outcomes: `04_longitudinal/01_repeated_measures.ipynb`

### Environmental Science
- Air quality: `06_case_studies/02_air_quality_gam.ipynb`
- Species distribution: `06_case_studies/03_species_distribution.ipynb`

### Psychology/Social Science
- Sleep study: `06_case_studies/04_sleep_study_gamm.ipynb`
- Hierarchical data: `00_quickstart/03_first_gamm.ipynb`

---

## 💾 Data Management

### Automatic Downloads

Case study notebooks automatically download real datasets on first run:
- **Insurance data** (CC0): Medical costs from ML-with-R-datasets
- **Sleep study** (GPL-2): Classic lme4 dataset
- **Air quality** (GPL-3): NYC environmental data

Data is cached in `data/` directory (git-ignored).

### Manual Download

If automatic download fails:

```bash
# From examples/ directory
cd data/

# Insurance
curl -o insurance.csv https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv

# Sleep study
curl -o sleepstudy.csv https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/lme4/sleepstudy.csv

# Air quality
curl -o airquality.csv https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv
```

See [`data/README.md`](data/README.md) for details.

---

## 🛠️ Notebook Features

### Consistent Structure
Each notebook follows:
1. **Overview** - What you'll learn
2. **Setup** - Imports and configuration
3. **Data** - Generation or loading
4. **EDA** - Exploratory visualizations
5. **Modeling** - Fit and interpret
6. **Diagnostics** - Check assumptions
7. **Comparison** - Alternative approaches
8. **Summary** - Key takeaways and next steps

### Self-Contained
- All notebooks run independently
- Synthetic data generated inline (no downloads needed for basic examples)
- Clear outputs and visualizations

### Progressive Complexity
- ⭐ Beginner: Minimal code, maximum explanation
- ⭐⭐ Intermediate: Complete workflows
- ⭐⭐⭐ Advanced: Specialized techniques

---

## 📝 Contributing Examples

Want to add a notebook? Follow the template:

```python
# Standard structure
1. Title with metadata (duration, level, topics)
2. Imports (numpy, pandas, matplotlib, aurora.*)
3. Data generation/loading
4. Visualization (seaborn style)
5. Model fitting (with convergence checks)
6. Interpretation (coefficients, metrics)
7. Diagnostics (residuals, etc.)
8. Summary with next steps
```

See existing notebooks as examples.

---

## 🔗 External Resources

### Learn More
- [Aurora-GLM Documentation](https://github.com/Matcraft94/Aurora-GLM)
- [GLM Theory (McCullagh & Nelder)](https://www.routledge.com/Generalized-Linear-Models/McCullagh-Nelder/p/book/9780412317606)
- [GAM with R (Wood)](https://www.routledge.com/Generalized-Additive-Models-An-Introduction-with-R-Second-Edition/Wood/p/book/9781498728331)
- [Mixed Models (Gelman & Hill)](http://www.stat.columbia.edu/~gelman/arm/)

### Datasets
- [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)
- [lme4 datasets](https://github.com/lme4/lme4)

---

## ⚖️ License

- **Example code**: MIT License (same as Aurora-GLM)
- **Datasets**: Follow their respective licenses (see `data/README.md`)
- **Notebooks**: MIT License

---

## 🐛 Issues

Found a bug or have suggestions?

- [Open an issue](https://github.com/Matcraft94/Aurora-GLM/issues)
- Include notebook name and error message
- Provide reproducible example

---

## ✨ Quick Reference

| Task | Notebook | Duration |
|------|----------|----------|
| First model | `00_quickstart/01_first_glm.ipynb` | 5 min |
| Binary classification | `02_classification/01_logistic_regression.ipynb` | 20 min |
| Count data | `03_count_data/01_poisson_regression.ipynb` | 15 min |
| Hierarchical data | `00_quickstart/03_first_gamm.ipynb` | 10 min |
| Non-linear relationships | `00_quickstart/02_first_gam.ipynb` | 7 min |
| Real-world example | `06_case_studies/01_insurance_pricing.ipynb` | 20 min |

**Total learning time:** ~6 hours for complete suite

---

**Author:** Lucy E. Arias
**Last Updated:** 2025-11-04
**Version:** 1.0.0

Happy modeling! 🎉
