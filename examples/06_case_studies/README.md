# Aurora-GLM Case Studies

Comprehensive, production-ready examples demonstrating Aurora-GLM capabilities across diverse domains.

**Total Case Studies**: 17
**Aurora-GLM Version**: 0.6.1
**Last Updated**: December 2024

---

## Quick Start

Each case study is a self-contained Jupyter notebook demonstrating:
- **Model selection** and mathematical justification
- **Data exploration** with publication-quality visualizations
- **Rigorous fitting** with convergence diagnostics
- **Effect interpretation** for domain experts
- **Multi-backend support** (NumPy, PyTorch, JAX)

**New to Aurora-GLM?** Start with:
1. **Case 01** (Insurance Pricing) - Classic Gamma GLM
2. **Case 16** (E-Commerce Conversion) - Binomial GAM with full 8-part structure
3. **Case 06** (Clinical Trial Longitudinal) - GAMM with temporal correlation

---

## All Case Studies

### Finance & Insurance (5 cases)

#### 01. Insurance Pricing with Gamma GLM
**Model**: Gamma GLM with log link
**Dataset**: 1,338 medical insurance records
**Key Features**:
- Gamma family for positive, right-skewed costs
- Log link for multiplicative effects (rate ratios)
- Comparison of link functions (log vs identity)
- Risk segmentation by smoking status

**Research Questions**:
1. What factors drive medical insurance costs?
2. How much more do smokers pay?
3. Which link function is most appropriate?

**Key Finding**: Smokers pay 3-4× more than non-smokers; log link preferred

---

#### 09. French Motor Claims (Poisson/Negative Binomial)
**Model**: Poisson GLM → Negative Binomial GLM
**Dataset**: French motor insurance claims
**Key Features**:
- Poisson for count data (number of claims)
- Overdispersion detection and handling
- Negative Binomial for extra-Poisson variation
- Dispersion parameter estimation

**Research Questions**:
1. What predicts claim frequency?
2. Is there overdispersion in claims?
3. How much does NB improve over Poisson?

**Key Finding**: Significant overdispersion; NB reduces AIC by 200+ points

---

#### 10. Medical Insurance Costs (Gamma GLM)
**Model**: Gamma GLM with log link
**Dataset**: Medical insurance costs dataset
**Key Features**:
- Similar to Case 01 but different dataset
- Feature engineering for medical conditions
- Interaction terms (age × BMI)

**Research Questions**:
1. How do pre-existing conditions affect costs?
2. Are there interaction effects?

---

#### 12. Telco Customer Churn (Binomial GLM)
**Model**: Binomial GLM with logit link
**Dataset**: Telecom customer churn (7,043 customers)
**Key Features**:
- Logistic regression for binary outcome (churn: yes/no)
- Odds ratios for customer retention factors
- ROC curves and classification metrics
- Customer lifetime value calculations

**Research Questions**:
1. What drives customer churn?
2. Which customers are highest risk?
3. What is the ROI of retention programs?

**Key Finding**: Contract type and tenure are strongest predictors; month-to-month customers churn 5× more

---

#### 16. E-Commerce Conversion Rate Optimization ⭐ NEW
**Model**: Binomial GAM with logit link
**Dataset**: 10,000 user sessions (synthetic)
**Key Features**:
- **Full 8-part structure** (template for all case studies)
- Smooth price effects s(price) - inverted U-shape
- Session duration s(duration) - diminishing returns
- Optimal price point discovery
- A/B testing framework and ROI calculations

**Research Questions**:
1. What is the optimal price for max conversion?
2. How does session duration affect conversion non-linearly?
3. What is ROI of UX improvements?

**Key Finding**: Optimal price ~$35; first 5 minutes of session matter most (10:1 ROI)

**Structure**: Complete 8-part template:
1. Setup and Data Loading
2. Exploratory Data Analysis
3. **Mathematical Specification** (rigorous formulas)
4. Model Fitting (GLM vs GAM)
5. Residual Diagnostics
6. Effect Interpretation
7. Multi-Backend Performance
8. Conclusions

---

### Healthcare & Medicine (4 cases)

#### 05. Clinical Trial (GLM/GAMM)
**Model**: Gaussian GAMM
**Dataset**: Clinical trial with treatment groups
**Key Features**:
- Treatment effect estimation
- Random effects for subjects
- Repeated measures analysis

**Research Questions**:
1. Is the treatment effective?
2. Are there individual differences in response?

---

#### 06. Clinical Trial Longitudinal (GAMM with AR1) ⭐
**Model**: Gaussian GAMM with AR1 covariance
**Dataset**: 12-week RCT with repeated measures
**Key Features**:
- **Temporal correlation** (AR1 structure)
- Treatment-by-time interactions
- Random intercepts and slopes
- Compound symmetry vs AR1 comparison
- Effect size calculations (Cohen's d)

**Research Questions**:
1. Does treatment reduce depression scores?
2. Is there a treatment-time interaction?
3. How much variation between patients?

**Key Finding**: Treatment shows progressive improvement (differential trajectories); ICC ≈ 59%

---

#### 13. Breast Cancer Survival (Survival Analysis)
**Model**: Survival GLM
**Dataset**: Breast cancer survival data
**Key Features**:
- Time-to-event analysis
- Hazard ratios
- Kaplan-Meier curves

**Research Questions**:
1. What factors predict survival?
2. How do treatment types compare?

---

#### 17. Restaurant Health Inspection Scores ⭐ NEW
**Model**: Beta Regression GAM
**Dataset**: 500 restaurants (synthetic)
**Key Features**:
- **Beta family** for continuous proportions (0,1)
- **Full 8-part structure** with rigorous math
- 2D spatial smooths te(lat, lon)
- Non-linear age effects (optimal ~10 years)
- Temporal trends (3-year improvement)
- Precision parameter (φ) estimation

**Research Questions**:
1. What predicts health inspection scores?
2. Are there spatial clusters (hot spots)?
3. How do scores change with restaurant age?

**Key Finding**: Previous violations strongest predictor; downtown scores 2-3 points lower; Beta superior to Gaussian

---

### Environmental & Ecology (2 cases)

#### 02. Air Quality Modeling (GAM)
**Model**: Gaussian GAM
**Dataset**: Daily air quality measurements
**Key Features**:
- Smooth temporal trends s(time)
- Non-linear temperature effects s(temp)
- Seasonal patterns (cyclic smooths)
- GCV vs REML smoothing selection

**Research Questions**:
1. How does air quality vary over time?
2. What is the effect of temperature?
3. Are there seasonal patterns?

**Key Finding**: Strong seasonal effect; temperature has non-linear impact

---

#### 03. Species Distribution Modeling (Binomial GAM)
**Model**: Binomial GAM with spatial smooths
**Dataset**: 500 species occurrence sites
**Key Features**:
- 2D spatial smooths te(lat, lon)
- Presence-absence data (binomial)
- Environmental covariates
- Habitat suitability mapping

**Research Questions**:
1. Where is the species most likely to occur?
2. What environmental factors predict presence?
3. Which areas need conservation priority?

**Key Finding**: Elevation and temperature are key predictors; identified 3 conservation priority zones

---

### Education & Psychology (3 cases)

#### 04. Sleep Study (GAMM) ⭐
**Model**: Gaussian GAMM
**Dataset**: Sleep deprivation study (repeated measures)
**Key Features**:
- Random intercepts and slopes
- Reaction time modeling
- Days of sleep deprivation effects

**Research Questions**:
1. How does sleep deprivation affect reaction time?
2. Are there individual differences?

**Key Finding**: Linear increase in reaction time (~10ms/day); large individual variability

---

#### 07. Psychometric Crossed Effects (GAMM)
**Model**: GAMM with crossed random effects
**Dataset**: Psychometric test data
**Key Features**:
- Crossed random effects (subjects × items)
- Partial pooling
- Variance component estimation

**Research Questions**:
1. How much variance is due to subjects vs items?
2. Are there item difficulty differences?

---

#### 08. Educational Multilevel (GAMM)
**Model**: Multilevel GAMM
**Dataset**: Student test scores (nested design)
**Key Features**:
- Hierarchical structure (students within schools)
- Between-school and within-school effects
- ICC calculation

**Research Questions**:
1. How much variance is between schools?
2. What school-level factors matter?

---

### Urban Planning & Energy (2 cases)

#### 11. Bike Sharing Demand (Poisson GAM) ⭐
**Model**: Poisson GAM
**Dataset**: 17,379 hourly bike rentals
**Key Features**:
- Smooth hourly patterns s(hour)
- Non-linear temperature effects s(temp)
- Count outcome (number of rentals)
- Cyclic smooths for temporal data

**Research Questions**:
1. What are the hourly demand patterns?
2. How does temperature affect ridership?
3. How much do smooths improve over linear?

**Key Finding**: Clear bimodal pattern (8am, 5pm peaks); GAM captures non-linearities GLM misses

---

#### 14. Wind Power Forecasting (Gaussian GAM)
**Model**: Gaussian GAM
**Dataset**: Wind power generation data
**Key Features**:
- Smooth wind speed effects
- Temporal trends
- Capacity factor modeling

**Research Questions**:
1. How does power output vary with wind speed?
2. Can we forecast generation?

---

### Public Safety (1 case)

#### 15. US Accidents Severity (Ordinal/Multinomial)
**Model**: Ordinal or Multinomial GLM
**Dataset**: US traffic accidents
**Key Features**:
- Ordered outcomes (severity levels)
- Proportional odds assumption
- Geographic and temporal factors

**Research Questions**:
1. What factors predict accident severity?
2. Are there spatial hot spots?

---

## Case Study Structure

### Existing Case Studies (01-15)
These follow a **flexible structure** with 4-6 main sections:
- Overview and research context
- Data loading and exploration
- Model fitting and comparison
- Diagnostics and interpretation
- Conclusions and recommendations

**Strengths**: Practical, accessible, domain-focused
**Use when**: Learning specific techniques, exploring domains

---

### New Case Studies (16-17) ⭐
These follow a **rigorous 8-part structure**:

#### Part 1: Setup and Data Loading
- Environment configuration
- Backend availability (NumPy, PyTorch, JAX)
- Data generation/loading with documentation
- Feature engineering with justification
- Summary statistics

#### Part 2: Exploratory Data Analysis
- Multi-panel visualizations (6-9 subplots)
- Distribution analysis
- Bivariate relationships
- Temporal/spatial patterns
- Key observations that motivate modeling

#### Part 3: Mathematical Specification ⭐ MOST IMPORTANT
**Rigorous mathematical justification**:
- **Why this distribution family?** (with statistical properties)
- **GLM formulation**: Link function, linear predictor, likelihood
- **GAM formulation** (if applicable): Smooth representation, basis functions
- **Penalized estimation**: Likelihood + smoothness penalty
- **Effective degrees of freedom (EDF)**: Interpretation
- **Model comparison metrics**: AIC, deviance, LRT
- **Domain-specific metrics**: Business/scientific interpretations

**Example content**:
```
### Why Gamma Distribution for Insurance Costs?

Medical charges are:
1. **Positive**: No zero or negative values → Gamma support (0, ∞)
2. **Right-skewed**: Mean > Median → Gamma shape parameter α < ...
3. **Heteroscedastic**: Var(Y) ∝ E[Y]² → Gamma variance = μ²/φ
4. **Multiplicative**: Effects as percentages → Log link natural

**Gamma GLM**:
$$
Y_i \\sim \\text{Gamma}(\\mu_i, \\phi)
$$
$$
\\log(\\mu_i) = \\beta_0 + \\beta_1 x_{1i} + ... + \\beta_p x_{pi}
$$

where:
- μᵢ is the mean charge for individual i
- φ is the shape parameter (constant across i)
- Variance: Var[Yᵢ] = μᵢ²/φ (increases with mean)
```

#### Part 4: Model Fitting
- GLM baseline (linear effects)
- GAM with smooth terms (if applicable)
- Model comparison (AIC/BIC/LRT)
- Convergence diagnostics
- Coefficient interpretation (on link scale and response scale)

#### Part 5: Residual Diagnostics
- Residuals vs fitted
- Q-Q plots (normality check)
- Scale-location (heteroscedasticity)
- Residuals vs predictors (check for patterns)
- Calibration plots
- Outlier detection and investigation

#### Part 6: Effect Interpretation
- Partial effect plots (GAM smooths)
- Optimal points identification
- Categorical comparisons (bar plots)
- Spatial visualizations (if applicable)
- Domain-specific insights extraction
- Business/policy recommendations

#### Part 7: Multi-Backend Performance
- NumPy benchmark (baseline)
- PyTorch CPU (if available)
- PyTorch GPU (if available)
- JAX (if available)
- Speedup comparisons
- Numerical equivalence checks (AIC should match)

#### Part 8: Conclusions
- **Research questions answered** (one by one)
- **Model performance summary** (R², RMSE, AIC)
- **Domain recommendations** (immediate actions + long-term strategy)
- **Statistical insights** (when to use this model)
- **Aurora-GLM capabilities highlighted**
- **Limitations and future work**
- **References** (academic papers, textbooks)

**Strengths**: Publication-ready, rigorous, complete mathematical documentation
**Use when**: Teaching, research, production deployment, academic writing

---

## Model Families Covered

| Family | Distribution | Link Functions | Use Cases | Case Studies |
|--------|-------------|----------------|-----------|--------------|
| **Gaussian** | Normal | Identity, log | Continuous unbounded | 02, 04, 05, 06, 07, 08, 14 |
| **Gamma** | Gamma | Log, identity, inverse | Positive continuous, right-skewed | **01**, 10 |
| **Poisson** | Poisson | Log | Count data (no overdispersion) | 09, **11** |
| **Negative Binomial** | NB | Log | Overdispersed counts | **09** |
| **Binomial** | Binomial | Logit, probit | Binary, proportions | 03, 12, **16** |
| **Beta** | Beta | Logit | Continuous proportions (0,1) | **17** |

**Bold** = Comprehensive example with full 8-part structure

---

## Aurora-GLM Features Demonstrated

### Core GLM Capabilities
- [x] **Distribution families**: Gaussian, Gamma, Poisson, Binomial, Beta, Negative Binomial
- [x] **Link functions**: Identity, log, logit, inverse, probit
- [x] **IRLS algorithm** for GLM fitting (fast, stable convergence)
- [x] **Model comparison**: AIC, BIC, deviance, likelihood ratio tests
- [x] **Coefficient inference**: Standard errors, z-scores, p-values
- [x] **Predictions**: Link scale and response scale

### GAM Enhancements
- [x] **Univariate smooths**: s(x) for non-linear effects
- [x] **2D tensor products**: te(x, y) for spatial data
- [x] **Cyclic smooths**: For temporal periodicity (hour, month)
- [x] **Smoothing parameter selection**: GCV, REML
- [x] **Effective degrees of freedom (EDF)**: Quantify non-linearity
- [x] **Partial effect plots**: Visualize smooth functions

### GAMM (Mixed Models)
- [x] **Random intercepts**: (1 | group)
- [x] **Random slopes**: (1 + x | group)
- [x] **Crossed effects**: (1 | subject) + (1 | item)
- [x] **Nested effects**: Students within schools
- [x] **Temporal correlation**: AR1, compound symmetry
- [x] **Variance components**: Between-group and within-group
- [x] **ICC calculation**: Intraclass correlation

### Multi-Backend Support
- [x] **NumPy**: CPU-based, standard scientific Python
- [x] **PyTorch**: GPU acceleration, automatic differentiation
- [x] **JAX**: JIT compilation, XLA optimization
- [x] **Numerical equivalence**: All backends produce identical results

### Advanced Features
- [x] **Sparse matrices**: For large-scale GAM/GAMM (6-8× memory reduction)
- [x] **Robust estimation**: Outlier detection, robust covariance
- [x] **Cross-validation**: K-fold for model selection
- [x] **Diagnostics**: Comprehensive residual analysis

---

## How to Use These Case Studies

### For Learning Aurora-GLM
**Recommended path**:
1. Start with **Case 01** (Insurance Pricing) - classic GLM
2. Move to **Case 16** (E-Commerce) - see full 8-part structure
3. Try **Case 06** (Clinical Trial) - mixed models with temporal correlation
4. Explore **Case 17** (Beta Regression) - advanced distribution family

**What you'll learn**:
- How to choose appropriate distribution families
- Link function selection and interpretation
- When to use GLM vs GAM vs GAMM
- Residual diagnostics and model checking
- Effect interpretation for stakeholders

### For Teaching
**Classroom use**:
- **Part 3** (Mathematical Specification) provides lecture-ready content
- **Parts 2 and 6** have publication-quality figures
- **Part 8** includes discussion questions
- Code is well-commented for self-study

**Assignment ideas**:
- Modify data generation for different scenarios
- Compare alternative link functions
- Add interaction terms
- Implement cross-validation

### For Research
**Publication preparation**:
- Part 3 provides mathematical rigor for Methods section
- Figures in Parts 2, 5, 6 are publication-ready
- Part 8 Conclusions template for Discussion
- References included

**Extensions**:
- Swap distribution families for your data type
- Add domain-specific covariates
- Implement validation against other packages (statsmodels, R's mgcv)
- Create custom link functions or families

### For Production Deployment
**Industry applications**:
- Part 7 benchmarks inform infrastructure choices
- Part 8 recommendations guide implementation
- Code is modular (extract functions for APIs)
- Models can be exported for real-time scoring

**Deployment checklist**:
1. Choose backend (NumPy for simplicity, PyTorch for GPU)
2. Optimize smoothing parameters on training data
3. Validate on hold-out test set
4. Monitor EDF values for drift detection
5. Retrain periodically with new data

---

## Dependencies

### Required
```bash
pip install numpy pandas matplotlib seaborn scipy
pip install aurora-glm>=0.6.1
```

### Optional (for multi-backend)
```bash
pip install torch>=1.10      # PyTorch backend
pip install jax>=0.3          # JAX backend
pip install requests          # Data downloading (Cases 01, 10, 11, 12)
```

### For Jupyter Notebooks
```bash
pip install jupyter ipykernel
```

---

## Running the Case Studies

### Option 1: Jupyter Notebook (Recommended)
```bash
cd examples/06_case_studies
jupyter notebook
# Open any .ipynb file and run cells sequentially
```

### Option 2: Convert to Python Script
```bash
jupyter nbconvert --to python 01_insurance_pricing.ipynb
python 01_insurance_pricing.py
```

### Option 3: Run in Google Colab
1. Upload `.ipynb` file to Google Drive
2. Right-click → Open with → Google Colaboratory
3. Install Aurora-GLM: `!pip install aurora-glm`
4. Run cells

---

## Citation

If you use these case studies in research or teaching:

```bibtex
@software{aurora_glm_cases,
  title={Aurora-GLM Case Studies: Comprehensive Examples for GLM, GAM, and GAMM},
  author={Aurora-GLM Development Team},
  year={2024},
  url={https://github.com/yourusername/Aurora-GLM},
  note={Version 0.6.1, 17 case studies across 7 domains}
}
```

---

## Contributing

To add new case studies:

1. **Follow the 8-part structure** (see Cases 16-17 as templates)
2. **Include rigorous mathematics** in Part 3
3. **Provide synthetic data** with known ground truth OR public datasets
4. **Add multi-backend benchmarks** in Part 7
5. **Write actionable conclusions** in Part 8
6. **Test on NumPy/PyTorch/JAX** (ensure numerical equivalence)

**Submission**:
- Create pull request with notebook + README entry
- Include sample outputs (screenshots)
- Document any new dependencies

---

## Roadmap

### Planned Case Studies (v0.7+)
- [ ] **18. Manufacturing Defect Prediction** (Negative Binomial GAMM with AR1)
- [ ] **19. Customer Lifetime Value** (Tweedie GAM for zero-inflated continuous)
- [ ] **20. Enzyme Kinetics** (Inverse Gaussian GAM, scientific computing)
- [ ] **21. Real Estate Pricing** (Gaussian GAM with thin plate splines)
- [ ] **22. Credit Default Risk** (Binomial GAM with interactions)
- [ ] **23. Hospital Readmissions** (Poisson GAMM with random effects)

### Documentation Improvements
- [ ] Interactive HTML version of all case studies
- [ ] Video tutorials for Cases 01, 16, 17
- [ ] Comparison tables (Aurora vs statsmodels vs R's mgcv)
- [ ] Cheat sheet for distribution family selection

---

## Support

**Questions?**
- GitHub Issues: https://github.com/anthropics/Aurora-GLM/issues
- Documentation: See `CLAUDE.md` in repository root
- Examples: This directory (`examples/06_case_studies/`)

**Bug Reports**:
- Include case study number and Aurora-GLM version
- Provide minimal reproducible example
- Share error messages and traceback

---

## License

These case studies are released under the MIT License (same as Aurora-GLM).

---

**Last Updated**: December 2024
**Aurora-GLM Version**: 0.6.1
**Case Studies**: 17 (2 with full 8-part structure)
**Lines of Code**: ~15,000 (across all notebooks)
