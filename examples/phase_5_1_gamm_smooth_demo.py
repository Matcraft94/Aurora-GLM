"""Demo: Non-Gaussian GAMM with Smooth Terms (Phase 5.1).

This example demonstrates the new Phase 5.1 functionality:
- Smooth functions in non-Gaussian GAMM via PQL
- Formula interface with s() terms
- Multiple smooth terms + random effects
- Custom smoothing parameters

Author: Aurora-GLM Team
Date: 2025
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aurora.models.gamm import fit_gamm

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("Phase 5.1 Demo: Non-Gaussian GAMM with Smooth Terms")
print("=" * 70)

# =============================================================================
# Example 1: Poisson GAMM with Single Smooth Term
# =============================================================================
print("\n" + "=" * 70)
print("Example 1: Poisson Count Data with Smooth Time Trend")
print("=" * 70)

# Generate data: hospital admissions over time with seasonal pattern
n_hospitals = 15
n_weeks = 52
n_years = 4
n = n_hospitals * n_weeks * n_years

# Time variable (weeks)
time = np.tile(np.arange(n_weeks), n_hospitals * n_years) / n_weeks

# Hospital IDs
hospital = np.repeat(np.arange(n_hospitals), n_weeks * n_years)

# True seasonal pattern (smooth function)
f_seasonal = 20 + 10 * np.sin(2 * np.pi * time)

# Hospital random effects (some hospitals have higher baseline)
hospital_effects = np.random.randn(n_hospitals) * 0.3
b_hospital = hospital_effects[hospital]

# Linear predictor
eta = np.log(f_seasonal) + b_hospital

# Generate Poisson counts
admissions = np.random.poisson(np.exp(eta))

# Create DataFrame
data_poisson = pd.DataFrame({
    'admissions': admissions,
    'week': time,
    'hospital': hospital
})

print(f"\nData: {n} observations from {n_hospitals} hospitals over {n_weeks * n_years} weeks")
print(f"Response range: [{admissions.min()}, {admissions.max()}]")

# Fit GAMM with smooth term
print("\nFitting: admissions ~ s(week) + (1 | hospital)")
print("Family: Poisson")

result_poisson = fit_gamm(
    formula='admissions ~ s(week) + (1 | hospital)',
    data=data_poisson,
    family='poisson',
    maxiter=15,
)

print(f"\nConverged: {result_poisson.converged}")
print(f"Iterations: {result_poisson.n_iterations}")
print(f"\nParametric coefficients:")
print(f"  Intercept: {result_poisson.beta_parametric[0]:.4f}")
print(f"\nSmooth term 's(week)':")
print(f"  EDF: {result_poisson.edf_smooth['s(week)']:.2f}")
print(f"  Lambda: {result_poisson.smoothing_parameters.get('s(week)', 'auto')}")
print(f"\nRandom effects variance:")
print(f"  σ²_hospital: {result_poisson.variance_components[0][0,0]:.4f}")

# =============================================================================
# Example 2: Binomial GAMM with Multiple Smooth Terms
# =============================================================================
print("\n" + "=" * 70)
print("Example 2: Binary Response with Multiple Predictors")
print("=" * 70)

# Generate data: disease occurrence based on age and exposure
n_regions = 20
n_per_region = 100
n = n_regions * n_per_region

# Predictors
age = np.random.uniform(20, 80, n)
exposure = np.random.uniform(0, 10, n)
region = np.repeat(np.arange(n_regions), n_per_region)

# True smooth functions
f_age = -2 + 0.05 * (age - 50) + 0.001 * (age - 50)**2  # Quadratic age effect
f_exposure = 0.3 * np.log(exposure + 1)  # Log exposure effect

# Region random effects
region_effects = np.random.randn(n_regions) * 0.5
b_region = region_effects[region]

# Linear predictor
eta = f_age + f_exposure + b_region
prob = 1 / (1 + np.exp(-eta))

# Generate binary response
disease = np.random.binomial(1, prob)

# Create DataFrame
data_binomial = pd.DataFrame({
    'disease': disease,
    'age': age,
    'exposure': exposure,
    'region': region
})

print(f"\nData: {n} individuals from {n_regions} regions")
print(f"Disease prevalence: {disease.mean():.2%}")

# Fit GAMM with two smooth terms
print("\nFitting: disease ~ s(age) + s(exposure) + (1 | region)")
print("Family: Binomial")

result_binomial = fit_gamm(
    formula='disease ~ s(age) + s(exposure) + (1 | region)',
    data=data_binomial,
    family='binomial',
    maxiter=12,
)

print(f"\nConverged: {result_binomial.converged}")
print(f"Iterations: {result_binomial.n_iterations}")
print(f"\nSmooth terms:")
print(f"  s(age) EDF: {result_binomial.edf_smooth['s(age)']:.2f}")
print(f"  s(exposure) EDF: {result_binomial.edf_smooth['s(exposure)']:.2f}")
print(f"\nTotal EDF: {result_binomial.edf_total:.2f}")
print(f"Random effects variance: {result_binomial.variance_components[0][0,0]:.4f}")

# =============================================================================
# Example 3: Gamma GAMM with Smooth and Parametric Terms
# =============================================================================
print("\n" + "=" * 70)
print("Example 3: Gamma Response with Mixed Effects")
print("=" * 70)

# Generate data: positive continuous outcome (e.g., claim amounts)
n_groups = 25
n_per_group = 80
n = n_groups * n_per_group

# Predictors
x_smooth = np.random.uniform(0, 10, n)  # Continuous predictor
x_categorical = np.random.binomial(1, 0.5, n)  # Binary predictor
group_id = np.repeat(np.arange(n_groups), n_per_group)

# True functions
f_smooth = 100 + 50 * np.sin(x_smooth / 5)  # Oscillating pattern
beta_cat = 30  # Categorical effect

# Group random effects
group_effects = np.random.randn(n_groups) * 0.2
b_group = group_effects[group_id]

# Linear predictor (log scale for Gamma)
eta = np.log(f_smooth + beta_cat * x_categorical) + b_group
mu = np.exp(eta)

# Generate Gamma response
shape = 5.0
scale = mu / shape
amount = np.random.gamma(shape, scale)

# Create DataFrame
data_gamma = pd.DataFrame({
    'amount': amount,
    'predictor': x_smooth,
    'category': x_categorical,
    'group': group_id
})

print(f"\nData: {n} observations from {n_groups} groups")
print(f"Amount range: [{amount.min():.1f}, {amount.max():.1f}]")

# Fit GAMM with smooth and parametric terms
print("\nFitting: amount ~ category + s(predictor) + (1 | group)")
print("Family: Gamma")

result_gamma = fit_gamm(
    formula='amount ~ category + s(predictor) + (1 | group)',
    data=data_gamma,
    family='gamma',
    maxiter=12,
)

print(f"\nConverged: {result_gamma.converged}")
print(f"Iterations: {result_gamma.n_iterations}")
print(f"\nParametric coefficients:")
print(f"  Intercept: {result_gamma.beta_parametric[0]:.4f}")
print(f"  category: {result_gamma.beta_parametric[1]:.4f}")
print(f"\nSmooth term 's(predictor)':")
print(f"  EDF: {result_gamma.edf_smooth['s(predictor)']:.2f}")

# =============================================================================
# Example 4: Custom Smooth Parameters
# =============================================================================
print("\n" + "=" * 70)
print("Example 4: Controlling Smoothness via Parameters")
print("=" * 70)

# Use same Poisson data from Example 1
print("\nComparing different numbers of basis functions:")

# Fit with fewer basis functions (less flexible)
result_k5 = fit_gamm(
    formula='admissions ~ s(week, k=5) + (1 | hospital)',
    data=data_poisson,
    family='poisson',
    maxiter=10,
)

# Fit with more basis functions (more flexible)
result_k15 = fit_gamm(
    formula='admissions ~ s(week, k=15) + (1 | hospital)',
    data=data_poisson,
    family='poisson',
    maxiter=10,
)

print(f"\nk=5 (less flexible):")
print(f"  EDF: {result_k5.edf_smooth['s(week)']:.2f}")
print(f"  Converged: {result_k5.converged}")

print(f"\nk=15 (more flexible):")
print(f"  EDF: {result_k15.edf_smooth['s(week)']:.2f}")
print(f"  Converged: {result_k15.converged}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Summary: Phase 5.1 Features Demonstrated")
print("=" * 70)

print("""
✅ Smooth terms in non-Gaussian GAMM via PQL
✅ Formula interface with s() notation
✅ Multiple families: Poisson, Binomial, Gamma
✅ Multiple smooth terms in one model
✅ Combining smooth and parametric terms
✅ Custom basis sizes via k= parameter
✅ Automatic smoothing parameter selection
✅ Random effects integration

Next Phase 5 features:
- Phase 5.1.3: Advanced smoothing parameter selection (GCV/REML)
- Phase 5.2: Laplace approximation for GLMM
- Phase 5.3: Additional covariance structures
- Phase 5.4: Visualization tools
- Phase 5.5: Heavy-tailed distributions
""")

print("=" * 70)
print("Demo completed successfully! 🎉")
print("=" * 70)
