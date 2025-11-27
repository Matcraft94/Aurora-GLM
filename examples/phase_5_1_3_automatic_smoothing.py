"""Demo: Automatic Smoothing Parameter Selection (Phase 5.1.3).

This example demonstrates automatic λ selection for smooth terms in
non-Gaussian GAMM using:
1. GCV (Generalized Cross-Validation)
2. Performance iteration

The user doesn't need to specify smoothing parameters - they are
selected automatically to optimize model fit.

Author: Aurora-GLM Team
Date: 2025
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aurora.models.gamm import fit_gamm

# Set random seed
np.random.seed(42)

print("=" * 70)
print("Phase 5.1.3 Demo: Automatic Smoothing Parameter Selection")
print("=" * 70)

# =============================================================================
# Example 1: Automatic λ Selection for Single Smooth Term
# =============================================================================
print("\n" + "=" * 70)
print("Example 1: Poisson GAMM with Automatic λ Selection")
print("=" * 70)

# Generate data: traffic accidents by time of day
n_locations = 20
n_hours = 24
n_days = 30
n = n_locations * n_hours * n_days

# Hour of day (0-23)
hour = np.tile(np.repeat(np.arange(n_hours), n_days), n_locations)

# Location IDs
location = np.repeat(np.arange(n_locations), n_hours * n_days)

# True pattern: more accidents during rush hours
f_hour = 5 + 3 * np.sin(2 * np.pi * (hour - 8) / 24) + \
         2 * np.sin(4 * np.pi * hour / 24)  # Double peak

# Location random effects
location_effects = np.random.randn(n_locations) * 0.4
b_location = location_effects[location]

# Generate counts
eta = np.log(np.abs(f_hour) + 1) + b_location
accidents = np.random.poisson(np.exp(eta))

# Create DataFrame
data = pd.DataFrame({
    'accidents': accidents,
    'hour': hour,
    'location': location
})

print(f"\nData: {n} observations")
print(f"Total accidents: {accidents.sum()}")
print(f"Accidents per hour: {accidents.sum() / (n_hours * n_days):.1f}")

# Fit model WITHOUT specifying λ (automatic selection)
print("\nFitting: accidents ~ s(hour) + (1 | location)")
print("Smoothing parameter: AUTO (via GCV)")

result_auto = fit_gamm(
    formula='accidents ~ s(hour) + (1 | location)',
    data=data,
    family='poisson',
    maxiter=10,
)

print(f"\nResults:")
print(f"  Converged: {result_auto.converged}")
print(f"  Selected λ: {result_auto.smoothing_parameters['s(hour)']:.4f}")
print(f"  EDF: {result_auto.edf_smooth['s(hour)']:.2f}")
print(f"  Random effects σ²: {result_auto.variance_components[0][0,0]:.4f}")

# Compare with manual λ selection
print("\n" + "-" * 70)
print("Comparison: Manual vs Automatic λ Selection")
print("-" * 70)

# Fit with very small λ (undersmoothing)
result_small = fit_gamm(
    formula='accidents ~ s(hour, sp=0.001) + (1 | location)',
    data=data,
    family='poisson',
    maxiter=8,
)

# Fit with very large λ (oversmoothing)
result_large = fit_gamm(
    formula='accidents ~ s(hour, sp=100.0) + (1 | location)',
    data=data,
    family='poisson',
    maxiter=8,
)

print(f"\nλ = 0.001 (undersmoothing):")
print(f"  EDF: {result_small.edf_smooth['s(hour)']:.2f}")

print(f"\nλ = {result_auto.smoothing_parameters['s(hour)']:.4f} (automatic - optimal):")
print(f"  EDF: {result_auto.edf_smooth['s(hour)']:.2f}")

print(f"\nλ = 100.0 (oversmoothing):")
print(f"  EDF: {result_large.edf_smooth['s(hour)']:.2f}")

print(f"\n→ Automatic selection found optimal balance!")

# =============================================================================
# Example 2: Multiple Smooth Terms with Auto λ
# =============================================================================
print("\n" + "=" * 70)
print("Example 2: Multiple Smooths with Individual λ Selection")
print("=" * 70)

# Generate data: disease risk by age and pollution
n_regions = 25
n_per_region = 150
n = n_regions * n_per_region

age = np.random.uniform(20, 80, n)
pollution = np.random.uniform(0, 100, n)  # PM2.5 levels
region = np.repeat(np.arange(n_regions), n_per_region)

# True effects (different smoothness levels)
f_age = -3 + 0.1 * (age - 50) + 0.002 * (age - 50)**2  # Smooth quadratic
f_pollution = 0.02 * pollution + 0.0001 * pollution**2  # Very smooth

# Region effects
region_effects = np.random.randn(n_regions) * 0.5
b_region = region_effects[region]

# Generate binary response
eta = f_age + f_pollution + b_region
prob = 1 / (1 + np.exp(-eta))
disease = np.random.binomial(1, prob)

data_multi = pd.DataFrame({
    'disease': disease,
    'age': age,
    'pollution': pollution,
    'region': region
})

print(f"\nData: {n} individuals, {n_regions} regions")
print(f"Disease prevalence: {disease.mean():.2%}")

# Fit with automatic λ for both smooths
print("\nFitting: disease ~ s(age) + s(pollution) + (1 | region)")
print("Both λ selected automatically")

result_multi = fit_gamm(
    formula='disease ~ s(age) + s(pollution) + (1 | region)',
    data=data_multi,
    family='binomial',
    maxiter=10,
)

print(f"\nResults:")
print(f"  Converged: {result_multi.converged}")
print(f"\n  s(age):")
print(f"    λ: {result_multi.smoothing_parameters['s(age)']:.4f}")
print(f"    EDF: {result_multi.edf_smooth['s(age)']:.2f}")
print(f"\n  s(pollution):")
print(f"    λ: {result_multi.smoothing_parameters['s(pollution)']:.4f}")
print(f"    EDF: {result_multi.edf_smooth['s(pollution)']:.2f}")

print(f"\n→ Each smooth gets its own optimal λ!")

# =============================================================================
# Example 3: Performance Iteration
# =============================================================================
print("\n" + "=" * 70)
print("Example 3: Advanced - Performance Iteration")
print("=" * 70)

# Use performance iteration for more refined λ selection
from aurora.models.gamm.smoothing_selection import select_smoothing_performance_iter
from aurora.smoothing.splines.bspline import BSplineBasis

# Generate simple data
n_simple = 150
n_groups_simple = 15
x_simple = np.linspace(0, 1, n_simple)
groups_simple = np.repeat(np.arange(n_groups_simple), n_simple // n_groups_simple)

f_simple = 2 * np.sin(2 * np.pi * x_simple)
b_simple = np.random.randn(n_groups_simple) * 0.4
eta_simple = f_simple + b_simple[groups_simple]
y_simple = np.random.poisson(np.exp(eta_simple))

# Create design matrices manually
knots = BSplineBasis.create_knots(x_simple, n_basis=10, degree=3)
basis = BSplineBasis(knots, degree=3)
B = basis.basis_matrix(x_simple)
S = basis.penalty_matrix(order=2)

Z_simple = np.zeros((n_simple, n_groups_simple))
Z_simple[np.arange(n_simple), groups_simple] = 1.0
Z_info_simple = [{'n_levels': n_groups_simple, 'dim': 1, 'type': 'intercept'}]

print(f"\nData: {n_simple} observations, {n_groups_simple} groups")
print(f"Running performance iteration (alternates between fitting and λ selection)...")

result_perf = select_smoothing_performance_iter(
    X_parametric=np.ones((n_simple, 1)),
    X_smooth_dict={'s(x)': B},
    S_smooth_dict={'s(x)': S},
    Z=Z_simple,
    Z_info=Z_info_simple,
    y=y_simple,
    family='poisson',
    max_iter=4,
    verbose=True,  # Show iteration progress
)

print(f"\nFinal Results:")
print(f"  Optimal λ: {result_perf['lambda_opt']['s(x)']:.4f}")
print(f"  EDF: {result_perf['edf_smooth']['s(x)']:.2f}")
print(f"  Converged: {result_perf['converged']}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Summary: Automatic Smoothing Parameter Selection")
print("=" * 70)

print("""
✅ GCV-based automatic λ selection
✅ Works for all non-Gaussian families (Poisson, Binomial, Gamma)
✅ Individual λ for each smooth term
✅ Performance iteration for refined selection
✅ No manual tuning required!

Key Benefits:
1. User doesn't need to specify λ
2. Optimal balance between fit and smoothness
3. Data-driven selection via cross-validation
4. Computationally efficient

Usage:
# Simply omit 'sp=' parameter - λ is selected automatically!
fit_gamm(formula='y ~ s(x) + (1 | group)', data=df, family='poisson')

# Or use performance iteration for best results
from aurora.models.gamm.smoothing_selection import select_smoothing_performance_iter
result = select_smoothing_performance_iter(...)
""")

print("=" * 70)
print("Phase 5.1.3 Complete! 🎉")
print("=" * 70)
