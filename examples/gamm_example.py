"""Example: Fitting GAMM (Generalized Additive Mixed Models) with Aurora-GLM.

This example demonstrates:
1. Random intercept model
2. Random intercept + slope model
3. Model comparison
4. Predictions (population vs conditional)
5. Diagnostics and interpretation

Based on simulated longitudinal data similar to the "sleepstudy" dataset
from lme4.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aurora.models import fit_gamm, predict_from_gamm
from aurora.models.gamm import RandomEffect

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("GAMM Example: Longitudinal Sleep Study")
print("=" * 70)
print()

# ============================================================================
# 1. Generate Synthetic Longitudinal Data
# ============================================================================

print("1. Generating synthetic data...")
print("-" * 70)

# Parameters
n_subjects = 18  # Number of subjects
n_days = 10  # Days of observation per subject
n = n_subjects * n_days

# Subject IDs
subject_id = np.repeat(np.arange(n_subjects), n_days)

# Days since baseline (0-9)
days = np.tile(np.arange(n_days), n_subjects)

# True population parameters
beta_0_true = 250.0  # Average baseline reaction time
beta_1_true = 10.0  # Average increase per day (sleep deprivation effect)

# Random effects (subject-specific deviations)
# Random intercepts (baseline variation)
sigma_intercept = 25.0
b0_true = np.random.randn(n_subjects) * sigma_intercept

# Random slopes (individual sensitivity to sleep deprivation)
sigma_slope = 6.0
b1_true = np.random.randn(n_subjects) * sigma_slope

# Correlation between random intercept and slope
rho = 0.3
b1_true = rho * b0_true / sigma_intercept * sigma_slope + np.sqrt(
    1 - rho**2
) * b1_true

# Residual variance
sigma_residual = 25.0

# Generate response: reaction time (ms)
reaction_time = (
    beta_0_true  # Population intercept
    + beta_1_true * days  # Population slope
    + b0_true[subject_id]  # Subject-specific intercept
    + b1_true[subject_id] * days  # Subject-specific slope
    + np.random.randn(n) * sigma_residual  # Residual
)

# Create DataFrame
df = pd.DataFrame(
    {"subject": subject_id, "days": days, "reaction_time": reaction_time}
)

print(f"Data shape: {df.shape}")
print(f"Number of subjects: {n_subjects}")
print(f"Observations per subject: {n_days}")
print(f"\nFirst few rows:")
print(df.head(15))
print()

# ============================================================================
# 2. Fit Random Intercept Model
# ============================================================================

print("2. Fitting random intercept model...")
print("-" * 70)

# Design matrix: intercept + days
X = df[["days"]].values
X = np.column_stack([np.ones(len(X)), X])

# Random intercept specification
re_intercept = RandomEffect(grouping="subject", include_intercept=True)

# Fit model
result_intercept = fit_gamm(
    y=df["reaction_time"].values,
    X=X,
    random_effects=[re_intercept],
    groups_data={"subject": df["subject"].values},
    covariance="identity",  # Single variance parameter
)

print(f"Converged: {result_intercept.converged}")
print(f"\nFixed Effects:")
print(f"  Intercept: {result_intercept.beta_parametric[0]:.2f} (true: {beta_0_true:.2f})")
print(f"  Days:      {result_intercept.beta_parametric[1]:.2f} (true: {beta_1_true:.2f})")

print(f"\nRandom Effects:")
print(f"  σ²_intercept: {result_intercept.variance_components[0,0]:.2f} (true: {sigma_intercept**2:.2f})")
print(f"  σ²_residual:  {result_intercept.residual_variance:.2f} (true: {sigma_residual**2:.2f})")

print(f"\nModel Fit:")
print(f"  Log-likelihood: {result_intercept.log_likelihood:.2f}")
print(f"  AIC: {result_intercept.aic:.2f}")
print(f"  BIC: {result_intercept.bic:.2f}")
print()

# ============================================================================
# 3. Fit Random Intercept + Slope Model
# ============================================================================

print("3. Fitting random intercept + slope model...")
print("-" * 70)

# Random intercept + slope on 'days' (variable index 1)
re_slope = RandomEffect(
    grouping="subject", variables=(1,), include_intercept=True, covariance="unstructured"
)

# Fit model
result_slope = fit_gamm(
    y=df["reaction_time"].values,
    X=X,
    random_effects=[re_slope],
    groups_data={"subject": df["subject"].values},
    covariance="unstructured",  # Full 2x2 covariance matrix
)

print(f"Converged: {result_slope.converged}")
print(f"\nFixed Effects:")
print(f"  Intercept: {result_slope.beta_parametric[0]:.2f} (true: {beta_0_true:.2f})")
print(f"  Days:      {result_slope.beta_parametric[1]:.2f} (true: {beta_1_true:.2f})")

print(f"\nRandom Effects Covariance (Ψ):")
psi = result_slope.variance_components
print(f"  Var(intercept): {psi[0,0]:.2f} (true: {sigma_intercept**2:.2f})")
print(f"  Cov(int,slope): {psi[0,1]:.2f}")
print(f"  Var(slope):     {psi[1,1]:.2f} (true: {sigma_slope**2:.2f})")

# Compute correlation
corr = psi[0, 1] / np.sqrt(psi[0, 0] * psi[1, 1])
print(f"  Corr(int,slope): {corr:.3f} (true: {rho:.3f})")

print(f"\nResidual variance:")
print(f"  σ²_residual: {result_slope.residual_variance:.2f} (true: {sigma_residual**2:.2f})")

print(f"\nModel Fit:")
print(f"  Log-likelihood: {result_slope.log_likelihood:.2f}")
print(f"  AIC: {result_slope.aic:.2f}")
print(f"  BIC: {result_slope.bic:.2f}")
print()

# ============================================================================
# 4. Model Comparison
# ============================================================================

print("4. Model Comparison")
print("-" * 70)

print(f"Random Intercept Model:")
print(f"  AIC: {result_intercept.aic:.2f}")
print(f"  BIC: {result_intercept.bic:.2f}")

print(f"\nRandom Intercept + Slope Model:")
print(f"  AIC: {result_slope.aic:.2f}")
print(f"  BIC: {result_slope.bic:.2f}")

aic_diff = result_intercept.aic - result_slope.aic
bic_diff = result_intercept.bic - result_slope.bic

print(f"\nΔAIC: {aic_diff:.2f} (lower is better)")
print(f"ΔBIC: {bic_diff:.2f} (lower is better)")

if aic_diff > 10:
    print("\nConclusion: Random slope model is STRONGLY preferred (ΔAIC > 10)")
elif aic_diff > 2:
    print("\nConclusion: Random slope model is preferred (ΔAIC > 2)")
else:
    print("\nConclusion: Models are similar")
print()

# ============================================================================
# 5. Extract and Examine Random Effects (BLUPs)
# ============================================================================

print("5. Random Effects (BLUPs)")
print("-" * 70)

# Extract random effects from preferred model
random_effects = result_slope.random_effects["subject"]

print("Subject-specific deviations (first 6 subjects):")
print(f"{'Subject':<10} {'Intercept':<12} {'Slope':<12} (estimated)")
for subject_idx in range(min(6, n_subjects)):
    b_subj = random_effects[subject_idx]
    print(f"{subject_idx:<10} {b_subj[0]:>11.2f} {b_subj[1]:>11.2f}")

print("\nTrue values (for comparison):")
print(f"{'Subject':<10} {'Intercept':<12} {'Slope':<12} (true)")
for subject_idx in range(min(6, n_subjects)):
    print(f"{subject_idx:<10} {b0_true[subject_idx]:>11.2f} {b1_true[subject_idx]:>11.2f}")
print()

# ============================================================================
# 6. Predictions
# ============================================================================

print("6. Predictions")
print("-" * 70)

# New data: predict reaction times for days 0-14 (extending beyond training)
days_new = np.arange(15)
n_new = len(days_new)
X_new = np.column_stack([np.ones(n_new), days_new])

# Population-level predictions (for a "typical" new subject)
pred_pop = predict_from_gamm(result_slope, X_new, include_random=False)

print("Population-level predictions (typical subject):")
for i in range(0, n_new, 3):
    print(f"  Day {days_new[i]:>2}: {pred_pop[i]:.1f} ms")

# Conditional predictions for an existing subject (e.g., subject 0)
subject_for_pred = 0
groups_new = np.full(n_new, subject_for_pred)
pred_cond = predict_from_gamm(
    result_slope, X_new, groups_new=groups_new, include_random=True
)

print(f"\nConditional predictions (subject {subject_for_pred}):")
for i in range(0, n_new, 3):
    print(f"  Day {days_new[i]:>2}: {pred_cond[i]:.1f} ms")
print()

# ============================================================================
# 7. Visualization
# ============================================================================

print("7. Creating visualization...")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Individual trajectories with population fit
ax = axes[0, 0]
for subj in range(min(6, n_subjects)):
    subj_data = df[df["subject"] == subj]
    ax.plot(
        subj_data["days"],
        subj_data["reaction_time"],
        "o-",
        alpha=0.3,
        label=f"Subj {subj}" if subj < 3 else "",
    )

# Population prediction
ax.plot(days_new, pred_pop, "r-", linewidth=2, label="Population mean")
ax.set_xlabel("Days")
ax.set_ylabel("Reaction Time (ms)")
ax.set_title("Individual Trajectories + Population Mean")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Random effects scatter
ax = axes[0, 1]
b_intercepts = [random_effects[i][0] for i in range(n_subjects)]
b_slopes = [random_effects[i][1] for i in range(n_subjects)]
ax.scatter(b_intercepts, b_slopes, alpha=0.6, s=80)
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.axvline(0, color="k", linestyle="--", alpha=0.3)
ax.set_xlabel("Random Intercept")
ax.set_ylabel("Random Slope")
ax.set_title(f"Random Effects (corr = {corr:.3f})")
ax.grid(True, alpha=0.3)

# Plot 3: Residuals vs fitted
ax = axes[1, 0]
ax.scatter(result_slope.fitted_values, result_slope.residuals, alpha=0.5)
ax.axhline(0, color="r", linestyle="--")
ax.set_xlabel("Fitted Values")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Fitted")
ax.grid(True, alpha=0.3)

# Plot 4: Q-Q plot
ax = axes[1, 1]
from scipy import stats

residuals_std = result_slope.residuals / np.sqrt(result_slope.residual_variance)
stats.probplot(residuals_std, dist="norm", plot=ax)
ax.set_title("Normal Q-Q Plot")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gamm_example.png", dpi=150, bbox_inches="tight")
print("Figure saved as 'gamm_example.png'")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 70)
print("Summary")
print("=" * 70)
print(
    """
This example demonstrated:

1. Fitting random intercept models for grouped/longitudinal data
2. Fitting random intercept + slope models with correlated effects
3. Model comparison using AIC/BIC
4. Extracting subject-specific effects (BLUPs)
5. Making population-level and conditional predictions
6. Basic diagnostic plots

Key findings:
- Random slope model provides better fit (lower AIC/BIC)
- Estimated parameters close to true values
- Individual subjects show variation in both baseline and slope
- Model captures correlation between random intercept and slope

For more information, see:
- aurora.models.gamm module documentation
- R packages: lme4, mgcv (for comparison)
"""
)
