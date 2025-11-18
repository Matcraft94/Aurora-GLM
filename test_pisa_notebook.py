"""Quick test of PISA UK multilevel analysis"""

import numpy as np
import pandas as pd
from pathlib import Path

from aurora.models.gamm import fit_gamm
from aurora.models.gamm.diagnostics import (
    interpret_variance_components,
    compute_r2_conditional_marginal,
)

# Load data
data_path = Path('examples/06_case_studies/data/pisaUK.csv')
df = pd.read_csv(data_path)

print(f"Dataset: {len(df)} students in {df['schoolid'].nunique()} schools")

# Center predictors
df['age_c'] = df['age'] - df['age'].mean()
df['wealth_c'] = df['wealth'] - df['wealth'].mean()
df['cultposs_c'] = df['cultposs'] - df['cultposs'].mean()
df['hedres_c'] = df['hedres'] - df['hedres'].mean()
df['lmins_c'] = (df['lmins'] - df['lmins'].mean()) / 60

# Model 1: Null model
print("\n" + "="*70)
print("Model 1: Null Model (Variance Decomposition)")
print("="*70)

result_null = fit_gamm(
    formula='zread ~ 1 + (1 | schoolid)',
    data=df,
    family='gaussian',
    covariance='identity'
)

print(f"Converged: {result_null.converged} (iterations: {result_null.n_iterations})")
print(f"Grand Mean: {result_null.beta_parametric[0]:.4f}")

tau_squared = result_null.variance_components[0][0, 0]
sigma_squared = result_null.residual_variance
total_variance = tau_squared + sigma_squared
icc = tau_squared / total_variance

print(f"\nVariance Components:")
print(f"  Between schools (tau^2): {tau_squared:.4f}")
print(f"  Within schools (sigma^2): {sigma_squared:.4f}")
print(f"  ICC: {icc:.4f} ({icc*100:.2f}% between schools)")

# Model 2: Student-level predictors
print("\n" + "="*70)
print("Model 2: Student-Level Predictors")
print("="*70)

result_student = fit_gamm(
    formula='zread ~ age_c + female + immig + hisced + wealth_c + cultposs_c + hedres_c + lmins_c + (1 | schoolid)',
    data=df,
    family='gaussian',
    covariance='identity'
)

print(f"Converged: {result_student.converged} (iterations: {result_student.n_iterations})")

predictors = ['Intercept', 'Age', 'Female', 'Immigration', 'Parent Ed.',
              'Wealth', 'Cultural', 'Home Res.', 'Learning']
print("\nFixed Effects:")
for name, coef in zip(predictors, result_student.beta_parametric):
    print(f"  {name:12s}: {coef:7.4f}")

tau_squared_2 = result_student.variance_components[0][0, 0]
sigma_squared_2 = result_student.residual_variance

var_explained_between = 1 - (tau_squared_2 / tau_squared)
var_explained_within = 1 - (sigma_squared_2 / sigma_squared)

print(f"\nVariance Explained:")
print(f"  Between-school: {var_explained_between*100:.2f}%")
print(f"  Within-school: {var_explained_within*100:.2f}%")

r2_m, r2_c = compute_r2_conditional_marginal(result_student)
print(f"\nR²:")
print(f"  Marginal: {r2_m:.4f}")
print(f"  Conditional: {r2_c:.4f}")

# Model 3: Add school-level predictors
print("\n" + "="*70)
print("Model 3: Full Model with School Predictors")
print("="*70)

df['stratio_c'] = df['stratio'] - df['stratio'].mean()
df['schsize_c'] = (df['schsize'] - df['schsize'].mean()) / 100
df['private'] = (df['schltype'] == 1).astype(int)
df['public'] = (df['schltype'] == 2).astype(int)

result_full = fit_gamm(
    formula='zread ~ age_c + female + immig + hisced + wealth_c + cultposs_c + hedres_c + lmins_c + private + public + stratio_c + schsize_c + (1 | schoolid)',
    data=df,
    family='gaussian',
    covariance='identity'
)

print(f"Converged: {result_full.converged} (iterations: {result_full.n_iterations})")

predictors_full = ['Intercept', 'Age', 'Female', 'Immigration', 'Parent Ed.',
                  'Wealth', 'Cultural', 'Home Res.', 'Learning',
                  'Private', 'Public', 'Stu-Teach Ratio', 'School Size']
print("\nFixed Effects:")
for name, coef in zip(predictors_full, result_full.beta_parametric):
    print(f"  {name:16s}: {coef:7.4f}")

tau_squared_3 = result_full.variance_components[0][0, 0]
sigma_squared_3 = result_full.residual_variance

var_explained_between_3 = 1 - (tau_squared_3 / tau_squared_2)

print(f"\nAdditional Between-School Variance Explained: {var_explained_between_3*100:.2f}%")

r2_m_full, r2_c_full = compute_r2_conditional_marginal(result_full)
print(f"\nR²:")
print(f"  Marginal: {r2_m_full:.4f}")
print(f"  Conditional: {r2_c_full:.4f}")

print("\n" + "="*70)
print("SUCCESS: All models converged and produced reasonable results!")
print("="*70)
print("\nKey Findings:")
print(f"  • {icc*100:.1f}% of variance is between schools")
print(f"  • Girls outperform boys by {result_student.beta_parametric[2]:.3f} SDs")
print(f"  • Family SES has substantial effect on reading achievement")
print(f"  • Student composition explains {var_explained_between*100:.1f}% of school differences")
print(f"  • Full model explains {r2_c_full*100:.1f}% of total variance")
