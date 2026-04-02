"""Debug PISA UK GAMM fitting"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aurora.models.gamm.fitting import fit_gamm_gaussian
from aurora.models.gamm.random_effects import RandomEffect

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Load data
data_path = Path("examples/06_case_studies/data/pisaUK.csv")
df = pd.read_csv(data_path)

print(f"Dataset: {len(df)} students in {df['schoolid'].nunique()} schools")
print(f"Reading score: mean={df['zread'].mean():.3f}, sd={df['zread'].std():.3f}")

# Prepare data for low-level API
y = df["zread"].values
X = np.ones((len(df), 1))  # Just intercept

# Create random effect for schools
school_ids = df["schoolid"].values
unique_schools = np.unique(school_ids)
print(f"\nUnique schools: {len(unique_schools)}")

# Map school IDs to indices 0, 1, 2, ...
school_map = {sid: i for i, sid in enumerate(unique_schools)}
school_indices = np.array([school_map[sid] for sid in school_ids])

print(f"School indices: min={school_indices.min()}, max={school_indices.max()}")
print(f"School index counts: {np.bincount(school_indices)[:5]}...")  # Show first 5

# Create random effect (just random intercept)
re = RandomEffect(grouping="schoolid", include_intercept=True, covariance="identity")

groups_data = {"schoolid": school_indices}

print("\nFitting model with low-level API...")
try:
    result = fit_gamm_gaussian(y=y, X=X, random_effects=[re], groups_data=groups_data, reml=True)

    print(f"\nSUCCESS! Converged: {result.converged}")
    print(f"Iterations: {result.n_iterations}")
    print(f"Intercept: {result.beta_parametric[0]:.4f} (expected ~{y.mean():.3f})")

    tau_squared = result.variance_components[0][0, 0]
    sigma_squared = result.residual_variance
    total_var = tau_squared + sigma_squared
    icc = tau_squared / total_var

    print("\nVariance components:")
    print(f"  Between schools (tau^2): {tau_squared:.4f}")
    print(f"  Within schools (sigma^2): {sigma_squared:.4f}")
    print(f"  Total: {total_var:.4f} (data variance: {y.var():.4f})")
    print(f"  ICC: {icc:.4f} ({icc * 100:.1f}% between schools)")

    # Check random effects
    if result.random_effects is not None and len(result.random_effects) > 0:
        school_effects = result.random_effects[0]
        print(f"\nRandom effects shape: {school_effects.shape}")
        print(f"Random effects range: [{school_effects.min():.3f}, {school_effects.max():.3f}]")
        print(f"Random effects SD: {school_effects.std():.3f}")

    # Manual ICC calculation for comparison
    school_means = df.groupby("schoolid")["zread"].mean()
    between_var = school_means.var()
    total_var_data = df["zread"].var()
    icc_naive = between_var / total_var_data

    print(f"\nNaive ICC (from raw data): {icc_naive:.4f}")
    print(f"Model ICC: {icc:.4f}")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback

    traceback.print_exc()
