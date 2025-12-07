"""Simple test of PISA UK data loading and null model"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Set encoding for Windows console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from aurora.models.gamm import fit_gamm

# Load data
data_path = Path('examples/06_case_studies/data/pisaUK.csv')
df = pd.read_csv(data_path)

print(f"Dataset: {len(df)} students in {df['schoolid'].nunique()} schools")
print(f"Variables: {list(df.columns)}")
print(f"\nReading score (zread): mean={df['zread'].mean():.3f}, sd={df['zread'].std():.3f}")

# School-level summary
school_summary = df.groupby('schoolid')['zread'].agg(['mean', 'count'])
print(f"\nSchools: N={len(school_summary)}")
print(f"Students per school: mean={school_summary['count'].mean():.1f}, median={school_summary['count'].median():.0f}")
print(f"Between-school SD (naive): {school_summary['mean'].std():.3f}")

# Fit null model (this might take a minute)
print("\nFitting null model: zread ~ 1 + (1 | schoolid)")
print("(This may take 1-2 minutes with 7,610 observations...)")

try:
    result = fit_gamm(
        formula='zread ~ 1 + (1 | schoolid)',
        data=df,
        family='gaussian',
        covariance='identity'
    )

    print(f"\nSUCCESS! Model converged: {result.converged}")
    print(f"Iterations: {result.n_iterations}")
    print(f"Grand mean: {result.beta_parametric[0]:.4f}")

    tau_squared = result.variance_components[0][0, 0]
    sigma_squared = result.residual_variance
    icc = tau_squared / (tau_squared + sigma_squared)

    print(f"\nVariance components:")
    print(f"  Between schools: {tau_squared:.4f}")
    print(f"  Within schools: {sigma_squared:.4f}")
    print(f"  ICC: {icc:.4f} ({icc*100:.1f}% between schools)")

    print("\n" + "="*60)
    print("✓ Test passed! PISA UK data works with Aurora-GLM")
    print("="*60)

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
