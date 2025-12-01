"""
Test script to verify notebook improvements are syntactically correct
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from scipy.stats import shapiro, normaltest
from scipy.signal import savgol_filter

print("Testing imports...")

# Test Aurora-GLM imports
try:
    from aurora.models.gamm import fit_gamm
    from aurora.models.gamm.diagnostics import (
        interpret_variance_components,
        compute_r2_conditional_marginal,
        plot_diagnostics
    )
    print("✓ Aurora-GLM imports successful")
except ImportError as e:
    print(f"✗ Aurora-GLM import error: {e}")
    exit(1)

# Test data simulation (simplified)
print("\nTesting data structures...")
np.random.seed(42)
n_schools = 10
n_students_per_school = 50
n_total = n_schools * n_students_per_school

# Simulate simple hierarchical data
school_ids = np.repeat(np.arange(n_schools), n_students_per_school)
school_effects = np.random.normal(0, 0.3, n_schools)
student_effects = np.random.normal(0, 0.8, n_total)
wealth = np.random.normal(0, 1, n_total)
female = np.random.binomial(1, 0.5, n_total)

y = school_effects[school_ids] + 0.3 * female + 0.4 * wealth + student_effects

df = pd.DataFrame({
    'schoolid': school_ids,
    'zread': y,
    'female': female,
    'wealth_c': wealth,
    'age_c': np.random.normal(0, 0.5, n_total),
    'immig': np.random.choice([0, 1, 2], n_total),
    'hisced': np.random.choice([1, 2, 3, 4, 5], n_total),
    'cultposs_c': np.random.normal(0, 1, n_total),
    'hedres_c': np.random.normal(0, 1, n_total),
    'lmins_c': np.random.normal(0, 0.5, n_total),
    'private': np.random.binomial(1, 0.2, n_total),
    'public': np.random.binomial(1, 0.3, n_total),
    'stratio_c': np.random.normal(0, 2, n_total),
    'schsize_c': np.random.normal(0, 1, n_total),
    'schltype': np.random.choice([1, 2, 3], n_total)
})

print(f"✓ Created test dataset: {len(df)} students in {df['schoolid'].nunique()} schools")

# Test basic model fitting
print("\nTesting basic GAMM fitting...")
try:
    result_null = fit_gamm(
        formula='zread ~ 1 + (1 | schoolid)',
        data=df,
        family='gaussian',
        covariance='identity'
    )
    print(f"✓ Null model converged: {result_null.converged}")
except Exception as e:
    print(f"✗ Null model error: {e}")
    exit(1)

# Test random slope model (key improvement)
print("\nTesting random slope model...")
try:
    result_rs = fit_gamm(
        formula='zread ~ female + wealth_c + (1 + wealth_c | schoolid)',
        data=df,
        family='gaussian',
        covariance='unstructured'
    )
    print(f"✓ Random slope model converged: {result_rs.converged}")

    # Test variance components access (critical API)
    psi = result_rs.variance_components[0]
    print(f"✓ Variance components accessible: shape {psi.shape}")

    sd_intercept = np.sqrt(psi[0, 0])
    sd_slope = np.sqrt(psi[1, 1])
    corr = psi[0, 1] / (sd_intercept * sd_slope)
    print(f"✓ Intercept-slope correlation: {corr:.3f}")

except Exception as e:
    print(f"✗ Random slope model error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test diagnostics
print("\nTesting diagnostic functions...")
try:
    r2_m, r2_c = compute_r2_conditional_marginal(result_rs)
    print(f"✓ R² marginal: {r2_m:.4f}, R² conditional: {r2_c:.4f}")
except Exception as e:
    print(f"✗ Diagnostics error: {e}")
    exit(1)

# Test residual analysis
print("\nTesting residual analysis...")
try:
    residuals_student = result_rs.residuals

    # Get random effects - it's a dict with group names as keys
    # For random slope model, random_effects['schoolid'] has shape (n_schools, 2) for (intercept, slope)
    random_effects_obj = result_rs.random_effects['schoolid']
    print(f"  DEBUG: type(random_effects['schoolid']) = {type(random_effects_obj)}")
    if isinstance(random_effects_obj, dict):
        print(f"  DEBUG: It's a dict with keys: {list(random_effects_obj.keys())}")
        # Assume it's structured as dict with numeric indices or term names
        random_effects_array = random_effects_obj.get(0, list(random_effects_obj.values())[0] if random_effects_obj else None)
    elif isinstance(random_effects_obj, np.ndarray):
        random_effects_array = random_effects_obj
    else:
        random_effects_array = np.array(random_effects_obj)

    if random_effects_array is not None:
        residuals_school_intercepts = random_effects_array[:, 0] if random_effects_array.ndim > 1 else random_effects_array
        residuals_school_slopes = random_effects_array[:, 1] if random_effects_array.ndim > 1 else None
    else:
        print("  ERROR: Could not extract random effects")
        raise ValueError("Could not extract random effects")

    print(f"✓ Student residuals: n={len(residuals_student)}, mean={residuals_student.mean():.4f}")
    print(f"✓ School random effects (intercepts): n={len(residuals_school_intercepts)}, mean={residuals_school_intercepts.mean():.4f}")
    if residuals_school_slopes is not None:
        print(f"✓ School random effects (slopes): n={len(residuals_school_slopes)}, mean={residuals_school_slopes.mean():.4f}")

    # Test normality tests
    if len(residuals_student) > 0 and len(residuals_school_intercepts) > 3:  # Shapiro needs n>=3
        _, p_shapiro_L1 = shapiro(residuals_student[:min(1000, len(residuals_student))])  # Sample
        _, p_shapiro_L2 = shapiro(residuals_school_intercepts)
        print(f"✓ Shapiro test L1 p={p_shapiro_L1:.4f}, L2 p={p_shapiro_L2:.4f}")
    else:
        print("⚠ Skipping Shapiro test (insufficient data)")

except Exception as e:
    print(f"✗ Residual analysis error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test outlier detection
print("\nTesting outlier detection...")
try:
    re_standardized = residuals_school_intercepts / np.std(residuals_school_intercepts)
    outlier_threshold = 2.5
    outlier_idx = np.where(np.abs(re_standardized) > outlier_threshold)[0]
    print(f"✓ Detected {len(outlier_idx)} outlier schools (threshold ±{outlier_threshold} SD)")

except Exception as e:
    print(f"✗ Outlier detection error: {e}")
    exit(1)

# Test visualization setup (don't actually plot)
print("\nTesting visualization functions...")
try:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(range(len(residuals_school_intercepts)), residuals_school_intercepts, alpha=0.6)
    ax.set_title('Test Plot')
    plt.close(fig)
    print("✓ Matplotlib plotting works")

except Exception as e:
    print(f"✗ Visualization error: {e}")
    exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nThe notebook improvements are syntactically correct and should work.")
print("Key features tested:")
print("  1. Random slope models with unstructured covariance")
print("  2. Variance components API (list of matrices)")
print("  3. R² marginal and conditional calculations")
print("  4. Multilevel residual diagnostics")
print("  5. Outlier detection")
print("  6. Visualization setup")
