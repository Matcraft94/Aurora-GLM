"""Demonstration of Non-Gaussian GAMM with PQL estimation.

This example shows how to fit Generalized Additive Mixed Models (GAMM)
with non-Gaussian response distributions using Penalized Quasi-Likelihood (PQL).

Examples include:
1. Poisson GAMM for count data with random intercepts
2. Binomial GAMM for binary data with random intercepts
3. Poisson GAMM using formula interface (lme4-style syntax)
4. Random slopes model for longitudinal count data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aurora.models.gamm import RandomEffect, fit_gamm, predict_from_gamm


def example_1_poisson_random_intercepts():
    """Example 1: Poisson GAMM with random intercepts.

    Scenario: Count of species observed at different sites over time.
    Random effects capture site-specific baseline abundance.
    """
    print("\n" + "="*70)
    print("Example 1: Poisson GAMM - Species Counts with Random Site Effects")
    print("="*70)

    np.random.seed(42)

    # Simulate data
    n_sites = 10
    n_per_site = 20
    n = n_sites * n_per_site

    sites = np.repeat(np.arange(n_sites), n_per_site)
    temperature = np.random.randn(n)  # Standardized temperature

    # Random site effects (some sites have more species)
    site_effects = np.random.randn(n_sites) * 0.3

    # True model: log(λ) = 1.5 + 0.3*temp + site_effect
    # Using smaller intercept to avoid overflow in PQL
    log_lambda = 1.5 + 0.3 * temperature + site_effects[sites]
    lambda_true = np.exp(log_lambda)

    # Generate Poisson counts
    counts = np.random.poisson(lambda_true)

    print(f"\nData Summary:")
    print(f"  Sites: {n_sites}")
    print(f"  Observations per site: {n_per_site}")
    print(f"  Mean count: {counts.mean():.2f}")
    print(f"  Count range: [{counts.min()}, {counts.max()}]")

    # Prepare data
    X = np.column_stack([np.ones(n), temperature])

    # Fit Poisson GAMM
    print("\nFitting Poisson GAMM with random intercepts...")

    re = RandomEffect(grouping='site')
    result = fit_gamm(
        y=counts,
        X=X,
        random_effects=[re],
        groups_data={'site': sites},
        family='poisson',
        covariance='identity',
        maxiter=20
    )

    # Display results
    print(f"\nModel Results:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.n_iterations}")
    print(f"\nFixed Effects:")
    print(f"  Intercept: {result.beta_parametric[0]:.4f} (true: 1.5)")
    print(f"  Temperature: {result.beta_parametric[1]:.4f} (true: 0.3)")
    print(f"\nRandom Effects:")
    print(f"  Site variance: {result.variance_components[0][0,0]:.4f} (true: 0.09)")
    print(f"\nModel Fit:")
    print(f"  Log-likelihood: {result.log_likelihood:.2f}")
    print(f"  AIC: {result.aic:.2f}")
    print(f"  BIC: {result.bic:.2f}")

    return result


def example_2_binomial_random_intercepts():
    """Example 2: Binomial GAMM with random intercepts.

    Scenario: Presence/absence of disease across different clinics.
    Random effects capture clinic-specific baseline risk.
    """
    print("\n" + "="*70)
    print("Example 2: Binomial GAMM - Disease Presence with Random Clinic Effects")
    print("="*70)

    np.random.seed(123)

    # Simulate data
    n_clinics = 8
    n_per_clinic = 30
    n = n_clinics * n_per_clinic

    clinics = np.repeat(np.arange(n_clinics), n_per_clinic)
    age = np.random.randn(n)  # Standardized age

    # Random clinic effects
    clinic_effects = np.random.randn(n_clinics) * 0.6

    # True model: logit(p) = -0.5 + 0.7*age + clinic_effect
    logit_p = -0.5 + 0.7 * age + clinic_effects[clinics]
    p_true = 1 / (1 + np.exp(-logit_p))

    # Generate binary outcomes
    disease = np.random.binomial(1, p_true)

    print(f"\nData Summary:")
    print(f"  Clinics: {n_clinics}")
    print(f"  Patients per clinic: {n_per_clinic}")
    print(f"  Disease prevalence: {disease.mean():.2%}")

    # Prepare data
    X = np.column_stack([np.ones(n), age])

    # Fit Binomial GAMM
    print("\nFitting Binomial GAMM with random intercepts...")

    re = RandomEffect(grouping='clinic')
    result = fit_gamm(
        y=disease,
        X=X,
        random_effects=[re],
        groups_data={'clinic': clinics},
        family='binomial',
        covariance='identity',
        maxiter=25
    )

    # Display results
    print(f"\nModel Results:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.n_iterations}")
    print(f"\nFixed Effects (log-odds scale):")
    print(f"  Intercept: {result.beta_parametric[0]:.4f} (true: -0.5)")
    print(f"  Age effect: {result.beta_parametric[1]:.4f} (true: 0.7)")
    print(f"\nRandom Effects:")
    print(f"  Clinic variance: {result.variance_components[0][0,0]:.4f} (true: 0.36)")

    # Compute predicted probabilities
    print(f"\nPredicted Probabilities:")
    print(f"  Mean: {result.fitted_values.mean():.3f}")
    print(f"  Range: [{result.fitted_values.min():.3f}, {result.fitted_values.max():.3f}]")

    return result


def example_3_formula_interface():
    """Example 3: Using R-style formula interface.

    Demonstrates the convenient formula syntax for specifying GAMM models.
    """
    print("\n" + "="*70)
    print("Example 3: Formula Interface - Vehicle Accidents")
    print("="*70)

    np.random.seed(456)

    # Simulate vehicle accident data
    n_regions = 6
    n_per_region = 25
    n = n_regions * n_per_region

    # Create DataFrame
    df = pd.DataFrame({
        'accidents': np.random.poisson(8, n),
        'traffic_volume': np.random.randn(n),
        'rain': np.random.randn(n),
        'region': np.repeat(np.arange(n_regions), n_per_region)
    })

    # Add region effects
    for i in range(n_regions):
        mask = df['region'] == i
        region_effect = np.random.randn() * 0.3
        df.loc[mask, 'accidents'] = np.random.poisson(
            np.exp(2.0 + 0.4*df.loc[mask, 'traffic_volume'] +
                   0.2*df.loc[mask, 'rain'] + region_effect),
            size=mask.sum()
        )

    print(f"\nData Summary:")
    print(f"  Regions: {n_regions}")
    print(f"  Observations: {n}")
    print(f"  Mean accidents: {df['accidents'].mean():.2f}")

    # Fit using formula syntax (like R's lme4)
    print("\nFitting model with formula: accidents ~ traffic_volume + rain + (1 | region)")

    result = fit_gamm(
        formula='accidents ~ traffic_volume + rain + (1 | region)',
        data=df,
        family='poisson',
        covariance='identity',
        maxiter=20
    )

    # Display results
    print(f"\nModel Results:")
    print(f"  Converged: {result.converged}")
    print(f"\nFixed Effects:")
    print(f"  Intercept: {result.beta_parametric[0]:.4f}")
    print(f"  Traffic volume: {result.beta_parametric[1]:.4f}")
    print(f"  Rain: {result.beta_parametric[2]:.4f}")
    print(f"\nRandom Effects:")
    print(f"  Region variance: {result.variance_components[0][0,0]:.4f}")

    return result, df


def example_4_random_slopes():
    """Example 4: Longitudinal data with random slopes.

    Scenario: Plant growth over time with plant-specific growth rates.
    """
    print("\n" + "="*70)
    print("Example 4: Random Slopes - Plant Growth Over Time")
    print("="*70)

    np.random.seed(789)

    # Simulate longitudinal growth data
    n_plants = 8
    n_timepoints = 15
    n = n_plants * n_timepoints

    plants = np.repeat(np.arange(n_plants), n_timepoints)
    time = np.tile(np.arange(n_timepoints), n_plants)

    # Random intercepts and slopes
    intercepts = np.random.randn(n_plants) * 0.15
    slopes = 0.05 + np.random.randn(n_plants) * 0.02  # Varying growth rates (smaller)

    # True model with count of new leaves (using smaller values to avoid overflow)
    log_lambda = 1.2 + slopes[plants] * time + intercepts[plants]
    lambda_true = np.exp(log_lambda)
    leaf_count = np.random.poisson(lambda_true)

    print(f"\nData Summary:")
    print(f"  Plants: {n_plants}")
    print(f"  Timepoints: {n_timepoints}")
    print(f"  Total observations: {n}")
    print(f"  Mean leaf count: {leaf_count.mean():.2f}")

    # Prepare data
    X = np.column_stack([np.ones(n), time])

    # Fit with random intercepts AND slopes
    print("\nFitting Poisson GAMM with random intercepts and slopes...")

    re = RandomEffect(
        grouping='plant',
        variables=(1,),  # Random slope on variable 1 (time)
        include_intercept=True,  # Also include random intercept
    )

    result = fit_gamm(
        y=leaf_count,
        X=X,
        random_effects=[re],
        groups_data={'plant': plants},
        family='poisson',
        covariance='diagonal',  # Independent variance for intercept and slope
        maxiter=20
    )

    # Display results
    print(f"\nModel Results:")
    print(f"  Converged: {result.converged}")
    print(f"\nFixed Effects:")
    print(f"  Intercept: {result.beta_parametric[0]:.4f}")
    print(f"  Time (growth rate): {result.beta_parametric[1]:.4f}")
    print(f"\nRandom Effects Variance-Covariance:")
    print(f"  Var(Intercept): {result.variance_components[0][0,0]:.4f}")
    print(f"  Var(Slope): {result.variance_components[0][1,1]:.4f}")

    # Make predictions for a new plant
    print(f"\nPredicting for a new (average) plant:")
    X_new = np.column_stack([np.ones(n_timepoints), np.arange(n_timepoints)])
    predictions = predict_from_gamm(result, X_new, include_random=False)
    print(f"  Time 0: {predictions[0]:.2f} leaves")
    print(f"  Time 5: {predictions[5]:.2f} leaves")
    print(f"  Time 10: {predictions[10]:.2f} leaves")

    return result


def main():
    """Run all examples."""
    print("\n" + "#"*70)
    print("# Non-Gaussian GAMM with Penalized Quasi-Likelihood (PQL)")
    print("# Aurora-GLM: Statistical Modeling Framework")
    print("#"*70)

    # Run examples
    result1 = example_1_poisson_random_intercepts()
    result2 = example_2_binomial_random_intercepts()
    result3, df3 = example_3_formula_interface()
    result4 = example_4_random_slopes()

    print("\n" + "="*70)
    print("Summary: All Non-Gaussian GAMM Examples Completed Successfully!")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Poisson GAMM for count data")
    print("  ✓ Binomial GAMM for binary data")
    print("  ✓ R-style formula interface: y ~ x + (1 | group)")
    print("  ✓ Random intercepts and random slopes")
    print("  ✓ PQL estimation for non-Gaussian families")
    print("  ✓ Predictions for new observations")

    print("\nImplementation Status:")
    print("  ✓ Gaussian, Poisson, Binomial, Gamma families supported")
    print("  ✓ PQL algorithm with REML variance estimation")
    print("  ✓ Multi-backend ready (NumPy, PyTorch, JAX)")
    print("  ⚠ Smooth terms in non-Gaussian GAMM: Coming in Phase 4.2")
    print("  ⚠ Laplace approximation: Planned for Phase 4.3")

    print("\n" + "#"*70 + "\n")


if __name__ == '__main__':
    main()
