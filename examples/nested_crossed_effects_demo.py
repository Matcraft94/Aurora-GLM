"""Demonstration of Nested and Crossed Random Effects in GAMM.

This example shows how to model complex hierarchical structures using:
1. Nested random effects: observations within groups within larger groups
2. Crossed random effects: observations belong to multiple independent grouping factors

Examples inspired by real-world scenarios in education, psychometrics, and ecology.
"""

import numpy as np
import pandas as pd

from aurora.models.gamm import fit_gamm


def example_1_nested_effects_education():
    """Example 1: Students nested within classrooms within schools.

    Hierarchical structure:
    - Level 3: Schools (e.g., 10 schools)
    - Level 2: Classrooms within schools (e.g., 3 classrooms per school)
    - Level 1: Students within classrooms (e.g., 20 students per classroom)

    Model: score ~ study_hours + (1 | school/classroom)

    This expands to random intercepts for schools AND for classrooms nested within schools.
    """
    print("\n" + "="*70)
    print("Example 1: Nested Random Effects - Students in Classrooms in Schools")
    print("="*70)

    np.random.seed(42)

    # Structure
    n_schools = 10
    n_classrooms_per_school = 3
    n_students_per_classroom = 20

    n_classrooms = n_schools * n_classrooms_per_school
    n_students = n_classrooms * n_students_per_classroom

    print(f"\nHierarchical Structure:")
    print(f"  Schools: {n_schools}")
    print(f"  Classrooms: {n_classrooms} ({n_classrooms_per_school} per school)")
    print(f"  Students: {n_students} ({n_students_per_classroom} per classroom)")

    # Generate hierarchical data
    school_ids = np.repeat(np.arange(n_schools), n_classrooms_per_school * n_students_per_classroom)
    classroom_ids = np.repeat(np.arange(n_classrooms), n_students_per_classroom)

    # Predictors
    study_hours = np.random.uniform(1, 5, n_students)

    # Random effects at each level
    school_effects = np.random.randn(n_schools) * 5  # School quality variation
    classroom_effects = np.random.randn(n_classrooms) * 3  # Teacher quality variation

    # Generate test scores
    baseline_score = 70
    study_effect = 4.0
    scores = (
        baseline_score +
        study_effect * study_hours +
        school_effects[school_ids] +
        classroom_effects[classroom_ids] +
        np.random.randn(n_students) * 5  # Individual variation
    )

    # Create DataFrame
    df = pd.DataFrame({
        'score': scores,
        'study_hours': study_hours,
        'school': school_ids,
        'classroom': classroom_ids
    })

    print(f"\nData Summary:")
    print(f"  Mean score: {scores.mean():.2f}")
    print(f"  Score range: [{scores.min():.1f}, {scores.max():.1f}]")
    print(f"  Study hours range: [{study_hours.min():.1f}, {study_hours.max():.1f}]")

    # Fit nested random effects model
    print("\nFitting model: score ~ study_hours + (1 | school/classroom)")
    print("  This creates random intercepts for:")
    print("    - Schools (Level 3)")
    print("    - Classrooms nested within schools (Level 2)")

    result = fit_gamm(
        formula='score ~ study_hours + (1 | school/classroom)',
        data=df,
        family='gaussian',
        covariance='identity'
    )

    print(f"\nModel Results:")
    print(f"  Converged: {result.converged}")
    print(f"\nFixed Effects:")
    print(f"  Intercept: {result.beta_parametric[0]:.2f} (true: {baseline_score})")
    print(f"  Study hours effect: {result.beta_parametric[1]:.2f} (true: {study_effect})")
    print(f"\nRandom Effects Variance:")
    print(f"  School level: {result.variance_components[0][0,0]:.2f} (true: 25)")
    print(f"  Classroom level: {result.variance_components[1][0,0]:.2f} (true: 9)")
    print(f"  Residual: {result.residual_variance:.2f} (true: 25)")

    # Variance partition coefficient
    total_var = (result.variance_components[0][0,0] +
                 result.variance_components[1][0,0] +
                 result.residual_variance)
    vpc_school = result.variance_components[0][0,0] / total_var
    vpc_classroom = result.variance_components[1][0,0] / total_var

    print(f"\nVariance Partition:")
    print(f"  School explains: {vpc_school*100:.1f}% of variance")
    print(f"  Classroom explains: {vpc_classroom*100:.1f}% of variance")
    print(f"  Student explains: {(1-vpc_school-vpc_classroom)*100:.1f}% of variance")

    return result


def example_2_crossed_effects_psychometrics():
    """Example 2: Crossed random effects - subjects and items.

    In psychometric studies, each subject rates multiple items,
    and each item is rated by multiple subjects. Subjects and items
    are not hierarchically nested - they are crossed.

    Model: rating ~ difficulty + (1 | subject) + (1 | item)
    """
    print("\n" + "="*70)
    print("Example 2: Crossed Random Effects - Subjects × Items")
    print("="*70)

    np.random.seed(123)

    # Structure
    n_subjects = 30
    n_items = 20

    print(f"\nCrossed Design:")
    print(f"  Subjects: {n_subjects}")
    print(f"  Items: {n_items}")
    print(f"  Design: Each subject rates all items")
    print(f"  Total observations: {n_subjects * n_items}")

    # Generate crossed data (each subject rates each item)
    subjects = np.repeat(np.arange(n_subjects), n_items)
    items = np.tile(np.arange(n_items), n_subjects)

    # Item difficulty (fixed effect predictor)
    item_difficulty = np.random.uniform(-1, 1, n_items)
    difficulty_values = item_difficulty[items]

    # Random effects
    subject_ability = np.random.randn(n_subjects) * 0.8  # Subject variation
    item_appeal = np.random.randn(n_items) * 0.6  # Item variation

    # Generate ratings (1-5 scale)
    baseline = 3.0
    difficulty_effect = -0.7  # Harder items get lower ratings
    ratings_continuous = (
        baseline +
        difficulty_effect * difficulty_values +
        subject_ability[subjects] +
        item_appeal[items] +
        np.random.randn(n_subjects * n_items) * 0.5
    )

    # Clip to valid range and round
    ratings = np.clip(np.round(ratings_continuous), 1, 5)

    # Create DataFrame
    df = pd.DataFrame({
        'rating': ratings,
        'difficulty': difficulty_values,
        'subject': subjects,
        'item': items
    })

    print(f"\nData Summary:")
    print(f"  Mean rating: {ratings.mean():.2f}")
    print(f"  Rating distribution:")
    for r in range(1, 6):
        count = (ratings == r).sum()
        pct = count / len(ratings) * 100
        print(f"    {r}: {count} ({pct:.1f}%)")

    # Fit crossed random effects model
    print("\nFitting model: rating ~ difficulty + (1 | subject) + (1 | item)")
    print("  Random intercepts for:")
    print("    - Subjects (captures individual rating tendencies)")
    print("    - Items (captures item-specific appeal beyond difficulty)")

    result = fit_gamm(
        formula='rating ~ difficulty + (1 | subject) + (1 | item)',
        data=df,
        family='gaussian',
        covariance='identity'
    )

    print(f"\nModel Results:")
    print(f"  Converged: {result.converged}")
    print(f"\nFixed Effects:")
    print(f"  Intercept: {result.beta_parametric[0]:.2f} (true: {baseline})")
    print(f"  Difficulty effect: {result.beta_parametric[1]:.2f} (true: {difficulty_effect})")
    print(f"\nRandom Effects Variance:")
    print(f"  Subject: {result.variance_components[0][0,0]:.3f} (true: 0.64)")
    print(f"  Item: {result.variance_components[1][0,0]:.3f} (true: 0.36)")
    print(f"  Residual: {result.residual_variance:.3f} (true: 0.25)")

    return result


def example_3_complex_crossed_nested():
    """Example 3: Both crossed and nested - clinical trial.

    Structure:
    - Patients nested within clinics
    - Patients crossed with time points
    - Each patient measured at each time point

    Model: outcome ~ treatment + time + (1 | clinic) + (1 | patient)
    """
    print("\n" + "="*70)
    print("Example 3: Mixed Crossed/Nested - Clinical Trial")
    print("="*70)

    np.random.seed(456)

    # Structure
    n_clinics = 5
    n_patients_per_clinic = 20
    n_timepoints = 4

    n_patients = n_clinics * n_patients_per_clinic
    n_obs = n_patients * n_timepoints

    print(f"\nStudy Design:")
    print(f"  Clinics: {n_clinics}")
    print(f"  Patients: {n_patients} ({n_patients_per_clinic} per clinic)")
    print(f"  Time points: {n_timepoints} (0, 1, 2, 3 months)")
    print(f"  Total observations: {n_obs}")

    # Generate data
    clinic_ids = np.repeat(np.arange(n_clinics), n_patients_per_clinic * n_timepoints)
    patient_ids = np.repeat(np.arange(n_patients), n_timepoints)
    timepoints = np.tile(np.arange(n_timepoints), n_patients)

    # Treatment assignment (randomized within clinics)
    treatment = np.random.binomial(1, 0.5, n_patients)
    treatment_values = treatment[patient_ids]

    # Random effects
    clinic_effects = np.random.randn(n_clinics) * 2.0
    patient_effects = np.random.randn(n_patients) * 3.0

    # Generate outcome
    baseline = 50
    treatment_effect = 5.0
    time_effect = -2.0  # Disease progression without treatment

    outcome = (
        baseline +
        treatment_effect * treatment_values +
        time_effect * timepoints +
        clinic_effects[clinic_ids] +
        patient_effects[patient_ids] +
        np.random.randn(n_obs) * 4.0
    )

    # Create DataFrame
    df = pd.DataFrame({
        'outcome': outcome,
        'treatment': treatment_values,
        'time': timepoints,
        'clinic': clinic_ids,
        'patient': patient_ids
    })

    print(f"\nData Summary:")
    print(f"  Mean outcome: {outcome.mean():.2f}")
    print(f"  Treatment groups: {treatment.sum()} treated, {n_patients - treatment.sum()} control")

    # Fit model
    print("\nFitting model: outcome ~ treatment + time + (1 | clinic) + (1 | patient)")
    print("  Structure:")
    print("    - Patients NESTED within clinics")
    print("    - Measurements CROSSED with time points")

    result = fit_gamm(
        formula='outcome ~ treatment + time + (1 | clinic) + (1 | patient)',
        data=df,
        family='gaussian',
        covariance='identity'
    )

    print(f"\nModel Results:")
    print(f"  Converged: {result.converged}")
    print(f"\nFixed Effects:")
    print(f"  Intercept: {result.beta_parametric[0]:.2f} (true: {baseline})")
    print(f"  Treatment effect: {result.beta_parametric[1]:.2f} (true: {treatment_effect})")
    print(f"  Time effect: {result.beta_parametric[2]:.2f} (true: {time_effect})")
    print(f"\nRandom Effects Variance:")
    print(f"  Clinic: {result.variance_components[0][0,0]:.2f} (true: 4.0)")
    print(f"  Patient: {result.variance_components[1][0,0]:.2f} (true: 9.0)")

    return result


def main():
    """Run all examples demonstrating nested and crossed random effects."""
    print("\n" + "#"*70)
    print("# Nested and Crossed Random Effects in GAMM")
    print("# Aurora-GLM: Advanced Hierarchical Modeling")
    print("#"*70)

    # Run examples
    result1 = example_1_nested_effects_education()
    result2 = example_2_crossed_effects_psychometrics()
    result3 = example_3_complex_crossed_nested()

    print("\n" + "="*70)
    print("Summary: All Nested/Crossed Effects Examples Completed!")
    print("="*70)

    print("\nKey Concepts Demonstrated:")
    print("  ✓ Nested effects: (1 | level1/level2)")
    print("    - Used when observations are hierarchically structured")
    print("    - Example: Students within classrooms within schools")
    print()
    print("  ✓ Crossed effects: (1 | factor1) + (1 | factor2)")
    print("    - Used when grouping factors are independent")
    print("    - Example: Subjects crossed with items in psychometrics")
    print()
    print("  ✓ Mixed structures: Combination of nested and crossed")
    print("    - Real-world complexity often requires both")
    print("    - Example: Longitudinal clinical trial data")

    print("\nFormula Syntax Supported:")
    print("  • (1 | group)              - Random intercept")
    print("  • (1 + x | group)          - Random intercept + slope")
    print("  • (1 | a/b)                - Nested: b within a")
    print("  • (1 | a) + (1 | b)        - Crossed: a and b independent")
    print("  • y ~ s(x) + (1 | group)   - With smooth terms (GAM + GAMM)")

    print("\n" + "#"*70 + "\n")


if __name__ == '__main__':
    main()
