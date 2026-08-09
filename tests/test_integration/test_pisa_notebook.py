"""Integration test: PISA UK multilevel analysis (null, student, full models)."""

from pathlib import Path

import pytest

DATA_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "06_case_studies" / "data" / "pisaUK.csv"
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def pisa_df():
    """Load PISA UK data and center student-level predictors."""
    if not DATA_PATH.exists():
        pytest.skip(f"PISA data not found at {DATA_PATH}")

    pd = pytest.importorskip("pandas")

    df = pd.read_csv(DATA_PATH)
    df["age_c"] = df["age"] - df["age"].mean()
    df["wealth_c"] = df["wealth"] - df["wealth"].mean()
    df["cultposs_c"] = df["cultposs"] - df["cultposs"].mean()
    df["hedres_c"] = df["hedres"] - df["hedres"].mean()
    df["lmins_c"] = (df["lmins"] - df["lmins"].mean()) / 60
    return df


@pytest.fixture(scope="module")
def result_null(pisa_df):
    """Model 1: Null model (variance decomposition)."""
    from aurora.models.gamm import fit_gamm

    return fit_gamm(
        formula="zread ~ 1 + (1 | schoolid)",
        data=pisa_df,
        family="gaussian",
        covariance="identity",
    )


@pytest.fixture(scope="module")
def result_student(pisa_df):
    """Model 2: Student-level predictors."""
    from aurora.models.gamm import fit_gamm

    return fit_gamm(
        formula=(
            "zread ~ age_c + female + immig + hisced + wealth_c + cultposs_c"
            " + hedres_c + lmins_c + (1 | schoolid)"
        ),
        data=pisa_df,
        family="gaussian",
        covariance="identity",
    )


@pytest.fixture(scope="module")
def result_full(pisa_df):
    """Model 3: Full model with school-level predictors."""
    from aurora.models.gamm import fit_gamm

    df = pisa_df.copy()
    df["stratio_c"] = df["stratio"] - df["stratio"].mean()
    df["schsize_c"] = (df["schsize"] - df["schsize"].mean()) / 100
    df["private"] = (df["schltype"] == 1).astype(int)
    df["public"] = (df["schltype"] == 2).astype(int)

    return fit_gamm(
        formula=(
            "zread ~ age_c + female + immig + hisced + wealth_c + cultposs_c"
            " + hedres_c + lmins_c + private + public + stratio_c + schsize_c"
            " + (1 | schoolid)"
        ),
        data=df,
        family="gaussian",
        covariance="identity",
    )


def test_null_model_variance_decomposition(result_null):
    """Null model converges and yields a sensible ICC."""
    assert result_null.converged, "Null model should converge"

    tau_squared = result_null.variance_components[0][0, 0]
    sigma_squared = result_null.residual_variance
    icc = tau_squared / (tau_squared + sigma_squared)

    assert tau_squared > 0, "Between-school variance should be positive"
    assert sigma_squared > 0, "Residual variance should be positive"
    assert 0 < icc < 1, f"ICC should be between 0 and 1, got {icc}"


def test_student_model_reduces_variance(result_null, result_student):
    """Student predictors converge and explain variance at both levels."""
    assert result_student.converged, "Student model should converge"
    assert len(result_student.beta_parametric) == 9

    tau_null = result_null.variance_components[0][0, 0]
    sigma_null = result_null.residual_variance
    tau_student = result_student.variance_components[0][0, 0]
    sigma_student = result_student.residual_variance

    assert tau_student < tau_null, "Student predictors should reduce between-school variance"
    assert sigma_student < sigma_null, "Student predictors should reduce within-school variance"


def test_student_model_r2(result_student):
    """Conditional R² exceeds marginal R² for the student model."""
    from aurora.models.gamm.diagnostics import compute_r2_conditional_marginal

    r2_m, r2_c = compute_r2_conditional_marginal(result_student)

    assert 0 <= r2_m <= 1, f"Marginal R² should be in [0, 1], got {r2_m}"
    assert 0 <= r2_c <= 1, f"Conditional R² should be in [0, 1], got {r2_c}"
    assert r2_c >= r2_m, "Conditional R² should be >= marginal R²"


def test_full_model_reduces_between_variance(result_student, result_full):
    """School predictors further reduce between-school variance."""
    assert result_full.converged, "Full model should converge"
    assert len(result_full.beta_parametric) == 13

    tau_student = result_student.variance_components[0][0, 0]
    tau_full = result_full.variance_components[0][0, 0]

    assert tau_full < tau_student, "School predictors should reduce between-school variance"


def test_full_model_r2(result_full):
    """Full model R² values are valid and conditional >= marginal."""
    from aurora.models.gamm.diagnostics import compute_r2_conditional_marginal

    r2_m, r2_c = compute_r2_conditional_marginal(result_full)

    assert 0 <= r2_m <= 1, f"Marginal R² should be in [0, 1], got {r2_m}"
    assert 0 <= r2_c <= 1, f"Conditional R² should be in [0, 1], got {r2_c}"
    assert r2_c >= r2_m, "Conditional R² should be >= marginal R²"
