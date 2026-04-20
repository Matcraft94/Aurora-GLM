"""Integration test: PISA UK data loading and null model fitting."""

from pathlib import Path

import pytest

DATA_PATH = Path("examples/06_case_studies/data/pisaUK.csv")


@pytest.mark.integration
@pytest.mark.slow
def test_pisa_null_model():
    """Test PISA UK data loading and null GAMM model (zread ~ 1 + (1|schoolid))."""
    if not DATA_PATH.exists():
        pytest.skip(f"PISA data not found at {DATA_PATH}")

    import pandas as pd

    from aurora.models.gamm import fit_gamm

    df = pd.read_csv(DATA_PATH)

    assert len(df) > 0
    assert "schoolid" in df.columns
    assert "zread" in df.columns

    result = fit_gamm(
        formula="zread ~ 1 + (1 | schoolid)", data=df, family="gaussian", covariance="identity"
    )

    assert result.converged, "Model should converge"

    tau_squared = result.variance_components[0][0, 0]
    sigma_squared = result.residual_variance
    icc = tau_squared / (tau_squared + sigma_squared)

    # Sanity checks
    assert 0 < icc < 1, f"ICC should be between 0 and 1, got {icc}"
    assert sigma_squared > 0, "Residual variance should be positive"
    assert len(result.beta_parametric) >= 1, "Should have at least intercept"
