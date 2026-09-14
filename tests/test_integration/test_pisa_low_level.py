"""Integration test: PISA UK low-level GAMM API with random effects."""

from pathlib import Path

import pytest

DATA_PATH = Path("examples/06_case_studies/data/pisaUK.csv")


@pytest.mark.integration
@pytest.mark.slow
def test_pisa_low_level_gamm():
    """Test PISA UK data with low-level fit_gamm_gaussian API."""
    if not DATA_PATH.exists():
        pytest.skip(f"PISA data not found at {DATA_PATH}")

    import numpy as np
    import pandas as pd

    from aurora.models.gamm.design import construct_Z_matrix
    from aurora.models.gamm.fitting import fit_gamm_gaussian
    from aurora.models.gamm.random_effects import RandomEffect

    df = pd.read_csv(DATA_PATH)
    y = df["zread"].values
    X = np.ones((len(df), 1))

    school_ids = df["schoolid"].values
    unique_schools = np.unique(school_ids)

    re = RandomEffect(grouping="schoolid", include_intercept=True, covariance="identity")
    groups_data = {"schoolid": school_ids}
    Z, Z_info = construct_Z_matrix(X, [re], groups_data)

    result = fit_gamm_gaussian(X_parametric=X, X_smooth=None, Z=Z, Z_info=Z_info, y=y)

    assert result.converged, "Model should converge"

    tau_squared = result.variance_components[0][0, 0]
    sigma_squared = result.residual_variance
    icc = tau_squared / (tau_squared + sigma_squared)

    # Sanity checks
    assert 0 < icc < 1, f"ICC should be between 0 and 1, got {icc}"
    assert sigma_squared > 0, "Residual variance should be positive"

    # Random effects should be present (mapping group -> coefficients)
    assert result.random_effects is not None
    assert len(result.random_effects) > 0
    school_effects = result.random_effects["schoolid"]
    assert len(school_effects) == len(unique_schools)
