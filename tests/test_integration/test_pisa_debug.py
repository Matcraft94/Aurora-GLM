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

    from aurora.models.gamm.fitting import fit_gamm_gaussian
    from aurora.models.gamm.random_effects import RandomEffect

    df = pd.read_csv(DATA_PATH)
    y = df["zread"].values
    X = np.ones((len(df), 1))

    school_ids = df["schoolid"].values
    unique_schools = np.unique(school_ids)
    school_map = {sid: i for i, sid in enumerate(unique_schools)}
    school_indices = np.array([school_map[sid] for sid in school_ids])

    re = RandomEffect(grouping="schoolid", include_intercept=True, covariance="identity")
    groups_data = {"schoolid": school_indices}

    result = fit_gamm_gaussian(y=y, X=X, random_effects=[re], groups_data=groups_data, reml=True)

    assert result.converged, "Model should converge"

    tau_squared = result.variance_components[0][0, 0]
    sigma_squared = result.residual_variance
    icc = tau_squared / (tau_squared + sigma_squared)

    # Sanity checks
    assert 0 < icc < 1, f"ICC should be between 0 and 1, got {icc}"
    assert sigma_squared > 0, "Residual variance should be positive"

    # Random effects should be present
    assert result.random_effects is not None
    assert len(result.random_effects) > 0
    school_effects = result.random_effects[0]
    assert school_effects.shape[0] == len(unique_schools)
