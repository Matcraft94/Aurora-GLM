"""Tests for negative binomial GLM with estimated dispersion (glm.nb flow)."""

import numpy as np
import pytest

from aurora.distributions.families import NegativeBinomialFamily
from aurora.models.glm.fitting import fit_glm


@pytest.fixture()
def nb_data():
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.normal(size=n)
    mu = np.exp(1.0 + 0.8 * x)
    theta_true = 2.0
    lam = rng.gamma(theta_true, mu / theta_true)
    y = rng.poisson(lam).astype(float)
    return x.reshape(-1, 1), y, theta_true


def test_theta_estimate_recovers_true_theta(nb_data):
    """fit_glm with theta='estimate' runs the two-step glm.nb flow."""
    X, y, theta_true = nb_data
    res = fit_glm(X, y, family=NegativeBinomialFamily(theta="estimate"))

    assert res.converged_
    assert hasattr(res, "theta_")
    assert res.theta_ == pytest.approx(theta_true, rel=0.2)
    beta = np.concatenate([[res.intercept_], res.coef_])
    assert beta[0] == pytest.approx(1.0, abs=0.1)
    assert beta[1] == pytest.approx(0.8, abs=0.1)


def test_theta_fixed_unchanged(nb_data):
    """A fixed theta must behave exactly as before (no two-step flow)."""
    X, y, _ = nb_data
    res = fit_glm(X, y, family=NegativeBinomialFamily(theta=2.0))

    assert res.converged_
    assert res.theta_ is None
