# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Validation of prior-weights propagation in GLM fitting against statsmodels.

Verifies that ``fit_glm(..., weights=w)`` follows the R ``glm`` convention
(McCullagh & Nelder §2.3): deviance D = Σ wᵢdᵢ and log-likelihood
ℓ = Σ wᵢℓᵢ, with weighted null deviance, AIC and BIC derived from them.
Reference: ``statsmodels.GLM(freq_weights=w)``, which shares these formulas
for frequency (integer-replicate) weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from aurora.distributions.families import GaussianFamily, PoissonFamily
from aurora.models.glm import fit_glm

TOL = 1e-8


@pytest.fixture
def weighted_gaussian_data():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 2))
    y = 0.5 + X @ np.array([1.5, -0.5]) + rng.standard_normal(n) * 2.0
    w = rng.integers(1, 5, n).astype(float)
    return X, y, w


@pytest.fixture
def weighted_poisson_data():
    rng = np.random.default_rng(123)
    n = 300
    X = rng.standard_normal((n, 2))
    mu = np.exp(0.2 + X @ np.array([0.5, -0.3]))
    y = rng.poisson(mu).astype(float)
    w = rng.integers(1, 4, n).astype(float)
    return X, y, w


class TestWeightedGaussianVsStatsmodels:
    def test_coefficients(self, weighted_gaussian_data):
        X, y, w = weighted_gaussian_data
        aurora = fit_glm(X, y, family=GaussianFamily(), weights=w)
        sm_res = sm.GLM(y, sm.add_constant(X), family=sm.families.Gaussian(), freq_weights=w).fit()
        np.testing.assert_allclose(aurora.coef_, sm_res.params[1:], rtol=TOL, atol=TOL)
        assert aurora.intercept_ == pytest.approx(sm_res.params[0], rel=TOL)

    def test_deviance_and_null_deviance(self, weighted_gaussian_data):
        X, y, w = weighted_gaussian_data
        aurora = fit_glm(X, y, family=GaussianFamily(), weights=w)
        sm_res = sm.GLM(y, sm.add_constant(X), family=sm.families.Gaussian(), freq_weights=w).fit()
        assert aurora.deviance_ == pytest.approx(sm_res.deviance, rel=TOL)
        assert aurora.null_deviance_ == pytest.approx(sm_res.null_deviance, rel=TOL)

    def test_log_likelihood_aic(self, weighted_gaussian_data):
        X, y, w = weighted_gaussian_data
        aurora = fit_glm(X, y, family=GaussianFamily(), weights=w)
        sm_res = sm.GLM(y, sm.add_constant(X), family=sm.families.Gaussian(), freq_weights=w).fit()
        assert aurora.log_likelihood_ == pytest.approx(sm_res.llf, abs=TOL)
        assert aurora.aic_ == pytest.approx(sm_res.aic, abs=1e-6)


class TestWeightedPoissonVsStatsmodels:
    def test_coefficients(self, weighted_poisson_data):
        X, y, w = weighted_poisson_data
        aurora = fit_glm(X, y, family=PoissonFamily(), weights=w)
        sm_res = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson(), freq_weights=w).fit()
        np.testing.assert_allclose(aurora.coef_, sm_res.params[1:], rtol=1e-7, atol=1e-8)
        assert aurora.intercept_ == pytest.approx(sm_res.params[0], rel=1e-7)

    def test_deviance_and_null_deviance(self, weighted_poisson_data):
        X, y, w = weighted_poisson_data
        aurora = fit_glm(X, y, family=PoissonFamily(), weights=w)
        sm_res = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson(), freq_weights=w).fit()
        assert aurora.deviance_ == pytest.approx(sm_res.deviance, rel=TOL)
        assert aurora.null_deviance_ == pytest.approx(sm_res.null_deviance, rel=TOL)

    def test_log_likelihood_aic_bic(self, weighted_poisson_data):
        X, y, w = weighted_poisson_data
        aurora = fit_glm(X, y, family=PoissonFamily(), weights=w)
        sm_res = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson(), freq_weights=w).fit()
        assert aurora.log_likelihood_ == pytest.approx(sm_res.llf, abs=TOL)
        assert aurora.aic_ == pytest.approx(sm_res.aic, abs=TOL)
        n = len(y)
        expected_bic = -2 * sm_res.llf + np.log(n) * len(sm_res.params)
        assert aurora.bic_ == pytest.approx(expected_bic, abs=1e-6)


class TestWeightsSemantics:
    """Prior weights must actually change the fit statistics."""

    def test_weights_change_fit_statistics(self, weighted_poisson_data):
        X, y, w = weighted_poisson_data
        unweighted = fit_glm(X, y, family=PoissonFamily())
        weighted = fit_glm(X, y, family=PoissonFamily(), weights=w)
        assert weighted.deviance_ != pytest.approx(unweighted.deviance_, rel=1e-3)
        assert weighted.log_likelihood_ != pytest.approx(unweighted.log_likelihood_, rel=1e-3)

    def test_unit_weights_reproduce_unweighted(self, weighted_poisson_data):
        X, y, _ = weighted_poisson_data
        unweighted = fit_glm(X, y, family=PoissonFamily())
        unit = fit_glm(X, y, family=PoissonFamily(), weights=np.ones(len(y)))
        np.testing.assert_allclose(unit.coef_, unweighted.coef_, rtol=1e-10)
        assert unit.deviance_ == pytest.approx(unweighted.deviance_, rel=1e-10)
        assert unit.log_likelihood_ == pytest.approx(unweighted.log_likelihood_, rel=1e-10)
        assert unit.aic_ == pytest.approx(unweighted.aic_, rel=1e-10)
