# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Validation of grouped binomial GLMs (n trials) against statsmodels.

Aurora follows the R ``binomial()`` convention: the response is a proportion
in [0, 1], the variance function is V(μ) = μ(1 − μ), and the number of
trials enters as a prior weight. statsmodels reference fits use the
two-column ``endog = [successes, failures]`` interface, whose log-likelihood
includes the binomial coefficient term log C(n, y).
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from aurora.distributions.families import BinomialFamily
from aurora.models.glm import fit_glm

TOL = 1e-8


@pytest.fixture
def grouped_binomial_data():
    rng = np.random.default_rng(7)
    n = 150
    X = rng.standard_normal((n, 2))
    eta = 0.3 + X @ np.array([0.8, -0.6])
    p = 1.0 / (1.0 + np.exp(-eta))
    trials = rng.integers(5, 20, n)
    successes = rng.binomial(trials, p)
    return X, successes, trials


def _fit_statsmodels_grouped(X, successes, trials):
    endog = np.column_stack([successes, trials - successes])
    return sm.GLM(endog, sm.add_constant(X), family=sm.families.Binomial()).fit()


class TestGroupedBinomialVsStatsmodels:
    def test_coefficients(self, grouped_binomial_data):
        X, successes, trials = grouped_binomial_data
        aurora = fit_glm(
            X, successes / trials, family=BinomialFamily(), weights=trials.astype(float)
        )
        sm_res = _fit_statsmodels_grouped(X, successes, trials)
        np.testing.assert_allclose(aurora.coef_, sm_res.params[1:], rtol=1e-7, atol=TOL)
        assert aurora.intercept_ == pytest.approx(sm_res.params[0], rel=1e-7)

    def test_deviance_and_null_deviance(self, grouped_binomial_data):
        X, successes, trials = grouped_binomial_data
        aurora = fit_glm(
            X, successes / trials, family=BinomialFamily(), weights=trials.astype(float)
        )
        sm_res = _fit_statsmodels_grouped(X, successes, trials)
        assert aurora.deviance_ == pytest.approx(sm_res.deviance, rel=TOL)
        assert aurora.null_deviance_ == pytest.approx(sm_res.null_deviance, rel=TOL)

    def test_log_likelihood_aic(self, grouped_binomial_data):
        """The log-likelihood must include the log C(n, k) term (R/statsmodels)."""
        X, successes, trials = grouped_binomial_data
        aurora = fit_glm(
            X, successes / trials, family=BinomialFamily(), weights=trials.astype(float)
        )
        sm_res = _fit_statsmodels_grouped(X, successes, trials)
        assert aurora.log_likelihood_ == pytest.approx(sm_res.llf, abs=TOL)
        assert aurora.aic_ == pytest.approx(sm_res.aic, abs=1e-6)

    def test_constant_trials_via_family_n(self, grouped_binomial_data):
        """BinomialFamily(n=k) must be equivalent to weights=k for constant trials."""
        X, successes, trials = grouped_binomial_data
        k = int(trials[0])
        # Restrict to a synthetic dataset with a constant trial count
        trials_const = np.full_like(trials, k)
        rng = np.random.default_rng(11)
        eta = 0.3 + X @ np.array([0.8, -0.6])
        p = 1.0 / (1.0 + np.exp(-eta))
        succ_const = rng.binomial(trials_const, p)
        props = succ_const / trials_const

        via_n = fit_glm(X, props, family=BinomialFamily(n=float(k)))
        via_w = fit_glm(X, props, family=BinomialFamily(), weights=trials_const.astype(float))
        np.testing.assert_allclose(via_n.coef_, via_w.coef_, rtol=1e-10)
        assert via_n.intercept_ == pytest.approx(via_w.intercept_, rel=1e-10)
        assert via_n.deviance_ == pytest.approx(via_w.deviance_, rel=1e-10)
        assert via_n.log_likelihood_ == pytest.approx(via_w.log_likelihood_, rel=1e-10)
        assert via_n.aic_ == pytest.approx(via_w.aic_, rel=1e-10)

    def test_count_response_rejected(self, grouped_binomial_data):
        """Raw counts (y > 1) must raise a clear error, not diverge silently."""
        X, successes, trials = grouped_binomial_data
        with pytest.raises(ValueError, match="proportion"):
            fit_glm(X, successes.astype(float), family=BinomialFamily())

    def test_bernoulli_unweighted_unchanged(self):
        """Binary data without weights keeps the plain Bernoulli likelihood."""
        rng = np.random.default_rng(3)
        n = 200
        X = rng.standard_normal((n, 2))
        p = 1.0 / (1.0 + np.exp(-(0.4 + X @ np.array([1.0, -0.5]))))
        y = rng.binomial(1, p).astype(float)
        aurora = fit_glm(X, y, family=BinomialFamily())
        sm_res = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
        assert aurora.log_likelihood_ == pytest.approx(sm_res.llf, abs=TOL)
        assert aurora.deviance_ == pytest.approx(sm_res.deviance, rel=1e-6)
