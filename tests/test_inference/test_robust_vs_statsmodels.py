# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Verification of robust (sandwich) covariance against statsmodels.

Conventions verified here:

- Poisson / Binomial GLMs: aurora HC0 matches
  ``statsmodels.GLM.fit(cov_type='HC0'...'HC3')`` exactly (statsmodels
  returns *identical* values for all four HC types on GLMs — it does not
  apply leverage corrections to GLM sandwiches; aurora HC1–HC4 implement
  the Zeileis (2006) weighted-leverage corrections instead).
- Gaussian/identity: aurora HC0–HC3 match
  ``statsmodels.OLS.fit(cov_type='HC0'...'HC3')`` (W = 1, Pearson =
  response residuals), which validates the leverage-correction machinery.
- Poisson HC1–HC3 are checked against an independent implementation of
  the weighted sandwich formula.
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from aurora.inference import bootstrap_inference, robust_covariance
from aurora.models import fit_glm


def _full_se(robust_result):
    return np.concatenate(([robust_result.intercept_std_error], robust_result.std_errors))


@pytest.fixture
def poisson_fit():
    rng = np.random.default_rng(3)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    mu = np.exp(0.4 + 0.7 * x1 - 0.3 * x2)
    y = rng.poisson(mu)
    return X, y, fit_glm(X[:, 1:], y, family="poisson")


@pytest.fixture
def binomial_fit():
    rng = np.random.default_rng(5)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    p = 1 / (1 + np.exp(-(0.2 + 1.0 * x1 - 0.6 * x2)))
    y = rng.binomial(1, p).astype(float)
    return X, y, fit_glm(X[:, 1:], y, family="binomial")


@pytest.fixture
def gaussian_fit():
    rng = np.random.default_rng(9)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    y = 1 + 2 * x1 - 0.5 * x2 + rng.normal(size=n) * (1 + 0.5 * np.abs(x1))
    return X, y, fit_glm(X[:, 1:], y, family="gaussian")


class TestHC0MatchesStatsmodelsGLM:
    @pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
    def test_poisson_hc0(self, poisson_fit, cov_type):
        X, y, result = poisson_fit
        se = _full_se(robust_covariance(result, hc_type="HC0"))
        ref = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type=cov_type)
        # statsmodels gives identical GLM sandwiches for HC0-HC3
        assert np.allclose(se, ref.bse, rtol=1e-6, atol=1e-12)

    @pytest.mark.parametrize("cov_type", ["HC0", "HC3"])
    def test_binomial_hc0(self, binomial_fit, cov_type):
        X, y, result = binomial_fit
        se = _full_se(robust_covariance(result, hc_type="HC0"))
        ref = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type=cov_type)
        assert np.allclose(se, ref.bse, rtol=1e-6, atol=1e-12)


class TestGaussianMatchesStatsmodelsOLS:
    """Gaussian/identity reduces to the OLS sandwich (W = 1)."""

    @pytest.mark.parametrize("hc_type", ["HC0", "HC1", "HC2", "HC3"])
    def test_gaussian_hc(self, gaussian_fit, hc_type):
        X, y, result = gaussian_fit
        se = _full_se(robust_covariance(result, hc_type=hc_type))
        ref = sm.OLS(y, X).fit(cov_type=hc_type)
        assert np.allclose(se, ref.bse, rtol=1e-6, atol=1e-12)


class TestGLMLeverageCorrections:
    """HC1-HC3 for non-Gaussian GLMs: weighted hat + Pearson residuals."""

    @pytest.mark.parametrize("hc_type", ["HC1", "HC2", "HC3"])
    def test_poisson_weighted_formula(self, poisson_fit, hc_type):
        X, y, result = poisson_fit
        robust = robust_covariance(result, hc_type=hc_type)

        # Independent reimplementation of the Zeileis (2006) GLM sandwich
        mu = np.asarray(result.mu_, dtype=float)
        deriv = np.asarray(result.link.derivative(mu), dtype=float)
        variance = np.asarray(result.family.variance(mu), dtype=float)
        w = 1.0 / (deriv**2 * variance)
        r_pearson = (y - mu) / np.sqrt(variance)

        xw = X * np.sqrt(w)[:, None]
        bread = np.linalg.inv(xw.T @ xw)
        h = np.einsum("ij,jk,ik->i", xw, bread, xw)

        omega = w * r_pearson**2
        n, p = X.shape
        if hc_type == "HC1":
            omega = omega * (n / (n - p))
        elif hc_type == "HC2":
            omega = omega / (1 - h)
        else:
            omega = omega / (1 - h) ** 2

        meat = X.T @ (X * omega[:, None])
        cov_expected = bread @ meat @ bread

        assert np.allclose(robust.coef_cov, cov_expected, rtol=1e-8, atol=1e-14)

    def test_hc3_at_least_hc0(self, poisson_fit):
        _, _, result = poisson_fit
        se0 = _full_se(robust_covariance(result, hc_type="HC0"))
        se3 = _full_se(robust_covariance(result, hc_type="HC3"))
        # Leverage corrections inflate SEs (up to small numerical noise)
        assert np.all(se3 >= se0 * 0.99)

    def test_hc4_finite_and_reasonable(self, poisson_fit):
        _, _, result = poisson_fit
        robust = robust_covariance(result, hc_type="HC4")
        se = _full_se(robust)
        assert np.all(np.isfinite(se))
        assert np.all(se > 0)


class TestBootstrapQuickWins:
    def test_reports_n_failed(self, gaussian_fit):
        _, _, result = gaussian_fit
        boot = bootstrap_inference(result, n_bootstrap=50, seed=42)
        assert "n_failed" in boot
        assert boot["n_failed"] == 0
        assert boot["boot_coefs"].shape[0] == 50

    def test_global_random_state_untouched(self, gaussian_fit):
        _, _, result = gaussian_fit
        np.random.seed(123)
        expected = np.random.random(4)
        np.random.seed(123)
        bootstrap_inference(result, n_bootstrap=20, seed=7)
        after = np.random.random(4)
        assert np.allclose(expected, after)

    def test_weights_propagated(self):
        rng = np.random.default_rng(21)
        n = 80
        x = rng.normal(size=n)
        y = 1 + 2 * x + rng.normal(size=n) * 0.5
        weights = rng.uniform(0.5, 2.0, size=n)
        result = fit_glm(x.reshape(-1, 1), y, family="gaussian", weights=weights)
        boot = bootstrap_inference(result, n_bootstrap=30, seed=1)
        assert boot["n_failed"] == 0
        assert boot["std_errors"][0] > 0

    def test_offset_propagated(self):
        rng = np.random.default_rng(33)
        n = 100
        x = rng.normal(size=n)
        exposure = rng.uniform(0.5, 2.0, size=n)
        mu = exposure * np.exp(0.3 + 0.6 * x)
        y = rng.poisson(mu)
        result = fit_glm(x.reshape(-1, 1), y, family="poisson", offset=np.log(exposure))
        assert result._offset is not None
        boot = bootstrap_inference(result, n_bootstrap=30, seed=1)
        assert boot["n_failed"] == 0
        # Slope should be recovered near 0.6 in most bootstrap samples
        assert 0.2 < np.median(boot["boot_coefs"][:, 1]) < 1.0
