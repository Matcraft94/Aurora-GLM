# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Tests for aurora.inference.anova module.

Value-based tests against statsmodels oracles:

- Single-model Gaussian ANOVA (type=3, F) matches
  ``statsmodels.stats.anova_lm(fit, typ=3)`` (squared t-statistics).
- Single-model Poisson Wald χ² matches the squared z-statistics of
  ``statsmodels.GLM``.
- Gaussian model comparison (F) matches ``anova_lm(reduced, full)``.
- Poisson model comparison (Chisq) matches the statsmodels deviance
  difference, which for unit-dispersion families equals 2ΔlogLik.
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sst
from statsmodels.stats.anova import anova_lm

from aurora.inference.anova import anova_glm, likelihood_ratio_test
from aurora.models import fit_glm


@pytest.fixture
def gaussian_data():
    rng = np.random.default_rng(7)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    y = 1 + 2 * x1 - 0.5 * x2 + rng.normal(size=n) * 0.7
    return X, y, {"y": y, "x1": x1, "x2": x2}


@pytest.fixture
def poisson_data():
    rng = np.random.default_rng(11)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    mu = np.exp(0.3 + 0.8 * x1 - 0.4 * x2)
    y = rng.poisson(mu)
    return X, y


class TestAnovaSingleGaussian:
    """Single-model type-3 table vs statsmodels anova_lm(typ=3)."""

    def test_f_statistics_match_statsmodels(self, gaussian_data):
        X, y, data = gaussian_data
        result = fit_glm(X[:, 1:], y, family="gaussian")
        tab = anova_glm(result, type=3, test="F")

        sm_tab = anova_lm(smf.ols("y ~ x1 + x2", data=data).fit(), typ=3)
        # Rows: intercept, X0 (x1), X1 (x2) — same order as statsmodels
        assert np.allclose(tab.f_statistic, sm_tab["F"].values[:-1], rtol=1e-6)
        assert np.allclose(tab.p_value, sm_tab["PR(>F)"].values[:-1], atol=1e-10)

    def test_residuals_row_matches_statsmodels(self, gaussian_data):
        X, y, data = gaussian_data
        result = fit_glm(X[:, 1:], y, family="gaussian")
        tab = anova_glm(result, type=3, test="F")

        ols = smf.ols("y ~ x1 + x2", data=data).fit()
        assert tab.residual_df == int(ols.df_resid)
        assert np.isclose(tab.residual_ss, np.sum(ols.resid**2), rtol=1e-10)

    def test_ss_are_type3_partial(self, gaussian_data):
        X, y, data = gaussian_data
        result = fit_glm(X[:, 1:], y, family="gaussian")
        tab = anova_glm(result, type=3, test="F")

        sm_tab = anova_lm(smf.ols("y ~ x1 + x2", data=data).fit(), typ=3)
        assert np.allclose(tab.ss, sm_tab["sum_sq"].values[:-1], rtol=1e-6)

    def test_chisq_uses_wald_reference(self, gaussian_data):
        X, y, _ = gaussian_data
        result = fit_glm(X[:, 1:], y, family="gaussian")
        tab = anova_glm(result, type=3, test="Chisq")

        cov = np.asarray(result.coef_cov_, dtype=float)
        beta = np.concatenate(([result.intercept_], np.asarray(result.coef_, dtype=float)))
        expected = beta**2 / np.diag(cov)
        assert np.allclose(tab.f_statistic, expected, rtol=1e-10)
        assert np.allclose(tab.p_value, sst.chi2.sf(expected, 1), atol=1e-12)

    def test_type_1_and_2_raise_for_single_model(self, gaussian_data):
        X, y, _ = gaussian_data
        result = fit_glm(X[:, 1:], y, family="gaussian")
        with pytest.raises(NotImplementedError, match="type=3"):
            anova_glm(result, type=1)
        with pytest.raises(NotImplementedError, match="type=3"):
            anova_glm(result, type=2)


class TestAnovaSinglePoisson:
    """Single-model Wald χ² table vs statsmodels GLM z-tests."""

    def test_wald_statistics_match_statsmodels(self, poisson_data):
        X, y = poisson_data
        result = fit_glm(X[:, 1:], y, family="poisson")
        tab = anova_glm(result, type=3, test="Chisq")

        glm = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        z2 = (glm.params / glm.bse) ** 2
        assert np.allclose(tab.f_statistic, z2, rtol=1e-6)
        assert np.allclose(tab.p_value, glm.pvalues, atol=1e-10)

    def test_f_test_falls_back_to_chisq_with_warning(self, poisson_data):
        X, y = poisson_data
        result = fit_glm(X[:, 1:], y, family="poisson")
        with pytest.warns(RuntimeWarning, match="chi-squared"):
            tab = anova_glm(result, type=3, test="F")
        assert tab.test == "Chisq"
        assert np.all(0 <= tab.p_value) and np.all(tab.p_value <= 1)


class TestAnovaCompareGaussian:
    """Nested Gaussian comparison vs statsmodels anova_lm(m1, m2)."""

    def test_f_test_matches_statsmodels(self, gaussian_data):
        X, y, data = gaussian_data
        reduced = fit_glm(X[:, 1:2], y, family="gaussian")
        full = fit_glm(X[:, 1:], y, family="gaussian")
        tab = anova_glm(reduced, full, test="F")

        o1 = smf.ols("y ~ x1", data=data).fit()
        o2 = smf.ols("y ~ x1 + x2", data=data).fit()
        sm_tab = anova_lm(o1, o2)

        assert tab.df[0] == 1
        assert np.isclose(tab.ss[0], sm_tab["ss_diff"].values[1], rtol=1e-8)
        assert np.isclose(tab.f_statistic[0], sm_tab["F"].values[1], rtol=1e-8)
        assert np.isclose(tab.p_value[0], sm_tab["Pr(>F)"].values[1], atol=1e-12)
        assert tab.test == "F"

    def test_deviance_is_rss_for_gaussian(self, gaussian_data):
        X, y, data = gaussian_data
        reduced = fit_glm(X[:, 1:2], y, family="gaussian")
        full = fit_glm(X[:, 1:], y, family="gaussian")
        tab = anova_glm(reduced, full, test="Chisq")

        o2 = smf.ols("y ~ x1 + x2", data=data).fit()
        assert np.isclose(tab.residual_ss, np.sum(o2.resid**2), rtol=1e-10)


class TestAnovaComparePoisson:
    """Nested Poisson comparison: deviance difference = LRT statistic."""

    def test_chisq_matches_deviance_difference(self, poisson_data):
        X, y = poisson_data
        reduced = fit_glm(X[:, 1:2], y, family="poisson")
        full = fit_glm(X[:, 1:], y, family="poisson")
        tab = anova_glm(reduced, full, test="Chisq")

        g1 = sm.GLM(y, X[:, :2], family=sm.families.Poisson()).fit()
        g2 = sm.GLM(y, X, family=sm.families.Poisson()).fit()

        ddev = g1.deviance - g2.deviance
        assert np.isclose(tab.ss[0], ddev, rtol=1e-8)
        # Unit dispersion: deviance difference equals 2ΔlogLik
        assert np.isclose(ddev, 2 * (g2.llf - g1.llf), rtol=1e-8)
        assert np.isclose(tab.p_value[0], sst.chi2.sf(ddev, 1), atol=1e-12)

    def test_models_sorted_by_complexity(self, poisson_data):
        X, y = poisson_data
        reduced = fit_glm(X[:, 1:2], y, family="poisson")
        full = fit_glm(X[:, 1:], y, family="poisson")
        # Passing full first must give the same table
        tab = anova_glm(full, reduced, test="Chisq")
        assert tab.ss[0] >= 0
        assert 0 <= tab.p_value[0] <= 1


class TestAnovaCompareEdgeCases:
    def test_non_gaussian_f_warns_and_uses_chisq(self, poisson_data):
        X, y = poisson_data
        reduced = fit_glm(X[:, 1:2], y, family="poisson")
        full = fit_glm(X[:, 1:], y, family="poisson")
        with pytest.warns(RuntimeWarning, match="chi-squared"):
            tab = anova_glm(reduced, full, test="F")
        assert tab.test == "Chisq"

    def test_invalid_test_raises(self, gaussian_data):
        X, y, _ = gaussian_data
        result = fit_glm(X[:, 1:], y, family="gaussian")
        with pytest.raises(ValueError, match="Unknown test"):
            anova_glm(result, test="Wald")

    def test_three_nested_models(self, gaussian_data):
        X, y, data = gaussian_data
        m0 = fit_glm(np.zeros((len(y), 0)), y, family="gaussian")
        m1 = fit_glm(X[:, 1:2], y, family="gaussian")
        m2 = fit_glm(X[:, 1:], y, family="gaussian")
        tab = anova_glm(m0, m1, m2, test="F")

        o0 = smf.ols("y ~ 1", data=data).fit()
        o1 = smf.ols("y ~ x1", data=data).fit()
        o2 = smf.ols("y ~ x1 + x2", data=data).fit()
        sm_tab = anova_lm(o0, o1, o2)

        assert len(tab.df) == 2
        assert np.allclose(tab.df, sm_tab["df_diff"].values[1:])
        assert np.allclose(tab.f_statistic, sm_tab["F"].values[1:], rtol=1e-6)
        assert np.allclose(tab.p_value, sm_tab["Pr(>F)"].values[1:], atol=1e-10)


class TestLRTValues:
    """likelihood_ratio_test against statsmodels log-likelihoods."""

    def test_lrt_matches_statsmodels(self, poisson_data):
        X, y = poisson_data
        reduced = fit_glm(X[:, 1:2], y, family="poisson")
        full = fit_glm(X[:, 1:], y, family="poisson")
        result = likelihood_ratio_test(reduced, full)

        g1 = sm.GLM(y, X[:, :2], family=sm.families.Poisson()).fit()
        g2 = sm.GLM(y, X, family=sm.families.Poisson()).fit()

        expected_stat = 2 * (g2.llf - g1.llf)
        assert np.isclose(result.statistic, expected_stat, rtol=1e-8)
        assert result.df == 1
        assert np.isclose(result.p_value, sst.chi2.sf(expected_stat, 1), atol=1e-12)

    def test_lrt_negative_difference_warns_and_truncates(self):
        class Reduced:
            log_likelihood_ = -90.0
            coef_ = np.array([1.0])
            intercept_ = 1.0

        class Full:
            log_likelihood_ = -100.0
            coef_ = np.array([1.0, 2.0])
            intercept_ = 1.0

        with pytest.warns(RuntimeWarning, match="not nested"):
            result = likelihood_ratio_test(Reduced(), Full())
        assert result.statistic == 0.0
        assert result.p_value == 1.0

    def test_lrt_same_model(self, gaussian_data):
        X, y, _ = gaussian_data
        result = fit_glm(X[:, 1:], y, family="gaussian")
        lrt_result = likelihood_ratio_test(result, result)
        assert lrt_result.statistic == 0.0
        assert lrt_result.p_value == 1.0
