# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Statistical validation of Aurora-GLM against statsmodels reference.

These tests verify that Aurora-GLM's log-likelihood, AIC, BIC, coefficients,
and deviance match statsmodels (the gold-standard Python GLM implementation)
to numerical precision. This is the primary reproducibility guarantee for
scientific use.

Tolerances chosen per quantity:
- Coefficients: rtol=1e-6 (algorithms agree to floating point)
- log_likelihood: atol=1e-6 (full formula with constants, should match exactly)
- AIC/BIC: atol=1e-6 (derived from log_likelihood)
- Deviance: rtol=1e-6
- Fitted values: rtol=1e-6

For Gamma: Aurora uses the standard R/mgcv shape parameterization (shape=alpha).
statsmodels uses a non-standard "scale" parameterization (scale=1/shape) for
its log-likelihood. Aurora's Gamma log-likelihood matches R dgamma() but may
differ from statsmodels' Gamma.llf by a constant per observation. See
docs/VALIDATION.md for the full derivation. Gamma is validated separately
below with R-style formula.
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from aurora.distributions.families import (
    BinomialFamily,
    GammaFamily,
    GaussianFamily,
    InverseGaussianFamily,
    PoissonFamily,
)
from aurora.distributions.links import LogLink
from aurora.models.glm import fit_glm

# Tolerances
COEF_RTOL = 1e-6
LLF_ATOL = 1e-6
AIC_ATOL = 1e-6
DEV_RTOL = 1e-6
FIT_RTOL = 1e-6


@pytest.fixture
def linear_data():
    """Gaussian data with known coefficients."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 2))
    true_beta = np.array([0.5, 1.5, -0.5])  # intercept + 2 slopes
    eta = true_beta[0] + X @ true_beta[1:]
    y = eta + rng.standard_normal(n) * 0.7
    return X, y


@pytest.fixture
def poisson_data():
    """Poisson count data."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.standard_normal((n, 2))
    true_beta = np.array([0.2, 0.5, -0.3])
    eta = true_beta[0] + X @ true_beta[1:]
    mu = np.exp(eta)
    y = rng.poisson(mu)
    return X, y


@pytest.fixture
def bernoulli_data():
    """Binary logistic regression data."""
    rng = np.random.default_rng(42)
    n = 400
    X = rng.standard_normal((n, 2))
    true_beta = np.array([0.5, 1.0, -0.5])
    eta = true_beta[0] + X @ true_beta[1:]
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p)
    return X, y


@pytest.fixture
def gamma_data():
    """Gamma-distributed positive continuous data."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 2))
    true_beta = np.array([0.3, 0.4, -0.2])
    eta = true_beta[0] + X @ true_beta[1:]
    mu = np.exp(eta)
    shape_true = 2.0
    y = rng.gamma(shape_true, mu / shape_true)
    return X, y


def _fit_statsmodels(X, y, sm_family):
    """Helper: fit GLM with statsmodels (with intercept)."""
    X_sm = sm.add_constant(X)
    return sm.GLM(y, X_sm, family=sm_family).fit()


# ============================================================================
# Gaussian tests
# ============================================================================


class TestGaussianVsStatsmodels:
    """Validate Aurora-GLM Gaussian family against statsmodels."""

    def test_coefficients_match(self, linear_data):
        X, y = linear_data
        aurora = fit_glm(X, y, family=GaussianFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Gaussian())
        np.testing.assert_allclose(aurora.coef_, sm_res.params[1:], rtol=COEF_RTOL)
        assert aurora.intercept_ == pytest.approx(sm_res.params[0], rel=COEF_RTOL)

    def test_log_likelihood_matches(self, linear_data):
        X, y = linear_data
        aurora = fit_glm(X, y, family=GaussianFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Gaussian())
        assert aurora.log_likelihood_ == pytest.approx(sm_res.llf, abs=LLF_ATOL)

    def test_aic_bic_match(self, linear_data):
        X, y = linear_data
        aurora = fit_glm(X, y, family=GaussianFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Gaussian())
        assert aurora.aic_ == pytest.approx(sm_res.aic, abs=AIC_ATOL)
        # BIC: statsmodels doesn't expose bic directly for GLM; compute manually
        n = len(y)
        expected_bic = -2 * sm_res.llf + np.log(n) * len(sm_res.params)
        assert aurora.bic_ == pytest.approx(expected_bic, abs=AIC_ATOL)

    def test_deviance_matches(self, linear_data):
        X, y = linear_data
        aurora = fit_glm(X, y, family=GaussianFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Gaussian())
        assert aurora.deviance_ == pytest.approx(sm_res.deviance, rel=DEV_RTOL)

    def test_fitted_values_match(self, linear_data):
        X, y = linear_data
        aurora = fit_glm(X, y, family=GaussianFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Gaussian())
        np.testing.assert_allclose(aurora.mu_, sm_res.fittedvalues, rtol=FIT_RTOL)


# ============================================================================
# Poisson tests
# ============================================================================


class TestPoissonVsStatsmodels:
    """Validate Aurora-GLM Poisson family against statsmodels."""

    def test_coefficients_match(self, poisson_data):
        X, y = poisson_data
        aurora = fit_glm(X, y, family=PoissonFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Poisson())
        np.testing.assert_allclose(aurora.coef_, sm_res.params[1:], rtol=COEF_RTOL)
        assert aurora.intercept_ == pytest.approx(sm_res.params[0], rel=COEF_RTOL)

    def test_log_likelihood_matches(self, poisson_data):
        X, y = poisson_data
        aurora = fit_glm(X, y, family=PoissonFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Poisson())
        # Full Poisson log-likelihood must include -log(y!) - both should match
        assert aurora.log_likelihood_ == pytest.approx(sm_res.llf, abs=LLF_ATOL)

    def test_aic_bic_match(self, poisson_data):
        X, y = poisson_data
        aurora = fit_glm(X, y, family=PoissonFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Poisson())
        assert aurora.aic_ == pytest.approx(sm_res.aic, abs=AIC_ATOL)
        n = len(y)
        expected_bic = -2 * sm_res.llf + np.log(n) * len(sm_res.params)
        assert aurora.bic_ == pytest.approx(expected_bic, abs=AIC_ATOL)

    def test_deviance_matches(self, poisson_data):
        X, y = poisson_data
        aurora = fit_glm(X, y, family=PoissonFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Poisson())
        assert aurora.deviance_ == pytest.approx(sm_res.deviance, rel=DEV_RTOL)


# ============================================================================
# Binomial (Bernoulli) tests
# ============================================================================


class TestBinomialVsStatsmodels:
    """Validate Aurora-GLM Binomial family against statsmodels (Bernoulli)."""

    def test_coefficients_match(self, bernoulli_data):
        X, y = bernoulli_data
        aurora = fit_glm(X, y, family=BinomialFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Binomial())
        np.testing.assert_allclose(aurora.coef_, sm_res.params[1:], rtol=COEF_RTOL)
        assert aurora.intercept_ == pytest.approx(sm_res.params[0], rel=COEF_RTOL)

    def test_log_likelihood_matches(self, bernoulli_data):
        X, y = bernoulli_data
        aurora = fit_glm(X, y, family=BinomialFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Binomial())
        # For Bernoulli (n=1), the binomial coefficient is 0, so values match.
        assert aurora.log_likelihood_ == pytest.approx(sm_res.llf, abs=LLF_ATOL)

    def test_aic_bic_match(self, bernoulli_data):
        X, y = bernoulli_data
        aurora = fit_glm(X, y, family=BinomialFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Binomial())
        assert aurora.aic_ == pytest.approx(sm_res.aic, abs=AIC_ATOL)
        n = len(y)
        expected_bic = -2 * sm_res.llf + np.log(n) * len(sm_res.params)
        assert aurora.bic_ == pytest.approx(expected_bic, abs=AIC_ATOL)


# ============================================================================
# Gamma tests - Aurora uses R/mgcv convention (shape parameterization).
# Statsmodels uses a non-standard scale parameterization that differs by a
# convention constant. We validate Aurora's Gamma log-likelihood against the
# R dgamma() formula directly.
# ============================================================================


class TestGammaAgainstRFormula:
    """Validate Aurora-GLM Gamma log-likelihood matches R dgamma formula.

    R's dgamma(x, shape, rate, log=TRUE) returns:
        shape*log(rate) - lgamma(shape) + (shape-1)*log(x) - rate*x
    With rate = shape/mu, this becomes Aurora's Gamma log-likelihood.
    """

    def test_log_likelihood_matches_r_formula(self, gamma_data):
        import scipy.special as sp

        X, y = gamma_data
        # Fit Aurora to get mu and use known shape
        shape_true = 2.0
        aurora = fit_glm(X, y, family=GammaFamily(shape=shape_true, link=None))

        # Compute expected using R dgamma formula
        mu = aurora.mu_
        rate = shape_true / mu
        expected_llf = float(
            np.sum(
                shape_true * np.log(rate)
                - sp.gammaln(shape_true)
                + (shape_true - 1) * np.log(y)
                - rate * y
            )
        )
        assert aurora.log_likelihood_ == pytest.approx(expected_llf, rel=1e-6)

    def test_coefficients_match_statsmodels(self, gamma_data):
        """Aurora coefficients should match statsmodels (algorithms agree)."""
        from aurora.distributions.links import LogLink

        X, y = gamma_data
        aurora = fit_glm(X, y, family=GammaFamily(shape=2.0, link=LogLink()))
        sm_res = _fit_statsmodels(X, y, sm.families.Gamma(link=sm.families.links.Log()))
        np.testing.assert_allclose(aurora.coef_, sm_res.params[1:], rtol=1e-4)
        assert aurora.intercept_ == pytest.approx(sm_res.params[0], rel=1e-4)


# ============================================================================
# Dispersion-scaled log-likelihood / AIC tests (default families).
# With the family left at its default dispersion parameter, fit_glm plugs the
# estimated φ̂ = D/(n − rank) into the log-likelihood (shape = 1/φ̂ for Gamma,
# λ = 1/φ̂ for inverse Gaussian), matching statsmodels' res.llf convention.
# Residual differences come from the scale estimator (deviance-based in
# aurora vs Pearson-based default in statsmodels), hence the loose tolerance.
# ============================================================================


class TestDispersionScaledLogLikelihood:
    """llf/AIC use the estimated dispersion, like statsmodels res.llf."""

    def test_gamma_llf_aic_match_statsmodels(self):
        rng = np.random.default_rng(1)
        n = 400
        x = rng.uniform(-1, 1, n)
        mu = np.exp(0.8 + 0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu / 5.0)

        aurora = fit_glm(x.reshape(-1, 1), y, family=GammaFamily(link=LogLink()))
        sm_res = _fit_statsmodels(
            x.reshape(-1, 1), y, sm.families.Gamma(link=sm.families.links.Log())
        )

        assert aurora.log_likelihood_ == pytest.approx(sm_res.llf, abs=0.5)
        assert aurora.aic_ == pytest.approx(sm_res.aic, abs=1.0)

    def test_inverse_gaussian_llf_aic_match_statsmodels(self):
        rng = np.random.default_rng(1)
        n = 400
        x = rng.uniform(-1, 1, n)
        mu = np.exp(0.8 + 0.5 * x)
        y = rng.wald(mean=mu, scale=8.0)

        aurora = fit_glm(x.reshape(-1, 1), y, family=InverseGaussianFamily(link=LogLink()))
        sm_res = _fit_statsmodels(
            x.reshape(-1, 1), y, sm.families.InverseGaussian(link=sm.families.links.Log())
        )

        assert aurora.log_likelihood_ == pytest.approx(sm_res.llf, abs=0.5)
        assert aurora.aic_ == pytest.approx(sm_res.aic, abs=1.0)

    def test_gamma_default_shape_uses_estimated_dispersion(self):
        """Default Gamma family: llf uses shape = 1/φ̂, not shape = 1."""
        rng = np.random.default_rng(1)
        n = 400
        x = rng.uniform(-1, 1, n)
        mu = np.exp(0.8 + 0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu / 5.0)

        aurora = fit_glm(x.reshape(-1, 1), y, family=GammaFamily(link=LogLink()))

        family = GammaFamily(link=LogLink())
        llf_shape1 = float(family.log_likelihood(y, np.asarray(aurora.mu_), shape=1.0))
        llf_estimated = float(
            family.log_likelihood(y, np.asarray(aurora.mu_), shape=1.0 / aurora.dispersion_)
        )
        assert aurora.log_likelihood_ == pytest.approx(llf_estimated, rel=1e-10)
        assert aurora.log_likelihood_ != pytest.approx(llf_shape1, abs=1.0)


# ============================================================================
# Cross-family consistency: AIC ordering should agree with statsmodels
# ============================================================================


class TestModelSelectionConsistency:
    """AIC-based model selection should agree between Aurora and statsmodels."""

    def test_poisson_model_selection_picks_same_model(self, poisson_data):
        """Fit two nested Poisson models; Aurora and statsmodels should agree on which is better."""
        X, y = poisson_data

        # Full model (2 predictors) vs reduced (1 predictor)
        aurora_full = fit_glm(X, y, family=PoissonFamily())
        aurora_reduced = fit_glm(X[:, :1], y, family=PoissonFamily())

        sm_full = _fit_statsmodels(X, y, sm.families.Poisson())
        sm_reduced = _fit_statsmodels(X[:, :1], y, sm.families.Poisson())

        aurora_picks_full = aurora_full.aic_ < aurora_reduced.aic_
        sm_picks_full = sm_full.aic < sm_reduced.aic
        assert aurora_picks_full == sm_picks_full


# ============================================================================
# Standard errors and dispersion (dispersion-scaled covariance)
# ============================================================================


class TestGaussianStandardErrorsVsStatsmodels:
    """Validate SEs, covariance and dispersion against statsmodels.

    Aurora estimates φ̂ = deviance/(n − rank) for dispersion families and
    scales the coefficient covariance by φ̂ (R ``summary.glm`` convention;
    identical to the statsmodels Gaussian default). Wald statistics are
    z-based, matching the statsmodels GLM default (use_t=False).
    """

    def test_standard_errors_match(self, linear_data):
        X, y = linear_data
        aurora = fit_glm(X, y, family=GaussianFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Gaussian())
        np.testing.assert_allclose(aurora.std_errors_, sm_res.bse[1:], rtol=COEF_RTOL)
        assert aurora.intercept_std_error_ == pytest.approx(sm_res.bse[0], rel=COEF_RTOL)

    def test_dispersion_matches_scale(self, linear_data):
        X, y = linear_data
        aurora = fit_glm(X, y, family=GaussianFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Gaussian())
        assert aurora.dispersion_ == pytest.approx(sm_res.scale, rel=COEF_RTOL)
        assert aurora.scale_ == aurora.dispersion_

    def test_covariance_and_p_values_match(self, linear_data):
        X, y = linear_data
        aurora = fit_glm(X, y, family=GaussianFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Gaussian())
        np.testing.assert_allclose(np.asarray(aurora.coef_cov_), sm_res.cov_params(), rtol=1e-5)
        np.testing.assert_allclose(aurora.p_values_, sm_res.pvalues[1:], rtol=1e-5)

    def test_large_noise_standard_errors(self):
        """With σ = 3 the SEs must be ~σ times larger than the φ = 1 values."""
        rng = np.random.default_rng(2024)
        n = 200
        X = rng.standard_normal((n, 2))
        y = 0.5 + X @ np.array([1.5, -0.5]) + rng.standard_normal(n) * 3.0
        aurora = fit_glm(X, y, family=GaussianFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Gaussian())
        np.testing.assert_allclose(aurora.std_errors_, sm_res.bse[1:], rtol=COEF_RTOL)
        assert aurora.intercept_std_error_ == pytest.approx(sm_res.bse[0], rel=COEF_RTOL)
        # Dispersion estimate recovers σ² = 9 up to sampling error
        assert aurora.dispersion_ == pytest.approx(9.0, rel=0.2)


class TestPoissonStandardErrorsVsStatsmodels:
    """Poisson has fixed dispersion φ ≡ 1; SEs must match statsmodels."""

    def test_standard_errors_match(self, poisson_data):
        X, y = poisson_data
        aurora = fit_glm(X, y, family=PoissonFamily())
        sm_res = _fit_statsmodels(X, y, sm.families.Poisson())
        assert aurora.dispersion_ == 1.0
        np.testing.assert_allclose(aurora.std_errors_, sm_res.bse[1:], rtol=1e-5)
        assert aurora.intercept_std_error_ == pytest.approx(sm_res.bse[0], rel=1e-5)
