"""Unit tests for Zero-Inflated Negative Binomial model components.

These tests verify individual components of the ZINB implementation
in isolation.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from aurora.models.zero_inflated import (
    fit_zinb,
    ZINBResult,
    ZeroInflatedNegBinFamily,
)


class TestZeroInflatedNegBinFamilyLogLikelihood:
    """Unit tests for ZINB log-likelihood computation."""

    @pytest.fixture
    def family(self):
        return ZeroInflatedNegBinFamily(theta=2.0)

    def test_log_likelihood_all_zeros(self, family):
        """Test log-likelihood for all-zero data."""
        y = np.zeros(10)
        mu = np.ones(10) * 2.0
        pi = np.ones(10) * 0.5
        theta = 2.0

        ll = family.log_likelihood(y, mu, pi, theta)

        # log[π + (1-π)(θ/(θ+μ))^θ] for each observation
        p_zero_nb = (theta / (theta + 2.0)) ** theta
        expected_per_obs = np.log(0.5 + 0.5 * p_zero_nb)
        expected_total = 10 * expected_per_obs

        assert_allclose(ll, expected_total, rtol=1e-5)

    def test_log_likelihood_no_zeros(self, family):
        """Test log-likelihood for positive counts only."""
        y = np.array([1, 2, 3, 4, 5])
        mu = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
        pi = np.ones(5) * 0.1
        theta = 2.0

        ll = family.log_likelihood(y, mu, pi, theta)

        assert np.isfinite(ll)
        assert ll < 0  # Log-likelihood should be negative

    def test_log_likelihood_mixed(self, family):
        """Test log-likelihood for mixed zeros and positives."""
        y = np.array([0, 1, 0, 2, 0, 3])
        mu = np.ones(6) * 2.0
        pi = np.ones(6) * 0.3
        theta = 2.0

        ll = family.log_likelihood(y, mu, pi, theta)

        assert np.isfinite(ll)
        assert ll < 0

    def test_log_likelihood_extreme_theta(self, family):
        """Test log-likelihood with extreme theta values."""
        y = np.array([0, 1, 2, 3])
        mu = np.ones(4) * 2.0
        pi = np.ones(4) * 0.3

        # High theta (near Poisson)
        ll_high_theta = family.log_likelihood(y, mu, pi, theta=100.0)
        assert np.isfinite(ll_high_theta)

        # Low theta (high overdispersion)
        ll_low_theta = family.log_likelihood(y, mu, pi, theta=0.5)
        assert np.isfinite(ll_low_theta)


class TestZINBEStep:
    """Unit tests for ZINB E-step computation."""

    @pytest.fixture
    def family(self):
        return ZeroInflatedNegBinFamily(theta=2.0)

    def test_e_step_positive_counts(self, family):
        """Test E-step assigns zero probability for y > 0."""
        y = np.array([1, 2, 3, 4, 5])
        mu = np.ones(5) * 2.0
        pi = np.ones(5) * 0.5

        z = family.e_step(y, mu, pi)

        # For y > 0, z should be 0 (must come from NB component)
        assert_allclose(z, 0.0)

    def test_e_step_zeros(self, family):
        """Test E-step for zero observations."""
        y = np.zeros(5)
        mu = np.ones(5) * 2.0
        pi = np.ones(5) * 0.5
        theta = 2.0

        z = family.e_step(y, mu, pi, theta)

        # z = π / [π + (1-π)(θ/(θ+μ))^θ]
        p_zero_nb = (theta / (theta + 2.0)) ** theta
        expected = 0.5 / (0.5 + 0.5 * p_zero_nb)
        assert_allclose(z, expected, rtol=1e-5)

    def test_e_step_bounded(self, family):
        """Test E-step returns values in [0, 1]."""
        y = np.array([0, 0, 0, 1, 2])
        mu = np.random.rand(5) * 3 + 0.1
        pi = np.random.rand(5) * 0.8 + 0.1

        z = family.e_step(y, mu, pi)

        assert np.all((z >= 0) & (z <= 1))


class TestZINBExpectedCount:
    """Unit tests for ZINB expected count computation."""

    @pytest.fixture
    def family(self):
        return ZeroInflatedNegBinFamily()

    def test_expected_count_formula(self, family):
        """Test E[Y] = (1-π)μ formula."""
        mu = np.array([1.0, 2.0, 3.0])
        pi = np.array([0.2, 0.3, 0.4])

        expected = family.expected_count(mu, pi)

        assert_allclose(expected, (1 - pi) * mu)


class TestZINBProbZero:
    """Unit tests for ZINB P(Y=0) computation."""

    @pytest.fixture
    def family(self):
        return ZeroInflatedNegBinFamily(theta=2.0)

    def test_prob_zero_formula(self, family):
        """Test P(Y=0) = π + (1-π)(θ/(θ+μ))^θ formula."""
        mu = np.array([1.0, 2.0, 3.0])
        pi = np.array([0.2, 0.3, 0.4])
        theta = 2.0

        p0 = family.prob_zero(mu, pi, theta)

        p_zero_nb = (theta / (theta + mu)) ** theta
        expected = pi + (1 - pi) * p_zero_nb
        assert_allclose(p0, expected)

    def test_prob_zero_bounded(self, family):
        """Test P(Y=0) is in [0, 1]."""
        mu = np.random.rand(100) * 5
        pi = np.random.rand(100)

        p0 = family.prob_zero(mu, pi)

        assert np.all((p0 >= 0) & (p0 <= 1))

    def test_prob_zero_greater_than_poisson(self):
        """Test that ZINB P(Y=0) > Poisson P(Y=0) when π > 0."""
        family = ZeroInflatedNegBinFamily(theta=2.0)
        mu = np.array([2.0, 3.0, 4.0])
        pi = np.array([0.2, 0.2, 0.2])

        p0_zinb = family.prob_zero(mu, pi)
        p0_poisson = np.exp(-mu)

        assert np.all(p0_zinb > p0_poisson)


class TestFitZINBConvergence:
    """Unit tests for ZINB fitting convergence."""

    @pytest.fixture
    def zinb_data(self):
        """Generate ZINB data."""
        np.random.seed(42)
        n = 300
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        beta_true = np.array([1.0, 0.5])
        gamma_true = np.array([-1.0])
        theta_true = 2.0

        mu = np.exp(X @ beta_true)
        pi = 1 / (1 + np.exp(-gamma_true[0]))

        # Generate ZINB counts
        from scipy.stats import nbinom
        structural_zero = np.random.binomial(1, pi, n)
        # NB parameterized as (n, p) where mean = n*(1-p)/p
        p_nb = theta_true / (theta_true + mu)
        nb_counts = nbinom.rvs(theta_true, p_nb)
        y = np.where(structural_zero, 0, nb_counts)

        return X, y, beta_true, gamma_true, theta_true

    def test_fit_converges(self, zinb_data):
        """Test that EM algorithm converges."""
        X, y, _, _, _ = zinb_data
        result = fit_zinb(X, y, max_iter=100, tol=1e-6)

        assert result.converged_ or result.n_iter_ < 100

    def test_fit_improves_likelihood(self, zinb_data):
        """Test that fitting improves log-likelihood."""
        X, y, _, _, _ = zinb_data

        # Fit with few iterations
        result_few = fit_zinb(X, y, max_iter=5, tol=1e-10)

        # Fit with many iterations
        result_many = fit_zinb(X, y, max_iter=100, tol=1e-6)

        # More iterations should give higher (less negative) log-likelihood
        assert result_many.log_likelihood_ >= result_few.log_likelihood_ - 1e-6


class TestZINBResultMethods:
    """Unit tests for ZINBResult methods."""

    @pytest.fixture
    def fitted_result(self):
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # Simple ZINB data
        from scipy.stats import nbinom
        mu = np.exp(X @ [1.0, 0.3])
        theta = 2.0
        pi = 0.3

        structural_zero = np.random.binomial(1, pi, n)
        p_nb = theta / (theta + mu)
        nb_counts = nbinom.rvs(theta, p_nb)
        y = np.where(structural_zero, 0, nb_counts)

        return fit_zinb(X, y)

    def test_predict_response(self, fitted_result):
        """Test response prediction (marginal mean)."""
        pred = fitted_result.predict(type="response")
        assert pred.shape == fitted_result.mu_.shape
        assert np.all(pred >= 0)

    def test_predict_count(self, fitted_result):
        """Test count prediction (NB mean)."""
        pred = fitted_result.predict(type="count")
        assert np.all(pred > 0)

    def test_predict_prob_zero(self, fitted_result):
        """Test P(Y=0) prediction."""
        pred = fitted_result.predict(type="prob_zero")
        assert np.all((pred >= 0) & (pred <= 1))

    def test_predict_prob_inflate(self, fitted_result):
        """Test inflation probability prediction."""
        pred = fitted_result.predict(type="prob_inflate")
        assert np.all((pred >= 0) & (pred <= 1))

    def test_summary_keys(self, fitted_result):
        """Test summary contains required keys."""
        summary = fitted_result.summary()

        required = ["n_obs", "n_zeros", "log_likelihood", "aic", "theta", "mean_pi"]
        for key in required:
            assert key in summary

    def test_theta_positive(self, fitted_result):
        """Test that estimated theta is positive."""
        assert fitted_result.theta_ > 0


class TestZINBEdgeCases:
    """Unit tests for ZINB edge cases."""

    def test_no_zeros_in_data(self):
        """Test fitting when data has no zeros."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = np.random.poisson(5, n)
        y[y == 0] = 1  # Remove zeros

        result = fit_zinb(X, y)

        # Should still fit, but with low inflation
        assert result.pi_.mean() < 0.2

    def test_high_proportion_zeros(self):
        """Test fitting with many zeros."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # Generate actual ZINB data with high inflation
        from scipy.stats import nbinom
        pi_true = 0.8
        mu_true = np.exp(X @ [1.5, 0.3])
        theta = 2.0
        p_nb = theta / (theta + mu_true)

        structural_zero = np.random.binomial(1, pi_true, n)
        nb_counts = nbinom.rvs(theta, p_nb)
        y = np.where(structural_zero, 0, nb_counts)

        result = fit_zinb(X, y)

        # Should detect high probability of zeros
        prob_zero = result.predict(type="prob_zero")
        assert prob_zero.mean() > 0.5

    def test_overdispersed_data(self):
        """Test that ZINB captures overdispersion."""
        np.random.seed(42)
        n = 300
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # Highly overdispersed data
        from scipy.stats import nbinom
        mu = np.exp(X @ [1.5, 0.3])
        theta_true = 0.5  # Low theta = high overdispersion
        p_nb = theta_true / (theta_true + mu)
        y = nbinom.rvs(theta_true, p_nb)

        result = fit_zinb(X, y)

        # Should estimate theta in reasonable range
        assert 0.01 < result.theta_ < 10.0


class TestZINBVsZIP:
    """Tests comparing ZINB to ZIP behavior."""

    def test_zinb_reduces_to_zip_high_theta(self):
        """Test that ZINB approaches ZIP as theta -> infinity."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # Generate ZIP data (no overdispersion)
        mu = np.exp(X @ [1.0, 0.3])
        pi = 0.3
        structural_zero = np.random.binomial(1, pi, n)
        poisson_counts = np.random.poisson(mu)
        y = np.where(structural_zero, 0, poisson_counts)

        # Fit ZINB
        result = fit_zinb(X, y, theta_init=10.0)

        # theta should be large (approaching Poisson limit)
        assert result.theta_ > 1.0
