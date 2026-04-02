"""Unit tests for Zero-Inflated Poisson model components.

These tests verify individual components of the ZIP implementation
in isolation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aurora.models.zero_inflated import (
    ZeroInflatedPoissonFamily,
    fit_zip,
)


class TestZeroInflatedPoissonFamilyLogLikelihood:
    """Unit tests for ZIP log-likelihood computation."""

    @pytest.fixture
    def family(self):
        return ZeroInflatedPoissonFamily()

    def test_log_likelihood_all_zeros(self, family):
        """Test log-likelihood for all-zero data."""
        y = np.zeros(10)
        mu = np.ones(10) * 2.0
        pi = np.ones(10) * 0.5

        ll = family.log_likelihood(y, mu, pi)

        # log[π + (1-π)exp(-μ)] for each observation
        expected_per_obs = np.log(0.5 + 0.5 * np.exp(-2.0))
        expected_total = 10 * expected_per_obs

        assert_allclose(ll, expected_total, rtol=1e-5)

    def test_log_likelihood_no_zeros(self, family):
        """Test log-likelihood for positive counts only."""
        y = np.array([1, 2, 3, 4, 5])
        mu = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
        pi = np.ones(5) * 0.1

        ll = family.log_likelihood(y, mu, pi)

        assert np.isfinite(ll)
        assert ll < 0  # Log-likelihood should be negative

    def test_log_likelihood_mixed(self, family):
        """Test log-likelihood for mixed zeros and positives."""
        y = np.array([0, 1, 0, 2, 0, 3])
        mu = np.ones(6) * 2.0
        pi = np.ones(6) * 0.3

        ll = family.log_likelihood(y, mu, pi)

        assert np.isfinite(ll)
        assert ll < 0

    def test_log_likelihood_extreme_pi(self, family):
        """Test log-likelihood with extreme pi values."""
        y = np.array([0, 1, 2])
        mu = np.array([2.0, 2.0, 2.0])

        # High inflation
        pi_high = np.array([0.99, 0.99, 0.99])
        ll_high = family.log_likelihood(y, mu, pi_high)
        assert np.isfinite(ll_high)

        # Low inflation
        pi_low = np.array([0.01, 0.01, 0.01])
        ll_low = family.log_likelihood(y, mu, pi_low)
        assert np.isfinite(ll_low)


class TestEStep:
    """Unit tests for E-step computation."""

    @pytest.fixture
    def family(self):
        return ZeroInflatedPoissonFamily()

    def test_e_step_positive_counts(self, family):
        """Test E-step assigns zero probability for y > 0."""
        y = np.array([1, 2, 3, 4, 5])
        mu = np.ones(5) * 2.0
        pi = np.ones(5) * 0.5

        z = family.e_step(y, mu, pi)

        # For y > 0, z should be 0 (must come from Poisson component)
        assert_allclose(z, 0.0)

    def test_e_step_zeros(self, family):
        """Test E-step for zero observations."""
        y = np.zeros(5)
        mu = np.ones(5) * 2.0
        pi = np.ones(5) * 0.5

        z = family.e_step(y, mu, pi)

        # z = π / [π + (1-π)exp(-μ)]
        expected = 0.5 / (0.5 + 0.5 * np.exp(-2.0))
        assert_allclose(z, expected, rtol=1e-5)

    def test_e_step_bounded(self, family):
        """Test E-step returns values in [0, 1]."""
        y = np.array([0, 0, 0, 1, 2])
        mu = np.random.rand(5) * 3 + 0.1
        pi = np.random.rand(5) * 0.8 + 0.1

        z = family.e_step(y, mu, pi)

        assert np.all((z >= 0) & (z <= 1))

    def test_e_step_high_pi(self, family):
        """Test E-step with high inflation probability."""
        y = np.zeros(5)
        mu = np.ones(5) * 0.5
        pi = np.ones(5) * 0.99

        z = family.e_step(y, mu, pi)

        # With high pi, zeros are mostly structural
        assert np.all(z > 0.9)


class TestExpectedCount:
    """Unit tests for expected count computation."""

    @pytest.fixture
    def family(self):
        return ZeroInflatedPoissonFamily()

    def test_expected_count_formula(self, family):
        """Test E[Y] = (1-π)μ formula."""
        mu = np.array([1.0, 2.0, 3.0])
        pi = np.array([0.2, 0.3, 0.4])

        expected = family.expected_count(mu, pi)

        assert_allclose(expected, (1 - pi) * mu)

    def test_expected_count_no_inflation(self, family):
        """Test expected count with no inflation."""
        mu = np.array([1.0, 2.0, 3.0])
        pi = np.zeros(3)

        expected = family.expected_count(mu, pi)

        assert_allclose(expected, mu)

    def test_expected_count_full_inflation(self, family):
        """Test expected count with full inflation."""
        mu = np.array([1.0, 2.0, 3.0])
        pi = np.ones(3)

        expected = family.expected_count(mu, pi)

        assert_allclose(expected, 0.0)


class TestProbZero:
    """Unit tests for P(Y=0) computation."""

    @pytest.fixture
    def family(self):
        return ZeroInflatedPoissonFamily()

    def test_prob_zero_formula(self, family):
        """Test P(Y=0) = π + (1-π)exp(-μ) formula."""
        mu = np.array([1.0, 2.0, 3.0])
        pi = np.array([0.2, 0.3, 0.4])

        p0 = family.prob_zero(mu, pi)

        expected = pi + (1 - pi) * np.exp(-mu)
        assert_allclose(p0, expected)

    def test_prob_zero_bounded(self, family):
        """Test P(Y=0) is in [0, 1]."""
        mu = np.random.rand(100) * 5
        pi = np.random.rand(100)

        p0 = family.prob_zero(mu, pi)

        assert np.all((p0 >= 0) & (p0 <= 1))

    def test_prob_zero_increases_with_pi(self, family):
        """Test P(Y=0) increases with inflation probability."""
        mu = np.ones(3) * 2.0

        p0_low = family.prob_zero(mu, np.array([0.1, 0.1, 0.1]))
        p0_high = family.prob_zero(mu, np.array([0.9, 0.9, 0.9]))

        assert np.all(p0_high > p0_low)


class TestFitZIPConvergence:
    """Unit tests for ZIP fitting convergence."""

    @pytest.fixture
    def zip_data(self):
        """Generate ZIP data."""
        np.random.seed(42)
        n = 300
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        beta_true = np.array([1.0, 0.5])
        gamma_true = np.array([-1.0])

        mu = np.exp(X @ beta_true)
        pi = 1 / (1 + np.exp(-gamma_true[0]))

        structural_zero = np.random.binomial(1, pi, n)
        poisson_counts = np.random.poisson(mu)
        y = np.where(structural_zero, 0, poisson_counts)

        return X, y, beta_true, gamma_true

    def test_fit_converges(self, zip_data):
        """Test that EM algorithm converges."""
        X, y, _, _ = zip_data
        result = fit_zip(X, y, max_iter=100, tol=1e-6)

        assert result.converged_ or result.n_iter_ < 100

    def test_fit_improves_likelihood(self, zip_data):
        """Test that fitting improves log-likelihood."""
        X, y, _, _ = zip_data

        # Fit with few iterations
        result_few = fit_zip(X, y, max_iter=5, tol=1e-10)

        # Fit with many iterations
        result_many = fit_zip(X, y, max_iter=100, tol=1e-6)

        # More iterations should give higher (less negative) log-likelihood
        assert result_many.log_likelihood_ >= result_few.log_likelihood_ - 1e-6


class TestZIPResultMethods:
    """Unit tests for ZIPResult methods."""

    @pytest.fixture
    def fitted_result(self):
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        mu = np.exp(X @ [1.0, 0.3])
        pi = 0.3
        structural_zero = np.random.binomial(1, pi, n)
        y = np.where(structural_zero, 0, np.random.poisson(mu))

        return fit_zip(X, y)

    def test_predict_response(self, fitted_result):
        """Test response prediction (marginal mean)."""
        pred = fitted_result.predict(type="response")
        assert pred.shape == fitted_result.mu_.shape
        assert np.all(pred >= 0)

    def test_predict_count(self, fitted_result):
        """Test count prediction (conditional mean)."""
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

        required = ["n_obs", "n_zeros", "log_likelihood", "aic", "mean_pi"]
        for key in required:
            assert key in summary

    def test_aic_bic_relationship(self, fitted_result):
        """Test AIC < BIC for n > e^2 (typically true)."""
        # For n > e^2 ≈ 7.4, BIC penalizes more than AIC
        assert fitted_result.bic_ > fitted_result.aic_


class TestZIPEdgeCases:
    """Unit tests for ZIP edge cases."""

    def test_no_zeros_in_data(self):
        """Test fitting when data has no zeros."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = np.random.poisson(5, n)
        y[y == 0] = 1  # Remove zeros

        result = fit_zip(X, y)

        # Should still fit, but with low inflation
        assert result.pi_.mean() < 0.2

    def test_high_proportion_zeros(self):
        """Test fitting with many zeros."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # Generate actual ZIP data with high inflation
        pi_true = 0.8
        mu_true = np.exp(X @ [1.5, 0.3])
        structural_zero = np.random.binomial(1, pi_true, n)
        poisson_counts = np.random.poisson(mu_true)
        y = np.where(structural_zero, 0, poisson_counts)

        result = fit_zip(X, y)

        # Should detect high inflation - check P(Y=0) is high
        prob_zero = result.predict(type="prob_zero")
        assert prob_zero.mean() > 0.5

    def test_all_zeros(self):
        """Test fitting with all zeros."""
        n = 100
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = np.zeros(n)

        result = fit_zip(X, y)

        # P(Y=0) should be very high
        prob_zero = result.predict(type="prob_zero")
        assert prob_zero.mean() > 0.99
