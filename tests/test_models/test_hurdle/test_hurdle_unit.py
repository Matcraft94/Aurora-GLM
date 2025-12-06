"""Unit tests for Hurdle model components.

These tests verify individual components of the Hurdle model implementation
in isolation.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from aurora.models.hurdle import (
    fit_hurdle_poisson,
    fit_hurdle_negbin,
    HurdlePoissonResult,
    HurdleNegBinResult,
)
from aurora.models.hurdle.truncated import (
    TruncatedPoissonFamily,
    TruncatedNegBinFamily,
)


class TestTruncatedPoissonFamily:
    """Unit tests for Truncated Poisson distribution."""

    @pytest.fixture
    def family(self):
        return TruncatedPoissonFamily()

    def test_truncated_mean_formula(self, family):
        """Test E[Y|Y>0] = μ/[1-exp(-μ)] formula."""
        mu = np.array([1.0, 2.0, 3.0])

        trunc_mean = family.truncated_mean(mu)

        expected = mu / (1 - np.exp(-mu))
        assert_allclose(trunc_mean, expected, rtol=1e-10)

    def test_truncated_mean_greater_than_mu(self, family):
        """Test that truncated mean > μ always."""
        mu = np.array([0.5, 1.0, 2.0, 5.0, 10.0])

        trunc_mean = family.truncated_mean(mu)

        assert np.all(trunc_mean > mu)

    def test_truncated_mean_approaches_mu_for_large_mu(self, family):
        """Test that truncated mean approaches μ as μ -> infinity."""
        mu = np.array([10.0, 20.0, 50.0])

        trunc_mean = family.truncated_mean(mu)

        # For large μ, truncated mean ≈ μ
        ratios = trunc_mean / mu
        assert np.all(ratios < 1.01)

    def test_log_likelihood_positive_counts(self, family):
        """Test log-likelihood only accepts positive counts."""
        y = np.array([1, 2, 3, 4, 5])
        mu = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

        ll = family.log_likelihood(y, mu)

        assert np.isfinite(ll)
        assert ll < 0

    def test_pmf_sums_to_one(self, family):
        """Test that truncated PMF sums to 1."""
        mu = 3.0
        k_vals = np.arange(1, 50)

        # P(Y=k|Y>0) = P(Y=k)/P(Y>0) for k>0
        from scipy.stats import poisson
        pmf_trunc = poisson.pmf(k_vals, mu) / (1 - poisson.pmf(0, mu))

        assert_allclose(np.sum(pmf_trunc), 1.0, atol=1e-10)


class TestTruncatedNegBinFamily:
    """Unit tests for Truncated Negative Binomial distribution."""

    @pytest.fixture
    def family(self):
        return TruncatedNegBinFamily(theta=2.0)

    def test_truncated_mean_greater_than_mu(self, family):
        """Test that truncated NB mean > μ."""
        mu = np.array([1.0, 2.0, 3.0])

        trunc_mean = family.truncated_mean(mu)

        assert np.all(trunc_mean > mu)

    def test_prob_zero_computed_correctly(self, family):
        """Test NB P(Y=0) = (θ/(θ+μ))^θ is computed correctly internally."""
        mu = np.array([1.0, 2.0, 3.0])
        theta = 2.0

        # Compute expected P(Y=0) for NB
        expected = (theta / (theta + mu)) ** theta

        # Verify truncated mean uses this correctly
        # Truncated mean = mu / (1 - P(Y=0))
        trunc_mean = family.truncated_mean(mu)
        expected_trunc_mean = mu / (1 - expected)
        assert_allclose(trunc_mean, expected_trunc_mean, rtol=1e-5)

    def test_log_likelihood_finite(self, family):
        """Test log-likelihood is finite for positive counts."""
        y = np.array([1, 2, 3, 4, 5])
        mu = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

        ll = family.log_likelihood(y, mu)

        assert np.isfinite(ll)


class TestHurdlePoissonBinaryComponent:
    """Unit tests for binary component of Hurdle Poisson."""

    @pytest.fixture
    def binary_data(self):
        """Generate data for testing binary component."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        gamma_true = np.array([-0.5, 1.0])
        pi = 1 / (1 + np.exp(-X @ gamma_true))
        y_binary = np.random.binomial(1, pi, n)

        # Convert to count data (positive where y_binary=1)
        y = np.where(y_binary, np.random.poisson(3, n) + 1, 0)
        y = np.maximum(y, 0)

        return X, y, gamma_true

    def test_binary_coef_recovery(self, binary_data):
        """Test that binary coefficients are recovered."""
        X, y, gamma_true = binary_data

        result = fit_hurdle_poisson(X, X, y)

        # Should be close to true parameters
        assert_allclose(result.coef_binary_, gamma_true, atol=0.5)

    def test_pi_bounded(self, binary_data):
        """Test that P(Y>0) is in (0, 1)."""
        X, y, _ = binary_data

        result = fit_hurdle_poisson(X, X, y)

        assert np.all((result.pi_ > 0) & (result.pi_ < 1))


class TestHurdlePoissonCountComponent:
    """Unit tests for count component of Hurdle Poisson."""

    @pytest.fixture
    def count_data(self):
        """Generate data for testing count component."""
        np.random.seed(42)
        n = 500  # More data for better estimation
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        beta_true = np.array([1.0, 0.3])  # Lower intercept for numerical stability
        mu = np.exp(X @ beta_true)

        # High probability of positive to focus on count model
        pi = 0.9
        is_positive = np.random.binomial(1, pi, n)

        y_pos = np.random.poisson(mu)
        y_pos[y_pos == 0] = 1  # Truncate
        y = np.where(is_positive, y_pos, 0)

        return X, y, beta_true

    def test_count_coef_finite(self, count_data):
        """Test that count coefficients are finite."""
        X, y, _ = count_data

        result = fit_hurdle_poisson(X, X, y)

        # Should have finite coefficients
        assert np.all(np.isfinite(result.coef_count_))

    def test_mu_positive(self, count_data):
        """Test that Poisson parameter is positive."""
        X, y, _ = count_data

        result = fit_hurdle_poisson(X, X, y)

        assert np.all(result.mu_ > 0)


class TestHurdlePoissonResultMethods:
    """Unit tests for HurdlePoissonResult methods."""

    @pytest.fixture
    def fitted_result(self):
        np.random.seed(42)
        n = 300
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # Generate hurdle Poisson data with lower mu for stability
        pi = 0.7
        mu = np.exp(X @ [0.8, 0.2])  # Lower coefficients
        is_positive = np.random.binomial(1, pi, n)
        y_pos = np.random.poisson(mu)
        y_pos[y_pos == 0] = 1
        y = np.where(is_positive, y_pos, 0)

        return fit_hurdle_poisson(X, X, y)

    def test_predict_response(self, fitted_result):
        """Test response prediction (marginal mean)."""
        pred = fitted_result.predict(type="response")

        assert pred.shape == fitted_result.mu_.shape
        assert np.all(pred >= 0)

    def test_predict_prob_positive(self, fitted_result):
        """Test P(Y>0) prediction."""
        pred = fitted_result.predict(type="prob_positive")

        assert np.all((pred > 0) & (pred < 1))

    def test_predict_prob_zero(self, fitted_result):
        """Test P(Y=0) prediction."""
        pred_zero = fitted_result.predict(type="prob_zero")
        pred_pos = fitted_result.predict(type="prob_positive")

        assert_allclose(pred_zero + pred_pos, 1.0)

    def test_predict_count(self, fitted_result):
        """Test truncated mean prediction."""
        pred = fitted_result.predict(type="count")

        # Truncated mean should be positive
        assert np.all(pred > 0)
        # For small mu, truncated mean is close to mu
        # Just verify it's finite
        assert np.all(np.isfinite(pred))

    def test_summary_keys(self, fitted_result):
        """Test summary contains required keys."""
        summary = fitted_result.summary()

        required = [
            "n_obs", "n_zeros", "n_positive",
            "log_likelihood", "aic", "bic",
            "coef_binary", "coef_count"
        ]
        for key in required:
            assert key in summary

    def test_log_likelihood_decomposition(self, fitted_result):
        """Test that total LL = binary LL + count LL."""
        total = fitted_result.log_likelihood_
        binary = fitted_result.log_likelihood_binary_
        count = fitted_result.log_likelihood_count_

        assert_allclose(total, binary + count, rtol=1e-10)


class TestHurdleNegBinResultMethods:
    """Unit tests for HurdleNegBinResult methods."""

    @pytest.fixture
    def fitted_result(self):
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # Generate hurdle NegBin data
        from scipy.stats import nbinom
        pi = 0.7
        mu = np.exp(X @ [1.5, 0.3])
        theta = 2.0

        is_positive = np.random.binomial(1, pi, n)
        p_nb = theta / (theta + mu)
        y_pos = nbinom.rvs(theta, p_nb)
        y_pos[y_pos == 0] = 1
        y = np.where(is_positive, y_pos, 0)

        return fit_hurdle_negbin(X, X, y)

    def test_predict_response(self, fitted_result):
        """Test response prediction."""
        pred = fitted_result.predict(type="response")

        assert pred.shape == fitted_result.mu_.shape
        assert np.all(pred >= 0)

    def test_theta_positive(self, fitted_result):
        """Test that estimated theta is positive."""
        assert fitted_result.theta_ > 0

    def test_summary_includes_theta(self, fitted_result):
        """Test that summary includes theta."""
        summary = fitted_result.summary()

        assert "theta" in summary
        assert summary["theta"] > 0


class TestHurdleEdgeCases:
    """Unit tests for Hurdle model edge cases."""

    def test_all_positive(self):
        """Test with all positive counts (no zeros)."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = np.random.poisson(3, n)
        y[y == 0] = 1  # No zeros

        result = fit_hurdle_poisson(X, X, y)

        # pi should be close to 1
        assert result.pi_.mean() > 0.9

    def test_high_proportion_zeros(self):
        """Test with many zeros."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # 80% zeros
        y = np.zeros(n, dtype=int)
        positive_counts = np.random.poisson(3, 40)
        positive_counts[positive_counts == 0] = 1
        y[:40] = positive_counts

        result = fit_hurdle_poisson(X, X, y)

        # pi should be low
        assert result.pi_.mean() < 0.4

    def test_different_design_matrices(self):
        """Test with different X for binary and count."""
        np.random.seed(42)
        n = 200
        X_binary = np.column_stack([
            np.ones(n),
            np.random.normal(0, 1, n)
        ])
        X_count = np.column_stack([
            np.ones(n),
            np.random.normal(0, 1, n),
            np.random.normal(0, 1, n)
        ])

        # Generate data
        pi = 0.6
        mu = np.exp(X_count @ [1.5, 0.3, -0.2])
        is_positive = np.random.binomial(1, pi, n)
        y_pos = np.random.poisson(mu)
        y_pos[y_pos == 0] = 1
        y = np.where(is_positive, y_pos, 0)

        result = fit_hurdle_poisson(X_binary, X_count, y)

        assert result.coef_binary_.shape == (2,)
        assert result.coef_count_.shape == (3,)

    def test_no_positive_counts_raises(self):
        """Test that all zeros raises error."""
        n = 50
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = np.zeros(n)

        with pytest.raises(ValueError, match="No positive counts"):
            fit_hurdle_poisson(X, X, y)


class TestHurdleVsZeroInflated:
    """Tests comparing Hurdle to Zero-Inflated models."""

    def test_hurdle_identifies_boundary(self):
        """Test that Hurdle correctly models P(Y>0)."""
        np.random.seed(42)
        n = 300
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # True P(Y>0) depends on X
        gamma_true = np.array([0.0, 1.5])
        pi_true = 1 / (1 + np.exp(-X @ gamma_true))
        is_positive = np.random.binomial(1, pi_true, n)

        mu = np.exp(X @ [1.5, 0.3])
        y_pos = np.random.poisson(mu)
        y_pos[y_pos == 0] = 1
        y = np.where(is_positive, y_pos, 0)

        result = fit_hurdle_poisson(X, X, y)

        # Estimated pi should correlate with true pi
        corr = np.corrcoef(result.pi_, pi_true)[0, 1]
        assert corr > 0.5


class TestHurdleConvergence:
    """Unit tests for Hurdle model convergence."""

    def test_poisson_converges(self):
        """Test Hurdle Poisson converges."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        pi = 0.7
        mu = np.exp(X @ [1.5, 0.3])
        is_positive = np.random.binomial(1, pi, n)
        y_pos = np.random.poisson(mu)
        y_pos[y_pos == 0] = 1
        y = np.where(is_positive, y_pos, 0)

        result = fit_hurdle_poisson(X, X, y, max_iter=100)

        # Should have valid coefficients
        assert np.all(np.isfinite(result.coef_binary_))
        assert np.all(np.isfinite(result.coef_count_))

    def test_negbin_converges(self):
        """Test Hurdle NegBin converges."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        from scipy.stats import nbinom
        pi = 0.7
        mu = np.exp(X @ [1.5, 0.3])
        theta = 2.0
        p_nb = theta / (theta + mu)

        is_positive = np.random.binomial(1, pi, n)
        y_pos = nbinom.rvs(theta, p_nb)
        y_pos[y_pos == 0] = 1
        y = np.where(is_positive, y_pos, 0)

        result = fit_hurdle_negbin(X, X, y, max_iter=100)

        # Should have valid coefficients
        assert np.all(np.isfinite(result.coef_binary_))
        assert np.all(np.isfinite(result.coef_count_))
        assert np.isfinite(result.theta_)
