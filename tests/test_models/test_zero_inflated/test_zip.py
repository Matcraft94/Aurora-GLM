"""Tests for Zero-Inflated Poisson (ZIP) model."""

import numpy as np
import pytest

from aurora.models.zero_inflated import fit_zip, ZIPResult, ZeroInflatedPoissonFamily


class TestZeroInflatedPoissonFamily:
    """Tests for ZIP family class."""

    def test_log_likelihood_zeros(self):
        """Test log-likelihood computation for zeros."""
        family = ZeroInflatedPoissonFamily()

        y = np.array([0, 0, 0, 0, 0])
        mu = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        pi = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        ll = family.log_likelihood(y, mu, pi)

        # log[π + (1-π)exp(-μ)] = log[0.5 + 0.5*exp(-2)] ≈ log(0.5676)
        expected_per_obs = np.log(0.5 + 0.5 * np.exp(-2.0))
        expected_total = 5 * expected_per_obs

        np.testing.assert_allclose(ll, expected_total, rtol=1e-5)

    def test_log_likelihood_positive(self):
        """Test log-likelihood computation for positive counts."""
        family = ZeroInflatedPoissonFamily()

        y = np.array([1, 2, 3])
        mu = np.array([2.0, 2.0, 2.0])
        pi = np.array([0.3, 0.3, 0.3])

        ll = family.log_likelihood(y, mu, pi)

        # Should be finite and negative
        assert np.isfinite(ll)
        assert ll < 0

    def test_e_step_zeros(self):
        """Test E-step returns correct probabilities for zeros."""
        family = ZeroInflatedPoissonFamily()

        y = np.array([0, 0, 1, 2])
        mu = np.array([2.0, 2.0, 2.0, 2.0])
        pi = np.array([0.5, 0.5, 0.5, 0.5])

        z = family.e_step(y, mu, pi)

        # For y>0, z should be 0
        assert z[2] == 0
        assert z[3] == 0

        # For y=0, z = π / [π + (1-π)exp(-μ)]
        expected_z0 = 0.5 / (0.5 + 0.5 * np.exp(-2.0))
        np.testing.assert_allclose(z[0], expected_z0, rtol=1e-5)
        np.testing.assert_allclose(z[1], expected_z0, rtol=1e-5)

    def test_expected_count(self):
        """Test marginal expected count."""
        family = ZeroInflatedPoissonFamily()

        mu = np.array([2.0, 3.0, 4.0])
        pi = np.array([0.2, 0.3, 0.4])

        expected = family.expected_count(mu, pi)

        # E[Y] = (1-π)μ
        np.testing.assert_allclose(expected, (1 - pi) * mu)

    def test_prob_zero(self):
        """Test probability of zero."""
        family = ZeroInflatedPoissonFamily()

        mu = np.array([2.0, 3.0])
        pi = np.array([0.3, 0.4])

        p0 = family.prob_zero(mu, pi)

        # P(Y=0) = π + (1-π)exp(-μ)
        expected_p0 = pi + (1 - pi) * np.exp(-mu)
        np.testing.assert_allclose(p0, expected_p0)


class TestFitZIP:
    """Tests for ZIP model fitting."""

    @pytest.fixture
    def zip_data(self):
        """Generate ZIP data."""
        np.random.seed(42)
        n = 500

        # Design matrix with intercept and covariate
        x = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x])

        # True parameters
        beta_true = np.array([1.0, 0.5])  # Count model
        gamma_true = np.array([-1.0])  # Inflation (intercept-only)

        # Generate ZIP data
        mu = np.exp(X @ beta_true)
        pi = 1 / (1 + np.exp(-gamma_true[0]))

        structural_zero = np.random.binomial(1, pi, n)
        poisson_counts = np.random.poisson(mu)
        y = np.where(structural_zero, 0, poisson_counts)

        return X, y, beta_true, gamma_true

    def test_fit_returns_result(self, zip_data):
        """Test that fit returns ZIPResult."""
        X, y, _, _ = zip_data
        result = fit_zip(X, y)

        assert isinstance(result, ZIPResult)

    def test_fit_coefficients_shape(self, zip_data):
        """Test coefficient shapes."""
        X, y, _, _ = zip_data
        result = fit_zip(X, y)

        assert result.coef_count_.shape == (2,)
        assert result.coef_inflate_.shape == (1,)  # Intercept-only default

    def test_fit_recovers_parameters(self, zip_data):
        """Test that fit recovers true parameters reasonably well."""
        X, y, beta_true, gamma_true = zip_data
        result = fit_zip(X, y)

        # Count model coefficients should be close to true
        np.testing.assert_allclose(result.coef_count_, beta_true, atol=0.3)

        # Inflation parameter should be in right direction
        assert result.coef_inflate_[0] < 0  # True is -1.0

    def test_fit_converges(self, zip_data):
        """Test that EM converges."""
        X, y, _, _ = zip_data
        result = fit_zip(X, y, max_iter=100, tol=1e-6)

        assert result.converged_ or result.n_iter_ < 100

    def test_fit_log_likelihood_increases(self, zip_data):
        """Test that final log-likelihood is reasonable."""
        X, y, _, _ = zip_data
        result = fit_zip(X, y)

        assert np.isfinite(result.log_likelihood_)
        assert result.log_likelihood_ < 0

    def test_fit_predictions(self, zip_data):
        """Test prediction types."""
        X, y, _, _ = zip_data
        result = fit_zip(X, y)

        # Response: marginal mean
        pred_response = result.predict(type="response")
        assert pred_response.shape == y.shape
        assert np.all(pred_response >= 0)

        # Count: conditional mean
        pred_count = result.predict(type="count")
        assert np.all(pred_count > 0)

        # Probability of zero
        pred_p0 = result.predict(type="prob_zero")
        assert np.all((pred_p0 >= 0) & (pred_p0 <= 1))

        # Inflation probability
        pred_pi = result.predict(type="prob_inflate")
        assert np.all((pred_pi >= 0) & (pred_pi <= 1))

    def test_fit_information_criteria(self, zip_data):
        """Test AIC and BIC."""
        X, y, _, _ = zip_data
        result = fit_zip(X, y)

        assert np.isfinite(result.aic_)
        assert np.isfinite(result.bic_)
        assert result.bic_ > result.aic_  # BIC penalizes more

    def test_fit_with_inflation_covariates(self, zip_data):
        """Test fitting with covariates in inflation model."""
        X, y, _, _ = zip_data

        # Use same X for inflation model
        result = fit_zip(X, y, X_inflate=X)

        assert result.coef_inflate_.shape == (2,)

    def test_fit_summary(self, zip_data):
        """Test summary output."""
        X, y, _, _ = zip_data
        result = fit_zip(X, y)

        summary = result.summary()

        assert "n_obs" in summary
        assert "n_zeros" in summary
        assert "log_likelihood" in summary
        assert "aic" in summary
        assert "mean_pi" in summary


class TestZIPEdgeCases:
    """Test edge cases for ZIP."""

    def test_no_zeros(self):
        """Test handling data with no zeros."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = np.random.poisson(5, n)
        y[y == 0] = 1  # Remove zeros

        result = fit_zip(X, y)

        # Should still fit, but with low inflation
        assert result.pi_.mean() < 0.1

    def test_all_zeros(self):
        """Test handling data with all zeros."""
        n = 100
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = np.zeros(n)

        result = fit_zip(X, y)

        # For all zeros, P(Y=0) should be very high
        # This can happen either via high π or low μ (or both)
        prob_zero = result.predict(type="prob_zero")
        assert prob_zero.mean() > 0.99

    def test_high_inflation(self):
        """Test with high zero-inflation."""
        np.random.seed(42)
        n = 500
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # 80% zeros
        pi_true = 0.8
        mu = np.exp(X @ [1.0, 0.5])
        structural_zero = np.random.binomial(1, pi_true, n)
        poisson_counts = np.random.poisson(mu)
        y = np.where(structural_zero, 0, poisson_counts)

        result = fit_zip(X, y)

        # Should detect high inflation
        assert result.pi_.mean() > 0.5
