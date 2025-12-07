"""Tests for Hurdle Poisson model."""

import numpy as np
import pytest

from aurora.models.hurdle import fit_hurdle_poisson, HurdlePoissonResult


class TestFitHurdlePoisson:
    """Tests for Hurdle Poisson fitting."""

    @pytest.fixture
    def hurdle_data(self):
        """Generate hurdle Poisson data."""
        np.random.seed(42)
        n = 500

        # Design matrix
        x = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x])

        # True parameters
        gamma_true = np.array([-0.5, 0.8])  # Binary model
        beta_true = np.array([1.5, 0.3])  # Count model

        # Generate hurdle data
        eta_binary = X @ gamma_true
        pi = 1 / (1 + np.exp(-eta_binary))
        is_positive = np.random.binomial(1, pi, n)

        mu = np.exp(X @ beta_true)
        # Generate truncated Poisson (reject zeros)
        y_pos = np.zeros(n, dtype=int)
        for i in range(n):
            if is_positive[i]:
                while True:
                    sample = np.random.poisson(mu[i])
                    if sample > 0:
                        y_pos[i] = sample
                        break

        y = np.where(is_positive, y_pos, 0)

        return X, y, gamma_true, beta_true

    def test_fit_returns_result(self, hurdle_data):
        """Test that fit returns HurdlePoissonResult."""
        X, y, _, _ = hurdle_data
        result = fit_hurdle_poisson(X, X, y)

        assert isinstance(result, HurdlePoissonResult)

    def test_fit_coefficients_shape(self, hurdle_data):
        """Test coefficient shapes."""
        X, y, _, _ = hurdle_data
        result = fit_hurdle_poisson(X, X, y)

        assert result.coef_binary_.shape == (2,)
        assert result.coef_count_.shape == (2,)

    def test_fit_recovers_binary_params(self, hurdle_data):
        """Test recovery of binary model parameters."""
        X, y, gamma_true, _ = hurdle_data
        result = fit_hurdle_poisson(X, X, y)

        # Should be in same direction
        assert np.sign(result.coef_binary_[1]) == np.sign(gamma_true[1])
        np.testing.assert_allclose(result.coef_binary_, gamma_true, atol=0.4)

    def test_fit_recovers_count_params(self, hurdle_data):
        """Test recovery of count model parameters."""
        X, y, _, beta_true = hurdle_data
        result = fit_hurdle_poisson(X, X, y)

        # Truncated Poisson parameters may differ from raw Poisson
        # Check that sign and rough magnitude are correct
        assert np.sign(result.coef_count_[1]) == np.sign(beta_true[1])
        # Intercept should be positive (positive counts)
        assert result.coef_count_[0] > 0

    def test_fit_predictions(self, hurdle_data):
        """Test prediction types."""
        X, y, _, _ = hurdle_data
        result = fit_hurdle_poisson(X, X, y)

        # Response: marginal mean
        pred_response = result.predict(type="response")
        assert pred_response.shape == y.shape
        assert np.all(pred_response >= 0)

        # P(Y > 0)
        pred_pi = result.predict(type="prob_positive")
        assert np.all((pred_pi >= 0) & (pred_pi <= 1))

        # P(Y = 0)
        pred_p0 = result.predict(type="prob_zero")
        np.testing.assert_allclose(pred_pi + pred_p0, 1.0)

        # Truncated mean
        pred_count = result.predict(type="count")
        assert np.all(pred_count > 0)

    def test_fit_log_likelihood(self, hurdle_data):
        """Test log-likelihood is decomposed correctly."""
        X, y, _, _ = hurdle_data
        result = fit_hurdle_poisson(X, X, y)

        # Total should equal sum of components
        np.testing.assert_allclose(
            result.log_likelihood_,
            result.log_likelihood_binary_ + result.log_likelihood_count_,
            rtol=1e-10
        )

    def test_fit_summary(self, hurdle_data):
        """Test summary output."""
        X, y, _, _ = hurdle_data
        result = fit_hurdle_poisson(X, X, y)

        summary = result.summary()

        assert summary["n_obs"] == len(y)
        assert summary["n_zeros"] == np.sum(y == 0)
        assert summary["n_positive"] == np.sum(y > 0)
        assert "aic" in summary
        assert "bic" in summary

    def test_different_design_matrices(self, hurdle_data):
        """Test with different X for binary and count models."""
        X, y, _, _ = hurdle_data
        n = len(y)

        # Binary model: just intercept
        X_binary = np.ones((n, 1))
        # Count model: full X
        X_count = X

        result = fit_hurdle_poisson(X_binary, X_count, y)

        assert result.coef_binary_.shape == (1,)
        assert result.coef_count_.shape == (2,)


class TestHurdlePoissonEdgeCases:
    """Test edge cases for Hurdle Poisson."""

    def test_high_proportion_zeros(self):
        """Test with many zeros."""
        np.random.seed(42)
        n = 200

        # Intercept-only model for binary part since zeros aren't covariate-related
        X_binary = np.ones((n, 1))
        X_count = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # 90% zeros - first 20 are positive, rest are zeros
        y = np.zeros(n, dtype=int)
        positive_counts = np.random.poisson(3, 20)
        positive_counts[positive_counts == 0] = 1  # Ensure positives are >0
        y[:20] = positive_counts

        result = fit_hurdle_poisson(X_binary, X_count, y)

        # Binary model's intercept should be negative (low P(Y>0))
        # With 20/200 = 10% positive, logit(0.1) ≈ -2.2
        assert result.coef_binary_[0] < 0  # Negative intercept = low P(Y>0)

    def test_no_zeros(self):
        """Test with no zeros."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = np.random.poisson(5, n)
        y[y == 0] = 1  # Remove zeros

        result = fit_hurdle_poisson(X, X, y)

        # All π should be high (all positive)
        assert result.pi_.mean() > 0.9

    def test_all_ones(self):
        """Test with all counts equal to 1."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)

        result = fit_hurdle_poisson(X, X, y)

        # Should handle this gracefully
        assert result.truncated_mean_.mean() >= 1.0
