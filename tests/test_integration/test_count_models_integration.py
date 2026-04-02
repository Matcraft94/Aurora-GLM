"""Integration tests for count data models.

These tests verify that Zero-Inflated and Hurdle models work correctly
in realistic scenarios and can be compared against each other.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestZIPVsZINBComparison:
    """Integration tests comparing ZIP and ZINB models."""

    @pytest.fixture
    def zero_inflated_data(self):
        """Generate zero-inflated count data."""
        np.random.seed(42)
        n = 500
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # True parameters
        beta_true = np.array([1.5, 0.5])
        pi_true = 0.3

        mu = np.exp(X @ beta_true)
        structural_zero = np.random.binomial(1, pi_true, n)
        poisson_counts = np.random.poisson(mu)
        y = np.where(structural_zero, 0, poisson_counts)

        return X, y, beta_true, pi_true

    def test_zip_fits_poisson_data(self, zero_inflated_data):
        """Test ZIP fits data without overdispersion."""
        from aurora.models.zero_inflated import fit_zip

        X, y, beta_true, _ = zero_inflated_data

        result = fit_zip(X, y)

        # Should recover count parameters reasonably
        assert_allclose(result.coef_count_, beta_true, atol=0.3)
        assert result.converged_

    def test_zinb_fits_poisson_data(self, zero_inflated_data):
        """Test ZINB also fits non-overdispersed data."""
        from aurora.models.zero_inflated import fit_zinb

        X, y, beta_true, _ = zero_inflated_data

        result = fit_zinb(X, y)

        # Should have high theta (approaching Poisson)
        assert result.theta_ > 1.0
        assert result.converged_

    def test_zip_vs_zinb_similar_predictions(self, zero_inflated_data):
        """Test ZIP and ZINB give similar predictions for Poisson data."""
        from aurora.models.zero_inflated import fit_zinb, fit_zip

        X, y, _, _ = zero_inflated_data

        zip_result = fit_zip(X, y)
        zinb_result = fit_zinb(X, y)

        # Predictions should be similar
        zip_pred = zip_result.predict(type="response")
        zinb_pred = zinb_result.predict(type="response")

        corr = np.corrcoef(zip_pred, zinb_pred)[0, 1]
        assert corr > 0.95


class TestHurdleVsZeroInflatedComparison:
    """Integration tests comparing Hurdle and Zero-Inflated models."""

    @pytest.fixture
    def count_data_with_zeros(self):
        """Generate count data with excess zeros."""
        np.random.seed(42)
        n = 500
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # True model parameters - lower values for numerical stability
        beta = np.array([0.8, 0.3])
        gamma = np.array([0.5, 0.8])

        # Probability of positive
        pi = 1 / (1 + np.exp(-X @ gamma))
        is_positive = np.random.binomial(1, pi, n)

        # Count values
        mu = np.exp(X @ beta)
        counts = np.random.poisson(mu)
        counts[counts == 0] = 1  # For hurdle, positive counts >= 1

        y = np.where(is_positive, counts, 0)

        return X, y, beta, gamma

    def test_hurdle_poisson_fits(self, count_data_with_zeros):
        """Test Hurdle Poisson fits correctly."""
        from aurora.models.hurdle import fit_hurdle_poisson

        X, y, _, _ = count_data_with_zeros

        result = fit_hurdle_poisson(X, X, y)

        # Check coefficients are finite
        assert np.all(np.isfinite(result.coef_count_))
        assert np.all(np.isfinite(result.coef_binary_))

    def test_zip_fits(self, count_data_with_zeros):
        """Test ZIP fits the same data."""
        from aurora.models.zero_inflated import fit_zip

        X, y, _, _ = count_data_with_zeros

        result = fit_zip(X, y)

        assert result.converged_
        assert np.all(np.isfinite(result.coef_count_))

    def test_both_models_predict_zeros_well(self, count_data_with_zeros):
        """Test both models predict P(Y=0) reasonably."""
        from aurora.models.hurdle import fit_hurdle_poisson
        from aurora.models.zero_inflated import fit_zip

        X, y, _, _ = count_data_with_zeros

        hurdle_result = fit_hurdle_poisson(X, X, y)
        zip_result = fit_zip(X, y)

        # Both should predict zeros
        hurdle_p0 = hurdle_result.predict(type="prob_zero")
        zip_p0 = zip_result.predict(type="prob_zero")

        # Should be positively correlated
        corr = np.corrcoef(hurdle_p0, zip_p0)[0, 1]
        assert corr > 0.5


class TestModelSelectionWorkflow:
    """Integration tests for model selection workflow."""

    @pytest.fixture
    def overdispersed_data(self):
        """Generate overdispersed count data with excess zeros."""
        np.random.seed(42)
        n = 500
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        from scipy.stats import nbinom

        beta = np.array([1.5, 0.5])
        pi = 0.3
        theta = 1.5  # Overdispersion

        mu = np.exp(X @ beta)
        structural_zero = np.random.binomial(1, pi, n)

        p_nb = theta / (theta + mu)
        nb_counts = nbinom.rvs(theta, p_nb)
        y = np.where(structural_zero, 0, nb_counts)

        return X, y

    def test_compare_models_by_aic(self, overdispersed_data):
        """Test comparing models using AIC."""
        from aurora.models.zero_inflated import fit_zinb, fit_zip

        X, y = overdispersed_data

        zip_result = fit_zip(X, y)
        zinb_result = fit_zinb(X, y)

        # ZINB should have better (lower) AIC for overdispersed data
        assert zinb_result.aic_ < zip_result.aic_

    def test_compare_models_by_bic(self, overdispersed_data):
        """Test comparing models using BIC."""
        from aurora.models.zero_inflated import fit_zinb, fit_zip

        X, y = overdispersed_data

        zip_result = fit_zip(X, y)
        zinb_result = fit_zinb(X, y)

        # ZINB should have better (lower) BIC for overdispersed data
        assert zinb_result.bic_ < zip_result.bic_


class TestEndToEndCountWorkflow:
    """End-to-end integration tests for count data analysis."""

    def test_full_zip_workflow(self):
        """Test complete ZIP analysis workflow."""
        from aurora.models.zero_inflated import fit_zip

        # 1. Generate data
        np.random.seed(42)
        n = 300
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        mu = np.exp(X @ [1.5, 0.5])
        pi = 0.3
        structural_zero = np.random.binomial(1, pi, n)
        y = np.where(structural_zero, 0, np.random.poisson(mu))

        # 2. Fit model
        result = fit_zip(X, y)
        assert result.converged_

        # 3. Get predictions
        pred_response = result.predict(type="response")
        pred_p0 = result.predict(type="prob_zero")

        assert pred_response.shape == (n,)
        assert np.all((pred_p0 >= 0) & (pred_p0 <= 1))

        # 4. Get summary
        summary = result.summary()
        assert "n_obs" in summary
        assert "n_zeros" in summary
        assert summary["n_zeros"] == np.sum(y == 0)

        # 5. Predictions on new data
        X_new = np.column_stack([np.ones(10), np.random.normal(0, 1, 10)])
        X_inflate_new = np.ones((10, 1))  # Intercept-only for inflation
        pred_new = result.predict(X_count=X_new, X_inflate=X_inflate_new, type="response")
        assert pred_new.shape == (10,)

    def test_full_hurdle_workflow(self):
        """Test complete Hurdle analysis workflow."""
        from aurora.models.hurdle import fit_hurdle_poisson

        # 1. Generate data
        np.random.seed(42)
        n = 300
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        pi = 1 / (1 + np.exp(-X @ [0.5, 0.8]))
        mu = np.exp(X @ [1.5, 0.3])
        is_positive = np.random.binomial(1, pi, n)
        y_pos = np.random.poisson(mu)
        y_pos[y_pos == 0] = 1
        y = np.where(is_positive, y_pos, 0)

        # 2. Fit model
        result = fit_hurdle_poisson(X, X, y)

        # 3. Get predictions
        pred_response = result.predict(type="response")
        pred_pi = result.predict(type="prob_positive")
        pred_count = result.predict(type="count")

        assert pred_response.shape == (n,)
        assert np.all((pred_pi > 0) & (pred_pi < 1))
        assert np.all(pred_count > 0)

        # 4. Get summary
        summary = result.summary()
        assert "n_obs" in summary
        assert "n_zeros" in summary
        assert "n_positive" in summary
        assert summary["n_obs"] == n

        # 5. Check log-likelihood decomposition
        assert_allclose(
            summary["log_likelihood"],
            summary["log_likelihood_binary"] + summary["log_likelihood_count"],
            rtol=1e-10,
        )


class TestSmoothingIntegration:
    """Integration tests for smoothing methods."""

    def test_pspline_fit_and_predict(self):
        """Test P-spline fitting and prediction."""
        from aurora.smoothing.splines import fit_pspline

        np.random.seed(42)
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x) + np.random.normal(0, 0.2, 100)

        # Fit with different smoothing selection methods
        for method in ["gcv", "aic"]:
            result = fit_pspline(x, y, n_basis=20, lambda_=method)

            assert result.fitted_values_.shape == (100,)
            assert result.lambda_ > 0
            assert result.edf_ > 0
            assert result.edf_ < 20

            # Predict at new points
            x_new = np.linspace(0, 2 * np.pi, 50)
            pred = result.predict(x_new)
            assert pred.shape == (50,)

    def test_loess_fit_and_predict(self):
        """Test LOESS fitting and prediction."""
        from aurora.smoothing.local import loess

        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.2, 100)

        # Fit with different spans
        for span in [0.3, 0.5, 0.7]:
            result = loess(x, y, span=span)

            assert result.fitted_values_.shape == (100,)

            # Predict at new points
            x_new = np.linspace(0, 10, 50)
            pred = result.predict(x_new)
            assert pred.shape == (50,)

    def test_robust_loess(self):
        """Test robust LOESS with outliers."""
        from aurora.smoothing.local import loess

        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        # Add outliers
        y_noisy = y.copy()
        y_noisy[20] = 10
        y_noisy[50] = -10

        result_standard = loess(x, y_noisy, span=0.3, robust=False)
        result_robust = loess(x, y_noisy, span=0.3, robust=True)

        # Robust should be closer to true signal
        true_y = np.sin(x)
        mse_standard = np.mean((result_standard.fitted_values_ - true_y) ** 2)
        mse_robust = np.mean((result_robust.fitted_values_ - true_y) ** 2)

        assert mse_robust < mse_standard


class TestDistributedIntegration:
    """Integration tests for distributed fitting."""

    def test_parallel_vs_sequential_gaussian(self):
        """Test parallel IRLS gives same result as sequential."""
        from aurora.models.distributed import fit_glm_parallel

        np.random.seed(42)
        n = 500
        p = 3

        X = np.random.randn(n, p)
        beta_true = np.array([1.0, -0.5, 0.3])
        y = X @ beta_true + np.random.normal(0, 0.5, n)

        # Single chunk (sequential)
        result_seq = fit_glm_parallel([X], [y], family="gaussian")

        # Multiple chunks (parallel)
        X_chunks = np.array_split(X, 10)
        y_chunks = np.array_split(y, 10)
        result_par = fit_glm_parallel(X_chunks, y_chunks, family="gaussian")

        # Should give identical results
        np.testing.assert_allclose(result_par.coef_, result_seq.coef_, rtol=1e-6)

    def test_sgd_convergence(self):
        """Test SGD converges on simple problem."""
        from aurora.models.distributed import fit_glm_sgd

        np.random.seed(42)
        n = 1000
        p = 3
        batch_size = 32

        X = np.random.randn(n, p)
        beta_true = np.array([1.0, -0.5, 0.3])
        y = X @ beta_true + np.random.normal(0, 0.5, n)

        # Create data iterator
        def data_iterator():
            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                yield X[i:end], y[i:end]

        result = fit_glm_sgd(
            data_iterator(),
            family="gaussian",
            optimizer="adam",
            learning_rate=0.01,
            max_epochs=50,
            n_features=p,
        )

        # Should be reasonably close to true parameters
        np.testing.assert_allclose(result.coef_, beta_true, atol=0.5)
