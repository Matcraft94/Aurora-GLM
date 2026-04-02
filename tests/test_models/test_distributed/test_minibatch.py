"""Tests for mini-batch SGD GLM fitting."""

import numpy as np
import pytest

from aurora.models.distributed import SGDResult, fit_glm_sgd


@pytest.fixture
def gaussian_data():
    """Generate Gaussian regression data."""
    np.random.seed(42)
    n = 500
    p = 3

    X = np.random.randn(n, p)
    beta_true = np.array([1.0, -0.5, 0.3])
    sigma = 0.5

    y = X @ beta_true + np.random.normal(0, sigma, n)

    return X, y, beta_true


@pytest.fixture
def poisson_data():
    """Generate Poisson regression data."""
    np.random.seed(42)
    n = 500
    p = 2

    X = np.random.randn(n, p)
    beta_true = np.array([0.5, -0.3])

    mu = np.exp(X @ beta_true)
    y = np.random.poisson(mu)

    return X, y, beta_true


@pytest.fixture
def binomial_data():
    """Generate binomial regression data."""
    np.random.seed(42)
    n = 500
    p = 2

    X = np.random.randn(n, p)
    beta_true = np.array([0.8, -0.5])

    prob = 1 / (1 + np.exp(-X @ beta_true))
    y = np.random.binomial(1, prob)

    return X, y, beta_true


def make_iterator(X, y, batch_size=100, n_epochs=1):
    """Create data iterator from arrays."""
    n = len(y)
    for _ in range(n_epochs):
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield X[start:end], y[start:end]


class TestFitGLMSGDGaussian:
    """Tests for SGD Gaussian regression."""

    def test_returns_result(self, gaussian_data):
        """Test that fit returns SGDResult."""
        X, y, _ = gaussian_data
        iterator = make_iterator(X, y, n_epochs=3)

        result = fit_glm_sgd(iterator, family="gaussian", max_epochs=3)

        assert isinstance(result, SGDResult)

    def test_coef_shape(self, gaussian_data):
        """Test coefficient shape."""
        X, y, _ = gaussian_data
        iterator = make_iterator(X, y, n_epochs=5)

        result = fit_glm_sgd(iterator, family="gaussian", max_epochs=5)

        assert result.coef_.shape == (X.shape[1],)

    def test_recovers_parameters(self, gaussian_data):
        """Test that SGD recovers true parameters."""
        X, y, beta_true = gaussian_data
        iterator = make_iterator(X, y, n_epochs=20)

        result = fit_glm_sgd(iterator, family="gaussian", learning_rate=0.01, max_epochs=20)

        # Should be reasonably close to true parameters
        np.testing.assert_allclose(result.coef_, beta_true, atol=0.3)

    def test_loss_decreases(self, gaussian_data):
        """Test that loss decreases over epochs."""
        X, y, _ = gaussian_data
        iterator = make_iterator(X, y, n_epochs=10)

        result = fit_glm_sgd(iterator, family="gaussian", max_epochs=10)

        # Loss should generally decrease
        assert result.loss_history_[-1] < result.loss_history_[0]


class TestFitGLMSGDPoisson:
    """Tests for SGD Poisson regression."""

    def test_poisson_fit(self, poisson_data):
        """Test Poisson regression fitting."""
        X, y, _ = poisson_data
        iterator = make_iterator(X, y, n_epochs=10)

        result = fit_glm_sgd(iterator, family="poisson", max_epochs=10)

        assert result.family == "poisson"
        assert result.link == "log"

    def test_poisson_predictions(self, poisson_data):
        """Test Poisson predictions are positive."""
        X, y, _ = poisson_data
        iterator = make_iterator(X, y, n_epochs=10)

        result = fit_glm_sgd(iterator, family="poisson", max_epochs=10)

        y_pred = result.predict(X)
        assert np.all(y_pred > 0)


class TestFitGLMSGDBinomial:
    """Tests for SGD binomial regression."""

    def test_binomial_fit(self, binomial_data):
        """Test binomial regression fitting."""
        X, y, _ = binomial_data
        iterator = make_iterator(X, y, n_epochs=10)

        result = fit_glm_sgd(iterator, family="binomial", max_epochs=10)

        assert result.family == "binomial"
        assert result.link == "logit"

    def test_binomial_predictions_bounded(self, binomial_data):
        """Test binomial predictions are in [0, 1]."""
        X, y, _ = binomial_data
        iterator = make_iterator(X, y, n_epochs=10)

        result = fit_glm_sgd(iterator, family="binomial", max_epochs=10)

        y_pred = result.predict(X)
        assert np.all((y_pred >= 0) & (y_pred <= 1))


class TestSGDOptimizers:
    """Test different optimizers."""

    def test_adam_optimizer(self, gaussian_data):
        """Test with Adam optimizer."""
        X, y, _ = gaussian_data
        iterator = make_iterator(X, y, n_epochs=5)

        result = fit_glm_sgd(iterator, family="gaussian", optimizer="adam", max_epochs=5)

        assert result.coef_ is not None

    def test_sgd_optimizer(self, gaussian_data):
        """Test with vanilla SGD."""
        X, y, _ = gaussian_data
        iterator = make_iterator(X, y, n_epochs=5)

        result = fit_glm_sgd(
            iterator, family="gaussian", optimizer="sgd", learning_rate=0.01, max_epochs=5
        )

        assert result.coef_ is not None

    def test_adagrad_optimizer(self, gaussian_data):
        """Test with AdaGrad optimizer."""
        X, y, _ = gaussian_data
        iterator = make_iterator(X, y, n_epochs=5)

        result = fit_glm_sgd(iterator, family="gaussian", optimizer="adagrad", max_epochs=5)

        assert result.coef_ is not None


class TestSGDResult:
    """Tests for SGDResult methods."""

    def test_predict(self, gaussian_data):
        """Test prediction method."""
        X, y, _ = gaussian_data
        iterator = make_iterator(X, y, n_epochs=5)

        result = fit_glm_sgd(iterator, family="gaussian", max_epochs=5)

        y_pred = result.predict(X)
        assert y_pred.shape == y.shape

    def test_predict_link(self, poisson_data):
        """Test prediction on link scale."""
        X, y, _ = poisson_data
        iterator = make_iterator(X, y, n_epochs=5)

        result = fit_glm_sgd(iterator, family="poisson", max_epochs=5)

        eta = result.predict(X, type="link")
        mu = result.predict(X, type="response")

        # For log link: mu = exp(eta)
        np.testing.assert_allclose(mu, np.exp(eta), rtol=1e-10)

    def test_summary(self, gaussian_data):
        """Test summary method."""
        X, y, _ = gaussian_data
        iterator = make_iterator(X, y, n_epochs=5)

        result = fit_glm_sgd(iterator, family="gaussian", max_epochs=5)

        summary = result.summary()

        assert "n_obs" in summary
        assert "n_features" in summary
        assert "n_epochs" in summary
        assert "final_loss" in summary
        assert "coef" in summary


class TestL2Regularization:
    """Tests for L2 regularization."""

    def test_l2_shrinks_coefficients(self, gaussian_data):
        """Test that L2 penalty shrinks coefficients."""
        X, y, _ = gaussian_data

        # Fit without regularization
        iterator1 = make_iterator(X, y, n_epochs=10)
        result1 = fit_glm_sgd(iterator1, family="gaussian", l2_penalty=0.0, max_epochs=10)

        # Fit with regularization
        iterator2 = make_iterator(X, y, n_epochs=10)
        result2 = fit_glm_sgd(iterator2, family="gaussian", l2_penalty=0.1, max_epochs=10)

        # Regularized coefficients should be smaller in magnitude
        assert np.linalg.norm(result2.coef_) < np.linalg.norm(result1.coef_)
