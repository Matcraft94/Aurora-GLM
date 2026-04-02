"""Tests for Bayesian GLM fitting."""

import numpy as np
import pytest

from aurora.models.bayes import (
    HAS_NUMPYRO,
    HAS_PYMC,
    BayesianGLMResult,
    Normal,
    PriorSpec,
    fit_glm_bayes,
)

# Skip all tests if no backend available
pytestmark = pytest.mark.skipif(
    not HAS_NUMPYRO and not HAS_PYMC,
    reason="No Bayesian backend available (install numpyro or pymc)",
)


@pytest.fixture
def gaussian_data():
    """Generate Gaussian regression data."""
    np.random.seed(42)
    n = 100
    p = 3

    X = np.random.randn(n, p)
    beta_true = np.array([1.0, -0.5, 0.3])
    sigma_true = 0.5

    y = X @ beta_true + np.random.normal(0, sigma_true, n)

    return X, y, beta_true, sigma_true


@pytest.fixture
def poisson_data():
    """Generate Poisson regression data."""
    np.random.seed(42)
    n = 100
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
    n = 100
    p = 2

    X = np.random.randn(n, p)
    beta_true = np.array([0.8, -0.5])

    prob = 1 / (1 + np.exp(-X @ beta_true))
    y = np.random.binomial(1, prob)

    return X, y, beta_true


class TestFitGLMBayesGaussian:
    """Tests for Bayesian Gaussian regression."""

    @pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not available")
    def test_fit_returns_result(self, gaussian_data):
        """Test that fit returns BayesianGLMResult."""
        X, y, _, _ = gaussian_data

        result = fit_glm_bayes(
            X, y, family="gaussian", backend="numpyro", draws=100, tune=50, chains=1
        )

        assert isinstance(result, BayesianGLMResult)

    @pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not available")
    def test_coef_shape(self, gaussian_data):
        """Test coefficient shape matches design matrix."""
        X, y, _, _ = gaussian_data

        result = fit_glm_bayes(
            X, y, family="gaussian", backend="numpyro", draws=100, tune=50, chains=1
        )

        assert result.coef_.shape == (X.shape[1],)

    @pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not available")
    def test_recovers_parameters(self, gaussian_data):
        """Test that posterior mean is close to true parameters."""
        X, y, beta_true, _ = gaussian_data

        result = fit_glm_bayes(
            X, y, family="gaussian", backend="numpyro", draws=500, tune=200, chains=2
        )

        # Posterior mean should be reasonably close to truth
        np.testing.assert_allclose(result.coef_, beta_true, atol=0.5)

    @pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not available")
    def test_credible_intervals(self, gaussian_data):
        """Test credible interval computation."""
        X, y, beta_true, _ = gaussian_data

        result = fit_glm_bayes(
            X, y, family="gaussian", backend="numpyro", draws=500, tune=200, chains=2
        )

        ci = result.credible_intervals(0.95)

        # CI should contain true values
        for i in range(len(beta_true)):
            assert ci["coef"][i, 0] <= beta_true[i] <= ci["coef"][i, 1]


class TestFitGLMBayesPoisson:
    """Tests for Bayesian Poisson regression."""

    @pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not available")
    def test_fit_poisson(self, poisson_data):
        """Test Poisson regression fitting."""
        X, y, beta_true = poisson_data

        result = fit_glm_bayes(
            X, y, family="poisson", backend="numpyro", draws=200, tune=100, chains=1
        )

        assert result.family == "poisson"
        assert result.link == "log"

    @pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not available")
    def test_poisson_predictions(self, poisson_data):
        """Test predictions from Poisson model."""
        X, y, _ = poisson_data

        result = fit_glm_bayes(
            X, y, family="poisson", backend="numpyro", draws=200, tune=100, chains=1
        )

        # Mean predictions
        y_pred = result.predict(X, type="mean")
        assert y_pred.shape == y.shape
        assert np.all(y_pred > 0)  # Log link means positive predictions


class TestFitGLMBayesBinomial:
    """Tests for Bayesian binomial regression."""

    @pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not available")
    def test_fit_binomial(self, binomial_data):
        """Test binomial regression fitting."""
        X, y, beta_true = binomial_data

        result = fit_glm_bayes(
            X, y, family="binomial", backend="numpyro", draws=200, tune=100, chains=1
        )

        assert result.family == "binomial"
        assert result.link == "logit"

    @pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not available")
    def test_binomial_predictions(self, binomial_data):
        """Test predictions from binomial model."""
        X, y, _ = binomial_data

        result = fit_glm_bayes(
            X, y, family="binomial", backend="numpyro", draws=200, tune=100, chains=1
        )

        y_pred = result.predict(X, type="mean")
        assert y_pred.shape == y.shape
        assert np.all((y_pred >= 0) & (y_pred <= 1))


class TestBayesianGLMResult:
    """Tests for BayesianGLMResult methods."""

    @pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not available")
    def test_summary(self, gaussian_data):
        """Test summary method."""
        X, y, _, _ = gaussian_data

        result = fit_glm_bayes(
            X, y, family="gaussian", backend="numpyro", draws=100, tune=50, chains=1
        )

        summary = result.summary()

        assert "coef" in summary
        assert "mean" in summary["coef"]
        assert "std" in summary["coef"]
        assert "percentiles" in summary["coef"]
        assert "n_obs" in summary
        assert "n_features" in summary

    @pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not available")
    def test_posterior_predictive(self, gaussian_data):
        """Test posterior predictive sampling."""
        X, y, _, _ = gaussian_data

        result = fit_glm_bayes(
            X, y, family="gaussian", backend="numpyro", draws=100, tune=50, chains=1
        )

        y_pred = result.posterior_predictive(X[:10], n_samples=50)

        assert y_pred.shape == (50, 10)

    @pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not available")
    def test_hdi_vs_eti(self, gaussian_data):
        """Test HDI vs ETI credible intervals."""
        X, y, _, _ = gaussian_data

        result = fit_glm_bayes(
            X, y, family="gaussian", backend="numpyro", draws=200, tune=100, chains=1
        )

        ci_hdi = result.credible_intervals(0.95, method="hdi")
        ci_eti = result.credible_intervals(0.95, method="eti")

        # Both should have same shape
        assert ci_hdi["coef"].shape == ci_eti["coef"].shape

        # HDI should be narrower or equal for unimodal distributions
        hdi_width = ci_hdi["coef"][:, 1] - ci_hdi["coef"][:, 0]
        eti_width = ci_eti["coef"][:, 1] - ci_eti["coef"][:, 0]

        # HDI width <= ETI width (approximately, for symmetric distributions they're equal)
        assert np.all(hdi_width <= eti_width + 0.1)


class TestPriorEffects:
    """Tests for prior specification effects."""

    @pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not available")
    def test_tight_prior_shrinks(self, gaussian_data):
        """Test that tight priors shrink estimates toward prior mean."""
        X, y, _, _ = gaussian_data

        # Fit with default priors
        result_default = fit_glm_bayes(
            X, y, family="gaussian", backend="numpyro", draws=200, tune=100, chains=1
        )

        # Fit with tight prior toward zero
        priors = PriorSpec()
        priors.coef_prior = Normal(0, 0.1)  # Very tight

        result_tight = fit_glm_bayes(
            X, y, family="gaussian", priors=priors, backend="numpyro", draws=200, tune=100, chains=1
        )

        # Tight prior should shrink coefficients toward zero
        assert np.abs(result_tight.coef_).mean() < np.abs(result_default.coef_).mean()


class TestInputValidation:
    """Tests for input validation."""

    def test_invalid_family(self, gaussian_data):
        """Test that invalid family raises error."""
        X, y, _, _ = gaussian_data

        with pytest.raises(ValueError, match="Unknown family"):
            fit_glm_bayes(X, y, family="invalid")

    def test_shape_mismatch(self, gaussian_data):
        """Test that X and y shape mismatch raises error."""
        X, y, _, _ = gaussian_data

        with pytest.raises(ValueError, match="rows"):
            fit_glm_bayes(X, y[:-10], family="gaussian")


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC not available")
class TestPyMCBackend:
    """Tests specific to PyMC backend."""

    def test_pymc_gaussian(self, gaussian_data):
        """Test PyMC backend for Gaussian regression."""
        X, y, _, _ = gaussian_data

        result = fit_glm_bayes(
            X, y, family="gaussian", backend="pymc", draws=100, tune=50, chains=1
        )

        assert result.backend_ == "pymc"
        assert result.coef_.shape == (X.shape[1],)
