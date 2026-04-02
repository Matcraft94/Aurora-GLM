"""Unit tests for Bayesian GLM components.

These tests verify individual components of the Bayesian module
in isolation, without requiring the full MCMC backends.
"""

import numpy as np
import pytest

from aurora.models.bayes import (
    BayesianGLMResult,
    Cauchy,
    Exponential,
    Gamma,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    Laplace,
    Normal,
    PriorSpec,
    StudentT,
    Uniform,
)


class TestNormalPrior:
    """Unit tests for Normal prior."""

    def test_default_parameters(self):
        """Test default Normal prior parameters."""
        prior = Normal()

        assert prior.mu == 0.0
        assert prior.sigma == 10.0

    def test_custom_parameters(self):
        """Test custom Normal prior parameters."""
        prior = Normal(mu=1.0, sigma=2.0)

        assert prior.mu == 1.0
        assert prior.sigma == 2.0

    def test_invalid_sigma_raises(self):
        """Test that negative sigma raises error."""
        with pytest.raises(ValueError):
            Normal(sigma=-1.0)

    def test_zero_sigma_raises(self):
        """Test that zero sigma raises error."""
        with pytest.raises(ValueError):
            Normal(sigma=0.0)


class TestCauchyPrior:
    """Unit tests for Cauchy prior."""

    def test_default_parameters(self):
        """Test default Cauchy prior parameters."""
        prior = Cauchy()

        assert prior.loc == 0.0
        assert prior.scale == 2.5

    def test_custom_parameters(self):
        """Test custom Cauchy prior parameters."""
        prior = Cauchy(loc=1.0, scale=5.0)

        assert prior.loc == 1.0
        assert prior.scale == 5.0

    def test_invalid_scale_raises(self):
        """Test that negative scale raises error."""
        with pytest.raises(ValueError):
            Cauchy(scale=-1.0)


class TestHalfNormalPrior:
    """Unit tests for Half-Normal prior."""

    def test_default_parameters(self):
        """Test default Half-Normal prior parameters."""
        prior = HalfNormal()

        assert prior.sigma == 1.0

    def test_custom_parameters(self):
        """Test custom Half-Normal prior parameters."""
        prior = HalfNormal(sigma=2.5)

        assert prior.sigma == 2.5

    def test_invalid_sigma_raises(self):
        """Test that negative sigma raises error."""
        with pytest.raises(ValueError):
            HalfNormal(sigma=-1.0)


class TestHalfCauchyPrior:
    """Unit tests for Half-Cauchy prior."""

    def test_default_parameters(self):
        """Test default Half-Cauchy prior parameters."""
        prior = HalfCauchy()

        assert prior.scale == 2.5

    def test_custom_parameters(self):
        """Test custom Half-Cauchy prior parameters."""
        prior = HalfCauchy(scale=5.0)

        assert prior.scale == 5.0

    def test_invalid_scale_raises(self):
        """Test that negative scale raises error."""
        with pytest.raises(ValueError):
            HalfCauchy(scale=-1.0)


class TestStudentTPrior:
    """Unit tests for Student-t prior."""

    def test_default_parameters(self):
        """Test default Student-t prior parameters."""
        prior = StudentT()

        assert prior.df == 3.0
        assert prior.loc == 0.0
        assert prior.scale == 1.0

    def test_custom_parameters(self):
        """Test custom Student-t prior parameters."""
        prior = StudentT(df=5.0, loc=1.0, scale=2.0)

        assert prior.df == 5.0
        assert prior.loc == 1.0
        assert prior.scale == 2.0

    def test_invalid_df_raises(self):
        """Test that negative df raises error."""
        with pytest.raises(ValueError):
            StudentT(df=-1.0)

    def test_invalid_scale_raises(self):
        """Test that negative scale raises error."""
        with pytest.raises(ValueError):
            StudentT(scale=-1.0)


class TestLaplacePrior:
    """Unit tests for Laplace prior."""

    def test_default_parameters(self):
        """Test default Laplace prior parameters."""
        prior = Laplace()

        assert prior.loc == 0.0
        assert prior.scale == 1.0

    def test_custom_parameters(self):
        """Test custom Laplace prior parameters."""
        prior = Laplace(loc=1.0, scale=0.5)

        assert prior.loc == 1.0
        assert prior.scale == 0.5

    def test_invalid_scale_raises(self):
        """Test that negative scale raises error."""
        with pytest.raises(ValueError):
            Laplace(scale=-1.0)


class TestExponentialPrior:
    """Unit tests for Exponential prior."""

    def test_default_parameters(self):
        """Test default Exponential prior parameters."""
        prior = Exponential()

        assert prior.rate == 1.0

    def test_custom_parameters(self):
        """Test custom Exponential prior parameters."""
        prior = Exponential(rate=0.5)

        assert prior.rate == 0.5

    def test_invalid_rate_raises(self):
        """Test that negative rate raises error."""
        with pytest.raises(ValueError):
            Exponential(rate=-1.0)


class TestGammaPrior:
    """Unit tests for Gamma prior."""

    def test_default_parameters(self):
        """Test default Gamma prior parameters."""
        prior = Gamma()

        assert prior.alpha == 1.0
        assert prior.beta == 1.0

    def test_custom_parameters(self):
        """Test custom Gamma prior parameters."""
        prior = Gamma(alpha=2.0, beta=0.5)

        assert prior.alpha == 2.0
        assert prior.beta == 0.5

    def test_invalid_alpha_raises(self):
        """Test that negative alpha raises error."""
        with pytest.raises(ValueError):
            Gamma(alpha=-1.0)

    def test_invalid_beta_raises(self):
        """Test that negative beta raises error."""
        with pytest.raises(ValueError):
            Gamma(beta=-1.0)


class TestInverseGammaPrior:
    """Unit tests for Inverse-Gamma prior."""

    def test_default_parameters(self):
        """Test default Inverse-Gamma prior parameters."""
        prior = InverseGamma()

        assert prior.alpha == 1.0
        assert prior.beta == 1.0

    def test_custom_parameters(self):
        """Test custom Inverse-Gamma prior parameters."""
        prior = InverseGamma(alpha=0.01, beta=0.01)

        assert prior.alpha == 0.01
        assert prior.beta == 0.01


class TestUniformPrior:
    """Unit tests for Uniform prior."""

    def test_default_parameters(self):
        """Test default Uniform prior parameters."""
        prior = Uniform()

        assert prior.lower == 0.0
        assert prior.upper == 1.0

    def test_custom_parameters(self):
        """Test custom Uniform prior parameters."""
        prior = Uniform(lower=-5.0, upper=5.0)

        assert prior.lower == -5.0
        assert prior.upper == 5.0

    def test_invalid_bounds_raises(self):
        """Test that lower >= upper raises error."""
        with pytest.raises(ValueError):
            Uniform(lower=5.0, upper=0.0)

    def test_equal_bounds_raises(self):
        """Test that equal bounds raises error."""
        with pytest.raises(ValueError):
            Uniform(lower=1.0, upper=1.0)


class TestPriorSpec:
    """Unit tests for PriorSpec."""

    def test_default_priors(self):
        """Test default PriorSpec creates default priors."""
        spec = PriorSpec()

        assert isinstance(spec.intercept_prior, Normal)
        assert isinstance(spec.coef_prior, Normal)
        assert isinstance(spec.scale_prior, HalfNormal)
        assert isinstance(spec.shape_prior, Exponential)

    def test_custom_coef_prior(self):
        """Test setting custom coefficient prior."""
        spec = PriorSpec()
        spec.coef_prior = Normal(0, 1)

        assert spec.coef_prior.mu == 0.0
        assert spec.coef_prior.sigma == 1.0

    def test_custom_scale_prior(self):
        """Test setting custom scale prior."""
        spec = PriorSpec()
        spec.scale_prior = HalfCauchy(2.5)

        assert isinstance(spec.scale_prior, HalfCauchy)
        assert spec.scale_prior.scale == 2.5

    def test_set_coef_prior_by_index(self):
        """Test setting prior for specific coefficient indices."""
        spec = PriorSpec()
        sparse_prior = Laplace(0, 0.5)
        spec.set_coef_prior(sparse_prior, indices=[1, 2, 3])

        # Default prior for index 0
        assert isinstance(spec.get_coef_prior(0), Normal)
        # Laplace prior for indices 1, 2, 3
        assert isinstance(spec.get_coef_prior(1), Laplace)
        assert isinstance(spec.get_coef_prior(2), Laplace)
        assert isinstance(spec.get_coef_prior(3), Laplace)

    def test_to_dict(self):
        """Test converting PriorSpec to dictionary."""
        spec = PriorSpec()
        d = spec.to_dict()

        assert "intercept_prior" in d
        assert "coef_prior" in d
        assert "scale_prior" in d
        assert "shape_prior" in d


class TestBayesianGLMResult:
    """Unit tests for BayesianGLMResult."""

    @pytest.fixture
    def mock_result(self):
        """Create a mock result with synthetic posterior samples."""
        np.random.seed(42)
        n_samples = 1000
        n_chains = 4
        n_obs = 100
        n_params = 3

        # Simulate posterior samples - beta has shape (n_samples, n_params)
        posterior_samples = {
            "beta": np.column_stack(
                [
                    np.random.normal(1.0, 0.1, n_samples),
                    np.random.normal(-0.5, 0.2, n_samples),
                    np.random.normal(0.3, 0.15, n_samples),
                ]
            ),
            "sigma": np.random.exponential(1.0, n_samples),
        }

        return BayesianGLMResult(
            posterior_samples_=posterior_samples,
            family="gaussian",
            link="identity",
            n_samples_=n_samples,
            n_chains_=n_chains,
            n_obs_=n_obs,
            n_features_=n_params,
            X_=np.random.randn(n_obs, n_params),
            y_=np.random.randn(n_obs),
            backend_="test",
        )

    def test_coef_shape(self, mock_result):
        """Test coefficient shape."""
        assert mock_result.coef_.shape == (3,)

    def test_coef_std_shape(self, mock_result):
        """Test coefficient std shape."""
        assert mock_result.coef_std_.shape == (3,)

    def test_posterior_samples_keys(self, mock_result):
        """Test posterior samples has correct keys."""
        assert "beta" in mock_result.posterior_samples_
        assert "sigma" in mock_result.posterior_samples_

    def test_posterior_samples_length(self, mock_result):
        """Test posterior samples have correct length."""
        assert len(mock_result.posterior_samples_["beta"]) == mock_result.n_samples_

    def test_summary_returns_dict(self, mock_result):
        """Test summary returns dictionary."""
        summary = mock_result.summary()

        assert isinstance(summary, dict)
        assert "n_samples" in summary
        assert "n_chains" in summary
        assert "family" in summary

    def test_credible_intervals_shape(self, mock_result):
        """Test credible intervals shape."""
        ci = mock_result.credible_intervals(level=0.95)

        assert "coef" in ci
        assert ci["coef"].shape == (3, 2)

    def test_credible_intervals_ordering(self, mock_result):
        """Test lower < upper for credible intervals."""
        ci = mock_result.credible_intervals(level=0.95)

        assert np.all(ci["coef"][:, 0] < ci["coef"][:, 1])

    def test_credible_intervals_contains_mean(self, mock_result):
        """Test 95% credible interval contains posterior mean."""
        ci = mock_result.credible_intervals(level=0.95)
        coef = mock_result.coef_

        # With 1000 samples, the mean should be within the 95% CI
        assert np.all(ci["coef"][:, 0] <= coef)
        assert np.all(ci["coef"][:, 1] >= coef)

    def test_credible_intervals_different_levels(self, mock_result):
        """Test wider interval with higher level."""
        ci_90 = mock_result.credible_intervals(level=0.90)
        ci_95 = mock_result.credible_intervals(level=0.95)

        # 95% interval should be wider than 90%
        width_90 = ci_90["coef"][:, 1] - ci_90["coef"][:, 0]
        width_95 = ci_95["coef"][:, 1] - ci_95["coef"][:, 0]

        assert np.all(width_95 >= width_90 - 1e-10)

    def test_eti_method(self, mock_result):
        """Test equal-tailed interval method."""
        ci = mock_result.credible_intervals(level=0.95, method="eti")

        assert "coef" in ci
        assert ci["coef"].shape == (3, 2)
        assert np.all(ci["coef"][:, 0] < ci["coef"][:, 1])

    def test_scale_property(self, mock_result):
        """Test scale property returns mean of sigma samples."""
        scale = mock_result.scale_
        assert scale is not None
        assert scale > 0


class TestPriorSpecPresets:
    """Test common prior specification patterns."""

    def test_weakly_informative_setup(self):
        """Test creating weakly informative priors."""
        spec = PriorSpec()
        spec.intercept_prior = Normal(0, 10)
        spec.coef_prior = Normal(0, 2.5)
        spec.scale_prior = HalfCauchy(2.5)

        assert spec.intercept_prior.sigma == 10.0
        assert spec.coef_prior.sigma == 2.5
        assert spec.scale_prior.scale == 2.5

    def test_regularizing_setup(self):
        """Test creating regularizing (strong) priors."""
        spec = PriorSpec()
        spec.coef_prior = Normal(0, 1)  # Strong regularization

        assert spec.coef_prior.sigma == 1.0

    def test_sparse_prior_setup(self):
        """Test creating sparsity-inducing priors."""
        spec = PriorSpec()
        spec.coef_prior = Laplace(0, 0.5)  # Lasso-like

        assert isinstance(spec.coef_prior, Laplace)
        assert spec.coef_prior.scale == 0.5

    def test_robust_prior_setup(self):
        """Test creating robust priors with heavy tails."""
        spec = PriorSpec()
        spec.coef_prior = StudentT(df=3, loc=0, scale=2.5)

        assert isinstance(spec.coef_prior, StudentT)
        assert spec.coef_prior.df == 3.0


class TestPriorValidation:
    """Test parameter validation for all priors."""

    def test_all_priors_reject_invalid_scale(self):
        """Test all scale-parameterized priors reject invalid values."""
        with pytest.raises(ValueError):
            Normal(sigma=-1)
        with pytest.raises(ValueError):
            Cauchy(scale=-1)
        with pytest.raises(ValueError):
            HalfNormal(sigma=-1)
        with pytest.raises(ValueError):
            HalfCauchy(scale=-1)
        with pytest.raises(ValueError):
            StudentT(scale=-1)
        with pytest.raises(ValueError):
            Laplace(scale=-1)
        with pytest.raises(ValueError):
            Exponential(rate=-1)
        with pytest.raises(ValueError):
            Gamma(alpha=-1)
        with pytest.raises(ValueError):
            InverseGamma(alpha=-1)


class TestPriorDataclass:
    """Test dataclass functionality of priors."""

    def test_normal_equality(self):
        """Test Normal prior equality."""
        p1 = Normal(0, 1)
        p2 = Normal(0, 1)
        p3 = Normal(0, 2)

        assert p1 == p2
        assert p1 != p3

    def test_cauchy_equality(self):
        """Test Cauchy prior equality."""
        p1 = Cauchy(0, 2.5)
        p2 = Cauchy(0, 2.5)
        p3 = Cauchy(1, 2.5)

        assert p1 == p2
        assert p1 != p3

    def test_prior_repr(self):
        """Test prior has reasonable repr."""
        prior = Normal(0, 1)
        repr_str = repr(prior)

        assert "Normal" in repr_str
        assert "mu=0" in repr_str or "mu=0.0" in repr_str
