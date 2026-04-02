"""Tests for prior specification classes."""

import pytest

from aurora.models.bayes import (
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
    """Tests for Normal prior."""

    def test_default_values(self):
        """Test default parameter values."""
        prior = Normal()
        assert prior.mu == 0.0
        assert prior.sigma == 10.0

    def test_custom_values(self):
        """Test custom parameter values."""
        prior = Normal(mu=1.0, sigma=2.0)
        assert prior.mu == 1.0
        assert prior.sigma == 2.0

    def test_invalid_sigma(self):
        """Test that negative sigma raises error."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            Normal(sigma=-1.0)

    def test_zero_sigma(self):
        """Test that zero sigma raises error."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            Normal(sigma=0.0)


class TestCauchyPrior:
    """Tests for Cauchy prior."""

    def test_default_values(self):
        """Test default parameter values."""
        prior = Cauchy()
        assert prior.loc == 0.0
        assert prior.scale == 2.5

    def test_invalid_scale(self):
        """Test that negative scale raises error."""
        with pytest.raises(ValueError, match="scale must be positive"):
            Cauchy(scale=-1.0)


class TestHalfNormalPrior:
    """Tests for HalfNormal prior."""

    def test_default_values(self):
        """Test default parameter values."""
        prior = HalfNormal()
        assert prior.sigma == 1.0

    def test_invalid_sigma(self):
        """Test that negative sigma raises error."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            HalfNormal(sigma=-1.0)


class TestHalfCauchyPrior:
    """Tests for HalfCauchy prior."""

    def test_default_values(self):
        """Test default parameter values."""
        prior = HalfCauchy()
        assert prior.scale == 2.5

    def test_invalid_scale(self):
        """Test that negative scale raises error."""
        with pytest.raises(ValueError, match="scale must be positive"):
            HalfCauchy(scale=0.0)


class TestExponentialPrior:
    """Tests for Exponential prior."""

    def test_default_values(self):
        """Test default parameter values."""
        prior = Exponential()
        assert prior.rate == 1.0

    def test_invalid_rate(self):
        """Test that non-positive rate raises error."""
        with pytest.raises(ValueError, match="rate must be positive"):
            Exponential(rate=0.0)


class TestGammaPrior:
    """Tests for Gamma prior."""

    def test_default_values(self):
        """Test default parameter values."""
        prior = Gamma()
        assert prior.alpha == 1.0
        assert prior.beta == 1.0

    def test_invalid_alpha(self):
        """Test that non-positive alpha raises error."""
        with pytest.raises(ValueError, match="alpha and beta must be positive"):
            Gamma(alpha=0.0)

    def test_invalid_beta(self):
        """Test that non-positive beta raises error."""
        with pytest.raises(ValueError, match="alpha and beta must be positive"):
            Gamma(beta=-1.0)


class TestInverseGammaPrior:
    """Tests for InverseGamma prior."""

    def test_default_values(self):
        """Test default parameter values."""
        prior = InverseGamma()
        assert prior.alpha == 1.0
        assert prior.beta == 1.0

    def test_invalid_params(self):
        """Test that non-positive params raise error."""
        with pytest.raises(ValueError, match="alpha and beta must be positive"):
            InverseGamma(alpha=-1.0)


class TestUniformPrior:
    """Tests for Uniform prior."""

    def test_default_values(self):
        """Test default parameter values."""
        prior = Uniform()
        assert prior.lower == 0.0
        assert prior.upper == 1.0

    def test_invalid_bounds(self):
        """Test that lower >= upper raises error."""
        with pytest.raises(ValueError, match="lower must be less than upper"):
            Uniform(lower=1.0, upper=0.0)


class TestStudentTPrior:
    """Tests for StudentT prior."""

    def test_default_values(self):
        """Test default parameter values."""
        prior = StudentT()
        assert prior.df == 3.0
        assert prior.loc == 0.0
        assert prior.scale == 1.0

    def test_invalid_df(self):
        """Test that non-positive df raises error."""
        with pytest.raises(ValueError, match="df and scale must be positive"):
            StudentT(df=0.0)


class TestLaplacePrior:
    """Tests for Laplace prior."""

    def test_default_values(self):
        """Test default parameter values."""
        prior = Laplace()
        assert prior.loc == 0.0
        assert prior.scale == 1.0

    def test_invalid_scale(self):
        """Test that non-positive scale raises error."""
        with pytest.raises(ValueError, match="scale must be positive"):
            Laplace(scale=-1.0)


class TestPriorSpec:
    """Tests for PriorSpec configuration."""

    def test_default_priors(self):
        """Test default prior specifications."""
        spec = PriorSpec()

        assert isinstance(spec.intercept_prior, Normal)
        assert spec.intercept_prior.sigma == 100

        assert isinstance(spec.coef_prior, Normal)
        assert spec.coef_prior.sigma == 10

        assert isinstance(spec.scale_prior, HalfNormal)
        assert isinstance(spec.shape_prior, Exponential)

    def test_set_coef_prior_all(self):
        """Test setting prior for all coefficients."""
        spec = PriorSpec()
        spec.coef_prior = Normal(0, 1)

        assert spec.coef_prior.sigma == 1.0
        assert spec.get_coef_prior(0).sigma == 1.0
        assert spec.get_coef_prior(5).sigma == 1.0

    def test_set_coef_prior_by_index(self):
        """Test setting prior for specific coefficients."""
        spec = PriorSpec()
        spec.set_coef_prior(Normal(0, 0.5), indices=[0, 1])
        spec.set_coef_prior(Normal(0, 2.0), indices=[2])

        assert spec.get_coef_prior(0).sigma == 0.5
        assert spec.get_coef_prior(1).sigma == 0.5
        assert spec.get_coef_prior(2).sigma == 2.0
        assert spec.get_coef_prior(3).sigma == 10.0  # Default

    def test_to_dict(self):
        """Test conversion to dictionary."""
        spec = PriorSpec()
        d = spec.to_dict()

        assert "intercept_prior" in d
        assert "coef_prior" in d
        assert "scale_prior" in d
        assert "shape_prior" in d
