"""Tests for heavy-tailed distribution families (Phase 5.5).

This module tests the new distribution families:
- StudentTFamily: Student's t for robust regression
- CauchyFamily: Cauchy distribution (t with df=1)
- NegativeBinomialFamily: Negative Binomial for overdispersed counts
- TweedieFamily: Tweedie for zero-inflated continuous data

References
----------
.. [1] Lange et al. (1989). Robust statistical modeling using the t distribution.
.. [2] Hilbe (2011). Negative Binomial Regression.
.. [3] Jørgensen (1987). Exponential dispersion models.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aurora.distributions.families import (
    CauchyFamily,
    NegativeBinomialFamily,
    StudentTFamily,
    TweedieFamily,
)
from aurora.distributions.links import InverseSquareLink, PowerLink, SqrtLink

# =============================================================================
# Test Student's t Family
# =============================================================================


class TestStudentTFamily:
    """Tests for Student's t distribution family."""

    def test_initialization_default(self):
        """Test default initialization."""
        family = StudentTFamily()
        assert family.df == 5.0
        assert family.name == "student_t"

    def test_initialization_custom_df(self):
        """Test initialization with custom degrees of freedom."""
        family = StudentTFamily(df=3.0)
        assert family.df == 3.0

        family = StudentTFamily(df=10.0)
        assert family.df == 10.0

    def test_initialization_invalid_df(self):
        """Test that negative df raises error."""
        with pytest.raises(ValueError):
            StudentTFamily(df=-1.0)

        with pytest.raises(ValueError):
            StudentTFamily(df=0.0)

    def test_variance_finite_df_gt_2(self):
        """Variance should be finite for df > 2."""
        family = StudentTFamily(df=5.0)
        mu = np.array([1.0, 2.0, 3.0])
        var = family.variance(mu, scale=1.0)

        # Var = σ² × df/(df-2) = 1 × 5/3 ≈ 1.667
        expected = np.full(3, 5.0 / 3.0)
        assert_allclose(var, expected)

    def test_variance_infinite_df_le_2(self):
        """Variance should be infinite for df <= 2."""
        family = StudentTFamily(df=2.0)
        mu = np.array([1.0, 2.0])
        var = family.variance(mu, scale=1.0)

        assert np.all(np.isinf(var))

    def test_initialize_uses_median(self):
        """Initialize should use median for robustness."""
        family = StudentTFamily()
        y = np.array([1.0, 2.0, 3.0, 100.0])  # Outlier at 100

        mu_init = family.initialize(y)

        # Should use median (2.5), not mean (26.5)
        # Returns array filled with median
        assert_allclose(mu_init[0], np.median(y))

    def test_log_likelihood_computation(self):
        """Test log-likelihood computation."""
        family = StudentTFamily(df=5.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])  # Perfect fit

        ll = family.log_likelihood(y, mu, scale=1.0)

        # Perfect fit should give maximum likelihood
        # Perturb mu and check ll decreases
        mu_worse = np.array([1.5, 2.5, 3.5])
        ll_worse = family.log_likelihood(y, mu_worse, scale=1.0)

        assert ll > ll_worse

    def test_deviance_computation(self):
        """Test deviance computation."""
        family = StudentTFamily(df=5.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])

        # Perfect fit: deviance should be 0
        dev = family.deviance(y, mu, scale=1.0)
        assert_allclose(dev, 0.0, atol=1e-10)

        # Bad fit: deviance should be positive
        mu_bad = np.array([5.0, 5.0, 5.0])
        dev_bad = family.deviance(y, mu_bad, scale=1.0)
        assert dev_bad > 0

    def test_weights_downweight_outliers(self):
        """IRLS weights should downweight outliers."""
        family = StudentTFamily(df=5.0)

        # Normal residuals
        y_normal = np.array([1.0, 1.1, 0.9, 1.0])
        mu = np.ones(4)
        weights_normal = family.weights(y_normal, mu, scale=0.1)

        # With outlier
        y_outlier = np.array([1.0, 1.1, 0.9, 10.0])  # Last is outlier
        weights_outlier = family.weights(y_outlier, mu, scale=0.1)

        # Outlier should have lower weight
        assert weights_outlier[3] < weights_normal[3]
        assert weights_outlier[3] < 0.5  # Much smaller weight

    def test_scale_estimation_robust(self):
        """Scale estimation should be robust to outliers."""
        family = StudentTFamily(df=5.0)

        # Clean data
        y_clean = np.random.randn(100)
        mu_clean = np.zeros(100)
        family.estimate_scale(y_clean, mu_clean)

        # With outliers
        y_outlier = np.concatenate([np.random.randn(95), np.array([10, -10, 15, -15, 20])])
        mu_outlier = np.zeros(100)
        scale_outlier = family.estimate_scale(y_outlier, mu_outlier)

        # Robust scale should not be dramatically affected by outliers
        assert scale_outlier < 3.0  # Regular std would be much larger

    def test_link_identity_default(self):
        """Default link should be identity."""
        from aurora.distributions.links import IdentityLink

        family = StudentTFamily()
        assert isinstance(family.default_link, IdentityLink)

    def test_link_log_option(self):
        """Log link should be available."""
        from aurora.distributions.links import LogLink

        family = StudentTFamily(link="log")
        assert isinstance(family.default_link, LogLink)


class TestCauchyFamily:
    """Tests for Cauchy distribution (t with df=1)."""

    def test_is_student_t_with_df1(self):
        """Cauchy is Student's t with df=1."""
        family = CauchyFamily()
        assert family.df == 1.0
        assert family.name == "cauchy"

    def test_variance_infinite(self):
        """Cauchy variance is always infinite."""
        family = CauchyFamily()
        mu = np.array([1.0, 2.0, 3.0])
        var = family.variance(mu)

        assert np.all(np.isinf(var))


# =============================================================================
# Test Negative Binomial Family
# =============================================================================


class TestNegativeBinomialFamily:
    """Tests for Negative Binomial distribution family."""

    def test_initialization_default(self):
        """Test default initialization."""
        family = NegativeBinomialFamily()
        assert family.theta == 1.0
        assert family.name == "negative_binomial"

    def test_initialization_custom_theta(self):
        """Test initialization with custom theta."""
        family = NegativeBinomialFamily(theta=5.0)
        assert family.theta == 5.0

    def test_initialization_estimate_theta(self):
        """Test initialization with theta='estimate'."""
        family = NegativeBinomialFamily(theta="estimate")
        assert family._estimate_theta is True

    def test_initialization_invalid_theta(self):
        """Test that invalid theta raises error."""
        with pytest.raises(ValueError):
            NegativeBinomialFamily(theta=-1.0)

        with pytest.raises(ValueError):
            NegativeBinomialFamily(theta=0.0)

    def test_variance_overdispersion(self):
        """Variance should show overdispersion: Var = μ + μ²/θ."""
        family = NegativeBinomialFamily(theta=2.0)
        mu = np.array([1.0, 2.0, 5.0])
        var = family.variance(mu)

        # Var = μ + μ²/θ
        expected = mu + mu**2 / 2.0
        assert_allclose(var, expected)

    def test_variance_large_theta_approaches_poisson(self):
        """Large theta should give variance close to Poisson (Var ≈ μ)."""
        family = NegativeBinomialFamily(theta=1000.0)
        mu = np.array([1.0, 5.0, 10.0])
        var = family.variance(mu)

        # Should be close to μ
        assert_allclose(var, mu, rtol=0.01)

    def test_log_likelihood_computation(self):
        """Test log-likelihood computation."""
        family = NegativeBinomialFamily(theta=2.0)
        y = np.array([0, 1, 2, 3, 5])
        mu = np.array([1.0, 1.0, 2.0, 3.0, 4.0])

        ll = family.log_likelihood(y, mu, theta=2.0)

        # Should be finite
        assert np.isfinite(ll)

        # Worse fit should have lower likelihood
        mu_bad = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        ll_bad = family.log_likelihood(y, mu_bad, theta=2.0)

        assert ll > ll_bad

    def test_deviance_computation(self):
        """Test deviance for NB."""
        family = NegativeBinomialFamily(theta=2.0)
        y = np.array([0, 1, 2, 5, 10])
        mu = y.astype(float) + 0.1  # Close to y

        dev = family.deviance(y, mu, theta=2.0)

        # Should be non-negative
        assert dev >= 0

        # Worse fit should have higher deviance
        mu_bad = np.full(5, 5.0)
        dev_bad = family.deviance(y, mu_bad, theta=2.0)

        assert dev_bad > dev

    def test_estimate_theta_moments(self):
        """Test method of moments theta estimation."""
        family = NegativeBinomialFamily(theta="estimate")

        # Simulate overdispersed data
        np.random.seed(42)
        n = 1000
        true_mu = 5.0
        true_theta = 2.0

        # Generate NB data using Poisson-Gamma mixture
        gamma_rates = np.random.gamma(true_theta, true_mu / true_theta, n)
        y = np.random.poisson(gamma_rates)
        mu = np.full(n, true_mu)

        theta_est = family.estimate_theta(y, mu, method="moments")

        # Should be in reasonable range
        assert 0.5 < theta_est < 10.0

    def test_link_log_default(self):
        """Default link should be log."""
        from aurora.distributions.links import LogLink

        family = NegativeBinomialFamily()
        assert isinstance(family.default_link, LogLink)


# =============================================================================
# Test Tweedie Family
# =============================================================================


class TestTweedieFamily:
    """Tests for Tweedie distribution family."""

    def test_initialization_default(self):
        """Test default initialization."""
        family = TweedieFamily()
        assert family.power == 1.5
        assert family.name == "tweedie"

    def test_initialization_custom_power(self):
        """Test initialization with custom power."""
        family = TweedieFamily(power=1.7)
        assert family.power == 1.7

    def test_initialization_invalid_power(self):
        """Power must be in (1, 2) for compound Poisson-Gamma."""
        with pytest.raises(ValueError):
            TweedieFamily(power=1.0)

        with pytest.raises(ValueError):
            TweedieFamily(power=2.0)

        with pytest.raises(ValueError):
            TweedieFamily(power=0.5)

    def test_variance_power_function(self):
        """Variance should follow V(μ) = μ^p."""
        family = TweedieFamily(power=1.6)
        mu = np.array([1.0, 2.0, 5.0])
        var = family.variance(mu)

        expected = mu**1.6
        assert_allclose(var, expected)

    def test_initialize_uses_positive_values(self):
        """Initialize should handle zeros appropriately."""
        family = TweedieFamily()
        y = np.array([0, 0, 0, 5, 10, 15])  # Many zeros

        mu_init = family.initialize(y)

        # Should use mean of positive values
        assert_allclose(mu_init, 10.0)  # mean of [5, 10, 15]

    def test_deviance_with_zeros(self):
        """Deviance should handle exact zeros."""
        family = TweedieFamily(power=1.5)
        y = np.array([0, 0, 1, 5])
        mu = np.array([0.5, 1.0, 1.0, 5.0])

        dev = family.deviance(y, mu)

        # Should be finite and non-negative
        assert np.isfinite(dev)
        assert dev >= 0

    def test_probability_zero(self):
        """Test probability of observing exactly zero."""
        family = TweedieFamily(power=1.5, phi=1.0)
        mu = np.array([0.5, 1.0, 2.0, 5.0])

        p_zero = family.probability_zero(mu, phi=1.0)

        # Probabilities should be valid
        assert np.all(p_zero >= 0)
        assert np.all(p_zero <= 1)

        # Higher mean should give lower P(Y=0)
        assert p_zero[0] > p_zero[3]

    def test_estimate_phi(self):
        """Test dispersion parameter estimation."""
        family = TweedieFamily(power=1.5)

        np.random.seed(42)
        n = 100
        y = np.abs(np.random.randn(n)) * 5
        mu = np.full(n, np.mean(y))

        phi = family.estimate_phi(y, mu)

        # Should be positive
        assert phi > 0

    def test_link_log_default(self):
        """Default link should be log."""
        from aurora.distributions.links import LogLink

        family = TweedieFamily()
        assert isinstance(family.default_link, LogLink)


# =============================================================================
# Test New Link Functions
# =============================================================================


class TestSqrtLink:
    """Tests for square root link function."""

    def test_link_inverse_roundtrip(self):
        """Link and inverse should be inverses."""
        link = SqrtLink()
        mu = np.array([1.0, 4.0, 9.0, 16.0])

        eta = link.link(mu)
        mu_back = link.inverse(eta)

        assert_allclose(mu_back, mu)

    def test_link_values(self):
        """Test link function values."""
        link = SqrtLink()
        mu = np.array([1.0, 4.0, 9.0])

        eta = link.link(mu)
        expected = np.array([1.0, 2.0, 3.0])

        assert_allclose(eta, expected)

    def test_derivative(self):
        """Test link derivative."""
        link = SqrtLink()
        mu = np.array([1.0, 4.0, 9.0])

        deriv = link.derivative(mu)
        expected = 0.5 / np.sqrt(mu)

        assert_allclose(deriv, expected)


class TestPowerLink:
    """Tests for power link function."""

    def test_power_1_is_identity(self):
        """Power link with power=1 should be identity."""
        link = PowerLink(power=1.0)
        mu = np.array([1.0, 2.0, 3.0])

        eta = link.link(mu)
        assert_allclose(eta, mu)

        mu_back = link.inverse(eta)
        assert_allclose(mu_back, mu)

    def test_power_0_is_log(self):
        """Power link with power=0 should be log."""
        link = PowerLink(power=0.0)
        mu = np.array([1.0, np.e, np.e**2])

        eta = link.link(mu)
        expected = np.log(mu)

        assert_allclose(eta, expected)

    def test_power_minus1_is_inverse(self):
        """Power link with power=-1 should be inverse."""
        link = PowerLink(power=-1.0)
        mu = np.array([1.0, 2.0, 4.0])

        eta = link.link(mu)
        expected = 1.0 / mu

        assert_allclose(eta, expected)

    def test_power_half_is_sqrt(self):
        """Power link with power=0.5 should be sqrt."""
        link = PowerLink(power=0.5)
        mu = np.array([1.0, 4.0, 9.0])

        eta = link.link(mu)
        expected = np.sqrt(mu)

        assert_allclose(eta, expected)


class TestInverseSquareLink:
    """Tests for inverse square link function."""

    def test_link_values(self):
        """Test link function values."""
        link = InverseSquareLink()
        mu = np.array([1.0, 2.0, 0.5])

        eta = link.link(mu)
        expected = 1.0 / mu**2

        assert_allclose(eta, expected)

    def test_link_inverse_roundtrip(self):
        """Link and inverse should be inverses."""
        link = InverseSquareLink()
        mu = np.array([0.5, 1.0, 2.0])

        eta = link.link(mu)
        mu_back = link.inverse(eta)

        assert_allclose(mu_back, mu)


# =============================================================================
# Integration Tests
# =============================================================================


class TestFamilyIntegration:
    """Integration tests for new families."""

    @pytest.mark.parametrize(
        "FamilyClass,kwargs",
        [
            (StudentTFamily, {"df": 5.0}),
            (StudentTFamily, {"df": 3.0}),
            (CauchyFamily, {}),
            (NegativeBinomialFamily, {"theta": 2.0}),
            (TweedieFamily, {"power": 1.5}),
            (TweedieFamily, {"power": 1.7}),
        ],
    )
    def test_all_families_have_required_methods(self, FamilyClass, kwargs):
        """All families should have required interface methods."""
        family = FamilyClass(**kwargs)

        # Check required methods exist
        assert hasattr(family, "variance")
        assert hasattr(family, "initialize")
        assert hasattr(family, "log_likelihood")
        assert hasattr(family, "deviance")
        assert hasattr(family, "default_link")

    @pytest.mark.parametrize(
        "FamilyClass,kwargs",
        [
            (StudentTFamily, {"df": 5.0}),
            (NegativeBinomialFamily, {"theta": 2.0}),
            (TweedieFamily, {"power": 1.5}),
        ],
    )
    def test_variance_positive(self, FamilyClass, kwargs):
        """Variance should be positive for valid mu."""
        family = FamilyClass(**kwargs)
        mu = np.array([0.5, 1.0, 2.0, 5.0])

        var = family.variance(mu)

        # Handle infinite variance for Cauchy
        if np.all(np.isfinite(var)):
            assert np.all(var > 0)

    @pytest.mark.parametrize(
        "FamilyClass,kwargs,y",
        [
            (StudentTFamily, {"df": 5.0}, np.array([-1.0, 0.0, 1.0, 2.0])),
            (NegativeBinomialFamily, {"theta": 2.0}, np.array([0, 1, 2, 5])),
            (TweedieFamily, {"power": 1.5}, np.array([0, 0.5, 1.0, 5.0])),
        ],
    )
    def test_deviance_nonnegative(self, FamilyClass, kwargs, y):
        """Deviance should be non-negative."""
        family = FamilyClass(**kwargs)
        mu = np.abs(y) + 0.1

        dev = family.deviance(y, mu)

        assert dev >= 0
