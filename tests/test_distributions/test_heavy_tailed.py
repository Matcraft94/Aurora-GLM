"""Tests for Phase 5.5: Heavy-Tailed Distribution Families.

This module tests the new heavy-tailed distribution families:
- Student's t (and Cauchy)
- Negative Binomial
- Tweedie (Compound Poisson-Gamma)

References
----------
.. [1] Lange et al. (1989). Robust statistical modeling using the t distribution.
.. [2] Hilbe (2011). Negative Binomial Regression.
.. [3] Jørgensen (1987). Exponential dispersion models.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less
from scipy import stats

from aurora.distributions.families import (
    CauchyFamily,
    NegativeBinomialFamily,
    NegBinFamily,
    StudentTFamily,
    TweedieFamily,
    CompoundPoissonGammaFamily,
)
from aurora.distributions.links import (
    IdentityLink,
    LogLink,
    SqrtLink,
    PowerLink,
    InverseSquareLink,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def continuous_data():
    """Generate continuous data with potential outliers."""
    np.random.seed(42)
    n = 100
    # Mixture: 90% normal, 10% outliers
    normal_data = np.random.randn(int(0.9 * n)) * 2 + 5
    outliers = np.random.randn(int(0.1 * n)) * 10 + 5
    return np.concatenate([normal_data, outliers])


@pytest.fixture
def count_data():
    """Generate overdispersed count data."""
    np.random.seed(42)
    n = 100
    # Negative binomial data (overdispersed counts)
    return np.random.negative_binomial(n=5, p=0.3, size=n)


@pytest.fixture
def zero_inflated_data():
    """Generate zero-inflated continuous data (like insurance claims)."""
    np.random.seed(42)
    n = 200
    # 60% zeros, 40% positive gamma-distributed values
    zeros = np.zeros(int(0.6 * n))
    positives = np.random.gamma(2, 100, size=int(0.4 * n))
    data = np.concatenate([zeros, positives])
    np.random.shuffle(data)
    return data


# =============================================================================
# Test Student's t Distribution
# =============================================================================

class TestStudentTFamily:
    """Test Student's t distribution family."""

    def test_initialization(self):
        """Test initialization with different df values."""
        t5 = StudentTFamily(df=5)
        assert t5.df == 5
        assert t5.name == 'student_t'
        
        t3 = StudentTFamily(df=3)
        assert t3.df == 3

    def test_invalid_df(self):
        """df must be positive."""
        with pytest.raises(ValueError):
            StudentTFamily(df=0)
        with pytest.raises(ValueError):
            StudentTFamily(df=-1)

    def test_variance(self, continuous_data):
        """Variance should be σ² × df/(df-2) for df > 2."""
        t5 = StudentTFamily(df=5)
        mu = np.full_like(continuous_data, np.mean(continuous_data))
        var = t5.variance(mu, scale=2.0)
        
        expected_var = 2.0**2 * 5 / (5 - 2)  # 4 * 5/3 = 20/3
        assert_allclose(var[0], expected_var)

    def test_variance_infinite_for_small_df(self):
        """Variance should be infinite for df <= 2."""
        t2 = StudentTFamily(df=2)
        mu = np.array([1.0, 2.0, 3.0])
        var = t2.variance(mu)
        
        assert np.all(np.isinf(var))

    def test_initialize_uses_median(self, continuous_data):
        """Initialization should use median for robustness."""
        t5 = StudentTFamily(df=5)
        mu_init = t5.initialize(continuous_data)
        
        assert_allclose(mu_init[0], np.median(continuous_data))

    def test_log_likelihood_computation(self, continuous_data):
        """Log-likelihood should be computable."""
        t5 = StudentTFamily(df=5)
        mu = np.full_like(continuous_data, np.mean(continuous_data))
        
        ll = t5.log_likelihood(continuous_data, mu, scale=np.std(continuous_data))
        
        assert np.isfinite(ll)
        assert ll < 0  # Log-likelihood is typically negative

    def test_weights_downweight_outliers(self):
        """Weights should downweight observations far from mean."""
        t5 = StudentTFamily(df=5)
        y = np.array([0.0, 0.0, 0.0, 10.0])  # One outlier
        mu = np.array([0.0, 0.0, 0.0, 0.0])
        
        weights = t5.weights(y, mu, scale=1.0)
        
        # Outlier should have lower weight
        assert weights[3] < weights[0]
        assert weights[3] < weights[1]
        assert weights[3] < weights[2]

    def test_deviance(self, continuous_data):
        """Deviance should be non-negative."""
        t5 = StudentTFamily(df=5)
        mu = np.full_like(continuous_data, np.mean(continuous_data))
        
        dev = t5.deviance(continuous_data, mu, scale=np.std(continuous_data))
        
        assert dev >= 0

    def test_scale_estimation(self, continuous_data):
        """Scale should be estimated robustly."""
        t5 = StudentTFamily(df=5)
        mu = np.full_like(continuous_data, np.median(continuous_data))
        
        scale = t5.estimate_scale(continuous_data, mu)
        
        assert scale > 0
        assert np.isfinite(scale)

    def test_link_functions(self):
        """Should support identity and log links."""
        t_identity = StudentTFamily(df=5, link='identity')
        t_log = StudentTFamily(df=5, link='log')
        
        assert isinstance(t_identity.default_link, IdentityLink)
        assert isinstance(t_log.default_link, LogLink)

    def test_invalid_link(self):
        """Invalid link should raise error."""
        with pytest.raises(ValueError):
            StudentTFamily(df=5, link='logit')


class TestCauchyFamily:
    """Test Cauchy distribution (t with df=1)."""

    def test_is_t_with_df1(self):
        """Cauchy should be Student's t with df=1."""
        cauchy = CauchyFamily()
        assert cauchy.df == 1.0
        assert cauchy.name == 'cauchy'

    def test_infinite_variance(self):
        """Cauchy has infinite variance."""
        cauchy = CauchyFamily()
        mu = np.array([1.0, 2.0, 3.0])
        var = cauchy.variance(mu)
        
        assert np.all(np.isinf(var))


# =============================================================================
# Test Negative Binomial Distribution
# =============================================================================

class TestNegativeBinomialFamily:
    """Test Negative Binomial distribution family."""

    def test_initialization(self):
        """Test initialization with different theta values."""
        nb = NegativeBinomialFamily(theta=2.0)
        assert nb.theta == 2.0
        assert nb.name == 'negative_binomial'

    def test_estimate_mode(self):
        """Should support 'estimate' mode for theta."""
        nb = NegativeBinomialFamily(theta='estimate')
        assert nb._estimate_theta is True

    def test_invalid_theta(self):
        """theta must be positive or 'estimate'."""
        with pytest.raises(ValueError):
            NegativeBinomialFamily(theta=0)
        with pytest.raises(ValueError):
            NegativeBinomialFamily(theta=-1)
        with pytest.raises(ValueError):
            NegativeBinomialFamily(theta='unknown')

    def test_variance_overdispersion(self, count_data):
        """Variance should be μ + μ²/θ (overdispersed)."""
        nb = NegativeBinomialFamily(theta=2.0)
        mu = np.full_like(count_data, np.mean(count_data), dtype=float)
        
        var = nb.variance(mu)
        
        # Var = μ + μ²/θ
        expected_var = mu + mu**2 / 2.0
        assert_allclose(var, expected_var)

    def test_variance_approaches_poisson(self, count_data):
        """Large theta should give variance ≈ μ (Poisson)."""
        nb_large = NegativeBinomialFamily(theta=1e6)
        mu = np.full_like(count_data, np.mean(count_data), dtype=float)
        
        var = nb_large.variance(mu)
        
        # Should be close to μ
        assert_allclose(var, mu, rtol=1e-3)

    def test_log_likelihood(self, count_data):
        """Log-likelihood should be computable."""
        nb = NegativeBinomialFamily(theta=2.0)
        mu = np.full_like(count_data, np.mean(count_data), dtype=float)
        
        ll = nb.log_likelihood(count_data, mu)
        
        assert np.isfinite(ll)

    def test_deviance(self, count_data):
        """Deviance should be non-negative."""
        nb = NegativeBinomialFamily(theta=2.0)
        mu = np.full_like(count_data, np.mean(count_data), dtype=float)
        
        dev = nb.deviance(count_data, mu)
        
        assert dev >= 0

    def test_theta_estimation_moments(self, count_data):
        """Theta estimation via moments should work."""
        nb = NegativeBinomialFamily(theta=2.0)
        mu = np.full_like(count_data, np.mean(count_data), dtype=float)
        
        theta_est = nb.estimate_theta(count_data, mu, method='moments')
        
        assert theta_est > 0
        assert np.isfinite(theta_est)

    def test_theta_estimation_ml(self, count_data):
        """Theta estimation via ML should work."""
        nb = NegativeBinomialFamily(theta=2.0)
        mu = np.full_like(count_data, np.mean(count_data), dtype=float)
        
        theta_est = nb.estimate_theta(count_data, mu, method='ml')
        
        assert theta_est > 0
        assert np.isfinite(theta_est)

    def test_link_functions(self):
        """Should support log, identity, sqrt links."""
        nb_log = NegativeBinomialFamily(theta=2.0, link='log')
        nb_id = NegativeBinomialFamily(theta=2.0, link='identity')
        nb_sqrt = NegativeBinomialFamily(theta=2.0, link='sqrt')
        
        assert isinstance(nb_log.default_link, LogLink)
        assert isinstance(nb_id.default_link, IdentityLink)
        assert isinstance(nb_sqrt.default_link, SqrtLink)

    def test_alias(self):
        """NegBinFamily should be an alias."""
        assert NegBinFamily is NegativeBinomialFamily


# =============================================================================
# Test Tweedie Distribution
# =============================================================================

class TestTweedieFamily:
    """Test Tweedie distribution family."""

    def test_initialization(self):
        """Test initialization with power parameter."""
        tw = TweedieFamily(power=1.5)
        assert tw.power == 1.5
        assert tw.name == 'tweedie'

    def test_invalid_power(self):
        """Power must be in (1, 2) for compound Poisson-Gamma."""
        with pytest.raises(ValueError):
            TweedieFamily(power=1.0)  # Poisson
        with pytest.raises(ValueError):
            TweedieFamily(power=2.0)  # Gamma
        with pytest.raises(ValueError):
            TweedieFamily(power=0.5)  # Invalid

    def test_variance_power_law(self, zero_inflated_data):
        """Variance should be μ^p."""
        tw = TweedieFamily(power=1.5)
        mu = np.full_like(zero_inflated_data, 10.0)
        
        var = tw.variance(mu)
        
        expected_var = 10.0 ** 1.5
        assert_allclose(var[0], expected_var)

    def test_deviance_with_zeros(self, zero_inflated_data):
        """Deviance should handle zeros correctly."""
        tw = TweedieFamily(power=1.5)
        mu = np.full_like(zero_inflated_data, np.mean(zero_inflated_data[zero_inflated_data > 0]))
        
        dev = tw.deviance(zero_inflated_data, mu)
        
        assert np.isfinite(dev)
        assert dev >= 0

    def test_probability_zero(self):
        """Should compute probability of zero correctly."""
        tw = TweedieFamily(power=1.5, phi=1.0)
        mu = np.array([1.0, 5.0, 10.0])
        
        p_zero = tw.probability_zero(mu)
        
        # All probabilities should be in [0, 1]
        assert np.all(p_zero >= 0)
        assert np.all(p_zero <= 1)
        
        # Larger μ should have smaller P(Y=0)
        assert p_zero[0] > p_zero[1] > p_zero[2]

    def test_phi_estimation(self, zero_inflated_data):
        """Dispersion parameter should be estimable."""
        tw = TweedieFamily(power=1.5)
        mu = np.full_like(zero_inflated_data, np.mean(zero_inflated_data[zero_inflated_data > 0]))
        
        phi = tw.estimate_phi(zero_inflated_data, mu)
        
        assert phi > 0
        assert np.isfinite(phi)

    def test_power_estimation(self, zero_inflated_data):
        """Power parameter should be estimable via profile likelihood."""
        tw = TweedieFamily(power=1.5)
        mu = np.full_like(zero_inflated_data, np.mean(zero_inflated_data[zero_inflated_data > 0]))
        
        p_est = tw.estimate_power(zero_inflated_data, mu)
        
        assert 1.0 < p_est < 2.0
        assert np.isfinite(p_est)

    def test_link_functions(self):
        """Should support log and identity links."""
        tw_log = TweedieFamily(power=1.5, link='log')
        tw_id = TweedieFamily(power=1.5, link='identity')
        
        assert isinstance(tw_log.default_link, LogLink)
        assert isinstance(tw_id.default_link, IdentityLink)


class TestCompoundPoissonGammaFamily:
    """Test Compound Poisson-Gamma (alias for Tweedie)."""

    def test_is_tweedie_subclass(self):
        """Should be a subclass of TweedieFamily."""
        cpg = CompoundPoissonGammaFamily(power=1.5)
        assert isinstance(cpg, TweedieFamily)
        assert cpg.name == 'compound_poisson_gamma'

    def test_get_poisson_rate(self):
        """Should compute underlying Poisson rate."""
        cpg = CompoundPoissonGammaFamily(power=1.5, phi=1.0)
        mu = np.array([5.0])
        
        rate = cpg.get_poisson_rate(mu)
        
        # λ = μ^(2-p) / [φ(2-p)]
        expected_rate = 5.0 ** 0.5 / 0.5
        assert_allclose(rate[0], expected_rate)

    def test_get_gamma_shape(self):
        """Should compute Gamma shape parameter."""
        cpg = CompoundPoissonGammaFamily(power=1.5)
        
        alpha = cpg.get_gamma_shape()
        
        # α = (2-p)/(p-1)
        expected_alpha = 0.5 / 0.5  # = 1.0
        assert_allclose(alpha, expected_alpha)


# =============================================================================
# Test New Link Functions
# =============================================================================

class TestSqrtLink:
    """Test square root link function."""

    def test_link(self):
        """g(μ) = √μ."""
        link = SqrtLink()
        mu = np.array([1.0, 4.0, 9.0])
        
        eta = link.link(mu)
        
        assert_allclose(eta, np.array([1.0, 2.0, 3.0]))

    def test_inverse(self):
        """g⁻¹(η) = η²."""
        link = SqrtLink()
        eta = np.array([1.0, 2.0, 3.0])
        
        mu = link.inverse(eta)
        
        assert_allclose(mu, np.array([1.0, 4.0, 9.0]))

    def test_derivative(self):
        """g'(μ) = 1/(2√μ)."""
        link = SqrtLink()
        mu = np.array([1.0, 4.0, 9.0])
        
        deriv = link.derivative(mu)
        
        assert_allclose(deriv, np.array([0.5, 0.25, 1/6]))


class TestPowerLink:
    """Test general power link function."""

    def test_link_power_2(self):
        """g(μ) = μ² for power=2."""
        link = PowerLink(power=2.0)
        mu = np.array([1.0, 2.0, 3.0])
        
        eta = link.link(mu)
        
        assert_allclose(eta, np.array([1.0, 4.0, 9.0]))

    def test_link_power_minus1(self):
        """g(μ) = 1/μ for power=-1 (inverse)."""
        link = PowerLink(power=-1.0)
        mu = np.array([1.0, 2.0, 4.0])
        
        eta = link.link(mu)
        
        assert_allclose(eta, np.array([1.0, 0.5, 0.25]))

    def test_link_power_zero_uses_log(self):
        """Power ≈ 0 should use log link."""
        link = PowerLink(power=0.0)
        mu = np.array([1.0, np.e, np.e**2])
        
        eta = link.link(mu)
        
        assert_allclose(eta, np.array([0.0, 1.0, 2.0]))

    def test_roundtrip(self):
        """link(inverse(η)) = η."""
        link = PowerLink(power=1.5)
        eta = np.array([1.0, 2.0, 3.0])
        
        mu = link.inverse(eta)
        eta_back = link.link(mu)
        
        assert_allclose(eta_back, eta)


class TestInverseSquareLink:
    """Test inverse square link function."""

    def test_link(self):
        """g(μ) = 1/μ²."""
        link = InverseSquareLink()
        mu = np.array([1.0, 2.0, 4.0])
        
        eta = link.link(mu)
        
        assert_allclose(eta, np.array([1.0, 0.25, 0.0625]))

    def test_inverse(self):
        """g⁻¹(η) = 1/√η."""
        link = InverseSquareLink()
        eta = np.array([1.0, 4.0, 16.0])
        
        mu = link.inverse(eta)
        
        assert_allclose(mu, np.array([1.0, 0.5, 0.25]))


# =============================================================================
# Integration Tests
# =============================================================================

class TestHeavyTailedIntegration:
    """Integration tests for heavy-tailed families."""

    def test_robustness_comparison(self, continuous_data):
        """Student's t should be more robust than Gaussian."""
        # Add more outliers
        data_with_outliers = np.concatenate([continuous_data, [50, -50, 100]])
        mu = np.full_like(data_with_outliers, np.median(data_with_outliers))
        
        t5 = StudentTFamily(df=5)
        weights = t5.weights(data_with_outliers, mu, scale=np.std(continuous_data))
        
        # Outliers should have low weights
        outlier_weights = weights[-3:]
        normal_weights = weights[:-3]
        
        assert np.mean(outlier_weights) < np.mean(normal_weights)

    def test_overdispersion_detection(self):
        """NB should detect overdispersion in simulated data."""
        np.random.seed(42)
        
        # Overdispersed data (variance >> mean)
        overdispersed = np.random.negative_binomial(2, 0.1, size=100)
        
        nb = NegativeBinomialFamily(theta=1.0)
        mu = np.full_like(overdispersed, np.mean(overdispersed), dtype=float)
        
        theta_est = nb.estimate_theta(overdispersed, mu, method='moments')
        
        # Should estimate a small theta (high overdispersion)
        assert theta_est < 10

    def test_zero_inflation_handling(self, zero_inflated_data):
        """Tweedie should handle zero-inflation naturally."""
        tw = TweedieFamily(power=1.5)
        
        # Should not error with zeros
        mu = np.full_like(zero_inflated_data, max(np.mean(zero_inflated_data), 0.1))
        
        dev = tw.deviance(zero_inflated_data, mu)
        ll = tw.log_likelihood(zero_inflated_data, mu)
        
        assert np.isfinite(dev)
        assert np.isfinite(ll)

    def test_all_families_have_required_methods(self):
        """All families should have required interface methods."""
        families = [
            StudentTFamily(df=5),
            CauchyFamily(),
            NegativeBinomialFamily(theta=2.0),
            TweedieFamily(power=1.5),
        ]
        
        required_methods = ['variance', 'initialize', 'log_likelihood', 'deviance']
        
        for family in families:
            for method in required_methods:
                assert hasattr(family, method), f"{family.name} missing {method}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
