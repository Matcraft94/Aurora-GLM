"""Unit tests for the Negative Binomial distribution family.

Mathematical Background
-----------------------
The Negative Binomial (NB2) distribution models overdispersed count data where
Var(Y) > E[Y]. This arises when:
- Unobserved heterogeneity exists across observations
- Data is clustered or hierarchical
- Temporal/spatial correlation is present

NB2 Parameterization
--------------------
The NB2 parameterization (Hilbe, 2011) uses:
    Y ~ NegBin(μ, θ)
where:
    E[Y] = μ (mean)
    Var(Y) = μ + μ²/θ (overdispersion controlled by θ)

The probability mass function is:
    P(Y=y) = Γ(y+θ) / [Γ(θ)y!] × (θ/(θ+μ))^θ × (μ/(θ+μ))^y

Special cases:
    θ → ∞: Converges to Poisson (Var = μ)
    θ → 0: Extreme overdispersion

Log-Likelihood
--------------
ℓ(μ, θ; y) = Σᵢ [log Γ(yᵢ+θ) - log Γ(θ) - log(yᵢ!) + θ log(θ/(θ+μᵢ)) + yᵢ log(μᵢ/(θ+μᵢ))]

Deviance
--------
The unit deviance for NB is:
    d(y, μ) = 2[y log(y/μ) - (y+θ) log((y+θ)/(μ+θ))]

with d(0, μ) = 2θ log(θ/(μ+θ))

References
----------
[1] Hilbe, J. M. (2011). Negative Binomial Regression (2nd ed.). Cambridge.
[2] Cameron, A. C., & Trivedi, P. K. (2013). Regression Analysis of Count Data.
[3] Lawless, J. F. (1987). Negative binomial and mixed Poisson regression.
    Canadian Journal of Statistics, 15(3), 209-225.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import special

from aurora.distributions.families.negative_binomial import (
    NegativeBinomialFamily,
    NegBinFamily,
)


class TestNegativeBinomialFamilyBasic:
    """Basic functionality tests for NegativeBinomialFamily."""

    def test_instantiation_default(self):
        """Test default instantiation with theta=1.0, link='log'."""
        from aurora.distributions.links import LogLink
        family = NegativeBinomialFamily()
        assert family.theta == 1.0
        assert family.name == 'negative_binomial'
        assert isinstance(family.default_link, LogLink)

    def test_instantiation_custom_theta(self):
        """Test instantiation with custom theta."""
        family = NegativeBinomialFamily(theta=2.5)
        assert family.theta == 2.5

    def test_instantiation_estimate_theta(self):
        """Test instantiation with theta='estimate'."""
        family = NegativeBinomialFamily(theta='estimate')
        with pytest.raises(ValueError, match="has not been estimated"):
            _ = family.theta

    def test_instantiation_invalid_theta_negative(self):
        """Test that negative theta raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            NegativeBinomialFamily(theta=-1.0)

    def test_instantiation_invalid_theta_zero(self):
        """Test that zero theta raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            NegativeBinomialFamily(theta=0.0)

    def test_instantiation_invalid_theta_string(self):
        """Test that invalid string theta raises ValueError."""
        with pytest.raises(ValueError, match="must be a number or 'estimate'"):
            NegativeBinomialFamily(theta='invalid')

    def test_link_identity(self):
        """Test identity link instantiation."""
        from aurora.distributions.links import IdentityLink
        family = NegativeBinomialFamily(link='identity')
        assert isinstance(family.default_link, IdentityLink)

    def test_link_sqrt(self):
        """Test sqrt link instantiation."""
        from aurora.distributions.links import SqrtLink
        family = NegativeBinomialFamily(link='sqrt')
        assert isinstance(family.default_link, SqrtLink)

    def test_link_invalid(self):
        """Test invalid link raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported link"):
            NegativeBinomialFamily(link='probit')

    def test_alias_negbin(self):
        """Test NegBinFamily alias works."""
        family = NegBinFamily(theta=3.0)
        assert isinstance(family, NegativeBinomialFamily)
        assert family.theta == 3.0

    def test_repr_fixed_theta(self):
        """Test string representation with fixed theta."""
        family = NegativeBinomialFamily(theta=2.5)
        repr_str = repr(family)
        assert 'NegativeBinomialFamily' in repr_str
        assert '2.5' in repr_str
        assert 'log' in repr_str.lower()  # Check link type mentioned

    def test_repr_estimate_theta(self):
        """Test string representation with estimate theta."""
        family = NegativeBinomialFamily(theta='estimate')
        repr_str = repr(family)
        assert 'estimate' in repr_str.lower()


class TestNegativeBinomialVariance:
    """Tests for the variance function V(μ) = μ + μ²/θ."""

    def test_variance_formula_basic(self):
        """Test variance function matches V(μ) = μ + μ²/θ."""
        family = NegativeBinomialFamily(theta=2.0)
        mu = np.array([1.0, 2.0, 5.0])

        expected = mu + mu**2 / 2.0
        result = family.variance(mu)

        np.testing.assert_allclose(result, expected)

    def test_variance_approaches_poisson_large_theta(self):
        """For large θ, Var(Y) ≈ μ (Poisson limit)."""
        family = NegativeBinomialFamily(theta=1e6)
        mu = np.array([1.0, 5.0, 10.0])

        result = family.variance(mu)
        # With θ=1e6, μ²/θ is negligible
        np.testing.assert_allclose(result, mu, rtol=1e-5)

    def test_variance_high_overdispersion_small_theta(self):
        """For small θ, variance is dominated by μ²/θ term."""
        family = NegativeBinomialFamily(theta=0.1)
        mu = np.array([5.0])

        result = family.variance(mu)
        expected = 5.0 + 25.0 / 0.1  # = 5 + 250 = 255

        np.testing.assert_allclose(result, expected)

    def test_variance_with_theta_override(self):
        """Test theta override in params."""
        family = NegativeBinomialFamily(theta=1.0)
        mu = np.array([2.0])

        result = family.variance(mu, theta=4.0)
        expected = 2.0 + 4.0 / 4.0  # = 3.0

        np.testing.assert_allclose(result, expected)


class TestNegativeBinomialLogLikelihood:
    """Tests for the log-likelihood function."""

    def test_log_likelihood_against_scipy(self):
        """Compare log-likelihood against scipy.stats.nbinom."""
        from scipy.stats import nbinom

        family = NegativeBinomialFamily(theta=3.0)
        y = np.array([0, 1, 2, 5, 10])
        mu = np.array([2.0, 2.5, 3.0, 4.0, 8.0])

        result = family.log_likelihood(y, mu)

        # scipy uses (n, p) parameterization where n=theta, p=theta/(theta+mu)
        theta = 3.0
        p = theta / (theta + mu)
        expected = np.sum(nbinom.logpmf(y, n=theta, p=p))

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_likelihood_zero_count(self):
        """Test log-likelihood handles y=0 correctly."""
        family = NegativeBinomialFamily(theta=2.0)
        y = np.array([0.0])
        mu = np.array([3.0])

        result = family.log_likelihood(y, mu)

        # Manual calculation: P(Y=0) = (θ/(θ+μ))^θ = (2/5)^2 = 0.16
        expected = 2.0 * np.log(2.0 / 5.0)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_likelihood_requires_theta(self):
        """Test log-likelihood raises when theta not specified."""
        family = NegativeBinomialFamily(theta='estimate')
        y = np.array([1, 2])
        mu = np.array([1.5, 2.5])

        with pytest.raises(ValueError, match="theta must be specified"):
            family.log_likelihood(y, mu)

    def test_log_likelihood_positive_counts(self):
        """Test log-likelihood for positive counts."""
        family = NegativeBinomialFamily(theta=5.0)
        y = np.array([3.0])
        mu = np.array([2.0])
        theta = 5.0

        result = family.log_likelihood(y, mu)

        # Manual: log Γ(8) - log Γ(5) - log(3!) + 5 log(5/7) + 3 log(2/7)
        expected = (
            special.gammaln(8) - special.gammaln(5) - special.gammaln(4) +
            5 * np.log(5/7) + 3 * np.log(2/7)
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestNegativeBinomialDeviance:
    """Tests for the deviance function."""

    def test_deviance_zero_at_saturated(self):
        """Deviance is zero when μ = y (saturated model)."""
        family = NegativeBinomialFamily(theta=2.0)
        y = np.array([1.0, 2.0, 5.0])

        result = family.deviance(y, y)

        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_deviance_positive_for_non_saturated(self):
        """Deviance is positive when μ ≠ y."""
        family = NegativeBinomialFamily(theta=3.0)
        y = np.array([1.0, 5.0, 10.0])
        mu = np.array([2.0, 4.0, 8.0])

        result = family.deviance(y, mu)

        assert result > 0

    def test_deviance_zero_count(self):
        """Test deviance handles y=0 correctly."""
        family = NegativeBinomialFamily(theta=2.0)
        y = np.array([0.0])
        mu = np.array([3.0])
        theta = 2.0

        result = family.deviance(y, mu)

        # d(0, μ) = 2 × [-θ log((0+θ)/(μ+θ))]
        # = 2 × [0 - (0+2) log((0+2)/(3+2))]
        # = 2 × [-2 log(2/5)]
        expected = -2 * (0 + theta) * np.log((0 + theta) / (mu[0] + theta))

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_deviance_formula_verification(self):
        """Verify deviance formula: d = 2[y log(y/μ) - (y+θ) log((y+θ)/(μ+θ))]."""
        family = NegativeBinomialFamily(theta=4.0)
        y = np.array([3.0, 7.0])
        mu = np.array([2.0, 5.0])
        theta = 4.0

        result = family.deviance(y, mu)

        # Manual calculation
        term1 = y * np.log(y / mu)
        term2 = (y + theta) * np.log((y + theta) / (mu + theta))
        expected = 2 * np.sum(term1 - term2)

        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestNegativeBinomialInitialize:
    """Tests for initialization."""

    def test_initialize_returns_positive(self):
        """Initialize returns positive values."""
        family = NegativeBinomialFamily()
        y = np.array([0, 0, 0, 1, 5])

        result = family.initialize(y)

        assert np.all(result > 0)

    def test_initialize_uses_mean(self):
        """Initialize uses sample mean."""
        family = NegativeBinomialFamily()
        y = np.array([2.0, 4.0, 6.0])  # mean = 4

        result = family.initialize(y)

        np.testing.assert_allclose(result, np.full_like(y, 4.0))

    def test_initialize_all_zeros(self):
        """Initialize handles all-zero response."""
        family = NegativeBinomialFamily()
        y = np.array([0, 0, 0, 0])

        result = family.initialize(y)

        # Should use minimum of 0.1
        assert np.all(result >= 0.1)


class TestNegativeBinomialThetaEstimation:
    """Tests for dispersion parameter estimation."""

    def test_estimate_theta_moments_overdispersed(self):
        """Test moments estimator detects overdispersion."""
        family = NegativeBinomialFamily(theta='estimate')

        # Generate overdispersed data (manually constructed)
        np.random.seed(42)
        y = np.array([0, 0, 1, 1, 2, 3, 5, 8, 12, 20])
        mu = np.full_like(y, np.mean(y), dtype=float)

        theta_est = family.estimate_theta(y, mu, method='moments')

        # Should be a reasonable positive value
        assert theta_est > 0
        assert theta_est < 1e6  # Not Poisson

    def test_estimate_theta_moments_poisson_like(self):
        """Test moments estimator for Poisson-like data."""
        family = NegativeBinomialFamily(theta='estimate')

        # Data with Var ≈ mean (Poisson-like)
        y = np.array([3, 4, 5, 4, 3, 5, 4, 3, 5, 4])
        mu = np.full_like(y, np.mean(y), dtype=float)

        theta_est = family.estimate_theta(y, mu, method='moments')

        # Should return large theta (near Poisson)
        assert theta_est > 100

    def test_estimate_theta_ml_basic(self):
        """Test ML estimator runs without error."""
        family = NegativeBinomialFamily(theta='estimate')

        y = np.array([0, 1, 1, 2, 3, 5, 7, 10])
        mu = np.array([1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 6.0, 9.0])

        theta_ml = family.estimate_theta(y, mu, method='ml')

        assert theta_ml > 0

    def test_estimate_theta_invalid_method(self):
        """Test invalid method raises ValueError."""
        family = NegativeBinomialFamily(theta='estimate')

        with pytest.raises(ValueError, match="Unknown method"):
            family.estimate_theta(np.array([1, 2]), np.array([1.5, 2.5]), method='unknown')

    def test_theta_setter(self):
        """Test theta can be set after instantiation."""
        family = NegativeBinomialFamily(theta='estimate')
        family.theta = 2.5

        assert family.theta == 2.5

    def test_theta_setter_invalid(self):
        """Test setting invalid theta raises ValueError."""
        family = NegativeBinomialFamily(theta=1.0)

        with pytest.raises(ValueError, match="must be positive"):
            family.theta = -1.0


class TestNegativeBinomialGradients:
    """Tests for gradient functions (d/dμ)."""

    def test_d_log_likelihood_direction(self):
        """Gradient points in correct direction."""
        family = NegativeBinomialFamily(theta=2.0)
        y = np.array([5.0])

        # When mu < y, gradient should be positive (increase mu)
        mu_low = np.array([3.0])
        grad_low = family.d_log_likelihood(y, mu_low)
        assert grad_low > 0

        # When mu > y, gradient should be negative (decrease mu)
        mu_high = np.array([8.0])
        grad_high = family.d_log_likelihood(y, mu_high)
        assert grad_high < 0

    def test_d_log_likelihood_formula(self):
        """Verify gradient formula: d/dμ = y/μ - (y+θ)/(μ+θ)."""
        family = NegativeBinomialFamily(theta=3.0)
        y = np.array([4.0])
        mu = np.array([2.5])
        theta = 3.0

        result = family.d_log_likelihood(y, mu)
        expected = y / mu - (y + theta) / (mu + theta)

        np.testing.assert_allclose(result, expected)

    def test_d2_log_likelihood_negative(self):
        """Second derivative (Hessian) is negative (concave)."""
        family = NegativeBinomialFamily(theta=2.0)
        y = np.array([3.0, 5.0, 8.0])
        mu = np.array([2.5, 4.5, 7.0])

        result = family.d2_log_likelihood(y, mu)

        assert np.all(result < 0)

    def test_gradient_numerical_verification(self):
        """Verify gradient numerically via finite differences."""
        family = NegativeBinomialFamily(theta=2.5)
        y = np.array([3.0])
        mu = np.array([2.0])
        eps = 1e-6

        # Numerical gradient
        ll_plus = family.log_likelihood(y, mu + eps)
        ll_minus = family.log_likelihood(y, mu - eps)
        numerical_grad = (ll_plus - ll_minus) / (2 * eps)

        # Analytical gradient
        analytical_grad = family.d_log_likelihood(y, mu)[0]

        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-4)


class TestNegativeBinomialEdgeCases:
    """Edge case tests."""

    def test_very_small_mu(self):
        """Test with very small mu values."""
        family = NegativeBinomialFamily(theta=2.0)
        y = np.array([0.0, 1.0])
        mu = np.array([1e-8, 1e-8])

        # Should not raise
        ll = family.log_likelihood(y, mu)
        dev = family.deviance(y, mu)

        assert np.isfinite(ll)
        assert np.isfinite(dev)

    def test_large_counts(self):
        """Test with large count values."""
        family = NegativeBinomialFamily(theta=10.0)
        y = np.array([100.0, 500.0, 1000.0])
        mu = np.array([120.0, 480.0, 950.0])

        ll = family.log_likelihood(y, mu)
        dev = family.deviance(y, mu)

        assert np.isfinite(ll)
        assert np.isfinite(dev)

    def test_mixed_zero_positive(self):
        """Test with mixed zeros and positive counts."""
        family = NegativeBinomialFamily(theta=1.5)
        y = np.array([0, 0, 0, 1, 3, 0, 5, 0])
        mu = np.array([0.5, 0.8, 1.0, 1.2, 2.5, 0.7, 4.0, 0.3])

        ll = family.log_likelihood(y, mu)
        dev = family.deviance(y, mu)
        var = family.variance(mu)

        assert np.isfinite(ll)
        assert np.isfinite(dev)
        assert np.all(np.isfinite(var))


class TestNegativeBinomialVsPoisson:
    """Comparison tests with Poisson (limiting case)."""

    def test_converges_to_poisson_variance(self):
        """As θ → ∞, variance converges to μ."""
        from aurora.distributions.families.poisson import PoissonFamily

        poisson = PoissonFamily()
        negbin = NegativeBinomialFamily(theta=1e8)

        mu = np.array([1.0, 5.0, 10.0])

        var_poisson = poisson.variance(mu)
        var_negbin = negbin.variance(mu)

        np.testing.assert_allclose(var_negbin, var_poisson, rtol=1e-6)

    def test_converges_to_poisson_deviance(self):
        """As θ → ∞, deviance converges to Poisson deviance."""
        from aurora.distributions.families.poisson import PoissonFamily

        poisson = PoissonFamily()
        negbin = NegativeBinomialFamily(theta=1e8)

        y = np.array([1.0, 3.0, 5.0])
        mu = np.array([1.5, 2.5, 6.0])

        dev_poisson = poisson.deviance(y, mu)
        dev_negbin = negbin.deviance(y, mu)

        np.testing.assert_allclose(dev_negbin, dev_poisson, rtol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
