"""Unit tests for the Tweedie distribution family.

Mathematical Background
-----------------------
The Tweedie distribution is a member of the exponential dispersion family
characterized by the variance function:

    Var(Y) = φ × μ^p

where p is the power parameter (also called index parameter).

Special Cases
-------------
| p     | Distribution              | Support          |
|-------|---------------------------|------------------|
| 0     | Gaussian (Normal)         | (-∞, ∞)          |
| 1     | Poisson                   | {0, 1, 2, ...}   |
| (1,2) | Compound Poisson-Gamma    | {0} ∪ (0, ∞)     |
| 2     | Gamma                     | (0, ∞)           |
| 3     | Inverse Gaussian          | (0, ∞)           |

For 1 < p < 2 (the "Tweedie proper" case), the distribution is:
    Y = X₁ + X₂ + ... + X_N
where:
    N ~ Poisson(λ)
    X_i ~ Gamma(α, β)  (iid)

This naturally produces:
- Exact zeros when N = 0 (probability: e^{-λ})
- Continuous positive values when N > 0

Deviance
--------
The unit deviance for Tweedie with power p ∈ (1, 2) is:

    d(y, μ) = 2 × [y^{2-p}/((1-p)(2-p)) - yμ^{1-p}/(1-p) + μ^{2-p}/(2-p)]

For y = 0:
    d(0, μ) = 2μ^{2-p}/(2-p)

References
----------
[1] Jørgensen, B. (1987). "Exponential dispersion models."
    Journal of the Royal Statistical Society: Series B, 49(2), 127-162.
[2] Smyth, G. K., & Jørgensen, B. (2002).
    "Fitting Tweedie's compound Poisson model to insurance claims data."
    ASTIN Bulletin, 32(1), 143-157.
[3] Dunn, P. K., & Smyth, G. K. (2005).
    "Series evaluation of Tweedie exponential dispersion model densities."
    Statistics and Computing, 15(4), 267-280.
"""
from __future__ import annotations

import numpy as np
import pytest

from aurora.distributions.families.tweedie import (
    TweedieFamily,
    CompoundPoissonGammaFamily,
)


class TestTweedieFamilyBasic:
    """Basic functionality tests for TweedieFamily."""

    def test_instantiation_default(self):
        """Test default instantiation with power=1.5."""
        from aurora.distributions.links import LogLink
        family = TweedieFamily()
        assert family.power == 1.5
        assert family.phi == 1.0
        assert family.name == 'tweedie'
        assert isinstance(family.default_link, LogLink)

    def test_instantiation_custom_power(self):
        """Test instantiation with custom power parameter."""
        family = TweedieFamily(power=1.7)
        assert family.power == 1.7

    def test_instantiation_boundary_power_low(self):
        """Test that power <= 1 raises ValueError."""
        with pytest.raises(ValueError, match="power must be in"):
            TweedieFamily(power=1.0)

    def test_instantiation_boundary_power_high(self):
        """Test that power >= 2 raises ValueError."""
        with pytest.raises(ValueError, match="power must be in"):
            TweedieFamily(power=2.0)

    def test_instantiation_power_near_boundaries(self):
        """Test power values near but within boundaries."""
        # Near 1
        family_low = TweedieFamily(power=1.01)
        assert family_low.power == 1.01

        # Near 2
        family_high = TweedieFamily(power=1.99)
        assert family_high.power == 1.99

    def test_link_identity(self):
        """Test identity link instantiation."""
        from aurora.distributions.links import IdentityLink
        family = TweedieFamily(link='identity')
        assert isinstance(family.default_link, IdentityLink)

    def test_link_power(self):
        """Test power link instantiation."""
        from aurora.distributions.links import PowerLink
        family = TweedieFamily(link='power0.5')
        assert isinstance(family.default_link, PowerLink)

    def test_link_invalid(self):
        """Test invalid link raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported link"):
            TweedieFamily(link='logit')

    def test_repr(self):
        """Test string representation."""
        family = TweedieFamily(power=1.6, phi=2.0)
        repr_str = repr(family)
        assert 'TweedieFamily' in repr_str
        assert '1.6' in repr_str
        assert '2' in repr_str


class TestTweedieVariance:
    """Tests for the variance function V(μ) = μ^p."""

    def test_variance_formula(self):
        """Test variance function V(μ) = μ^p."""
        family = TweedieFamily(power=1.5)
        mu = np.array([1.0, 2.0, 4.0])

        result = family.variance(mu)
        expected = mu ** 1.5

        np.testing.assert_allclose(result, expected)

    def test_variance_different_powers(self):
        """Test variance with different power values."""
        mu = np.array([2.0])

        # p = 1.2 (more Poisson-like)
        family_12 = TweedieFamily(power=1.2)
        var_12 = family_12.variance(mu)
        np.testing.assert_allclose(var_12, 2.0 ** 1.2)

        # p = 1.8 (more Gamma-like)
        family_18 = TweedieFamily(power=1.8)
        var_18 = family_18.variance(mu)
        np.testing.assert_allclose(var_18, 2.0 ** 1.8)

    def test_variance_handles_small_mu(self):
        """Test variance handles small mu values (clipped to avoid issues)."""
        family = TweedieFamily(power=1.5)
        mu = np.array([1e-12, 1e-8, 1e-4])

        result = family.variance(mu)

        # Should not have NaN or Inf
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)


class TestTweedieDeviance:
    """Tests for the deviance function."""

    def test_deviance_zero_at_saturated(self):
        """Deviance is zero when μ = y (for y > 0)."""
        family = TweedieFamily(power=1.5)
        y = np.array([1.0, 2.0, 5.0])

        result = family.deviance(y, y)

        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_deviance_positive_for_non_saturated(self):
        """Deviance is positive when μ ≠ y."""
        family = TweedieFamily(power=1.5)
        y = np.array([1.0, 3.0, 5.0])
        mu = np.array([1.5, 2.5, 4.5])

        result = family.deviance(y, mu)

        assert result > 0

    def test_deviance_zero_response(self):
        """Test deviance handles y=0 correctly.

        For y=0: d(0, μ) = 2μ^{2-p}/(2-p)
        """
        family = TweedieFamily(power=1.5)
        y = np.array([0.0])
        mu = np.array([2.0])
        p = 1.5

        result = family.deviance(y, mu)

        # d(0, μ) = 2μ^{2-p}/(2-p) = 2×2^{0.5}/0.5 = 4×√2 ≈ 5.657
        expected = 2 * (2.0 ** 0.5) / 0.5

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_deviance_formula_positive_y(self):
        """Verify deviance formula for positive y values.

        d(y, μ) = 2[y^{2-p}/((1-p)(2-p)) - yμ^{1-p}/(1-p) + μ^{2-p}/(2-p)]
        """
        family = TweedieFamily(power=1.6)
        y = np.array([3.0])
        mu = np.array([2.0])
        p = 1.6

        result = family.deviance(y, mu)

        # Manual calculation
        term1 = y**(2-p) / ((1-p) * (2-p))
        term2 = y * mu**(1-p) / (1-p)
        term3 = mu**(2-p) / (2-p)
        expected = 2 * np.sum(term1 - term2 + term3)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_deviance_mixed_zero_positive(self):
        """Test deviance with mixed zeros and positive values."""
        family = TweedieFamily(power=1.5)
        y = np.array([0.0, 0.0, 1.5, 3.0])
        mu = np.array([0.5, 1.0, 2.0, 2.5])

        result = family.deviance(y, mu)

        assert np.isfinite(result)
        assert result > 0


class TestTweedieInitialize:
    """Tests for initialization."""

    def test_initialize_uses_positive_mean(self):
        """Initialize uses mean of positive values."""
        family = TweedieFamily()
        # Mix of zeros and positives
        y = np.array([0.0, 0.0, 2.0, 4.0, 6.0])

        result = family.initialize(y)

        # Mean of positive values: (2+4+6)/3 = 4
        expected = np.full_like(y, 4.0)
        np.testing.assert_allclose(result, expected)

    def test_initialize_all_zeros(self):
        """Initialize handles all-zero response."""
        family = TweedieFamily()
        y = np.array([0.0, 0.0, 0.0])

        result = family.initialize(y)

        # Should use fallback value
        assert np.all(result > 0)
        assert np.all(result == 0.1)

    def test_initialize_returns_positive(self):
        """Initialize always returns positive values."""
        family = TweedieFamily()
        y = np.array([0.0, 1.0, 5.0, 10.0])

        result = family.initialize(y)

        assert np.all(result > 0)


class TestTweedieLogLikelihood:
    """Tests for the quasi-log-likelihood approximation."""

    def test_log_likelihood_decreases_with_deviance(self):
        """Log-likelihood should be -0.5 × deviance / φ."""
        family = TweedieFamily(power=1.5, phi=2.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.5, 2.5, 2.8])

        ll = family.log_likelihood(y, mu)
        dev = family.deviance(y, mu)

        # ll ≈ -0.5 × D / φ
        expected = -0.5 * dev / 2.0

        np.testing.assert_allclose(ll, expected)

    def test_log_likelihood_phi_scaling(self):
        """Log-likelihood scales inversely with dispersion."""
        family1 = TweedieFamily(power=1.5, phi=1.0)
        family2 = TweedieFamily(power=1.5, phi=2.0)

        y = np.array([1.0, 2.0])
        mu = np.array([1.2, 1.8])

        ll1 = family1.log_likelihood(y, mu)
        ll2 = family2.log_likelihood(y, mu)

        # ll1/ll2 should be approximately 2 (inverse of phi ratio)
        np.testing.assert_allclose(ll1 / ll2, 2.0, rtol=1e-10)


class TestTweedieWeights:
    """Tests for IRLS weights."""

    def test_weights_inverse_variance(self):
        """Weights should be 1/V(μ) = 1/μ^p."""
        family = TweedieFamily(power=1.5)
        y = np.array([1.0, 2.0])  # Not used in weight calculation
        mu = np.array([2.0, 4.0])

        weights = family.weights(y, mu)

        expected = 1.0 / (mu ** 1.5)
        np.testing.assert_allclose(weights, expected, rtol=1e-10)


class TestTweedieGradients:
    """Tests for gradient functions."""

    def test_d_log_likelihood_direction(self):
        """Gradient points in correct direction."""
        family = TweedieFamily(power=1.5)
        y = np.array([5.0])

        # When mu < y, gradient should be positive
        mu_low = np.array([3.0])
        grad_low = family.d_log_likelihood(y, mu_low)
        assert grad_low > 0

        # When mu > y, gradient should be negative
        mu_high = np.array([8.0])
        grad_high = family.d_log_likelihood(y, mu_high)
        assert grad_high < 0

    def test_d_log_likelihood_formula(self):
        """Verify gradient formula: (y - μ) / (φ μ^p)."""
        family = TweedieFamily(power=1.6, phi=2.0)
        y = np.array([4.0])
        mu = np.array([3.0])

        result = family.d_log_likelihood(y, mu)
        expected = (y - mu) / (2.0 * mu ** 1.6)

        np.testing.assert_allclose(result, expected)

    def test_d2_log_likelihood_negative(self):
        """Second derivative (negative Fisher info) should be negative."""
        family = TweedieFamily(power=1.5)
        y = np.array([1.0, 3.0, 5.0])
        mu = np.array([1.5, 2.5, 4.5])

        result = family.d2_log_likelihood(y, mu)

        assert np.all(result < 0)


class TestTweedieParameterEstimation:
    """Tests for parameter estimation."""

    def test_estimate_phi_basic(self):
        """Test dispersion parameter estimation."""
        family = TweedieFamily(power=1.5)
        np.random.seed(42)

        # Generate data with known properties
        y = np.array([0.0, 0.5, 1.0, 2.0, 3.5, 5.0])
        mu = np.array([0.3, 0.8, 1.2, 1.8, 3.0, 4.5])

        phi_est = family.estimate_phi(y, mu, ddof=1)

        assert phi_est > 0
        assert np.isfinite(phi_est)

    def test_estimate_power_in_range(self):
        """Test power parameter estimation stays in valid range."""
        family = TweedieFamily(power=1.5)
        y = np.array([0.0, 0.0, 1.0, 2.0, 5.0, 10.0])
        mu = np.array([0.5, 1.0, 1.5, 2.5, 4.0, 8.0])

        power_est = family.estimate_power(y, mu, power_range=(1.1, 1.9))

        assert 1.1 <= power_est <= 1.9


class TestTweedieProbabilityZero:
    """Tests for probability of zero (P(Y=0))."""

    def test_probability_zero_formula(self):
        """Test P(Y=0) = exp(-λ) where λ = μ^{2-p}/(φ(2-p))."""
        family = TweedieFamily(power=1.5, phi=1.0)
        mu = np.array([2.0])
        p = 1.5
        phi = 1.0

        result = family.probability_zero(mu)

        # λ = 2^{0.5} / (1 × 0.5) = √2 / 0.5 = 2√2
        # P(Y=0) = exp(-2√2) ≈ 0.059
        lambda_ = 2.0 ** 0.5 / (1.0 * 0.5)
        expected = np.exp(-lambda_)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_probability_zero_increases_with_power(self):
        """P(Y=0) increases as power approaches 1 (more Poisson-like)."""
        mu = np.array([2.0])

        family_12 = TweedieFamily(power=1.2)
        family_18 = TweedieFamily(power=1.8)

        p_zero_12 = family_12.probability_zero(mu)
        p_zero_18 = family_18.probability_zero(mu)

        # Lower power = more probability of zero
        assert p_zero_12 > p_zero_18

    def test_probability_zero_in_valid_range(self):
        """P(Y=0) should be in [0, 1]."""
        family = TweedieFamily(power=1.5)
        mu = np.array([0.5, 1.0, 2.0, 5.0, 10.0])

        result = family.probability_zero(mu)

        assert np.all(result >= 0)
        assert np.all(result <= 1)


class TestCompoundPoissonGammaFamily:
    """Tests for the CompoundPoissonGammaFamily alias."""

    def test_is_tweedie_subclass(self):
        """CompoundPoissonGammaFamily is a TweedieFamily subclass."""
        family = CompoundPoissonGammaFamily(power=1.5)
        assert isinstance(family, TweedieFamily)

    def test_name(self):
        """Check the name attribute."""
        family = CompoundPoissonGammaFamily()
        assert family.name == 'compound_poisson_gamma'

    def test_poisson_rate(self):
        """Test Poisson rate λ = μ^{2-p}/(φ(2-p))."""
        family = CompoundPoissonGammaFamily(power=1.5, phi=2.0)
        mu = np.array([4.0])
        p = 1.5
        phi = 2.0

        result = family.get_poisson_rate(mu)

        expected = 4.0 ** (2-p) / (phi * (2-p))
        np.testing.assert_allclose(result, expected)

    def test_gamma_shape(self):
        """Test Gamma shape α = (2-p)/(p-1)."""
        family = CompoundPoissonGammaFamily(power=1.5)

        result = family.get_gamma_shape()

        # α = (2-1.5)/(1.5-1) = 0.5/0.5 = 1
        np.testing.assert_allclose(result, 1.0)

    def test_gamma_rate(self):
        """Test Gamma rate β = φ(p-1)μ^{p-1}."""
        family = CompoundPoissonGammaFamily(power=1.5, phi=2.0)
        mu = np.array([4.0])
        p = 1.5
        phi = 2.0

        result = family.get_gamma_rate(mu)

        expected = phi * (p - 1) * mu ** (p - 1)
        np.testing.assert_allclose(result, expected)


class TestTweedieEdgeCases:
    """Edge case tests."""

    def test_very_small_mu(self):
        """Test with very small mu values."""
        family = TweedieFamily(power=1.5)
        y = np.array([0.0, 0.1])
        mu = np.array([1e-8, 1e-6])

        # Should not raise
        dev = family.deviance(y, mu)
        var = family.variance(mu)
        ll = family.log_likelihood(y, mu)

        assert np.isfinite(dev)
        assert np.all(np.isfinite(var))
        assert np.isfinite(ll)

    def test_large_values(self):
        """Test with large values."""
        family = TweedieFamily(power=1.5)
        y = np.array([100.0, 500.0, 1000.0])
        mu = np.array([120.0, 480.0, 950.0])

        dev = family.deviance(y, mu)
        var = family.variance(mu)

        assert np.isfinite(dev)
        assert np.all(np.isfinite(var))

    def test_many_zeros(self):
        """Test with data containing many zeros (typical for insurance)."""
        family = TweedieFamily(power=1.7)
        # 70% zeros, typical for insurance claims
        y = np.array([0, 0, 0, 0, 0, 0, 0, 1.5, 3.2, 8.0])
        mu = np.full(10, 1.27)  # Overall mean

        dev = family.deviance(y, mu)
        p_zero = family.probability_zero(mu[:1])

        assert np.isfinite(dev)
        assert dev > 0
        assert 0 < p_zero < 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
