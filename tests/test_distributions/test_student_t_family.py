"""Unit tests for the Student's t distribution family.

Mathematical Background
-----------------------
The Student's t-distribution is used for robust regression, being less
sensitive to outliers than the Gaussian distribution due to its heavier tails.

Probability Density Function
----------------------------
For location μ, scale σ, and ν degrees of freedom:

    f(y; μ, σ, ν) = Γ((ν+1)/2) / [√(νπ) σ Γ(ν/2)] × (1 + z²/ν)^{-(ν+1)/2}

where z = (y - μ) / σ is the standardized residual.

Properties
----------
- E[Y] = μ (for ν > 1)
- Var(Y) = σ² × ν/(ν-2) (for ν > 2)
- For ν ≤ 2, variance is infinite
- For ν = 1, this is the Cauchy distribution (no finite moments)

Special Cases
-------------
| ν     | Distribution        | Properties                    |
|-------|---------------------|-------------------------------|
| 1     | Cauchy              | No finite moments             |
| 2-4   | Very heavy tails    | Robust, slow convergence      |
| 5-10  | Moderate tails      | Good balance robustness/eff.  |
| 30+   | Near-Gaussian       | Light tails                   |
| ∞     | Gaussian (Normal)   | No outlier robustness         |

Robust Regression via IRLS
--------------------------
The t-distribution enables robust regression through adaptive weights:

    w_i = (ν + 1) / (ν + z_i²)

These weights downweight observations with large residuals (outliers).

References
----------
[1] Lange, K. L., Little, R. J., & Taylor, J. M. (1989).
    "Robust statistical modeling using the t distribution."
    Journal of the American Statistical Association, 84(408), 881-896.
[2] Fernandez, C., & Steel, M. F. (1999).
    "Multivariate Student-t regression models."
    Journal of the Royal Statistical Society: Series B, 61(3), 579-602.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special
from scipy.stats import t as t_dist

from aurora.distributions.families.student_t import (
    CauchyFamily,
    StudentTFamily,
)


class TestStudentTFamilyBasic:
    """Basic functionality tests for StudentTFamily."""

    def test_instantiation_default(self):
        """Test default instantiation with df=5, link='identity'."""
        from aurora.distributions.links import IdentityLink

        family = StudentTFamily()
        assert family.df == 5.0
        assert family.name == "student_t"
        assert isinstance(family.default_link, IdentityLink)

    def test_instantiation_custom_df(self):
        """Test instantiation with custom degrees of freedom."""
        family = StudentTFamily(df=10.0)
        assert family.df == 10.0

    def test_instantiation_df_one(self):
        """Test df=1 (Cauchy distribution)."""
        family = StudentTFamily(df=1.0)
        assert family.df == 1.0

    def test_instantiation_invalid_df_zero(self):
        """Test that df=0 raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            StudentTFamily(df=0.0)

    def test_instantiation_invalid_df_negative(self):
        """Test that negative df raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            StudentTFamily(df=-5.0)

    def test_link_log(self):
        """Test log link instantiation."""
        from aurora.distributions.links import LogLink

        family = StudentTFamily(link="log")
        assert isinstance(family.default_link, LogLink)

    def test_link_invalid(self):
        """Test invalid link raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported link"):
            StudentTFamily(link="logit")

    def test_repr(self):
        """Test string representation."""
        family = StudentTFamily(df=5.0)
        repr_str = repr(family)
        assert "StudentTFamily" in repr_str
        assert "5" in repr_str
        assert "identity" in repr_str


class TestStudentTVariance:
    """Tests for the variance function."""

    def test_variance_formula_df_gt_2(self):
        """Test variance = σ² × ν/(ν-2) for ν > 2."""
        family = StudentTFamily(df=5.0)
        family._scale = 2.0  # Set scale
        mu = np.array([0.0, 1.0, -1.0])

        result = family.variance(mu, scale=2.0)

        # Var = σ² × df/(df-2) = 4 × 5/3 = 20/3 ≈ 6.667
        expected = 4.0 * 5.0 / 3.0
        np.testing.assert_allclose(result, np.full_like(mu, expected))

    def test_variance_infinite_df_le_2(self):
        """Test variance is infinite for df ≤ 2."""
        family = StudentTFamily(df=2.0)
        mu = np.array([0.0, 1.0])

        result = family.variance(mu)

        assert np.all(np.isinf(result))

    def test_variance_df_1_cauchy(self):
        """Test variance is infinite for df=1 (Cauchy)."""
        family = StudentTFamily(df=1.0)
        mu = np.array([0.0])

        result = family.variance(mu)

        assert np.all(np.isinf(result))


class TestStudentTLogLikelihood:
    """Tests for the log-likelihood function."""

    def test_log_likelihood_against_scipy(self):
        """Compare log-likelihood against scipy.stats.t."""
        family = StudentTFamily(df=5.0)
        family._scale = 1.5
        y = np.array([-2.0, 0.0, 1.0, 3.0])
        mu = np.array([0.0, 0.5, 1.5, 2.5])

        result = family.log_likelihood(y, mu, scale=1.5)

        # scipy.stats.t with loc=mu, scale=scale
        expected = np.sum(t_dist.logpdf(y, df=5.0, loc=mu, scale=1.5))

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_likelihood_formula(self):
        """Verify log-likelihood formula directly."""
        family = StudentTFamily(df=3.0)
        y = np.array([1.0])
        mu = np.array([0.0])
        scale = 2.0
        df = 3.0

        result = family.log_likelihood(y, mu, scale=scale)

        # Manual calculation
        z = (y - mu) / scale
        log_const = (
            special.gammaln((df + 1) / 2)
            - special.gammaln(df / 2)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
        )
        log_kernel = -(df + 1) / 2 * np.log(1 + z**2 / df)
        expected = np.sum(log_const + log_kernel)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_likelihood_symmetric(self):
        """Log-likelihood should be symmetric around mean."""
        family = StudentTFamily(df=5.0)
        mu = np.array([0.0])
        y_pos = np.array([2.0])
        y_neg = np.array([-2.0])

        ll_pos = family.log_likelihood(y_pos, mu, scale=1.0)
        ll_neg = family.log_likelihood(y_neg, mu, scale=1.0)

        np.testing.assert_allclose(ll_pos, ll_neg)


class TestStudentTDeviance:
    """Tests for the deviance function."""

    def test_deviance_zero_at_saturated(self):
        """Deviance is zero when μ = y."""
        family = StudentTFamily(df=5.0)
        y = np.array([1.0, 2.0, -1.0])

        result = family.deviance(y, y, scale=1.0)

        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_deviance_positive_for_non_saturated(self):
        """Deviance is positive when μ ≠ y."""
        family = StudentTFamily(df=5.0)
        y = np.array([0.0, 2.0, -1.0])
        mu = np.array([0.5, 1.5, 0.0])

        result = family.deviance(y, mu, scale=1.0)

        assert result > 0

    def test_deviance_equals_minus_2_log_ratio(self):
        """Deviance = -2 × (ll_model - ll_saturated)."""
        family = StudentTFamily(df=5.0)
        y = np.array([1.0, 3.0, -2.0])
        mu = np.array([0.5, 2.5, -1.0])
        scale = 1.5

        ll_model = family.log_likelihood(y, mu, scale=scale)
        ll_sat = family.log_likelihood(y, y, scale=scale)
        dev = family.deviance(y, mu, scale=scale)

        expected = -2 * (ll_model - ll_sat)
        np.testing.assert_allclose(dev, expected, rtol=1e-10)


class TestStudentTInitialize:
    """Tests for initialization."""

    def test_initialize_uses_median(self):
        """Initialize uses median for robustness."""
        family = StudentTFamily()
        y = np.array([1.0, 2.0, 100.0])  # Outlier at 100

        result = family.initialize(y)

        # Median = 2.0, not mean ≈ 34.3
        expected = np.full_like(y, 2.0)
        np.testing.assert_allclose(result, expected)


class TestStudentTWeights:
    """Tests for IRLS weights (robust weighting)."""

    def test_weights_formula(self):
        """Test weight formula: w = (ν + 1) / (ν + z²)."""
        family = StudentTFamily(df=5.0)
        y = np.array([0.0, 1.0, 5.0])  # Last one is an outlier
        mu = np.array([0.0, 0.0, 0.0])
        scale = 1.0

        result = family.weights(y, mu, scale=scale)

        z = (y - mu) / scale
        expected = (5.0 + 1) / (5.0 + z**2)

        np.testing.assert_allclose(result, expected)

    def test_weights_downweight_outliers(self):
        """Weights should downweight outliers (large residuals)."""
        family = StudentTFamily(df=5.0)
        mu = np.array([0.0, 0.0, 0.0])
        y_normal = np.array([0.5])  # Normal observation
        y_outlier = np.array([10.0])  # Outlier

        w_normal = family.weights(y_normal, mu[:1], scale=1.0)
        w_outlier = family.weights(y_outlier, mu[:1], scale=1.0)

        # Outlier should have much smaller weight
        assert w_outlier < w_normal
        assert w_outlier < 0.5  # Should be substantially downweighted

    def test_weights_bounded(self):
        """Weights should be in (0, (df+1)/df] for standardized residuals.

        Maximum weight is at z=0: w_max = (df+1)/df
        For df=5: w_max = 6/5 = 1.2
        """
        family = StudentTFamily(df=5.0)
        y = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        mu = np.zeros_like(y)

        result = family.weights(y, mu, scale=1.0)

        # Weights are always positive
        assert np.all(result > 0)
        # Maximum weight is (df+1)/df when z=0
        max_weight = (5.0 + 1) / 5.0  # = 1.2
        assert np.all(result <= max_weight + 1e-10)


class TestStudentTGradients:
    """Tests for gradient functions."""

    def test_d_log_likelihood_direction(self):
        """Gradient points in correct direction."""
        family = StudentTFamily(df=5.0)
        y = np.array([3.0])

        # When mu < y, gradient should be positive
        mu_low = np.array([1.0])
        grad_low = family.d_log_likelihood(y, mu_low, scale=1.0)
        assert grad_low > 0

        # When mu > y, gradient should be negative
        mu_high = np.array([5.0])
        grad_high = family.d_log_likelihood(y, mu_high, scale=1.0)
        assert grad_high < 0

    def test_d_log_likelihood_formula(self):
        """Verify gradient formula: (ν+1)z / [σ(ν + z²)]."""
        family = StudentTFamily(df=5.0)
        y = np.array([2.0])
        mu = np.array([1.0])
        scale = 1.5
        df = 5.0

        result = family.d_log_likelihood(y, mu, scale=scale)

        z = (y - mu) / scale
        expected = (df + 1) * z / (scale * (df + z**2))

        np.testing.assert_allclose(result, expected)

    def test_d_log_likelihood_zero_at_y_equals_mu(self):
        """Gradient is zero when y = μ."""
        family = StudentTFamily(df=5.0)
        y = np.array([2.5])
        mu = np.array([2.5])

        result = family.d_log_likelihood(y, mu, scale=1.0)

        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_d2_log_likelihood_negative_at_mode(self):
        """Second derivative is negative at mode (concave)."""
        family = StudentTFamily(df=5.0)
        y = np.array([2.0])
        mu = y  # At the mode

        result = family.d2_log_likelihood(y, mu, scale=1.0)

        assert np.all(result < 0)

    def test_gradient_numerical_verification(self):
        """Verify gradient numerically via finite differences."""
        family = StudentTFamily(df=5.0)
        y = np.array([2.0])
        mu = np.array([1.5])
        scale = 1.0
        eps = 1e-6

        # Numerical gradient
        ll_plus = family.log_likelihood(y, mu + eps, scale=scale)
        ll_minus = family.log_likelihood(y, mu - eps, scale=scale)
        numerical_grad = (ll_plus - ll_minus) / (2 * eps)

        # Analytical gradient
        analytical_grad = family.d_log_likelihood(y, mu, scale=scale)[0]

        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-4)


class TestStudentTScaleEstimation:
    """Tests for scale parameter estimation."""

    def test_estimate_scale_positive(self):
        """Scale estimate should be positive."""
        family = StudentTFamily(df=5.0)
        y = np.array([1.0, 2.0, 3.0, 4.0, 10.0])  # With outlier
        mu = np.full_like(y, 3.0)

        result = family.estimate_scale(y, mu)

        assert result > 0

    def test_estimate_scale_robust_to_outliers(self):
        """MAD-based scale should be robust to outliers."""
        family = StudentTFamily(df=5.0)
        y_normal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_with_outlier = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # Extreme outlier
        mu = np.full(5, 3.0)

        scale_normal = family.estimate_scale(y_normal, mu)
        scale_outlier = family.estimate_scale(y_with_outlier, mu)

        # With MAD, the outlier should have limited impact
        # The ratio should not be extreme
        ratio = scale_outlier / scale_normal
        assert ratio < 5  # Should be reasonably similar


class TestCauchyFamily:
    """Tests for CauchyFamily (Student's t with df=1)."""

    def test_is_student_t_subclass(self):
        """CauchyFamily is a StudentTFamily subclass."""
        family = CauchyFamily()
        assert isinstance(family, StudentTFamily)

    def test_df_equals_one(self):
        """Cauchy has df=1."""
        family = CauchyFamily()
        assert family.df == 1.0

    def test_name(self):
        """Check the name attribute."""
        family = CauchyFamily()
        assert family.name == "cauchy"

    def test_variance_infinite(self):
        """Cauchy has infinite variance."""
        family = CauchyFamily()
        mu = np.array([0.0, 1.0, -1.0])

        result = family.variance(mu)

        assert np.all(np.isinf(result))

    def test_repr(self):
        """Test string representation."""
        family = CauchyFamily()
        repr_str = repr(family)
        assert "CauchyFamily" in repr_str
        assert "identity" in repr_str

    def test_weights_extreme_downweighting(self):
        """Cauchy weights should downweight outliers more aggressively."""
        cauchy = CauchyFamily()
        student_t5 = StudentTFamily(df=5.0)

        y_outlier = np.array([5.0])
        mu = np.array([0.0])
        scale = 1.0

        w_cauchy = cauchy.weights(y_outlier, mu, scale=scale)
        w_t5 = student_t5.weights(y_outlier, mu, scale=scale)

        # Cauchy should downweight more
        assert w_cauchy < w_t5


class TestStudentTEdgeCases:
    """Edge case tests."""

    def test_very_large_df_approaches_gaussian(self):
        """Large df should approach Gaussian behavior."""
        family_large_df = StudentTFamily(df=1000.0)
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([0.5, 1.5, 2.5])
        scale = 1.0

        # Log-likelihood should be close to Gaussian
        ll_t = family_large_df.log_likelihood(y, mu, scale=scale)

        # Gaussian log-likelihood (up to constants)
        from scipy.stats import norm

        ll_gaussian = np.sum(norm.logpdf(y, loc=mu, scale=scale))

        # Should be very close
        np.testing.assert_allclose(ll_t, ll_gaussian, rtol=0.01)

    def test_small_df_heavy_tails(self):
        """Small df should have much heavier tails."""
        family_df3 = StudentTFamily(df=3.0)
        family_df30 = StudentTFamily(df=30.0)

        # For an extreme outlier, df=3 should give higher likelihood
        y_outlier = np.array([10.0])
        mu = np.array([0.0])
        scale = 1.0

        ll_df3 = family_df3.log_likelihood(y_outlier, mu, scale=scale)
        ll_df30 = family_df30.log_likelihood(y_outlier, mu, scale=scale)

        # Heavy tails = higher probability for outliers
        assert ll_df3 > ll_df30

    def test_numerical_stability_large_residuals(self):
        """Test stability with very large residuals."""
        family = StudentTFamily(df=3.0)
        y = np.array([1000.0])
        mu = np.array([0.0])
        scale = 1.0

        ll = family.log_likelihood(y, mu, scale=scale)
        dev = family.deviance(y, mu, scale=scale)
        weights = family.weights(y, mu, scale=scale)

        assert np.isfinite(ll)
        assert np.isfinite(dev)
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0)


class TestStudentTVsGaussian:
    """Comparison tests between Student's t and Gaussian."""

    def test_weights_comparison(self):
        """t-distribution weights should be less than 1 for residuals > 0."""
        family = StudentTFamily(df=5.0)
        y = np.array([2.0])
        mu = np.array([0.0])

        weights = family.weights(y, mu, scale=1.0)

        # For any non-zero residual, weight < 1
        assert weights < 1.0

    def test_outlier_influence_reduced(self):
        """Outliers should have reduced influence in t regression."""
        family = StudentTFamily(df=5.0)
        mu = np.array([0.0])

        # Small residual
        w_small = family.weights(np.array([0.5]), mu, scale=1.0)

        # Large residual (outlier)
        w_large = family.weights(np.array([5.0]), mu, scale=1.0)

        # Outlier weight is much smaller
        assert w_large / w_small < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
