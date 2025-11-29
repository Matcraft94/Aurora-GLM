"""Unit tests for the Inverse Gaussian distribution family with multi-backend support."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.distributions.families.inverse_gaussian import InverseGaussianFamily, WaldFamily

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None  # type: ignore[assignment]


# ============================================================================
# Helper Functions
# ============================================================================

def _as_array(xp, data):
    """Convert data to array in specified namespace."""
    if xp is np:
        return np.asarray(data, dtype=float)
    if torch is not None and xp is torch:
        return torch.tensor(data, dtype=torch.float64)
    if jnp is not None and xp is jnp:
        return jnp.array(data, dtype=jnp.float64)
    raise TypeError("Unsupported namespace")


def _to_scalar(value, xp):
    """Convert array to Python scalar."""
    if xp is np:
        return float(value)
    if torch is not None and xp is torch:
        return value.item()
    if jnp is not None and xp is jnp:
        return float(value)
    raise TypeError("Unsupported namespace")


def _allclose(actual, expected, xp, rtol=1e-6, atol=1e-8):
    """Check if arrays are close with backend-specific comparison."""
    if xp is np:
        return np.allclose(actual, expected, rtol=rtol, atol=atol)
    if torch is not None and xp is torch:
        expected_tensor = torch.tensor(expected, dtype=actual.dtype, device=actual.device)
        return torch.allclose(actual, expected_tensor, rtol=rtol, atol=atol)
    if jnp is not None and xp is jnp:
        expected_array = jnp.array(expected, dtype=actual.dtype)
        return jnp.allclose(actual, expected_array, rtol=rtol, atol=atol)
    raise TypeError("Unsupported namespace")


def _torch_available():
    return torch is not None


def _jax_available():
    return jnp is not None


# Build list of available backends
AVAILABLE_BACKENDS = [np]
if _torch_available():
    AVAILABLE_BACKENDS.append(torch)
if _jax_available():
    AVAILABLE_BACKENDS.append(jnp)


# ============================================================================
# Test Functions - Basic Properties
# ============================================================================

class TestInverseGaussianFamilyBasicProperties:
    """Tests for basic Inverse Gaussian family properties."""

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_variance_function_mu_cubed(self, xp):
        """Test variance function: V(μ) = μ³."""
        family = InverseGaussianFamily(lambda_=2.0)
        mu = _as_array(xp, [1.0, 2.0, 3.0])

        result = family.variance(mu)

        # Expected: μ³
        expected = [1.0**3, 2.0**3, 3.0**3]

        assert _allclose(result, expected, xp, rtol=1e-10)

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_full_variance_formula(self, xp):
        """Test full variance: Var(Y) = μ³/λ."""
        lambda_ = 2.0
        family = InverseGaussianFamily(lambda_=lambda_)
        mu = _as_array(xp, [1.0, 2.0, 3.0])

        result = family.full_variance(mu)

        # Expected: μ³/λ
        expected = [1.0 / 2.0, 8.0 / 2.0, 27.0 / 2.0]

        assert _allclose(result, expected, xp, rtol=1e-10)

    def test_invalid_lambda_raises_error(self):
        """Test that non-positive lambda raises ValueError."""
        with pytest.raises(ValueError, match="lambda_ must be positive"):
            InverseGaussianFamily(lambda_=-1.0)

        with pytest.raises(ValueError, match="lambda_ must be positive"):
            InverseGaussianFamily(lambda_=0.0)

    def test_estimate_lambda_option(self):
        """Test that 'estimate' is valid for lambda."""
        family = InverseGaussianFamily(lambda_='estimate')
        assert family.lambda_ == 'estimate'

    def test_wald_alias(self):
        """Test that WaldFamily is an alias for InverseGaussianFamily."""
        assert WaldFamily is InverseGaussianFamily


# ============================================================================
# Test Functions - Log-Likelihood
# ============================================================================

class TestInverseGaussianFamilyLogLikelihood:
    """Tests for Inverse Gaussian family log-likelihood computation."""

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_log_likelihood_finite(self, xp):
        """Test log-likelihood is finite for valid inputs."""
        family = InverseGaussianFamily(lambda_=2.0)

        y = _as_array(xp, [0.5, 1.0, 2.0, 3.0])
        mu = _as_array(xp, [0.6, 1.2, 1.8, 2.5])

        ll = family.log_likelihood(y, mu)
        ll_scalar = _to_scalar(ll, xp)

        assert np.isfinite(ll_scalar), f"Log-likelihood not finite: {ll_scalar}"

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_log_likelihood_increases_near_mean(self, xp):
        """Test that log-likelihood is higher when y is closer to mu."""
        family = InverseGaussianFamily(lambda_=5.0)

        y = _as_array(xp, [1.0])
        mu_close = _as_array(xp, [1.0])
        mu_far = _as_array(xp, [2.0])

        ll_close = _to_scalar(family.log_likelihood(y, mu_close), xp)
        ll_far = _to_scalar(family.log_likelihood(y, mu_far), xp)

        assert ll_close > ll_far, f"Expected {ll_close} > {ll_far}"

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_log_likelihood_small_y_handling(self, xp):
        """Test that log-likelihood handles small y values."""
        family = InverseGaussianFamily(lambda_=3.0)

        # Small positive values
        y = _as_array(xp, [0.01, 0.1, 0.5])
        mu = _as_array(xp, [0.1, 0.2, 0.6])

        ll = family.log_likelihood(y, mu)
        ll_scalar = _to_scalar(ll, xp)

        assert np.isfinite(ll_scalar), f"Log-likelihood not finite for small y: {ll_scalar}"

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS) 
    def test_log_likelihood_known_values(self, xp):
        """Test log-likelihood against manually computed values."""
        # For inverse Gaussian with μ=1, λ=2, y=1:
        # f(y) = √(λ/(2πy³)) × exp(-λ(y-μ)²/(2μ²y))
        # f(1) = √(2/(2π)) × exp(0) = √(1/π) ≈ 0.564
        # log f(1) ≈ -0.572
        family = InverseGaussianFamily(lambda_=2.0)

        y = _as_array(xp, [1.0])
        mu = _as_array(xp, [1.0])

        ll = _to_scalar(family.log_likelihood(y, mu), xp)

        # Check against expected value
        expected = 0.5 * (np.log(2) - np.log(2 * np.pi) - 3 * np.log(1))  # - 0 for (y-μ)²
        expected_approx = 0.5 * np.log(2 / (2 * np.pi))

        assert abs(ll - expected_approx) < 0.01, f"Expected {expected_approx}, got {ll}"


# ============================================================================
# Test Functions - Deviance
# ============================================================================

class TestInverseGaussianFamilyDeviance:
    """Tests for Inverse Gaussian family deviance computation."""

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_deviance_zero_for_perfect_fit(self, xp):
        """Test deviance is zero when y = mu."""
        family = InverseGaussianFamily(lambda_=2.0)

        y = _as_array(xp, [1.0, 2.0, 3.0])
        mu = _as_array(xp, [1.0, 2.0, 3.0])

        dev = family.deviance(y, mu)
        dev_scalar = _to_scalar(dev, xp)

        assert abs(dev_scalar) < 1e-10, f"Deviance not zero for perfect fit: {dev_scalar}"

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_deviance_positive_for_imperfect_fit(self, xp):
        """Test deviance is positive when y ≠ mu."""
        family = InverseGaussianFamily(lambda_=2.0)

        y = _as_array(xp, [1.0, 2.0, 3.0])
        mu = _as_array(xp, [1.5, 2.5, 2.5])

        dev = family.deviance(y, mu)
        dev_scalar = _to_scalar(dev, xp)

        assert dev_scalar > 0, f"Deviance should be positive: {dev_scalar}"

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_deviance_formula(self, xp):
        """Test deviance formula: d(y,μ) = (y-μ)²/(μ²y)."""
        family = InverseGaussianFamily(lambda_=1.0)

        y = _as_array(xp, [2.0])
        mu = _as_array(xp, [1.0])

        dev = _to_scalar(family.deviance(y, mu), xp)

        # Expected: (2-1)²/(1²×2) = 1/2 = 0.5
        expected = 0.5

        assert abs(dev - expected) < 1e-10, f"Expected {expected}, got {dev}"


# ============================================================================
# Test Functions - Initialization
# ============================================================================

class TestInverseGaussianFamilyInitialization:
    """Tests for Inverse Gaussian family initialization."""

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_initialize_returns_positive_values(self, xp):
        """Test that initialize returns positive values."""
        family = InverseGaussianFamily(lambda_=2.0)

        y = _as_array(xp, [0.5, 1.0, 2.0, 5.0])

        mu_init = family.initialize(y)

        if xp is np:
            mu_np = mu_init
        elif torch is not None and xp is torch:
            mu_np = mu_init.cpu().numpy()
        else:
            mu_np = np.array(mu_init)

        assert np.all(mu_np > 0), f"Initialize has non-positive values: {mu_np}"

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_initialize_uses_sample_mean(self, xp):
        """Test that initialize returns array filled with sample mean."""
        family = InverseGaussianFamily(lambda_=1.0)

        y = _as_array(xp, [1.0, 2.0, 3.0, 4.0])

        mu_init = family.initialize(y)

        if xp is np:
            mu_np = mu_init
        elif torch is not None and xp is torch:
            mu_np = mu_init.cpu().numpy()
        else:
            mu_np = np.array(mu_init)

        # All values should be the mean (2.5)
        assert np.allclose(mu_np, 2.5, rtol=1e-10)


# ============================================================================
# Test Functions - Lambda Estimation
# ============================================================================

class TestInverseGaussianFamilyLambdaEstimation:
    """Tests for Inverse Gaussian family lambda estimation."""

    def test_estimate_lambda_from_data(self):
        """Test deviance-based lambda estimation."""
        # For IG(μ, λ), the ML estimate of λ is n / Σ[(y-μ)²/(μ²y)]
        np.random.seed(42)

        # Generate data from IG distribution (approximate)
        # True μ = 1, λ = 2
        mu_true = 1.0
        y = np.abs(np.random.normal(mu_true, 0.5, size=500))  # Approximate
        y = np.maximum(y, 0.01)  # Ensure positive

        family = InverseGaussianFamily(lambda_='estimate')

        # Estimate lambda
        lambda_est = family.estimate_lambda(y, np.full_like(y, mu_true))

        # Should be a reasonable positive value
        assert lambda_est > 0, f"Lambda estimate not positive: {lambda_est}"


# ============================================================================
# Test Functions - Multi-Backend Consistency
# ============================================================================

class TestInverseGaussianFamilyMultiBackend:
    """Tests for multi-backend consistency."""

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_matches_numpy_reference(self, xp):
        """Test that PyTorch/JAX match NumPy within tolerance."""
        if xp is np:
            pytest.skip("Reference backend")

        family = InverseGaussianFamily(lambda_=3.0)

        y_np = np.array([0.5, 1.0, 1.5, 2.0])
        mu_np = np.array([0.6, 1.1, 1.4, 2.1])

        y_xp = _as_array(xp, y_np)
        mu_xp = _as_array(xp, mu_np)

        # Log-likelihood
        ll_np = float(family.log_likelihood(y_np, mu_np))
        ll_xp = _to_scalar(family.log_likelihood(y_xp, mu_xp), xp)

        # Deviance
        dev_np = float(family.deviance(y_np, mu_np))
        dev_xp = _to_scalar(family.deviance(y_xp, mu_xp), xp)

        # Variance
        var_np = family.variance(mu_np)
        var_xp = family.variance(mu_xp)

        tol = 1e-5 if jnp is not None and xp is jnp else 1e-6

        assert abs(ll_np - ll_xp) < tol, f"Log-likelihood mismatch: {ll_np} vs {ll_xp}"
        assert abs(dev_np - dev_xp) < tol, f"Deviance mismatch: {dev_np} vs {dev_xp}"
        assert _allclose(var_xp, var_np, xp, rtol=tol)


# ============================================================================
# Test Functions - Default Link
# ============================================================================

class TestInverseGaussianFamilyDefaultLink:
    """Tests for default link function."""

    def test_default_link_is_inverse_square(self):
        """Test that default link is InverseSquare."""
        from aurora.distributions.links import InverseSquareLink

        family = InverseGaussianFamily()
        assert isinstance(family.default_link, InverseSquareLink)

    def test_custom_link_preserved(self):
        """Test that custom link is preserved."""
        from aurora.distributions.links import LogLink

        log = LogLink()
        family = InverseGaussianFamily(link=log)
        assert family.default_link is log

    def test_log_link_alternative(self):
        """Test that log link works as alternative."""
        from aurora.distributions.links import LogLink

        family = InverseGaussianFamily(lambda_=2.0, link=LogLink())

        mu = np.array([1.0, 2.0, 3.0])
        y = np.array([1.1, 2.1, 2.9])

        ll = family.log_likelihood(y, mu)
        assert np.isfinite(float(ll))


# ============================================================================
# Test Functions - Edge Cases
# ============================================================================

class TestInverseGaussianFamilyEdgeCases:
    """Tests for edge cases and numerical stability."""

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_extreme_lambda_values(self, xp):
        """Test with extreme lambda values."""
        mu = _as_array(xp, [1.0, 2.0])
        y = _as_array(xp, [1.1, 1.9])

        # Very high lambda (low variance)
        family_high = InverseGaussianFamily(lambda_=100.0)
        ll_high = family_high.log_likelihood(y, mu)
        assert np.isfinite(_to_scalar(ll_high, xp))

        # Low lambda (high variance)
        family_low = InverseGaussianFamily(lambda_=0.1)
        ll_low = family_low.log_likelihood(y, mu)
        assert np.isfinite(_to_scalar(ll_low, xp))

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_large_mu_values(self, xp):
        """Test with large mu values (variance grows as μ³)."""
        family = InverseGaussianFamily(lambda_=1.0)

        mu = _as_array(xp, [10.0, 100.0])
        y = _as_array(xp, [11.0, 95.0])

        ll = family.log_likelihood(y, mu)
        ll_scalar = _to_scalar(ll, xp)

        assert np.isfinite(ll_scalar), f"Log-likelihood not finite for large μ: {ll_scalar}"
