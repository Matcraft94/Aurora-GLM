"""Unit tests for the Beta distribution family with multi-backend support."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.distributions.families.beta import BetaFamily

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


def _allclose(actual, expected, xp, rtol=1e-5, atol=1e-7):
    """Check if arrays are close with backend-specific comparison."""
    if xp is np:
        return np.allclose(actual, expected, rtol=rtol, atol=atol)
    if torch is not None and xp is torch:
        expected_tensor = torch.tensor(expected, dtype=actual.dtype, device=actual.device)
        return torch.allclose(actual, expected_tensor, rtol=rtol, atol=atol)
    if jnp is not None and xp is jnp:
        # JAX uses float32 by default, need looser tolerance
        expected_array = jnp.array(expected, dtype=actual.dtype)
        return jnp.allclose(actual, expected_array, rtol=1e-4, atol=1e-5)
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

class TestBetaFamilyBasicProperties:
    """Tests for basic Beta family properties."""

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_variance_formula(self, xp):
        """Test variance function: V(μ) = μ(1-μ)."""
        family = BetaFamily(phi=5.0)
        mu = _as_array(xp, [0.2, 0.5, 0.8])

        result = family.variance(mu)

        # Expected: μ(1-μ)
        expected = [0.2 * 0.8, 0.5 * 0.5, 0.8 * 0.2]

        assert _allclose(result, expected, xp, rtol=1e-10)

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_full_variance_formula(self, xp):
        """Test full variance: Var(Y) = μ(1-μ)/(φ+1)."""
        phi = 4.0
        family = BetaFamily(phi=phi)
        mu = _as_array(xp, [0.3, 0.5, 0.7])

        result = family.full_variance(mu)

        # Expected: μ(1-μ)/(φ+1)
        expected = [
            0.3 * 0.7 / 5.0,
            0.5 * 0.5 / 5.0,
            0.7 * 0.3 / 5.0
        ]

        assert _allclose(result, expected, xp, rtol=1e-10)

    def test_invalid_phi_raises_error(self):
        """Test that non-positive phi raises ValueError."""
        with pytest.raises(ValueError, match="phi must be positive"):
            BetaFamily(phi=-1.0)

        with pytest.raises(ValueError, match="phi must be positive"):
            BetaFamily(phi=0.0)

    def test_estimate_phi_option(self):
        """Test that 'estimate' is valid for phi."""
        family = BetaFamily(phi='estimate')
        assert family.phi == 'estimate'

    def test_invalid_phi_string_raises_error(self):
        """Test that invalid string for phi raises ValueError."""
        with pytest.raises(ValueError, match="phi must be a positive float or 'estimate'"):
            BetaFamily(phi='invalid')


# ============================================================================
# Test Functions - Log-Likelihood
# ============================================================================

class TestBetaFamilyLogLikelihood:
    """Tests for Beta family log-likelihood computation."""

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_log_likelihood_finite(self, xp):
        """Test log-likelihood is finite for valid inputs."""
        family = BetaFamily(phi=2.0)

        y = _as_array(xp, [0.2, 0.5, 0.8])
        mu = _as_array(xp, [0.3, 0.5, 0.7])

        ll = family.log_likelihood(y, mu)
        ll_scalar = _to_scalar(ll, xp)

        assert np.isfinite(ll_scalar), f"Log-likelihood not finite: {ll_scalar}"

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_log_likelihood_increases_with_concentration(self, xp):
        """Test that log-likelihood is higher when y is closer to mu."""
        family = BetaFamily(phi=5.0)

        y = _as_array(xp, [0.5])
        mu_close = _as_array(xp, [0.5])
        mu_far = _as_array(xp, [0.2])

        ll_close = _to_scalar(family.log_likelihood(y, mu_close), xp)
        ll_far = _to_scalar(family.log_likelihood(y, mu_far), xp)

        assert ll_close > ll_far, f"Expected {ll_close} > {ll_far}"

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_log_likelihood_boundary_handling(self, xp):
        """Test that log-likelihood handles near-boundary values."""
        family = BetaFamily(phi=3.0)

        # Near-boundary values
        y = _as_array(xp, [0.01, 0.99])
        mu = _as_array(xp, [0.1, 0.9])

        ll = family.log_likelihood(y, mu)
        ll_scalar = _to_scalar(ll, xp)

        assert np.isfinite(ll_scalar), f"Log-likelihood not finite at boundary: {ll_scalar}"


# ============================================================================
# Test Functions - Deviance
# ============================================================================

class TestBetaFamilyDeviance:
    """Tests for Beta family deviance computation."""

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_deviance_zero_for_perfect_fit(self, xp):
        """Test deviance is zero when y = mu."""
        family = BetaFamily(phi=2.0)

        y = _as_array(xp, [0.2, 0.5, 0.8])
        mu = _as_array(xp, [0.2, 0.5, 0.8])

        dev = family.deviance(y, mu)
        dev_scalar = _to_scalar(dev, xp)

        # JAX float32 has lower precision, so use relaxed tolerance
        assert abs(dev_scalar) < 1e-6, f"Deviance not zero for perfect fit: {dev_scalar}"

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_deviance_positive_for_imperfect_fit(self, xp):
        """Test deviance is positive when y ≠ mu."""
        family = BetaFamily(phi=2.0)

        # Use values that are clearly different
        y = _as_array(xp, [0.25, 0.50, 0.75])
        mu = _as_array(xp, [0.35, 0.45, 0.65])

        dev = family.deviance(y, mu)
        dev_scalar = _to_scalar(dev, xp)

        # Deviance should be strictly positive when y != mu
        assert dev_scalar > 0, f"Deviance should be positive: {dev_scalar}"


# ============================================================================
# Test Functions - Initialization
# ============================================================================

class TestBetaFamilyInitialization:
    """Tests for Beta family initialization."""

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_initialize_returns_valid_probabilities(self, xp):
        """Test that initialize returns values in (0, 1)."""
        family = BetaFamily(phi=2.0)

        y = _as_array(xp, [0.1, 0.3, 0.5, 0.7, 0.9])

        mu_init = family.initialize(y)

        # Convert to numpy for checking
        if xp is np:
            mu_np = mu_init
        elif torch is not None and xp is torch:
            mu_np = mu_init.cpu().numpy()
        elif jnp is not None and xp is jnp:
            mu_np = np.array(mu_init)

        # All values should be in (0, 1)
        assert np.all(mu_np > 0), f"Initialize has values ≤ 0: {mu_np}"
        assert np.all(mu_np < 1), f"Initialize has values ≥ 1: {mu_np}"

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_initialize_clamps_boundary_values(self, xp):
        """Test that boundary values are properly clamped."""
        family = BetaFamily(phi=2.0)

        # Include values that would be at boundaries
        y = _as_array(xp, [0.001, 0.5, 0.999])

        mu_init = family.initialize(y)

        if xp is np:
            mu_np = mu_init
        elif torch is not None and xp is torch:
            mu_np = mu_init.cpu().numpy()
        else:
            mu_np = np.array(mu_init)

        assert np.all(mu_np >= 0.01), "Values not clamped away from 0"
        assert np.all(mu_np <= 0.99), "Values not clamped away from 1"


# ============================================================================
# Test Functions - Phi Estimation
# ============================================================================

class TestBetaFamilyPhiEstimation:
    """Tests for Beta family phi estimation."""

    def test_estimate_phi_from_data(self):
        """Test method-of-moments phi estimation."""
        # Generate beta data with known parameters
        np.random.seed(42)
        true_alpha, true_beta = 3.0, 7.0  # True mean = 0.3
        y = np.random.beta(true_alpha, true_beta, size=500)

        family = BetaFamily(phi='estimate')

        # Estimate phi
        phi_est = family.estimate_phi(y)

        # True phi = alpha + beta = 10
        assert phi_est > 5 and phi_est < 20, f"Phi estimate {phi_est} too far from true value 10"

    def test_auto_phi_estimation_in_log_likelihood(self):
        """Test that phi is estimated when computing log-likelihood with 'estimate'."""
        np.random.seed(123)
        y = np.random.beta(2, 8, size=100)  # Mean ≈ 0.2

        family = BetaFamily(phi='estimate')
        mu = np.full_like(y, 0.2)

        # Should not raise error even though phi='estimate'
        ll = family.log_likelihood(y, mu)
        assert np.isfinite(float(ll))


# ============================================================================
# Test Functions - Multi-Backend Consistency
# ============================================================================

class TestBetaFamilyMultiBackend:
    """Tests for multi-backend consistency."""

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_matches_numpy_reference(self, xp):
        """Test that PyTorch/JAX match NumPy within tolerance."""
        if xp is np:
            pytest.skip("Reference backend")

        family = BetaFamily(phi=3.0)

        y_np = np.array([0.2, 0.4, 0.6, 0.8])
        mu_np = np.array([0.25, 0.45, 0.55, 0.75])

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

        # Check tolerances
        tol = 1e-5 if jnp is not None and xp is jnp else 1e-6

        assert abs(ll_np - ll_xp) < tol, f"Log-likelihood mismatch: {ll_np} vs {ll_xp}"
        assert abs(dev_np - dev_xp) < tol, f"Deviance mismatch: {dev_np} vs {dev_xp}"
        assert _allclose(var_xp, var_np, xp, rtol=tol)


# ============================================================================
# Test Functions - Edge Cases
# ============================================================================

class TestBetaFamilyEdgeCases:
    """Tests for edge cases and numerical stability."""

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_extreme_phi_values(self, xp):
        """Test with extreme phi values (very concentrated/dispersed)."""
        mu = _as_array(xp, [0.3, 0.5, 0.7])
        y = _as_array(xp, [0.35, 0.55, 0.65])

        # Very concentrated (high phi)
        family_high = BetaFamily(phi=100.0)
        ll_high = family_high.log_likelihood(y, mu)
        assert np.isfinite(_to_scalar(ll_high, xp))

        # Dispersed (low phi)
        family_low = BetaFamily(phi=0.5)
        ll_low = family_low.log_likelihood(y, mu)
        assert np.isfinite(_to_scalar(ll_low, xp))

    @pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
    def test_symmetric_around_half(self, xp):
        """Test symmetry property: Beta with μ=0.5 should be symmetric."""
        family = BetaFamily(phi=4.0)

        # Symmetric points around 0.5
        y_low = _as_array(xp, [0.3])
        y_high = _as_array(xp, [0.7])
        mu = _as_array(xp, [0.5])

        ll_low = _to_scalar(family.log_likelihood(y_low, mu), xp)
        ll_high = _to_scalar(family.log_likelihood(y_high, mu), xp)

        # Should be equal by symmetry (use looser tolerance for JAX float32)
        tol = 1e-4 if jnp is not None and xp is jnp else 1e-10
        assert abs(ll_low - ll_high) < tol, f"Not symmetric: {ll_low} vs {ll_high}"


# ============================================================================
# Test Functions - Default Link
# ============================================================================

class TestBetaFamilyDefaultLink:
    """Tests for default link function."""

    def test_default_link_is_logit(self):
        """Test that default link is Logit."""
        from aurora.distributions.links import LogitLink

        family = BetaFamily()
        assert isinstance(family.default_link, LogitLink)

    def test_custom_link_preserved(self):
        """Test that custom link is preserved."""
        from aurora.distributions.links import ProbitLink

        probit = ProbitLink()
        family = BetaFamily(link=probit)
        assert family.default_link is probit
