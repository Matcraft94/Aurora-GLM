"""Unit tests for the Gamma distribution family with multi-backend support."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.distributions.families.gamma import GammaFamily

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
    """Check if PyTorch is available."""
    return torch is not None


def _jax_available():
    """Check if JAX is available."""
    return jnp is not None


# Build list of available backends
AVAILABLE_BACKENDS = [np]
if _torch_available():
    AVAILABLE_BACKENDS.append(torch)
if _jax_available():
    AVAILABLE_BACKENDS.append(jnp)


# ============================================================================
# Test Functions
# ============================================================================


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_gamma_variance_equals_mu_squared_over_shape(xp):
    """Test Gamma variance formula: Var(Y) = mu^2 / shape.

    For the Gamma distribution, the variance is mu^2 / shape,
    where shape is the shape parameter (often denoted alpha or k).
    """
    shape = 2.0
    family = GammaFamily(shape=shape)
    mu = _as_array(xp, [1.0, 2.0, 3.0, 4.0])

    result = family.variance(mu)

    # Expected: mu^2 / shape
    expected = [1.0**2 / 2.0, 2.0**2 / 2.0, 3.0**2 / 2.0, 4.0**2 / 2.0]

    assert _allclose(result, expected, xp, rtol=1e-10)


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_gamma_log_likelihood_finite_for_positive_y(xp):
    """Test Gamma log-likelihood is finite for positive y and mu.

    The Gamma log-likelihood requires both y > 0 and mu > 0.
    This test verifies numerical stability for valid inputs.
    """
    family = GammaFamily(shape=2.0)

    # All positive values
    y = _as_array(xp, [0.5, 1.0, 2.0, 3.0])
    mu = _as_array(xp, [0.6, 1.2, 1.8, 2.5])

    ll = family.log_likelihood(y, mu)
    ll_scalar = _to_scalar(ll, xp)

    # Should be finite
    assert np.isfinite(ll_scalar), f"Log-likelihood is not finite: {ll_scalar}"


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_gamma_deviance_zero_for_perfect_fit(xp):
    """Test Gamma deviance is zero when y = mu (perfect fit).

    Deviance formula: D = 2 * sum((y - mu)/mu - log(y/mu))
    When y = mu, this should be exactly 0.
    """
    family = GammaFamily(shape=1.0)

    # Perfect fit: y = mu
    y = _as_array(xp, [1.0, 2.0, 3.0, 4.0])
    mu = _as_array(xp, [1.0, 2.0, 3.0, 4.0])

    dev = family.deviance(y, mu)
    dev_scalar = _to_scalar(dev, xp)

    # Deviance should be ~0 for perfect fit
    assert abs(dev_scalar) < 1e-10, f"Deviance not zero for perfect fit: {dev_scalar}"


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_gamma_matches_numpy_reference(xp):
    """Test that PyTorch and JAX match NumPy within tolerance.

    This verifies numerical consistency across backends for the
    Gamma family's log-likelihood and deviance computations.
    """
    if xp is np:
        pytest.skip("Reference backend, nothing to compare")

    shape = 1.5
    family_np = GammaFamily(shape=shape)
    family_xp = GammaFamily(shape=shape)

    # Test data
    y_np = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    mu_np = np.array([0.6, 1.1, 1.4, 2.1, 2.3])

    y_xp = _as_array(xp, y_np)
    mu_xp = _as_array(xp, mu_np)

    # Log-likelihood
    ll_np = float(family_np.log_likelihood(y_np, mu_np))
    ll_xp = _to_scalar(family_xp.log_likelihood(y_xp, mu_xp), xp)

    # Deviance
    dev_np = float(family_np.deviance(y_np, mu_np))
    dev_xp = _to_scalar(family_xp.deviance(y_xp, mu_xp), xp)

    # Should match within reasonable tolerance
    # PyTorch may have slightly different precision (~1e-6)
    # JAX uses float32 by default, so needs more tolerance (~1e-5)
    if torch is not None and xp is torch:
        assert abs(ll_np - ll_xp) < 1e-6, f"Log-likelihood mismatch: {ll_np} vs {ll_xp}"
        assert abs(dev_np - dev_xp) < 1e-6, f"Deviance mismatch: {dev_np} vs {dev_xp}"
    elif jnp is not None and xp is jnp:
        # JAX without x64 mode uses float32
        assert abs(ll_np - ll_xp) < 1e-5, f"Log-likelihood mismatch: {ll_np} vs {ll_xp}"
        assert abs(dev_np - dev_xp) < 1e-5, f"Deviance mismatch: {dev_np} vs {dev_xp}"


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_gamma_initialize_returns_positive_values(xp):
    """Test Gamma initialization returns positive values.

    The Gamma distribution requires positive support, so the
    initialize method should return positive values even for
    edge cases.
    """
    family = GammaFamily(shape=1.0)

    # Test with various positive values
    y = _as_array(xp, [0.001, 0.1, 1.0, 10.0, 100.0])

    mu_init = family.initialize(y)

    # Convert to NumPy for checking
    if xp is np:
        mu_np = mu_init
    elif torch is not None and xp is torch:
        mu_np = mu_init.cpu().numpy()
    elif jnp is not None and xp is jnp:
        mu_np = np.array(mu_init)

    # All values should be positive
    assert np.all(mu_np > 0), f"Initialize returned non-positive values: {mu_np}"

    # Should be close to y (Gamma initializes to y itself)
    y_np = (
        np.array(y) if xp is jnp else (y.cpu().numpy() if torch is not None and xp is torch else y)
    )
    assert np.allclose(mu_np, y_np, rtol=1e-6)
