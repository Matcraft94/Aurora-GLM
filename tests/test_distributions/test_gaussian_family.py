"""Unit tests for the Gaussian distribution family with multi-backend support."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.distributions.families.gaussian import GaussianFamily

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
def test_gaussian_variance_is_constant(xp):
    """Test Gaussian variance is constant (does not depend on mu).

    For the Gaussian/Normal distribution, the variance is constant
    and does not depend on the mean parameter mu.
    """
    variance = 2.5
    family = GaussianFamily(variance=variance)

    # Different mu values
    mu = _as_array(xp, [0.0, 1.0, -1.0, 10.0, -10.0])

    result = family.variance(mu)

    # Expected: all values equal to variance
    expected = [variance] * 5

    assert _allclose(result, expected, xp, rtol=1e-10)


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_gaussian_log_likelihood_is_normal_pdf(xp):
    """Test Gaussian log-likelihood matches normal distribution formula.

    The Gaussian log-likelihood (ignoring constant terms) is:
    -0.5 * sum((y - mu)^2 / variance)

    This is proportional to the log of the normal probability density.
    """
    variance = 1.0
    family = GaussianFamily(variance=variance)

    y = _as_array(xp, [0.0, 1.0, 2.0, 3.0])
    mu = _as_array(xp, [0.1, 0.9, 2.1, 2.9])

    ll = family.log_likelihood(y, mu)
    ll_scalar = _to_scalar(ll, xp)

    # Reference calculation (without constant terms)
    y_np = np.array([0.0, 1.0, 2.0, 3.0])
    mu_np = np.array([0.1, 0.9, 2.1, 2.9])
    expected = float(np.sum(-0.5 * (y_np - mu_np)**2 / variance))

    assert pytest.approx(expected, abs=1e-10) == ll_scalar


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_gaussian_deviance_is_rss(xp):
    """Test Gaussian deviance is the residual sum of squares (scaled).

    Deviance for Gaussian family: sum((y - mu)^2) / variance

    This is equivalent to the residual sum of squares divided by variance.
    """
    variance = 1.0
    family = GaussianFamily(variance=variance)

    y = _as_array(xp, [1.0, 2.0, 3.0, 4.0])
    mu = _as_array(xp, [1.1, 1.9, 3.2, 3.8])

    dev = family.deviance(y, mu)
    dev_scalar = _to_scalar(dev, xp)

    # Expected: RSS / variance
    y_np = np.array([1.0, 2.0, 3.0, 4.0])
    mu_np = np.array([1.1, 1.9, 3.2, 3.8])
    expected = float(np.sum((y_np - mu_np)**2) / variance)

    assert pytest.approx(expected, abs=1e-10) == dev_scalar


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_gaussian_matches_numpy_reference(xp):
    """Test that PyTorch and JAX match NumPy within tolerance.

    This verifies numerical consistency across backends for the
    Gaussian family's log-likelihood and deviance computations.
    """
    if xp is np:
        pytest.skip("Reference backend, nothing to compare")

    variance = 1.5
    family_np = GaussianFamily(variance=variance)
    family_xp = GaussianFamily(variance=variance)

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

    # Should match at machine precision for Gaussian (no complex operations)
    assert abs(ll_np - ll_xp) < 1e-10, f"Log-likelihood mismatch: {ll_np} vs {ll_xp}"
    assert abs(dev_np - dev_xp) < 1e-10, f"Deviance mismatch: {dev_np} vs {dev_xp}"


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_gaussian_initialize_returns_mean(xp):
    """Test Gaussian initialization returns y itself.

    For Gaussian family, the natural starting value for mu is y
    (the observed values), since they are the best initial estimate
    of the mean.
    """
    family = GaussianFamily(variance=1.0)

    # Test data
    y = _as_array(xp, [-1.0, 0.0, 1.0, 2.0, 3.0])

    mu_init = family.initialize(y)

    # Should equal y
    assert _allclose(mu_init, y, xp, rtol=1e-12, atol=1e-12)
