"""Unit tests for the Binomial distribution family with multi-backend support."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.distributions.families.binomial import BinomialFamily

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
def test_binomial_log_likelihood(xp):
    """Test binomial log-likelihood computation across backends.

    Verifies that log-likelihood is computed correctly for binary outcomes
    and matches the reference NumPy implementation.
    """
    family = BinomialFamily(n=1.0)  # Binary case
    y = _as_array(xp, [0.0, 1.0, 1.0, 0.0])
    mu = _as_array(xp, [0.3, 0.7, 0.8, 0.2])

    result = family.log_likelihood(y, mu)

    # Reference calculation (manual)
    # log L = sum(y * log(p) + (1-y) * log(1-p))
    y_np = np.array([0.0, 1.0, 1.0, 0.0])
    p_np = np.array([0.3, 0.7, 0.8, 0.2])
    expected = float(np.sum(y_np * np.log(p_np) + (1 - y_np) * np.log(1 - p_np)))

    assert pytest.approx(expected, rel=1e-6) == _to_scalar(result, xp)


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_binomial_deviance(xp):
    """Test binomial deviance computation across backends.

    Deviance formula: D = 2 * sum(y * log(y/mu) + (n-y) * log((n-y)/(n-mu)))
    For binary case (n=1), this simplifies appropriately.
    """
    family = BinomialFamily(n=1.0)
    y = _as_array(xp, [0.0, 1.0, 1.0, 0.0])
    mu = _as_array(xp, [0.3, 0.7, 0.8, 0.2])

    result = family.deviance(y, mu)

    # Reference calculation
    y_np = np.array([0.0, 1.0, 1.0, 0.0])
    mu_np = np.array([0.3, 0.7, 0.8, 0.2])
    eps = 1e-12
    y_safe = np.clip(y_np, eps, 1 - eps)
    mu_safe = np.clip(mu_np, eps, 1 - eps)

    dev_terms = []
    for yi, yi_safe, mui_safe in zip(y_np, y_safe, mu_safe):
        term1 = yi * np.log(yi_safe / mui_safe)
        term2 = (1 - yi) * np.log((1 - yi_safe) / (1 - mui_safe))
        dev_terms.append(2.0 * (term1 + term2))

    expected = float(sum(dev_terms))

    assert pytest.approx(expected, rel=1e-5) == _to_scalar(result, xp)


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_binomial_variance(xp):
    """Test binomial variance function across backends.

    Variance formula: V(mu) = n * p * (1-p) where p = mu/n
    For binary case (n=1), this is mu * (1-mu).
    """
    family = BinomialFamily(n=1.0)
    mu = _as_array(xp, [0.2, 0.5, 0.8])

    result = family.variance(mu)

    # Expected: mu * (1 - mu)
    expected = [0.2 * 0.8, 0.5 * 0.5, 0.8 * 0.2]

    assert _allclose(result, expected, xp, rtol=1e-6)


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_binomial_initialize(xp):
    """Test binomial initialization across backends.

    Initialization uses (y + 0.5) / (n + 1) to avoid 0 and 1,
    then returns n * p_init.
    """
    family = BinomialFamily(n=1.0)
    y = _as_array(xp, [0.0, 1.0, 0.0, 1.0])

    init = family.initialize(y)

    # Expected: (y + 0.5) / (1 + 1) = (y + 0.5) / 2
    expected = [(0.0 + 0.5) / 2.0, (1.0 + 0.5) / 2.0, (0.0 + 0.5) / 2.0, (1.0 + 0.5) / 2.0]

    assert _allclose(init, expected, xp, rtol=1e-6)

    # Verify all values are strictly between 0 and 1 (for n=1)
    if xp is np:
        assert np.all(init > 0.0)
        assert np.all(init < 1.0)
    elif torch is not None and xp is torch:
        assert torch.all(init > 0.0)
        assert torch.all(init < 1.0)
    elif jnp is not None and xp is jnp:
        assert jnp.all(init > 0.0)
        assert jnp.all(init < 1.0)


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_binomial_boundary_cases(xp):
    """Test binomial family handles boundary cases (y=0, y=n) correctly.

    This test verifies that the epsilon handling in _safe_log() prevents
    numerical issues when probabilities are at the boundaries.
    """
    family = BinomialFamily(n=1.0)

    # Test with exact 0 and 1 values (boundary cases)
    y = _as_array(xp, [0.0, 1.0])
    mu = _as_array(xp, [0.0001, 0.9999])  # Near boundaries

    # Should not raise errors
    ll = family.log_likelihood(y, mu)
    dev = family.deviance(y, mu)
    var = family.variance(mu)

    # All should return finite values
    assert np.isfinite(_to_scalar(ll, xp))
    assert np.isfinite(_to_scalar(dev, xp))

    if xp is np:
        assert np.all(np.isfinite(var))
    elif torch is not None and xp is torch:
        assert torch.all(torch.isfinite(var))
    elif jnp is not None and xp is jnp:
        assert jnp.all(jnp.isfinite(var))


@pytest.mark.parametrize("xp", AVAILABLE_BACKENDS)
def test_binomial_epsilon_consistency(xp):
    """Test that epsilon handling is consistent across backends.

    This test verifies that our fix (using 1e-12 epsilon) produces
    numerically consistent results across NumPy, PyTorch, and JAX.
    """
    family = BinomialFamily(n=1.0)

    # Use same data for all backends
    y_data = [0.0, 1.0, 1.0, 0.0, 1.0]
    mu_data = [0.25, 0.75, 0.65, 0.35, 0.85]

    y = _as_array(xp, y_data)
    mu = _as_array(xp, mu_data)

    ll = _to_scalar(family.log_likelihood(y, mu), xp)
    dev = _to_scalar(family.deviance(y, mu), xp)

    # Reference values computed with NumPy
    y_ref = np.array(y_data)
    mu_ref = np.array(mu_data)
    ll_ref = float(family.log_likelihood(y_ref, mu_ref))
    dev_ref = float(family.deviance(y_ref, mu_ref))

    # Should match within reasonable tolerance
    # PyTorch may have slightly different numerical precision (~1e-7)
    # JAX uses float32 by default (without x64), needs more tolerance
    if xp is np:
        # NumPy vs NumPy should be exact
        assert pytest.approx(ll_ref, abs=1e-14) == ll
        assert pytest.approx(dev_ref, abs=1e-14) == dev
    elif torch is not None and xp is torch:
        # PyTorch may differ slightly due to different BLAS
        assert pytest.approx(ll_ref, abs=1e-6) == ll
        assert pytest.approx(dev_ref, abs=1e-6) == dev
    elif jnp is not None and xp is jnp:
        # JAX without x64 mode uses float32
        assert pytest.approx(ll_ref, abs=1e-5) == ll
        assert pytest.approx(dev_ref, abs=1e-5) == dev
