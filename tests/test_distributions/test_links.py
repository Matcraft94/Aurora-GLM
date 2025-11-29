"""Unit tests for standard link functions."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.distributions.links import (
    CLogLogLink,
    IdentityLink,
    InverseLink,
    InverseSquareLink,
    LogLink,
    LogitLink,
    PowerLink,
    ProbitLink,
    SqrtLink,
)

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def _namespaces():
    libs = [np]
    if torch is not None:
        libs.append(torch)
    return tuple(libs)


def _as_array(xp, data):
    if xp is np:
        return np.asarray(data, dtype=float)
    if torch is not None and xp is torch:
        return torch.tensor(data, dtype=torch.float64)
    raise TypeError("Unsupported namespace")


def _allclose(actual, expected, xp, atol=1e-8, rtol=1e-5):
    if xp is np:
        return np.allclose(actual, expected, atol=atol, rtol=rtol)
    if torch is not None and xp is torch:
        expected_tensor = torch.tensor(expected, dtype=actual.dtype)
        return torch.allclose(actual, expected_tensor, atol=atol, rtol=rtol)
    raise TypeError("Unsupported namespace")


@pytest.mark.parametrize("xp", _namespaces())
def test_identity_link_roundtrip(xp):
    link = IdentityLink()
    mu = _as_array(xp, [0.2, -1.5, 3.7])
    eta = link.link(mu)
    mu_back = link.inverse(eta)
    assert _allclose(mu_back, [0.2, -1.5, 3.7], xp)
    deriv = link.derivative(mu)
    expected = np.ones(3)
    assert _allclose(deriv, expected, xp)


@pytest.mark.parametrize("xp", _namespaces())
def test_log_link_behaviour(xp):
    link = LogLink()
    mu_values = [0.5, 1.2, 3.4]
    mu = _as_array(xp, mu_values)
    eta = link.link(mu)
    expected_eta = np.log(mu_values)
    assert _allclose(eta, expected_eta, xp)
    mu_back = link.inverse(eta)
    assert _allclose(mu_back, mu_values, xp)
    deriv = link.derivative(mu)
    expected_deriv = [1.0 / v for v in mu_values]
    assert _allclose(deriv, expected_deriv, xp)


@pytest.mark.parametrize("xp", _namespaces())
def test_logit_link_behaviour(xp):
    link = LogitLink()
    mu_values = [0.2, 0.4, 0.8]
    mu = _as_array(xp, mu_values)
    eta = link.link(mu)
    expected_eta = np.log(np.asarray(mu_values) / (1.0 - np.asarray(mu_values)))
    assert _allclose(eta, expected_eta, xp)
    mu_back = link.inverse(eta)
    assert _allclose(mu_back, mu_values, xp)
    deriv = link.derivative(mu)
    expected_deriv = [1.0 / (v * (1.0 - v)) for v in mu_values]
    assert _allclose(deriv, expected_deriv, xp)


@pytest.mark.parametrize("xp", _namespaces())
def test_logit_link_clips_extremes(xp):
    link = LogitLink()
    mu = _as_array(xp, [0.0, 1.0])
    eta = link.link(mu)
    if xp is np:
        assert np.all(np.isfinite(eta))
    else:
        assert torch.all(torch.isfinite(eta))


@pytest.mark.parametrize("xp", _namespaces())
def test_inverse_link_behaviour(xp):
    link = InverseLink()
    mu_values = [0.5, 2.0, 4.5]
    mu = _as_array(xp, mu_values)
    eta = link.link(mu)
    expected_eta = 1.0 / np.asarray(mu_values)
    assert _allclose(eta, expected_eta, xp)
    mu_back = link.inverse(eta)
    assert _allclose(mu_back, mu_values, xp)
    deriv = link.derivative(mu)
    expected_deriv = [-1.0 / (v**2) for v in mu_values]
    assert _allclose(deriv, expected_deriv, xp)


@pytest.mark.parametrize("xp", _namespaces())
def test_inverse_link_clips_small_mu(xp):
    link = InverseLink()
    mu = _as_array(xp, [0.0, 1e-9])
    eta = link.link(mu)
    if xp is np:
        assert np.all(np.isfinite(eta))
    else:
        assert torch.all(torch.isfinite(eta))


@pytest.mark.parametrize("xp", _namespaces())
def test_cloglog_link_behaviour(xp):
    link = CLogLogLink()
    mu_values = [0.2, 0.6, 0.85]
    mu = _as_array(xp, mu_values)
    eta = link.link(mu)
    expected_eta = np.log(-np.log(1.0 - np.asarray(mu_values)))
    assert _allclose(eta, expected_eta, xp)
    mu_back = link.inverse(eta)
    assert _allclose(mu_back, mu_values, xp)
    deriv = link.derivative(mu)
    log_term = -np.log(1.0 - np.asarray(mu_values))
    expected_deriv = 1.0 / (log_term * (1.0 - np.asarray(mu_values)))
    assert _allclose(deriv, expected_deriv, xp)


@pytest.mark.parametrize("xp", _namespaces())
def test_cloglog_link_clips_extremes(xp):
    link = CLogLogLink()
    mu = _as_array(xp, [0.0, 1.0])
    eta = link.link(mu)
    if xp is np:
        assert np.all(np.isfinite(eta))
    else:
        assert torch.all(torch.isfinite(eta))


# ============================================================================
# ProbitLink Tests
# ============================================================================

@pytest.mark.parametrize("xp", _namespaces())
def test_probit_link_roundtrip(xp):
    """Test that inverse(link(mu)) ≈ mu for Probit link."""
    link = ProbitLink()
    mu_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    mu = _as_array(xp, mu_values)
    
    eta = link.link(mu)
    mu_back = link.inverse(eta)
    
    assert _allclose(mu_back, mu_values, xp, rtol=1e-6)


@pytest.mark.parametrize("xp", _namespaces())
def test_probit_link_known_values(xp):
    """Test Probit link against known values.
    
    For standard normal:
    - Φ^{-1}(0.5) = 0
    - Φ^{-1}(0.84134) ≈ 1
    - Φ^{-1}(0.15866) ≈ -1
    """
    from scipy.stats import norm
    
    link = ProbitLink()
    mu = _as_array(xp, [0.5, 0.8413447, 0.1586553])
    
    eta = link.link(mu)
    expected_eta = [0.0, 1.0, -1.0]
    
    assert _allclose(eta, expected_eta, xp, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("xp", _namespaces())
def test_probit_inverse_known_values(xp):
    """Test Probit inverse against known values.
    
    For standard normal:
    - Φ(0) = 0.5
    - Φ(1) ≈ 0.8413
    - Φ(-1) ≈ 0.1587
    """
    link = ProbitLink()
    eta = _as_array(xp, [0.0, 1.0, -1.0])
    
    mu = link.inverse(eta)
    expected_mu = [0.5, 0.8413447, 0.1586553]
    
    assert _allclose(mu, expected_mu, xp, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("xp", _namespaces())
def test_probit_link_derivative_positive(xp):
    """Test that Probit derivative is always positive."""
    link = ProbitLink()
    mu = _as_array(xp, [0.1, 0.3, 0.5, 0.7, 0.9])
    
    deriv = link.derivative(mu)
    
    if xp is np:
        assert np.all(deriv > 0)
    else:
        assert torch.all(deriv > 0)


@pytest.mark.parametrize("xp", _namespaces())
def test_probit_link_derivative_formula(xp):
    """Test Probit derivative: dη/dμ = 1/φ(Φ^{-1}(μ))."""
    from scipy.stats import norm
    
    link = ProbitLink()
    mu_values = [0.2, 0.5, 0.8]
    mu = _as_array(xp, mu_values)
    
    deriv = link.derivative(mu)
    
    # Expected: 1/φ(Φ^{-1}(μ))
    z = norm.ppf(mu_values)
    expected_deriv = 1.0 / norm.pdf(z)
    
    assert _allclose(deriv, expected_deriv, xp, rtol=1e-5)


@pytest.mark.parametrize("xp", _namespaces())
def test_probit_link_clips_extremes(xp):
    """Test that Probit handles extreme values (near 0 and 1)."""
    link = ProbitLink()
    mu = _as_array(xp, [0.001, 0.999])
    
    eta = link.link(mu)
    
    if xp is np:
        assert np.all(np.isfinite(eta))
    else:
        assert torch.all(torch.isfinite(eta))


@pytest.mark.parametrize("xp", _namespaces())
def test_probit_vs_logit_similarity(xp):
    """Test that Probit and Logit are similar in the middle range.
    
    For μ ∈ [0.2, 0.8], Probit ≈ (π/√3) × Logit at μ ≠ 0.5
    At μ = 0.5, both are 0, so we exclude it from ratio comparison.
    """
    probit = ProbitLink()
    logit = LogitLink()
    
    # Exclude 0.5 where both are 0 (causes division issues)
    mu = _as_array(xp, [0.3, 0.4, 0.6, 0.7])
    
    eta_probit = probit.link(mu)
    eta_logit = logit.link(mu)
    
    # Scaling factor: π/√3 ≈ 1.814
    scale = np.pi / np.sqrt(3)
    
    scaled_logit = eta_logit / scale
    
    # Should be approximately equal (within 15% - approximation is rough)
    if xp is np:
        ratio = eta_probit / scaled_logit
        assert np.all(np.abs(ratio - 1) < 0.15), f"Ratio too far from 1: {ratio}"
    else:
        ratio = eta_probit / scaled_logit
        assert torch.all(torch.abs(ratio - 1) < 0.15), f"Ratio too far from 1: {ratio}"


# ============================================================================
# InverseSquareLink Tests
# ============================================================================

@pytest.mark.parametrize("xp", _namespaces())
def test_inverse_square_link_roundtrip(xp):
    """Test that inverse(link(mu)) ≈ mu for InverseSquare link."""
    link = InverseSquareLink()
    mu_values = [0.5, 1.0, 2.0, 4.0]
    mu = _as_array(xp, mu_values)
    
    eta = link.link(mu)
    mu_back = link.inverse(eta)
    
    assert _allclose(mu_back, mu_values, xp, rtol=1e-6)


@pytest.mark.parametrize("xp", _namespaces())
def test_inverse_square_link_formula(xp):
    """Test InverseSquare link: η = 1/μ²."""
    link = InverseSquareLink()
    mu_values = [0.5, 1.0, 2.0]
    mu = _as_array(xp, mu_values)
    
    eta = link.link(mu)
    expected_eta = [4.0, 1.0, 0.25]
    
    assert _allclose(eta, expected_eta, xp)


@pytest.mark.parametrize("xp", _namespaces())
def test_inverse_square_link_derivative(xp):
    """Test InverseSquare derivative: dη/dμ = -2/μ³."""
    link = InverseSquareLink()
    mu_values = [0.5, 1.0, 2.0]
    mu = _as_array(xp, mu_values)
    
    deriv = link.derivative(mu)
    expected_deriv = [-2.0 / (v**3) for v in mu_values]
    
    assert _allclose(deriv, expected_deriv, xp)


# ============================================================================
# SqrtLink Tests
# ============================================================================

@pytest.mark.parametrize("xp", _namespaces())
def test_sqrt_link_roundtrip(xp):
    """Test that inverse(link(mu)) ≈ mu for Sqrt link."""
    link = SqrtLink()
    mu_values = [0.25, 1.0, 4.0, 9.0]
    mu = _as_array(xp, mu_values)
    
    eta = link.link(mu)
    mu_back = link.inverse(eta)
    
    assert _allclose(mu_back, mu_values, xp)


@pytest.mark.parametrize("xp", _namespaces())
def test_sqrt_link_formula(xp):
    """Test Sqrt link: η = √μ."""
    link = SqrtLink()
    mu_values = [1.0, 4.0, 9.0, 16.0]
    mu = _as_array(xp, mu_values)
    
    eta = link.link(mu)
    expected_eta = [1.0, 2.0, 3.0, 4.0]
    
    assert _allclose(eta, expected_eta, xp)


# ============================================================================
# PowerLink Tests
# ============================================================================

@pytest.mark.parametrize("xp", _namespaces())
def test_power_link_roundtrip(xp):
    """Test PowerLink roundtrip for various powers."""
    for power in [0.5, 1.0, 2.0, -1.0]:
        link = PowerLink(power=power)
        mu_values = [0.5, 1.0, 2.0]
        mu = _as_array(xp, mu_values)
        
        eta = link.link(mu)
        mu_back = link.inverse(eta)
        
        assert _allclose(mu_back, mu_values, xp, rtol=1e-5)


@pytest.mark.parametrize("xp", _namespaces())
def test_power_link_zero_power_is_log(xp):
    """Test that power=0 acts like log link."""
    power_link = PowerLink(power=0.0)
    log_link = LogLink()
    
    mu = _as_array(xp, [0.5, 1.0, 2.0])
    
    eta_power = power_link.link(mu)
    eta_log = log_link.link(mu)
    
    assert _allclose(eta_power, eta_log, xp, rtol=1e-5)
