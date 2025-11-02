"""Unit tests for standard link functions."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.distributions.links import CLogLogLink, IdentityLink, InverseLink, LogLink, LogitLink

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
