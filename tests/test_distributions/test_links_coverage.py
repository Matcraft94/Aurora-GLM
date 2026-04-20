"""Additional tests for link functions to improve coverage of common.py."""

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


class TestPowerLinkEdgeCases:
    """Cover PowerLink branches: power=0 derivative, power>0 inverse, etc."""

    def test_derivative_power_zero_is_reciprocal(self):
        """PowerLink(power=0) derivative should be 1/mu (same as log link)."""
        link = PowerLink(power=0.0)
        mu = np.array([0.5, 1.0, 2.0])
        deriv = link.derivative(mu)
        expected = 1.0 / mu
        np.testing.assert_allclose(deriv, expected, rtol=1e-10)

    def test_inverse_power_zero_is_exp(self):
        """PowerLink(power=0) inverse should be exp(eta)."""
        link = PowerLink(power=0.0)
        eta = np.array([0.0, 1.0, 2.0])
        mu = link.inverse(eta)
        np.testing.assert_allclose(mu, np.exp(eta), rtol=1e-10)

    def test_inverse_positive_power_clamps_eta(self):
        """For power > 0, inverse should ensure positive eta."""
        link = PowerLink(power=2.0)
        eta = np.array([1.0, 4.0, 9.0])
        mu = link.inverse(eta)
        expected = eta ** (1.0 / 2.0)
        np.testing.assert_allclose(mu, expected, rtol=1e-10)

    def test_derivative_nonzero_power(self):
        """PowerLink(power=2) derivative: 2 * mu^1."""
        link = PowerLink(power=2.0)
        mu = np.array([1.0, 2.0, 3.0])
        deriv = link.derivative(mu)
        expected = 2.0 * mu ** (2.0 - 1)
        np.testing.assert_allclose(deriv, expected, rtol=1e-10)

    def test_negative_power_inverse(self):
        """PowerLink(power=-1): inverse should be eta^(1/-1) = 1/eta."""
        link = PowerLink(power=-1.0)
        eta = np.array([0.5, 1.0, 2.0])
        mu = link.inverse(eta)
        expected = eta ** (1.0 / -1.0)
        np.testing.assert_allclose(mu, expected, rtol=1e-10)

    def test_power_link_name(self):
        link = PowerLink(power=0.5)
        assert link.name == "power0.5"
        assert link.power == 0.5


class TestSqrtLinkDerivative:
    """Cover SqrtLink.derivative."""

    def test_derivative_formula(self):
        """d(sqrt(mu))/dmu = 0.5 / sqrt(mu)."""
        link = SqrtLink()
        mu = np.array([1.0, 4.0, 9.0, 16.0])
        deriv = link.derivative(mu)
        expected = 0.5 / np.sqrt(mu)
        np.testing.assert_allclose(deriv, expected, rtol=1e-10)

    def test_inverse_squares(self):
        """SqrtLink.inverse(eta) = eta^2."""
        link = SqrtLink()
        eta = np.array([1.0, 2.0, 3.0])
        mu = link.inverse(eta)
        np.testing.assert_allclose(mu, eta**2, rtol=1e-10)


class TestCLogLogDerivative:
    """Additional CLogLogLink derivative coverage."""

    def test_derivative_formula(self):
        link = CLogLogLink()
        mu = np.array([0.2, 0.5, 0.8])
        deriv = link.derivative(mu)
        one_minus = 1.0 - mu
        log_term = -np.log(one_minus)
        expected = 1.0 / (log_term * one_minus)
        np.testing.assert_allclose(deriv, expected, rtol=1e-10)

    def test_inverse_formula(self):
        link = CLogLogLink()
        eta = np.array([0.0, 1.0, -1.0])
        mu = link.inverse(eta)
        expected = 1.0 - np.exp(-np.exp(eta))
        np.testing.assert_allclose(mu, expected, rtol=1e-10)


class TestInverseSquareDerivative:
    """Additional InverseSquareLink derivative coverage."""

    def test_derivative_negative(self):
        link = InverseSquareLink()
        mu = np.array([1.0, 2.0, 4.0])
        deriv = link.derivative(mu)
        assert np.all(deriv < 0), "InverseSquare derivative should be negative"

    def test_inverse_formula(self):
        link = InverseSquareLink()
        eta = np.array([1.0, 4.0, 9.0])
        mu = link.inverse(eta)
        expected = 1.0 / np.sqrt(eta)
        np.testing.assert_allclose(mu, expected, rtol=1e-10)


class TestLogLinkInverseClamping:
    """Cover the backend-specific clamping in LogLink.inverse."""

    def test_inverse_extreme_values(self):
        """Large eta values should be clamped to prevent overflow."""
        link = LogLink()
        eta = np.array([800.0, -800.0])
        mu = link.inverse(eta)
        assert np.all(np.isfinite(mu))


class TestInverseLinkClamping:
    """Cover InverseLink edge cases with small values."""

    def test_link_near_zero(self):
        link = InverseLink()
        mu = np.array([1e-10])
        eta = link.link(mu)
        assert np.isfinite(eta[0])

    def test_derivative_near_zero(self):
        link = InverseLink()
        mu = np.array([1e-10])
        deriv = link.derivative(mu)
        assert np.isfinite(deriv[0])


class TestIdentityLinkDerivative:
    """Cover IdentityLink.derivative (returns ones)."""

    def test_derivative_is_ones(self):
        link = IdentityLink()
        mu = np.array([1.0, 2.0, 3.0])
        deriv = link.derivative(mu)
        np.testing.assert_array_equal(deriv, [1.0, 1.0, 1.0])
