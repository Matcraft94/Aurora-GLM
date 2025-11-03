"""Tests for default link behaviour in distribution families."""
from __future__ import annotations

import pytest

from aurora.distributions.families import BinomialFamily, GaussianFamily, PoissonFamily
from aurora.distributions.links import IdentityLink, LogLink, LogitLink
from aurora.distributions.base import LinkFunction


class _ScaledIdentity(LinkFunction):
    """Minimal custom link used for testing injection."""

    def __init__(self, scale: float = 2.0) -> None:
        self._scale = scale

    def link(self, mu):  # noqa: ANN001 - testing stub
        return mu * self._scale

    def inverse(self, eta):  # noqa: ANN001 - testing stub
        return eta / self._scale

    def derivative(self, mu):  # noqa: ANN001 - testing stub
        return 1.0 * self._scale


@pytest.mark.parametrize(
    "family_cls, expected_link_type",
    [
        (GaussianFamily, IdentityLink),
        (PoissonFamily, LogLink),
        (BinomialFamily, LogitLink),
    ],
)
def test_default_link_types(family_cls, expected_link_type):
    family = family_cls()
    assert isinstance(family.default_link, expected_link_type)


@pytest.mark.parametrize("family_cls", [GaussianFamily, PoissonFamily, BinomialFamily])
def test_custom_link_is_respected(family_cls):
    custom_link = _ScaledIdentity()
    family = family_cls(link=custom_link)
    assert family.default_link is custom_link
