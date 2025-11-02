"""Base protocols for probability distributions and link functions."""
from __future__ import annotations

from typing import Protocol

from ..core.types import ArrayLike


class Distribution(Protocol):
    """Protocol describing the interface for Aurora-GLM distribution families."""

    def log_likelihood(self, y: ArrayLike, mu: ArrayLike, **params) -> float:
        ...

    def deviance(self, y: ArrayLike, mu: ArrayLike, **params) -> float:
        ...

    def variance(self, mu: ArrayLike, **params) -> ArrayLike:
        ...

    def initialize(self, y: ArrayLike):
        ...


class LinkFunction(Protocol):
    """Protocol describing the interface for link functions."""

    def link(self, mu: ArrayLike) -> ArrayLike:
        ...

    def inverse(self, eta: ArrayLike) -> ArrayLike:
        ...

    def derivative(self, mu: ArrayLike) -> ArrayLike:
        ...


__all__ = ["Distribution", "LinkFunction"]
