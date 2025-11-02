"""Abstract base classes for GLM distribution families and link functions."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import Array, Scalar


class LinkFunction(ABC):
    """Abstract base class for link functions."""

    @abstractmethod
    def link(self, mu: Array) -> Array:
        """Apply the link function ``g(mu)``."""

    @abstractmethod
    def inverse(self, eta: Array) -> Array:
        """Apply the inverse link ``g^{-1}(eta)``."""

    @abstractmethod
    def derivative(self, mu: Array) -> Array:
        """Return the derivative ``dg/dmu`` evaluated at ``mu``."""


class Family(ABC):
    """Abstract base class for probability distribution families."""

    @abstractmethod
    def log_likelihood(self, y: Array, mu: Array, **params) -> Scalar:
        """Return the log-likelihood of observations ``y`` given mean ``mu``."""

    @abstractmethod
    def deviance(self, y: Array, mu: Array, **params) -> Scalar:
        """Return the deviance contribution for observations ``y`` and mean ``mu``."""

    @abstractmethod
    def variance(self, mu: Array, **params) -> Array:
        """Return the variance function evaluated at ``mu``."""

    @abstractmethod
    def initialize(self, y: Array) -> Array:
        """Return starting values for the mean parameter ``mu`` given data ``y``."""

    @property
    @abstractmethod
    def default_link(self) -> LinkFunction:
        """Return the canonical link for this family."""


__all__ = ["Family", "LinkFunction"]
