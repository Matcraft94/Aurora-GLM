"""Validation utilities for user inputs."""
from __future__ import annotations

from typing import Iterable

from ..exceptions import ConfigurationError


def ensure_positive(value: float, *, name: str) -> None:
    """Validate that a numeric value is strictly positive."""
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive; received {value}.")


def ensure_non_empty(sequence: Iterable[object], *, name: str) -> None:
    """Validate that an iterable contains at least one element."""
    if not any(True for _ in sequence):  # pragma: no branch - generator short-circuits on first element
        raise ConfigurationError(f"{name} cannot be empty.")


__all__ = ["ensure_positive", "ensure_non_empty"]
