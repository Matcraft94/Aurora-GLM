# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Validation utilities for user inputs.

This module provides:
- Simple validation functions (ensure_positive, ensure_non_empty)
- Decorators for function argument validation (@validate_array, etc.)

Usage
-----
>>> from aurora.utils.validation import ensure_positive, validate_array
>>>
>>> # Function-style validation
>>> ensure_positive(x, name='alpha')
>>>
>>> # Decorator-style validation
>>> @validate_array('X', ndim=2, dtype_check='numeric')
>>> @validate_positive('alpha')
>>> def fit(X, y, alpha=1.0):
...     pass
"""

from __future__ import annotations

from typing import Iterable

from ..exceptions import ConfigurationError

# Import decorators
from .decorators import (
    ValidationError,
    validate_array,
    validate_positive,
    validate_non_negative,
    validate_probability,
    validate_in_range,
    validate_type,
    validate_callable,
    validate_not_none,
    validate_one_of,
    validated,
)


def ensure_positive(value: float, *, name: str) -> None:
    """Validate that a numeric value is strictly positive."""
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive; received {value}.")


def ensure_non_empty(sequence: Iterable[object], *, name: str) -> None:
    """Validate that an iterable contains at least one element."""
    if not any(
        True for _ in sequence
    ):  # pragma: no branch - generator short-circuits on first element
        raise ConfigurationError(f"{name} cannot be empty.")


__all__ = [
    # Simple validation functions
    "ensure_positive",
    "ensure_non_empty",
    # Decorators
    "ValidationError",
    "validate_array",
    "validate_positive",
    "validate_non_negative",
    "validate_probability",
    "validate_in_range",
    "validate_type",
    "validate_callable",
    "validate_not_none",
    "validate_one_of",
    "validated",
]
