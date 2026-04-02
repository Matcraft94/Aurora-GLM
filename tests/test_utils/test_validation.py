"""Tests for validation utilities."""

from __future__ import annotations

import pytest

from aurora.utils.exceptions import ConfigurationError
from aurora.utils.validation import ensure_non_empty, ensure_positive


def test_ensure_positive_accepts_positive_values():
    """ensure_positive should accept strictly positive values."""
    ensure_positive(1.0, name="test_param")
    ensure_positive(0.001, name="test_param")
    ensure_positive(1000.0, name="test_param")
    # Should not raise


def test_ensure_positive_rejects_zero():
    """ensure_positive should reject zero."""
    with pytest.raises(ConfigurationError, match="test_param must be positive"):
        ensure_positive(0.0, name="test_param")


def test_ensure_positive_rejects_negative():
    """ensure_positive should reject negative values."""
    with pytest.raises(ConfigurationError, match="test_param must be positive"):
        ensure_positive(-1.0, name="test_param")

    with pytest.raises(ConfigurationError, match="alpha must be positive"):
        ensure_positive(-0.001, name="alpha")


def test_ensure_non_empty_accepts_non_empty_sequences():
    """ensure_non_empty should accept sequences with at least one element."""
    ensure_non_empty([1], name="test_list")
    ensure_non_empty([1, 2, 3], name="test_list")
    ensure_non_empty(range(5), name="test_range")
    ensure_non_empty((x for x in [1, 2]), name="test_generator")
    # Should not raise


def test_ensure_non_empty_rejects_empty_sequences():
    """ensure_non_empty should reject empty sequences."""
    with pytest.raises(ConfigurationError, match="test_list cannot be empty"):
        ensure_non_empty([], name="test_list")

    with pytest.raises(ConfigurationError, match="items cannot be empty"):
        ensure_non_empty(iter([]), name="items")

    with pytest.raises(ConfigurationError, match="data cannot be empty"):
        ensure_non_empty(range(0), name="data")
