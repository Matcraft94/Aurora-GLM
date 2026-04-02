"""Tests for TensorTerm specification."""

from __future__ import annotations

import pytest

from aurora.models.gam import TensorTerm


def test_tensor_term_basic():
    """TensorTerm should create with basic parameters."""
    term = TensorTerm(variables=(0, 1))

    assert term.variables == (0, 1)
    assert term.basis_types == ("bspline", "bspline")
    assert term.n_basis == (10, 10)
    assert term.lambdas is None


def test_tensor_term_named_variables():
    """TensorTerm should work with named variables."""
    term = TensorTerm(variables=("x1", "x2"))

    assert term.variables == ("x1", "x2")


def test_tensor_term_custom_n_basis():
    """TensorTerm should accept custom basis sizes."""
    term = TensorTerm(variables=(0, 1), n_basis=(12, 15))

    assert term.n_basis == (12, 15)


def test_tensor_term_custom_basis_types():
    """TensorTerm should accept different basis types."""
    term = TensorTerm(variables=(0, 1), basis_types=("bspline", "cubic"))

    assert term.basis_types == ("bspline", "cubic")


def test_tensor_term_with_lambdas():
    """TensorTerm should accept fixed smoothing parameters."""
    term = TensorTerm(variables=(0, 1), lambdas=(0.1, 0.5))

    assert term.lambdas == (0.1, 0.5)


def test_tensor_term_three_variables():
    """TensorTerm should work with 3+ variables."""
    term = TensorTerm(variables=(0, 1, 2), n_basis=(8, 10, 12))

    assert len(term.variables) == 3
    assert term.n_basis == (8, 10, 12)


def test_tensor_term_validation_min_vars():
    """TensorTerm should require at least 2 variables."""
    with pytest.raises(ValueError, match="at least 2 variables"):
        TensorTerm(variables=(0,))


def test_tensor_term_validation_basis_types_length():
    """TensorTerm should validate basis_types length."""
    with pytest.raises(ValueError, match="same length as variables"):
        TensorTerm(
            variables=(0, 1),
            basis_types=("bspline",),  # Too short
        )


def test_tensor_term_validation_n_basis_length():
    """TensorTerm should validate n_basis length."""
    with pytest.raises(ValueError, match="same length as variables"):
        TensorTerm(
            variables=(0, 1),
            n_basis=(10, 12, 8),  # Too long
        )


def test_tensor_term_validation_lambdas_length():
    """TensorTerm should validate lambdas length."""
    with pytest.raises(ValueError, match="same length as variables"):
        TensorTerm(
            variables=(0, 1),
            lambdas=(0.1,),  # Too short
        )


def test_tensor_term_validation_min_n_basis():
    """TensorTerm should validate minimum n_basis."""
    with pytest.raises(ValueError, match="at least 3"):
        TensorTerm(
            variables=(0, 1),
            n_basis=(10, 2),  # Second one too small
        )


def test_tensor_term_validation_negative_lambda():
    """TensorTerm should reject negative lambdas."""
    with pytest.raises(ValueError, match="non-negative"):
        TensorTerm(
            variables=(0, 1),
            lambdas=(0.1, -0.5),  # Second one negative
        )


def test_tensor_term_validation_unsupported_basis():
    """TensorTerm should reject unsupported basis types."""
    with pytest.raises(NotImplementedError, match="not supported"):
        TensorTerm(
            variables=(0, 1),
            basis_types=("bspline", "tp"),  # tp not yet supported
        )


def test_tensor_term_repr():
    """TensorTerm should have informative repr."""
    term = TensorTerm(variables=("x1", "x2"), n_basis=(12, 15))

    repr_str = repr(term)
    assert "TensorTerm" in repr_str
    assert "x1" in repr_str or "0" in repr_str
