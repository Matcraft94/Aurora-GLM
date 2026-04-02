"""Tests for R-style formula parsing."""

from __future__ import annotations

import pytest

from aurora.models.gam import parse_formula


def test_parse_formula_simple():
    """parse_formula should handle simple formula."""
    spec = parse_formula("y ~ s(x1) + s(x2)")

    assert spec.response == "y"
    assert len(spec.smooth_terms) == 2
    assert len(spec.parametric_terms) == 0

    assert spec.smooth_terms[0].variable == "x1"
    assert spec.smooth_terms[1].variable == "x2"


def test_parse_formula_with_indices():
    """parse_formula should handle integer column indices."""
    spec = parse_formula("y ~ s(0) + s(1)")

    assert spec.response == "y"
    assert len(spec.smooth_terms) == 2

    assert spec.smooth_terms[0].variable == 0
    assert spec.smooth_terms[1].variable == 1


def test_parse_formula_mixed():
    """parse_formula should handle mixed smooth and parametric terms."""
    spec = parse_formula("y ~ s(x1) + s(x2) + x3 + x4")

    assert spec.response == "y"
    assert len(spec.smooth_terms) == 2
    assert len(spec.parametric_terms) == 2

    assert spec.smooth_terms[0].variable == "x1"
    assert spec.smooth_terms[1].variable == "x2"
    assert spec.parametric_terms[0].variable == "x3"
    assert spec.parametric_terms[1].variable == "x4"


def test_parse_formula_with_options():
    """parse_formula should parse smooth term options."""
    spec = parse_formula("y ~ s(x1, n_basis=15)")

    assert len(spec.smooth_terms) == 1
    assert spec.smooth_terms[0].variable == "x1"
    assert spec.smooth_terms[0].n_basis == 15


def test_parse_formula_with_basis_type():
    """parse_formula should parse basis type option."""
    spec = parse_formula("y ~ s(x1, basis='cubic')")

    assert len(spec.smooth_terms) == 1
    assert spec.smooth_terms[0].variable == "x1"
    assert spec.smooth_terms[0].basis_type == "cubic"


def test_parse_formula_with_multiple_options():
    """parse_formula should parse multiple smooth term options."""
    spec = parse_formula("y ~ s(x1, n_basis=12, basis='bspline', penalty_order=3)")

    assert len(spec.smooth_terms) == 1
    term = spec.smooth_terms[0]
    assert term.variable == "x1"
    assert term.n_basis == 12
    assert term.basis_type == "bspline"
    assert term.penalty_order == 3


def test_parse_formula_with_lambda():
    """parse_formula should parse lambda parameter."""
    spec = parse_formula("y ~ s(x1, lambda=0.5)")

    assert len(spec.smooth_terms) == 1
    assert spec.smooth_terms[0].variable == "x1"
    assert spec.smooth_terms[0].lambda_ == 0.5


def test_parse_formula_whitespace():
    """parse_formula should handle extra whitespace."""
    spec = parse_formula("  y  ~  s( x1 )  +  s( x2 )  +  x3  ")

    assert spec.response == "y"
    assert len(spec.smooth_terms) == 2
    assert len(spec.parametric_terms) == 1


def test_parse_formula_no_tilde():
    """parse_formula should raise error if no tilde."""
    with pytest.raises(ValueError, match="must contain '~'"):
        parse_formula("y + x1 + x2")


def test_parse_formula_no_response():
    """parse_formula should raise error if no response."""
    with pytest.raises(ValueError, match="must specify response"):
        parse_formula("~ s(x1) + x2")


def test_parse_formula_no_predictors():
    """parse_formula should raise error if no predictors."""
    with pytest.raises(ValueError, match="must specify at least one predictor"):
        parse_formula("y ~")


def test_parse_formula_multiple_tildes():
    """parse_formula should raise error if multiple tildes."""
    with pytest.raises(ValueError, match="exactly one '~'"):
        parse_formula("y ~ x1 ~ x2")


def test_parse_formula_invalid_smooth_syntax():
    """parse_formula should raise error for invalid smooth syntax."""
    with pytest.raises(ValueError, match="Invalid smooth term"):
        parse_formula("y ~ s(x1")  # Missing closing paren


def test_parse_formula_smooth_no_variable():
    """parse_formula should raise error if smooth has no variable."""
    with pytest.raises(ValueError, match="must specify variable"):
        parse_formula("y ~ s()")


def test_parse_formula_invalid_option():
    """parse_formula should raise error for invalid option syntax."""
    with pytest.raises(ValueError, match="Invalid smooth term option"):
        parse_formula("y ~ s(x1, 10)")  # Missing key=value


def test_parse_formula_unknown_parameter():
    """parse_formula should raise error for unknown parameters."""
    with pytest.raises(ValueError, match="Unknown smooth term parameter"):
        parse_formula("y ~ s(x1, unknown_param=10)")


def test_parse_formula_complex():
    """parse_formula should handle complex real-world formula."""
    spec = parse_formula(
        "response ~ s(temp, n_basis=15) + s(pressure, basis='cubic') + humidity + elevation"
    )

    assert spec.response == "response"
    assert len(spec.smooth_terms) == 2
    assert len(spec.parametric_terms) == 2

    # Check smooth terms
    assert spec.smooth_terms[0].variable == "temp"
    assert spec.smooth_terms[0].n_basis == 15

    assert spec.smooth_terms[1].variable == "pressure"
    assert spec.smooth_terms[1].basis_type == "cubic"

    # Check parametric terms
    assert spec.parametric_terms[0].variable == "humidity"
    assert spec.parametric_terms[1].variable == "elevation"


def test_parse_formula_numeric_variables():
    """parse_formula should handle all numeric column indices."""
    spec = parse_formula("0 ~ s(1) + s(2) + 3")

    assert spec.response == "0"
    assert len(spec.smooth_terms) == 2
    assert len(spec.parametric_terms) == 1

    assert spec.smooth_terms[0].variable == 1
    assert spec.smooth_terms[1].variable == 2
    assert spec.parametric_terms[0].variable == 3


def test_parse_formula_default_values():
    """Parsed smooth terms should have correct default values."""
    spec = parse_formula("y ~ s(x1)")

    term = spec.smooth_terms[0]
    assert term.variable == "x1"
    assert term.basis_type == "bspline"  # Default
    assert term.n_basis == 10  # Default
    assert term.penalty_order == 2  # Default
    assert term.lambda_ is None  # Default


def test_parse_formula_double_quotes():
    """parse_formula should handle double quotes in options."""
    spec = parse_formula('y ~ s(x1, basis="cubic")')

    assert spec.smooth_terms[0].basis_type == "cubic"
