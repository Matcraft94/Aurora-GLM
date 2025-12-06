"""Tests for R-style formula parsing."""
from __future__ import annotations

import pytest

from aurora.models.gam.formula import FormulaSpec, parse_formula
from aurora.models.gamm.random_effects import RandomEffect


def test_parse_simple_gam():
    """Parse simple GAM formula without random effects."""
    spec = parse_formula("y ~ s(x1) + s(x2) + x3")

    assert spec.response == "y"
    assert len(spec.smooth_terms) == 2
    assert len(spec.parametric_terms) == 1
    assert len(spec.random_effects) == 0

    # Check smooth terms
    assert spec.smooth_terms[0].variable == "x1"
    assert spec.smooth_terms[1].variable == "x2"

    # Check parametric term
    assert spec.parametric_terms[0].variable == "x3"


def test_parse_random_intercept():
    """Parse formula with random intercept: (1 | group)."""
    spec = parse_formula("y ~ x1 + (1 | subject)")

    assert spec.response == "y"
    assert len(spec.parametric_terms) == 1
    assert len(spec.random_effects) == 1

    re = spec.random_effects[0]
    assert isinstance(re, RandomEffect)
    assert re.grouping == "subject"
    assert re.include_intercept is True
    assert re.variables == ()
    assert re.n_effects == 1


def test_parse_random_slope_no_intercept():
    """Parse formula with random slope only: (x | group)."""
    spec = parse_formula("y ~ x1 + (time | subject)")

    assert len(spec.random_effects) == 1

    re = spec.random_effects[0]
    assert re.grouping == "subject"
    assert re.include_intercept is False
    assert re.variables == ("time",)
    assert re.n_effects == 1


def test_parse_random_intercept_and_slope():
    """Parse formula with random intercept + slope: (1 + x | group)."""
    spec = parse_formula("y ~ x1 + (1 + time | subject)")

    assert len(spec.random_effects) == 1

    re = spec.random_effects[0]
    assert re.grouping == "subject"
    assert re.include_intercept is True
    assert re.variables == ("time",)
    assert re.n_effects == 2


def test_parse_multiple_random_slopes():
    """Parse formula with multiple random slopes: (x + y | group)."""
    spec = parse_formula("y ~ (time + age | subject)")

    assert len(spec.random_effects) == 1

    re = spec.random_effects[0]
    assert re.grouping == "subject"
    assert re.include_intercept is False
    assert re.variables == ("time", "age")
    assert re.n_effects == 2


def test_parse_random_intercept_multiple_slopes():
    """Parse formula with random intercept + multiple slopes."""
    spec = parse_formula("y ~ (1 + time + age | subject)")

    assert len(spec.random_effects) == 1

    re = spec.random_effects[0]
    assert re.grouping == "subject"
    assert re.include_intercept is True
    assert re.variables == ("time", "age")
    assert re.n_effects == 3


def test_parse_crossed_random_effects():
    """Parse formula with crossed random effects: (1 | a) + (1 | b)."""
    spec = parse_formula("y ~ x1 + (1 | subject) + (1 | clinic)")

    assert len(spec.random_effects) == 2

    re1 = spec.random_effects[0]
    assert re1.grouping == "subject"
    assert re1.include_intercept is True
    assert re1.variables == ()

    re2 = spec.random_effects[1]
    assert re2.grouping == "clinic"
    assert re2.include_intercept is True
    assert re2.variables == ()


def test_parse_nested_random_effects():
    """Parse formula with nested random effects: (1 | a/b)."""
    spec = parse_formula("y ~ x1 + (1 | clinic/subject)")

    # Nested syntax creates multiple random effects
    assert len(spec.random_effects) == 2

    # First level: clinic
    re1 = spec.random_effects[0]
    assert re1.grouping == "clinic"
    assert re1.include_intercept is True

    # Second level: subject (nested within clinic)
    re2 = spec.random_effects[1]
    assert re2.grouping == "subject"
    assert re2.include_intercept is True


def test_parse_complex_gamm():
    """Parse complex GAMM formula with smooths, parametric, and random effects."""
    spec = parse_formula("y ~ s(x1) + s(x2, n_basis=15) + x3 + (1 + time | subject)")

    assert spec.response == "y"
    assert len(spec.smooth_terms) == 2
    assert len(spec.parametric_terms) == 1
    assert len(spec.random_effects) == 1

    # Check smooth with custom basis
    assert spec.smooth_terms[1].variable == "x2"
    assert spec.smooth_terms[1].n_basis == 15

    # Check random effect
    re = spec.random_effects[0]
    assert re.grouping == "subject"
    assert re.include_intercept is True
    assert re.variables == ("time",)


def test_parse_integer_variable_names():
    """Parse formula with integer column indices."""
    # Note: '1' in formulas means intercept (not column 1), so use 4 as parametric term
    spec = parse_formula("y ~ s(0) + 4 + (1 + 2 | 3)")

    # Smooth term with column index
    assert spec.smooth_terms[0].variable == 0

    # Parametric term with column index
    assert spec.parametric_terms[0].variable == 4

    # Random effect with column indices
    re = spec.random_effects[0]
    assert re.grouping == '3'  # Grouping is parsed as string
    assert re.variables == (2,)


def test_parse_whitespace_handling():
    """Parse formula with various whitespace patterns."""
    spec1 = parse_formula("y~s(x1)+x2+(1|subject)")
    spec2 = parse_formula("y ~ s(x1) + x2 + (1 | subject)")
    spec3 = parse_formula("y  ~  s( x1 )  +  x2  +  ( 1  |  subject )")

    # All should parse identically
    for spec in [spec1, spec2, spec3]:
        assert spec.response == "y"
        assert len(spec.smooth_terms) == 1
        assert len(spec.parametric_terms) == 1
        assert len(spec.random_effects) == 1


def test_parse_no_tilde_raises():
    """Formula without ~ should raise error."""
    with pytest.raises(ValueError, match="must contain '~'"):
        parse_formula("y s(x1) + x2")


def test_parse_empty_response_raises():
    """Formula with empty response should raise error."""
    with pytest.raises(ValueError, match="must specify response"):
        parse_formula("~ s(x1) + x2")


def test_parse_empty_predictors_raises():
    """Formula with empty predictors should raise error."""
    with pytest.raises(ValueError, match="must specify at least one predictor"):
        parse_formula("y ~ ")


def test_parse_invalid_random_effect_no_bar():
    """Random effect without | should raise error."""
    # Parenthesized terms without | are invalid - must use (1 | group) syntax
    with pytest.raises(ValueError, match="Parenthesized term.*without.*invalid"):
        parse_formula("y ~ x1 + (1 subject)")


def test_parse_invalid_random_effect_no_parens():
    """The '1' in formula represents intercept, not a parametric term."""
    # '1' is the intercept marker, not column 1, so it's skipped
    # Intercept is added automatically by the fitting functions
    spec = parse_formula("y ~ 1")
    assert len(spec.parametric_terms) == 0  # '1' is intercept, not a parametric term
    assert len(spec.random_effects) == 0

    # To have an actual parametric term, use variable names or other column indices
    spec2 = parse_formula("y ~ x1")
    assert len(spec2.parametric_terms) == 1
    assert spec2.parametric_terms[0].variable == "x1"


def test_parse_smooth_with_all_options():
    """Parse smooth term with all available options."""
    spec = parse_formula("y ~ s(x1, n_basis=20, basis='cubic', lambda=0.5)")

    smooth = spec.smooth_terms[0]
    assert smooth.variable == "x1"
    assert smooth.n_basis == 20
    assert smooth.basis_type == "cubic"
    assert smooth.lambda_ == 0.5


def test_parse_multiple_nested_levels():
    """Parse formula with deeply nested random effects."""
    spec = parse_formula("y ~ x1 + (1 | country/state/city)")

    # Should create 3 random effects for 3 levels
    assert len(spec.random_effects) == 3
    assert spec.random_effects[0].grouping == "country"
    assert spec.random_effects[1].grouping == "state"
    assert spec.random_effects[2].grouping == "city"


def test_parse_mixed_crossed_and_nested():
    """Parse formula with both crossed and nested random effects."""
    spec = parse_formula("y ~ x1 + (1 | clinic/subject) + (1 | time)")

    # Nested clinic/subject creates 2 REs, plus crossed time = 3 total
    assert len(spec.random_effects) == 3


def test_formula_spec_empty_lists():
    """FormulaSpec with no terms should have empty lists."""
    spec = FormulaSpec(
        response="y",
        smooth_terms=[],
        parametric_terms=[],
        random_effects=[],
    )

    assert spec.response == "y"
    assert len(spec.smooth_terms) == 0
    assert len(spec.parametric_terms) == 0
    assert len(spec.random_effects) == 0


def test_parse_zero_removes_intercept():
    """Formula with 0 in random effect should not include intercept."""
    spec = parse_formula("y ~ x1 + (0 + time | subject)")

    re = spec.random_effects[0]
    assert re.include_intercept is False
    assert re.variables == ("time",)


def test_parse_both_one_and_zero():
    """Formula with both 1 and 0: 0 removes intercept (R behavior)."""
    spec = parse_formula("y ~ x1 + (1 + 0 + time | subject)")

    re = spec.random_effects[0]
    # 0 explicitly removes intercept, even if 1 is present (R formula convention)
    assert re.include_intercept is False
    assert re.variables == ("time",)


def test_parse_empty_random_effect_parens():
    """Empty parentheses should raise error."""
    # Empty parentheses are invalid - use proper random effect syntax
    with pytest.raises(ValueError, match="Empty parentheses"):
        parse_formula("y ~ x1 + ()")


def test_parse_random_effect_only_bar():
    """Random effect with only | should raise error."""
    with pytest.raises(ValueError):
        parse_formula("y ~ x1 + (|)")


def test_parse_multiple_bars_in_random_effect():
    """Random effect with multiple | should raise error."""
    with pytest.raises(ValueError, match="exactly one '|'"):
        parse_formula("y ~ x1 + (1 | a | b)")


def test_split_formula_terms_simple():
    """Test _split_formula_terms with simple formula."""
    from aurora.models.gam.formula import _split_formula_terms

    terms = _split_formula_terms("x1 + x2 + x3")
    assert terms == ["x1", "x2", "x3"]


def test_split_formula_terms_with_parens():
    """Test _split_formula_terms respects parentheses."""
    from aurora.models.gam.formula import _split_formula_terms

    terms = _split_formula_terms("x1 + (1 + x2 | group) + x3")
    assert terms == ["x1", "(1 + x2 | group)", "x3"]


def test_split_formula_terms_nested_parens():
    """Test _split_formula_terms with nested parentheses."""
    from aurora.models.gam.formula import _split_formula_terms

    # Should handle complex nesting
    terms = _split_formula_terms("x1 + (1 | a/b) + (1 | c)")
    assert terms == ["x1", "(1 | a/b)", "(1 | c)"]


def test_parse_random_effects_formula_intercept_only():
    """Test _parse_random_effects_formula with intercept only."""
    from aurora.models.gam.formula import _parse_random_effects_formula

    include_intercept, variables = _parse_random_effects_formula("1")
    assert include_intercept is True
    assert variables == ()


def test_parse_random_effects_formula_slope_only():
    """Test _parse_random_effects_formula with slope only."""
    from aurora.models.gam.formula import _parse_random_effects_formula

    include_intercept, variables = _parse_random_effects_formula("x")
    assert include_intercept is False
    assert variables == ("x",)


def test_parse_random_effects_formula_intercept_and_slope():
    """Test _parse_random_effects_formula with intercept + slope."""
    from aurora.models.gam.formula import _parse_random_effects_formula

    include_intercept, variables = _parse_random_effects_formula("1 + x")
    assert include_intercept is True
    assert variables == ("x",)


def test_parse_random_effects_formula_multiple_slopes():
    """Test _parse_random_effects_formula with multiple slopes."""
    from aurora.models.gam.formula import _parse_random_effects_formula

    include_intercept, variables = _parse_random_effects_formula("x + y + z")
    assert include_intercept is False
    assert variables == ("x", "y", "z")


def test_parse_random_effects_formula_all_together():
    """Test _parse_random_effects_formula with intercept + multiple slopes."""
    from aurora.models.gam.formula import _parse_random_effects_formula

    include_intercept, variables = _parse_random_effects_formula("1 + x + y")
    assert include_intercept is True
    assert variables == ("x", "y")
