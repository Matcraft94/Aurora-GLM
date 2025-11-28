"""R-style formula parsing for GAM and GAMM models.

This module provides a formula parser that supports syntax like:
    y ~ s(x1) + s(x2) + x3
    y ~ s(x1, n_basis=15) + s(x2, basis="cubic") + x3 + x4
    y ~ s(x1) + x2 + (1 | subject)  # With random effects
    y ~ s(x1) + (1 + time | subject)  # Random intercept + slope

The parser converts formula strings into term specifications that can be
used with fit_additive_gam() or fit_gamm().
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from aurora.models.gam.terms import ParametricTerm, SmoothTerm


@dataclass
class FormulaSpec:
    """Parsed formula specification.

    Attributes
    ----------
    response : str
        Name of the response variable.
    smooth_terms : list of SmoothTerm
        Smooth term specifications.
    parametric_terms : list of ParametricTerm
        Parametric term specifications.
    random_effects : list of RandomEffect
        Random effects specifications (for GAMM models).
    """
    response: str
    smooth_terms: list[SmoothTerm]
    parametric_terms: list[ParametricTerm]
    random_effects: list = field(default_factory=list)


def parse_formula(formula: str) -> FormulaSpec:
    """Parse R-style GAM/GAMM formula into term specifications.

    Parameters
    ----------
    formula : str
        Formula string in R-style syntax, e.g.:
        - "y ~ s(x1) + s(x2)"
        - "y ~ s(x1, n_basis=10) + s(x2, basis='cubic') + x3"
        - "y ~ s(x1) + x2 + (1 | subject)"  # Random intercept
        - "y ~ s(x1) + (1 + time | subject)"  # Random intercept + slope
        - "y ~ s(x1) + (1 | clinic/subject)"  # Nested random effects
        - "y ~ s(x1) + (1 | subject) + (1 | clinic)"  # Crossed random effects

    Returns
    -------
    spec : FormulaSpec
        Parsed formula specification with response, smooth terms, parametric
        terms, and random effects.

    Examples
    --------
    >>> from aurora.models.gam.formula import parse_formula
    >>> spec = parse_formula("y ~ s(x1) + s(x2) + x3")
    >>> spec.response
    'y'
    >>> len(spec.smooth_terms)
    2
    >>> len(spec.parametric_terms)
    1

    >>> # With random effects
    >>> spec = parse_formula("y ~ s(x1) + x2 + (1 | subject)")
    >>> len(spec.random_effects)
    1
    >>> spec.random_effects[0].grouping
    'subject'

    Notes
    -----
    Supported smooth term syntax:
    - s(variable): Default B-spline with 10 basis functions
    - s(variable, n_basis=15): Custom number of basis functions
    - s(variable, basis='cubic'): Cubic spline basis
    - s(variable, basis='bspline', n_basis=12): Combined options

    Supported random effects syntax (lme4-style):
    - (1 | group): Random intercept
    - (x | group): Random slope (no intercept)
    - (1 + x | group): Random intercept + slope
    - (x + y | group): Multiple random slopes
    - (1 | a/b): Nested random effects (a within b)
    - (1 | a) + (1 | b): Crossed random effects

    Variable names can be column names (when using DataFrames) or column
    indices (when using arrays).

    The intercept is always included automatically for fixed effects.
    """
    # Remove whitespace
    formula = formula.strip()

    # Split into response and predictors
    if '~' not in formula:
        raise ValueError("Formula must contain '~' separating response and predictors")

    parts = formula.split('~')
    if len(parts) != 2:
        raise ValueError("Formula must have exactly one '~'")

    response = parts[0].strip()
    predictors = parts[1].strip()

    if not response:
        raise ValueError("Formula must specify response variable")

    if not predictors:
        raise ValueError("Formula must specify at least one predictor")

    # Split predictors by '+', but be careful with parentheses in random effects
    term_strings = _split_formula_terms(predictors)

    smooth_terms = []
    parametric_terms = []
    random_effects = []

    for term_str in term_strings:
        if not term_str:
            continue

        # Check if it's a random effect (...)
        if term_str.startswith('(') and '|' in term_str:
            random_effect = _parse_random_effect_term(term_str)
            # Could be list if nested (e.g., (1 | a/b))
            if isinstance(random_effect, list):
                random_effects.extend(random_effect)
            else:
                random_effects.append(random_effect)
        # Check if it's a smooth term s(...)
        elif term_str.startswith('s('):
            smooth_term = _parse_smooth_term(term_str)
            smooth_terms.append(smooth_term)
        else:
            # Parametric term (just variable name)
            # Skip '1' as it represents the intercept which is added automatically
            if term_str.strip() == '1':
                continue
            parametric_term = _parse_parametric_term(term_str)
            parametric_terms.append(parametric_term)

    return FormulaSpec(
        response=response,
        smooth_terms=smooth_terms,
        parametric_terms=parametric_terms,
        random_effects=random_effects,
    )


def _split_formula_terms(formula_str: str) -> list[str]:
    """Split formula by '+' while respecting parentheses.

    Parameters
    ----------
    formula_str : str
        Formula string (predictor side only)

    Returns
    -------
    terms : list of str
        List of term strings

    Examples
    --------
    >>> _split_formula_terms("x1 + x2 + x3")
    ['x1', 'x2', 'x3']
    >>> _split_formula_terms("x1 + (1 | group)")
    ['x1', '(1 | group)']
    >>> _split_formula_terms("x1 + (1 + x2 | group) + x3")
    ['x1', '(1 + x2 | group)', 'x3']
    """
    terms = []
    current_term = []
    paren_depth = 0

    for char in formula_str:
        if char == '(':
            paren_depth += 1
            current_term.append(char)
        elif char == ')':
            paren_depth -= 1
            current_term.append(char)
        elif char == '+' and paren_depth == 0:
            # Split here
            terms.append(''.join(current_term).strip())
            current_term = []
        else:
            current_term.append(char)

    # Add last term
    if current_term:
        terms.append(''.join(current_term).strip())

    return terms


def _parse_random_effect_term(term_str: str):
    """Parse a random effect term like '(1 | group)' or '(1 + x | group)'.

    Parameters
    ----------
    term_str : str
        Random effect term string

    Returns
    -------
    random_effect : RandomEffect or list of RandomEffect
        Parsed random effect(s). Returns list for nested effects like (1 | a/b).

    Examples
    --------
    >>> from aurora.models.gamm import RandomEffect
    >>> re = _parse_random_effect_term("(1 | subject)")
    >>> re.grouping
    'subject'
    >>> re.include_intercept
    True
    >>> re.variables
    ()

    >>> re = _parse_random_effect_term("(1 + time | subject)")
    >>> re.include_intercept
    True
    >>> re.variables
    ('time',)

    >>> re = _parse_random_effect_term("(time | subject)")
    >>> re.include_intercept
    False
    >>> re.variables
    ('time',)
    """
    # Import here to avoid circular import
    from aurora.models.gamm.random_effects import RandomEffect

    # Remove outer parentheses
    if not (term_str.startswith('(') and term_str.endswith(')')):
        raise ValueError(f"Random effect term must be enclosed in parentheses: {term_str}")

    content = term_str[1:-1].strip()

    if '|' not in content:
        raise ValueError(f"Random effect term must contain '|': {term_str}")

    # Split by '|'
    parts = content.split('|')
    if len(parts) != 2:
        raise ValueError(f"Random effect term must have exactly one '|': {term_str}")

    effects_part = parts[0].strip()
    grouping_part = parts[1].strip()

    # Parse grouping (could be nested like a/b)
    if '/' in grouping_part:
        # Nested random effects: (1 | a/b) means (1 | b) + (1 | a:b)
        grouping_vars = [g.strip() for g in grouping_part.split('/')]
        # Create random effects for nested structure
        # For simplicity, we'll create one RE per grouping level
        random_effects_list = []
        for grouping in grouping_vars:
            # Parse the effects part
            include_intercept, variables = _parse_random_effects_formula(effects_part)
            random_effects_list.append(
                RandomEffect(
                    grouping=grouping,
                    variables=variables,
                    include_intercept=include_intercept,
                )
            )
        return random_effects_list
    else:
        # Simple random effect
        grouping = grouping_part

        # Parse the effects part (left of |)
        include_intercept, variables = _parse_random_effects_formula(effects_part)

        return RandomEffect(
            grouping=grouping,
            variables=variables,
            include_intercept=include_intercept,
        )


def _parse_random_effects_formula(formula_str: str) -> tuple[bool, tuple]:
    """Parse the left side of a random effect term (before |).

    Parameters
    ----------
    formula_str : str
        Effects formula like "1", "x", "1 + x", "x + y"

    Returns
    -------
    include_intercept : bool
        Whether to include random intercept
    variables : tuple
        Variables with random slopes

    Examples
    --------
    >>> _parse_random_effects_formula("1")
    (True, ())
    >>> _parse_random_effects_formula("x")
    (False, ('x',))
    >>> _parse_random_effects_formula("1 + x")
    (True, ('x',))
    >>> _parse_random_effects_formula("x + y")
    (False, ('x', 'y'))
    """
    # Split by '+'
    parts = [p.strip() for p in formula_str.split('+')]

    include_intercept = False
    variables = []

    for part in parts:
        if not part:
            continue

        if part == '1':
            include_intercept = True
        elif part == '0':
            # Explicit removal of intercept (not common but valid)
            include_intercept = False
        else:
            # Variable name or index
            try:
                var = int(part)
            except ValueError:
                var = part
            variables.append(var)

    return include_intercept, tuple(variables)


def _parse_smooth_term(term_str: str) -> SmoothTerm:
    """Parse a smooth term like 's(x1, n_basis=10, basis="cubic")'."""
    # Extract content inside s(...)
    match = re.match(r's\((.*)\)', term_str)
    if not match:
        raise ValueError(f"Invalid smooth term syntax: {term_str}")

    content = match.group(1).strip()

    if not content:
        raise ValueError(f"Smooth term must specify variable: {term_str}")

    # Split by comma
    parts = [p.strip() for p in content.split(',')]

    if len(parts) == 0 or not parts[0]:
        raise ValueError(f"Smooth term must specify variable: {term_str}")

    # First part is the variable
    variable_str = parts[0]

    # Try to parse as integer (column index) or keep as string (column name)
    try:
        variable = int(variable_str)
    except ValueError:
        variable = variable_str

    # Parse options
    kwargs = {}
    for part in parts[1:]:
        if '=' not in part:
            raise ValueError(f"Invalid smooth term option: {part}")

        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()

        # Remove quotes from string values
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            value = value[1:-1]
        # Try to parse as int
        elif value.isdigit():
            value = int(value)
        # Try to parse as float
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string

        # Map parameter names (with R/mgcv-style aliases)
        if key == 'basis':
            kwargs['basis_type'] = value
        elif key in ('n_basis', 'k'):  # 'k' is the mgcv-style parameter
            kwargs['n_basis'] = value
        elif key in ('penalty_order', 'm'):  # 'm' is mgcv-style penalty order
            kwargs['penalty_order'] = value
        elif key in ('lambda', 'sp'):  # 'sp' is mgcv-style smoothing parameter
            kwargs['lambda_'] = value
        else:
            raise ValueError(f"Unknown smooth term parameter: {key}")

    return SmoothTerm(variable=variable, **kwargs)


def _parse_parametric_term(term_str: str) -> ParametricTerm:
    """Parse a parametric term like 'x1' or '2'."""
    # Try to parse as integer (column index)
    try:
        variable = int(term_str)
    except ValueError:
        # Keep as string (column name)
        variable = term_str

    return ParametricTerm(variable=variable)


__all__ = ["parse_formula", "FormulaSpec"]
