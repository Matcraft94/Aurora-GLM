"""R-style formula parsing for GAM models.

This module provides a simple formula parser that supports syntax like:
    y ~ s(x1) + s(x2) + x3
    y ~ s(x1, n_basis=15) + s(x2, basis="cubic") + x3 + x4

The parser converts formula strings into term specifications that can be
used with fit_additive_gam().
"""
from __future__ import annotations

import re
from dataclasses import dataclass

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
    """
    response: str
    smooth_terms: list[SmoothTerm]
    parametric_terms: list[ParametricTerm]


def parse_formula(formula: str) -> FormulaSpec:
    """Parse R-style GAM formula into term specifications.

    Parameters
    ----------
    formula : str
        Formula string in R-style syntax, e.g.:
        - "y ~ s(x1) + s(x2)"
        - "y ~ s(x1, n_basis=10) + s(x2, basis='cubic') + x3"
        - "y ~ s(x1) + x2 + x3"

    Returns
    -------
    spec : FormulaSpec
        Parsed formula specification with response name and term lists.

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

    Notes
    -----
    Supported smooth term syntax:
    - s(variable): Default B-spline with 10 basis functions
    - s(variable, n_basis=15): Custom number of basis functions
    - s(variable, basis='cubic'): Cubic spline basis
    - s(variable, basis='bspline', n_basis=12): Combined options

    Variable names can be column names (when using DataFrames) or column
    indices (when using arrays).

    The intercept is always included automatically.
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

    # Split predictors by '+'
    term_strings = [t.strip() for t in predictors.split('+')]

    smooth_terms = []
    parametric_terms = []

    for term_str in term_strings:
        if not term_str:
            continue

        # Check if it's a smooth term s(...)
        if term_str.startswith('s('):
            smooth_term = _parse_smooth_term(term_str)
            smooth_terms.append(smooth_term)
        else:
            # Parametric term (just variable name)
            parametric_term = _parse_parametric_term(term_str)
            parametric_terms.append(parametric_term)

    return FormulaSpec(
        response=response,
        smooth_terms=smooth_terms,
        parametric_terms=parametric_terms,
    )


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

        # Map parameter names
        if key == 'basis':
            kwargs['basis_type'] = value
        elif key == 'n_basis':
            kwargs['n_basis'] = value
        elif key == 'penalty_order':
            kwargs['penalty_order'] = value
        elif key == 'lambda':
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
