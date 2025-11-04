"""Data structures for GAM smooth terms specification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class SmoothTerm:
    """Specification of a smooth term s(x, ...)

    This class defines how a predictor variable should be smoothed
    in a GAM model.

    Parameters
    ----------
    variable : str or int
        Variable name (if using DataFrame) or column index (if using array).
    basis_type : {'bspline', 'cubic', 'tp', 'cr', 'ps'}, default='bspline'
        Type of spline basis:
        - 'bspline': B-spline basis (default)
        - 'cubic': Natural cubic splines
        - 'tp': Thin plate splines (future)
        - 'cr': Cubic regression splines (future)
        - 'ps': P-splines (future)
    n_basis : int, default=10
        Number of basis functions (k in mgcv terminology).
    penalty_order : int, default=2
        Order of difference penalty (usually 2 for second derivative).
    by : str or int, optional
        Factor variable for by-factor smooths (not yet implemented).
    lambda_ : float, optional
        Fixed smoothing parameter. If None, selected automatically.
    knot_method : {'quantile', 'uniform'}, default='quantile'
        Method for placing knots.

    Examples
    --------
    >>> # Smooth term with default settings
    >>> s1 = SmoothTerm(variable='x1')

    >>> # Smooth term with custom settings
    >>> s2 = SmoothTerm(
    ...     variable='x2',
    ...     basis_type='cubic',
    ...     n_basis=15,
    ...     lambda_=0.1
    ... )

    >>> # Using column index instead of name
    >>> s3 = SmoothTerm(variable=0, n_basis=12)

    Notes
    -----
    This is inspired by mgcv's s() function in R but simplified for
    initial implementation.
    """

    variable: str | int
    basis_type: Literal["bspline", "cubic", "tp", "cr", "ps"] = "bspline"
    n_basis: int = 10
    penalty_order: int = 2
    by: str | int | None = None
    lambda_: float | None = None
    knot_method: Literal["quantile", "uniform"] = "quantile"

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.n_basis < 3:
            raise ValueError(f"n_basis must be at least 3, got {self.n_basis}")

        if self.penalty_order < 1:
            raise ValueError(
                f"penalty_order must be positive, got {self.penalty_order}"
            )

        if self.basis_type not in ["bspline", "cubic", "tp", "cr", "ps"]:
            raise ValueError(f"Unknown basis_type: {self.basis_type}")

        if self.basis_type not in ["bspline", "cubic"]:
            raise NotImplementedError(
                f"basis_type='{self.basis_type}' not yet implemented. "
                f"Use 'bspline' or 'cubic'."
            )

        if self.lambda_ is not None and self.lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative, got {self.lambda_}")

        if self.by is not None:
            raise NotImplementedError("by-factor smooths not yet implemented")


@dataclass
class ParametricTerm:
    """Specification of a parametric (linear) term.

    Parameters
    ----------
    variable : str or int
        Variable name or column index.

    Examples
    --------
    >>> p1 = ParametricTerm(variable='age')
    >>> p2 = ParametricTerm(variable=2)  # column index
    """

    variable: str | int


@dataclass
class TensorTerm:
    """Specification of a tensor product smooth te(x1, x2, ...).

    Tensor product smooths model interactions between variables using
    the tensor product of marginal basis functions.

    Parameters
    ----------
    variables : tuple of (str or int)
        Tuple of variable names or column indices (length 2 or more).
    basis_types : tuple of str, optional
        Basis type for each variable. If None, uses 'bspline' for all.
    n_basis : tuple of int, optional
        Number of basis functions for each variable. If None, uses 10 for all.
    lambdas : tuple of float, optional
        Fixed smoothing parameters for each direction. If None, selected automatically.

    Examples
    --------
    >>> # 2D tensor product with defaults
    >>> te1 = TensorTerm(variables=(0, 1))

    >>> # Custom basis sizes
    >>> te2 = TensorTerm(variables=('x1', 'x2'), n_basis=(12, 12))

    >>> # Different basis types
    >>> te3 = TensorTerm(
    ...     variables=(0, 1),
    ...     basis_types=('bspline', 'cubic'),
    ...     n_basis=(10, 8)
    ... )

    Notes
    -----
    Tensor products are useful for modeling interactions. They are more
    flexible than additive models but require more data and computation.

    The tensor product te(x1, x2) creates a surface f(x1, x2) rather than
    the sum f1(x1) + f2(x2).
    """

    variables: tuple[str | int, ...]
    basis_types: tuple[str, ...] | None = None
    n_basis: tuple[int, ...] | None = None
    lambdas: tuple[float, ...] | None = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        if len(self.variables) < 2:
            raise ValueError("TensorTerm requires at least 2 variables")

        # Set defaults
        n_vars = len(self.variables)

        if self.basis_types is None:
            self.basis_types = ('bspline',) * n_vars
        elif len(self.basis_types) != n_vars:
            raise ValueError(
                f"basis_types must have same length as variables, "
                f"got {len(self.basis_types)} vs {n_vars}"
            )

        if self.n_basis is None:
            self.n_basis = (10,) * n_vars
        elif len(self.n_basis) != n_vars:
            raise ValueError(
                f"n_basis must have same length as variables, "
                f"got {len(self.n_basis)} vs {n_vars}"
            )

        if self.lambdas is not None and len(self.lambdas) != n_vars:
            raise ValueError(
                f"lambdas must have same length as variables, "
                f"got {len(self.lambdas)} vs {n_vars}"
            )

        # Validate basis types
        for bt in self.basis_types:
            if bt not in ['bspline', 'cubic']:
                raise NotImplementedError(
                    f"basis_type='{bt}' not supported in tensor products yet. "
                    f"Use 'bspline' or 'cubic'."
                )

        # Validate n_basis
        for nb in self.n_basis:
            if nb < 3:
                raise ValueError(f"Each n_basis must be at least 3, got {nb}")

        # Validate lambdas
        if self.lambdas is not None:
            for lam in self.lambdas:
                if lam < 0:
                    raise ValueError(f"All lambdas must be non-negative, got {lam}")


__all__ = ["SmoothTerm", "ParametricTerm", "TensorTerm"]
