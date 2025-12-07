# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Input validation decorators for Aurora-GLM.

This module provides decorators for validating function inputs, reducing
boilerplate and ensuring consistent error messages across the library.

Usage
-----
>>> from aurora.utils.validation.decorators import validate_array, validate_positive
>>>
>>> @validate_array('X', ndim=2)
>>> @validate_array('y', ndim=1)
>>> @validate_positive('alpha')
>>> def fit_model(X, y, alpha=1.0):
...     pass

Available Decorators
--------------------
- @validate_array: Validate numpy arrays (shape, dtype, finiteness)
- @validate_positive: Validate positive numeric values
- @validate_non_negative: Validate non-negative numeric values
- @validate_probability: Validate values in [0, 1]
- @validate_in_range: Validate values in a specified range
- @validate_type: Validate argument type
- @validate_callable: Validate callable arguments
- @validate_not_none: Validate required arguments
- @validated: Composite decorator for multiple validations

Examples
--------
>>> @validate_array('X', ndim=2, min_rows=1)
>>> @validate_array('y', ndim=1, dtype_check='numeric')
>>> @validate_positive('lambda_', name='regularization parameter')
>>> @validate_in_range('alpha', 0, 1, inclusive='both')
>>> def fit_with_penalty(X, y, lambda_=1.0, alpha=0.5):
...     '''Fit model with elastic net penalty.'''
...     pass
"""

from __future__ import annotations

import functools
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    ParamSpec,
    Sequence,
    TypeVar,
)

import numpy as np

if TYPE_CHECKING:
    pass

P = ParamSpec("P")
R = TypeVar("R")


# =============================================================================
# Exception Types
# =============================================================================


class ValidationError(ValueError):
    """Raised when input validation fails.

    Attributes
    ----------
    param_name : str
        Name of the parameter that failed validation.
    message : str
        Detailed error message.
    """

    def __init__(self, param_name: str, message: str):
        self.param_name = param_name
        self.message = message
        super().__init__(f"Invalid value for '{param_name}': {message}")


# =============================================================================
# Array Validation Decorator
# =============================================================================


def validate_array(
    param: str,
    *,
    ndim: int | tuple[int, ...] | None = None,
    min_rows: int | None = None,
    min_cols: int | None = None,
    dtype_check: Literal["numeric", "float", "int", "bool", None] = None,
    allow_1d: bool = True,
    ensure_2d: bool = False,
    check_finite: bool = True,
    allow_none: bool = False,
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to validate numpy array arguments.

    Parameters
    ----------
    param : str
        Name of the parameter to validate.
    ndim : int or tuple of int, optional
        Required number of dimensions. Can be a tuple for multiple options.
    min_rows : int, optional
        Minimum number of rows (first dimension).
    min_cols : int, optional
        Minimum number of columns (second dimension).
    dtype_check : {'numeric', 'float', 'int', 'bool'}, optional
        Required dtype category.
    allow_1d : bool, default=True
        Whether to allow 1D arrays.
    ensure_2d : bool, default=False
        If True, automatically reshape 1D arrays to 2D column vectors.
    check_finite : bool, default=True
        Whether to check for NaN/Inf values.
    allow_none : bool, default=False
        Whether None is a valid value.
    name : str, optional
        Human-readable name for error messages.

    Returns
    -------
    decorator : callable
        Decorator function.

    Examples
    --------
    >>> @validate_array('X', ndim=2, min_rows=1, dtype_check='numeric')
    >>> def fit(X, y):
    ...     pass
    """
    display_name = name or param

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Get the argument value
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            value = bound.arguments.get(param)

            # Handle None
            if value is None:
                if allow_none:
                    return func(*args, **kwargs)
                raise ValidationError(param, f"{display_name} cannot be None")

            # Convert to array if needed
            try:
                arr = np.asarray(value)
            except Exception as e:
                raise ValidationError(param, f"Cannot convert to array: {e}")

            # Check ndim
            if ndim is not None:
                valid_ndims = (ndim,) if isinstance(ndim, int) else ndim
                if arr.ndim not in valid_ndims:
                    raise ValidationError(
                        param, f"{display_name} must be {ndim}D, got {arr.ndim}D"
                    )

            # Check 1D allowed
            if not allow_1d and arr.ndim == 1:
                raise ValidationError(param, f"{display_name} cannot be 1D")

            # Handle ensure_2d
            if ensure_2d and arr.ndim == 1:
                arr = arr.reshape(-1, 1)
                bound.arguments[param] = arr

            # Check shape
            if min_rows is not None and arr.shape[0] < min_rows:
                raise ValidationError(
                    param,
                    f"{display_name} must have at least {min_rows} rows, got {arr.shape[0]}",
                )

            if min_cols is not None and arr.ndim >= 2 and arr.shape[1] < min_cols:
                raise ValidationError(
                    param,
                    f"{display_name} must have at least {min_cols} columns, got {arr.shape[1]}",
                )

            # Check dtype
            if dtype_check is not None:
                if dtype_check == "numeric":
                    if not np.issubdtype(arr.dtype, np.number):
                        raise ValidationError(
                            param, f"{display_name} must be numeric, got {arr.dtype}"
                        )
                elif dtype_check == "float":
                    if not np.issubdtype(arr.dtype, np.floating):
                        raise ValidationError(
                            param, f"{display_name} must be float, got {arr.dtype}"
                        )
                elif dtype_check == "int":
                    if not np.issubdtype(arr.dtype, np.integer):
                        raise ValidationError(
                            param, f"{display_name} must be integer, got {arr.dtype}"
                        )
                elif dtype_check == "bool":
                    if not np.issubdtype(arr.dtype, np.bool_):
                        raise ValidationError(
                            param, f"{display_name} must be boolean, got {arr.dtype}"
                        )

            # Check finite
            if check_finite and np.issubdtype(arr.dtype, np.number):
                if not np.all(np.isfinite(arr)):
                    n_nan = np.sum(np.isnan(arr))
                    n_inf = np.sum(np.isinf(arr))
                    raise ValidationError(
                        param,
                        f"{display_name} contains {n_nan} NaN and {n_inf} Inf values",
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Numeric Validation Decorators
# =============================================================================


def validate_positive(
    param: str,
    *,
    strict: bool = True,
    allow_none: bool = False,
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to validate that a numeric argument is positive.

    Parameters
    ----------
    param : str
        Name of the parameter to validate.
    strict : bool, default=True
        If True, value must be > 0. If False, value must be >= 0.
    allow_none : bool, default=False
        Whether None is a valid value.
    name : str, optional
        Human-readable name for error messages.

    Examples
    --------
    >>> @validate_positive('alpha')
    >>> @validate_positive('tol', strict=False)
    >>> def fit(X, y, alpha=1.0, tol=0.0):
    ...     pass
    """
    display_name = name or param

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            value = bound.arguments.get(param)

            if value is None:
                if allow_none:
                    return func(*args, **kwargs)
                raise ValidationError(param, f"{display_name} cannot be None")

            if strict:
                if value <= 0:
                    raise ValidationError(
                        param, f"{display_name} must be positive, got {value}"
                    )
            else:
                if value < 0:
                    raise ValidationError(
                        param, f"{display_name} must be non-negative, got {value}"
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_non_negative(
    param: str,
    *,
    allow_none: bool = False,
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to validate that a numeric argument is non-negative (>= 0)."""
    return validate_positive(param, strict=False, allow_none=allow_none, name=name)


def validate_probability(
    param: str,
    *,
    allow_zero: bool = True,
    allow_one: bool = True,
    allow_none: bool = False,
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to validate that a value is a valid probability in [0, 1].

    Parameters
    ----------
    param : str
        Name of the parameter to validate.
    allow_zero : bool, default=True
        Whether 0 is a valid value.
    allow_one : bool, default=True
        Whether 1 is a valid value.
    allow_none : bool, default=False
        Whether None is a valid value.
    name : str, optional
        Human-readable name for error messages.
    """
    display_name = name or param

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            value = bound.arguments.get(param)

            if value is None:
                if allow_none:
                    return func(*args, **kwargs)
                raise ValidationError(param, f"{display_name} cannot be None")

            # Check bounds
            lower = 0 if allow_zero else 0 + np.finfo(float).eps
            upper = 1 if allow_one else 1 - np.finfo(float).eps

            if not (lower <= value <= upper):
                bounds_str = f"[{0 if allow_zero else '(0'}, {1 if allow_one else '1)'}"
                raise ValidationError(
                    param, f"{display_name} must be in {bounds_str}, got {value}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_in_range(
    param: str,
    lower: float | None = None,
    upper: float | None = None,
    *,
    inclusive: Literal["both", "left", "right", "neither"] = "both",
    allow_none: bool = False,
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to validate that a value is within a specified range.

    Parameters
    ----------
    param : str
        Name of the parameter to validate.
    lower : float, optional
        Lower bound.
    upper : float, optional
        Upper bound.
    inclusive : {'both', 'left', 'right', 'neither'}, default='both'
        Which bounds are inclusive.
    allow_none : bool, default=False
        Whether None is a valid value.
    name : str, optional
        Human-readable name for error messages.

    Examples
    --------
    >>> @validate_in_range('x', 0, 10, inclusive='left')  # [0, 10)
    >>> @validate_in_range('ratio', 0, 1)  # [0, 1]
    """
    display_name = name or param

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            value = bound.arguments.get(param)

            if value is None:
                if allow_none:
                    return func(*args, **kwargs)
                raise ValidationError(param, f"{display_name} cannot be None")

            # Check lower bound
            if lower is not None:
                if inclusive in ("both", "left"):
                    if value < lower:
                        raise ValidationError(
                            param, f"{display_name} must be >= {lower}, got {value}"
                        )
                else:
                    if value <= lower:
                        raise ValidationError(
                            param, f"{display_name} must be > {lower}, got {value}"
                        )

            # Check upper bound
            if upper is not None:
                if inclusive in ("both", "right"):
                    if value > upper:
                        raise ValidationError(
                            param, f"{display_name} must be <= {upper}, got {value}"
                        )
                else:
                    if value >= upper:
                        raise ValidationError(
                            param, f"{display_name} must be < {upper}, got {value}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Type Validation Decorators
# =============================================================================


def validate_type(
    param: str,
    expected_type: type | tuple[type, ...],
    *,
    allow_none: bool = False,
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to validate that an argument is of a specific type.

    Parameters
    ----------
    param : str
        Name of the parameter to validate.
    expected_type : type or tuple of types
        Expected type(s) for the argument.
    allow_none : bool, default=False
        Whether None is a valid value.
    name : str, optional
        Human-readable name for error messages.
    """
    display_name = name or param

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            value = bound.arguments.get(param)

            if value is None:
                if allow_none:
                    return func(*args, **kwargs)
                raise ValidationError(param, f"{display_name} cannot be None")

            if not isinstance(value, expected_type):
                if isinstance(expected_type, tuple):
                    type_names = " or ".join(t.__name__ for t in expected_type)
                else:
                    type_names = expected_type.__name__
                raise ValidationError(
                    param,
                    f"{display_name} must be {type_names}, got {type(value).__name__}",
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_callable(
    param: str,
    *,
    allow_none: bool = False,
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to validate that an argument is callable."""
    display_name = name or param

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            value = bound.arguments.get(param)

            if value is None:
                if allow_none:
                    return func(*args, **kwargs)
                raise ValidationError(param, f"{display_name} cannot be None")

            if not callable(value):
                raise ValidationError(
                    param,
                    f"{display_name} must be callable, got {type(value).__name__}",
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_not_none(
    *params: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to validate that arguments are not None.

    Parameters
    ----------
    *params : str
        Names of parameters that must not be None.

    Examples
    --------
    >>> @validate_not_none('X', 'y')
    >>> def fit(X, y, weights=None):
    ...     pass
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param in params:
                if bound.arguments.get(param) is None:
                    raise ValidationError(
                        param, f"'{param}' is required and cannot be None"
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_one_of(
    param: str,
    choices: Sequence[Any],
    *,
    allow_none: bool = False,
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to validate that an argument is one of a set of choices.

    Parameters
    ----------
    param : str
        Name of the parameter to validate.
    choices : sequence
        Valid values for the parameter.
    allow_none : bool, default=False
        Whether None is a valid value.
    name : str, optional
        Human-readable name for error messages.

    Examples
    --------
    >>> @validate_one_of('method', ['gcv', 'reml', 'ml'])
    >>> def fit(X, y, method='gcv'):
    ...     pass
    """
    display_name = name or param

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            value = bound.arguments.get(param)

            if value is None:
                if allow_none:
                    return func(*args, **kwargs)
                raise ValidationError(param, f"{display_name} cannot be None")

            if value not in choices:
                choices_str = ", ".join(repr(c) for c in choices)
                raise ValidationError(
                    param,
                    f"{display_name} must be one of [{choices_str}], got {value!r}",
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Composite Decorator
# =============================================================================


def validated(
    **validations: dict[str, Any],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Composite decorator for multiple validations.

    Parameters
    ----------
    **validations : dict
        Mapping of parameter names to validation specifications.
        Each specification is a dict with validation options.

    Examples
    --------
    >>> @validated(
    ...     X={'type': 'array', 'ndim': 2, 'dtype': 'numeric'},
    ...     y={'type': 'array', 'ndim': 1},
    ...     alpha={'type': 'positive'},
    ...     method={'choices': ['gcv', 'reml']}
    ... )
    >>> def fit(X, y, alpha=1.0, method='gcv'):
    ...     pass
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        decorated = func

        for param, spec in reversed(validations.items()):
            if not isinstance(spec, dict):
                raise ValueError(f"Validation spec for '{param}' must be a dict")

            val_type = spec.get("type", "any")

            if val_type == "array":
                decorated = validate_array(
                    param,
                    ndim=spec.get("ndim"),
                    min_rows=spec.get("min_rows"),
                    min_cols=spec.get("min_cols"),
                    dtype_check=spec.get("dtype"),
                    check_finite=spec.get("check_finite", True),
                    allow_none=spec.get("allow_none", False),
                )(decorated)

            elif val_type == "positive":
                decorated = validate_positive(
                    param,
                    strict=spec.get("strict", True),
                    allow_none=spec.get("allow_none", False),
                )(decorated)

            elif val_type == "probability":
                decorated = validate_probability(
                    param,
                    allow_zero=spec.get("allow_zero", True),
                    allow_one=spec.get("allow_one", True),
                    allow_none=spec.get("allow_none", False),
                )(decorated)

            elif val_type == "range":
                decorated = validate_in_range(
                    param,
                    lower=spec.get("lower"),
                    upper=spec.get("upper"),
                    inclusive=spec.get("inclusive", "both"),
                    allow_none=spec.get("allow_none", False),
                )(decorated)

            elif "choices" in spec:
                decorated = validate_one_of(
                    param,
                    spec["choices"],
                    allow_none=spec.get("allow_none", False),
                )(decorated)

        return decorated

    return decorator


__all__ = [
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
