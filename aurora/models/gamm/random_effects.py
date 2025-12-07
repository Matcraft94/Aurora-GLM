# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Random effects structures for GAMMs.

This module provides dataclasses and utilities for specifying and working with
random effects in Generalized Additive Mixed Models.

References
----------
.. [1] Pinheiro & Bates (2000). Mixed-Effects Models in S and S-PLUS.
.. [2] Wood (2017). Generalized Additive Models: An Introduction with R, 2nd ed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class RandomEffect:
    """Specification of a random effect term.

    Random effects model within-group correlation in hierarchical or
    longitudinal data. Each random effect term specifies:
    - Which variables have random coefficients
    - What grouping structure to use
    - Whether to include a random intercept
    - The covariance structure among random effects

    Parameters
    ----------
    grouping : str or int
        Grouping variable name or column index. All observations with
        the same value of this variable belong to the same group.
    variables : tuple of (str or int), optional
        Variables with random slopes. If empty, only includes random
        intercept (if include_intercept=True).
    include_intercept : bool, default=True
        Whether to include a random intercept (1 | group).
    covariance : str, default='unstructured'
        Covariance structure among random effects. Options:
        - 'unstructured': Full covariance matrix (q(q+1)/2 parameters)
        - 'diagonal': Independent random effects (q parameters)
        - 'identity': Equal variances, no correlation (1 parameter)
        - 'ar1': Autoregressive AR(1) for temporal correlation
        - 'compound_symmetry' or 'cs': Exchangeable correlation (ICC model)
        - 'exponential': Spatial exponential decay (requires coordinates)
        - 'matern': Matérn spatial covariance (requires coordinates)

        For temporal/longitudinal data, use 'ar1'. For clustered data with
        equal within-cluster correlation, use 'compound_symmetry'.

    Attributes
    ----------
    n_effects : int
        Total number of random effects (intercept + slopes)

    Examples
    --------
    >>> # Random intercept: (1 | subject)
    >>> re1 = RandomEffect(grouping='subject')
    >>> re1.n_effects
    1

    >>> # Random intercept + slope: (1 + time | subject)
    >>> re2 = RandomEffect(grouping='subject', variables=('time',))
    >>> re2.n_effects
    2

    >>> # Random slopes only: (time + age | subject)
    >>> re3 = RandomEffect(
    ...     grouping='subject',
    ...     variables=('time', 'age'),
    ...     include_intercept=False
    ... )
    >>> re3.n_effects
    2

    >>> # Diagonal covariance (independent effects)
    >>> re4 = RandomEffect(
    ...     grouping='subject',
    ...     variables=('time',),
    ...     covariance='diagonal'
    ... )

    Notes
    -----
    The total number of random effects is:
        q = (1 if include_intercept else 0) + len(variables)

    Nested random effects like (1 | clinic/subject) should be specified
    as two separate RandomEffect instances with appropriate grouping.

    Crossed random effects like (1 | subject) + (1 | clinic) are
    specified as multiple RandomEffect instances in a list.
    """

    grouping: str | int
    variables: tuple[str | int, ...] = field(default_factory=tuple)
    include_intercept: bool = True
    covariance: Literal[
        "unstructured",
        "diagonal",
        "identity",
        "ar1",
        "compound_symmetry",
        "cs",
        "exponential",
        "matern",
    ] = "unstructured"

    def __post_init__(self):
        """Validate random effect specification."""
        # Check grouping is specified
        if self.grouping is None:
            raise ValueError("grouping must be specified")

        # Validate covariance structure
        valid_cov = {
            "unstructured",
            "diagonal",
            "identity",
            "ar1",
            "compound_symmetry",
            "cs",  # Temporal correlation
            "exponential",
            "matern",  # Spatial correlation
        }
        if self.covariance not in valid_cov:
            raise ValueError(
                f"covariance must be one of {valid_cov}, got '{self.covariance}'"
            )

        # Check that we have at least one effect
        if not self.include_intercept and len(self.variables) == 0:
            raise ValueError(
                "Random effect must include either intercept or at least one variable. "
                "Set include_intercept=True or provide variables."
            )

        # Ensure variables is a tuple
        if not isinstance(self.variables, tuple):
            if isinstance(self.variables, (list, str, int)):
                # Convert to tuple
                if isinstance(self.variables, (str, int)):
                    self.variables = (self.variables,)
                else:
                    self.variables = tuple(self.variables)
            else:
                raise TypeError(
                    f"variables must be tuple, list, str, or int, got {type(self.variables)}"
                )

    @property
    def n_effects(self) -> int:
        """Number of random effects in this term.

        Returns
        -------
        int
            Number of random effects (intercept + slopes)
        """
        n = len(self.variables)
        if self.include_intercept:
            n += 1
        return n

    def __repr__(self) -> str:
        """String representation."""
        # Build R-style formula representation
        parts = []
        if self.include_intercept:
            parts.append("1")
        parts.extend(str(v) for v in self.variables)

        formula = " + ".join(parts)
        result = f"({formula} | {self.grouping})"

        if self.covariance != "unstructured":
            result += f" [{self.covariance}]"

        return f"RandomEffect({result})"


def validate_random_effects(
    random_effects: list[RandomEffect],
) -> None:
    """Validate a list of random effects specifications.

    Parameters
    ----------
    random_effects : list of RandomEffect
        Random effects to validate

    Raises
    ------
    TypeError
        If random_effects is not a list or contains non-RandomEffect items
    ValueError
        If random effects specifications are invalid

    Examples
    --------
    >>> re1 = RandomEffect(grouping='subject')
    >>> re2 = RandomEffect(grouping='clinic')
    >>> validate_random_effects([re1, re2])  # OK

    >>> validate_random_effects([re1, "invalid"])  # Raises TypeError
    Traceback (most recent call last):
        ...
    TypeError: All items in random_effects must be RandomEffect instances

    Notes
    -----
    This function checks:
    - All items are RandomEffect instances
    - Each RandomEffect is internally valid (via __post_init__)
    """
    if not isinstance(random_effects, list):
        raise TypeError(f"random_effects must be a list, got {type(random_effects)}")

    for i, re in enumerate(random_effects):
        if not isinstance(re, RandomEffect):
            raise TypeError(
                f"All items in random_effects must be RandomEffect instances. "
                f"Item {i} is {type(re)}"
            )


def get_group_indices(
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Get unique groups and indices for each group.

    Parameters
    ----------
    groups : ndarray, shape (n,)
        Group indicators (integers)

    Returns
    -------
    unique_groups : ndarray
        Sorted array of unique group values
    group_indices : dict
        Dictionary mapping group_id -> array of observation indices

    Examples
    --------
    >>> groups = np.array([1, 1, 2, 2, 2, 3])
    >>> unique, indices = get_group_indices(groups)
    >>> unique
    array([1, 2, 3])
    >>> indices[1]
    array([0, 1])
    >>> indices[2]
    array([2, 3, 4])
    """
    unique_groups = np.unique(groups)
    group_indices = {}

    for group in unique_groups:
        group_indices[group] = np.where(groups == group)[0]

    return unique_groups, group_indices


def count_random_effects(
    random_effects: list[RandomEffect],
) -> tuple[int, dict[str | int, int]]:
    """Count total number of random effect coefficients.

    Parameters
    ----------
    random_effects : list of RandomEffect
        Random effects specifications

    Returns
    -------
    total_effects : int
        Total number of random effect coefficients across all terms
    effects_per_term : dict
        Dictionary mapping grouping variable -> number of effects

    Examples
    --------
    >>> re1 = RandomEffect(grouping='subject', variables=('time',))
    >>> re2 = RandomEffect(grouping='clinic')
    >>> total, per_term = count_random_effects([re1, re2])
    >>> total
    3
    >>> per_term['subject']
    2
    >>> per_term['clinic']
    1

    Notes
    -----
    For each random effect term with q effects and G groups, we have
    q * G random coefficients in total.
    """
    validate_random_effects(random_effects)

    total = 0
    effects_per_term = {}

    for re in random_effects:
        n = re.n_effects
        total += n
        effects_per_term[re.grouping] = n

    return total, effects_per_term


__all__ = [
    "RandomEffect",
    "validate_random_effects",
    "get_group_indices",
    "count_random_effects",
]
