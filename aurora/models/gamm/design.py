"""Design matrix construction for random effects.

This module provides functions to construct the random effects design
matrix Z from data and RandomEffect specifications.

References
----------
.. [1] Pinheiro & Bates (2000). Mixed-Effects Models in S and S-PLUS.
.. [2] Wood (2017). Generalized Additive Models: An Introduction with R, 2nd ed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aurora.models.gamm.random_effects import RandomEffect, get_group_indices, validate_random_effects


def construct_Z_matrix(
    X: np.ndarray,
    random_effects: list[RandomEffect],
    groups_data: dict[str | int, np.ndarray],
) -> tuple[np.ndarray, list[dict]]:
    """Construct random effects design matrix Z.

    For each random effect term, Z has a block for each group. Within
    each block, columns correspond to the random effects (intercept,
    slopes) for that group.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Fixed effects design matrix (including intercept if needed)
    random_effects : list of RandomEffect
        Random effect specifications
    groups_data : dict
        Dictionary mapping grouping variable -> group indicators array.
        Keys should match RandomEffect.grouping values.

    Returns
    -------
    Z : ndarray, shape (n, q)
        Random effects design matrix where q = sum over terms of
        (n_groups_i * n_effects_i)
    Z_info : list of dict
        List of dictionaries (one per random effect term) containing:
        - 'grouping': grouping variable
        - 'n_effects': number of random effects per group
        - 'n_groups': number of groups
        - 'groups': array of unique group values
        - 'start_col': starting column in Z
        - 'end_col': ending column in Z (exclusive)

    Examples
    --------
    >>> n = 6
    >>> X = np.ones((n, 1))  # Intercept only
    >>> groups = np.array([1, 1, 2, 2, 3, 3])
    >>> groups_data = {'subject': groups}
    >>>
    >>> # Random intercept: (1 | subject)
    >>> re = RandomEffect(grouping='subject')
    >>> Z, info = construct_Z_matrix(X, [re], groups_data)
    >>> Z.shape
    (6, 3)  # 3 groups, 1 effect per group
    >>> Z
    array([[1., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [0., 0., 1.]])

    >>> # Random intercept + slope: (1 + x1 | subject)
    >>> X_with_x1 = np.column_stack([np.ones(n), np.arange(n)])
    >>> re = RandomEffect(grouping='subject', variables=(1,))  # x1 is column 1
    >>> Z, info = construct_Z_matrix(X_with_x1, [re], groups_data)
    >>> Z.shape
    (6, 6)  # 3 groups, 2 effects per group

    Notes
    -----
    Z is constructed as a block diagonal matrix where each block corresponds
    to one group in one random effect term:

    For random intercept (1 | group):
        Z_ij = 1 if observation i is in group j, else 0

    For random intercept + slope (1 + x | group):
        Z has two columns per group: [1, x_i] for each observation i
        in that group, zero elsewhere.

    Multiple random effect terms are concatenated horizontally.
    """
    validate_random_effects(random_effects)
    n = X.shape[0]

    # Check that all grouping variables are provided
    for re in random_effects:
        if re.grouping not in groups_data:
            raise ValueError(
                f"Grouping variable '{re.grouping}' not found in groups_data. "
                f"Available: {list(groups_data.keys())}"
            )

    # Build Z blocks for each random effect term
    Z_blocks = []
    Z_info = []
    current_col = 0

    for re in random_effects:
        # Get group indicators for this term
        groups = groups_data[re.grouping]
        if len(groups) != n:
            raise ValueError(
                f"Length of groups for '{re.grouping}' ({len(groups)}) "
                f"does not match number of observations ({n})"
            )

        # Get unique groups and indices
        unique_groups, group_indices = get_group_indices(groups)
        n_groups = len(unique_groups)

        # Determine if this is a temporal covariance structure
        is_temporal = re.covariance in ('ar1', 'compound_symmetry', 'cs')

        if is_temporal:
            # For temporal covariance, each observation within a group gets its own random effect
            # This requires balanced design (same number of observations per group)
            group_sizes = [len(group_indices[g]) for g in unique_groups]
            if len(set(group_sizes)) > 1:
                raise ValueError(
                    f"Temporal covariance '{re.covariance}' requires balanced design. "
                    f"Found varying group sizes: {set(group_sizes)}. "
                    f"All groups in '{re.grouping}' must have the same number of observations."
                )
            n_effects = group_sizes[0]  # Number of time points per group
        else:
            n_effects = re.n_effects  # Standard: intercept + slopes

        # Construct Z block for this random effect term
        Z_block = np.zeros((n, n_groups * n_effects))

        # Fill in Z block
        for g_idx, group_val in enumerate(unique_groups):
            obs_indices = group_indices[group_val]
            col_start = g_idx * n_effects
            col_end = col_start + n_effects

            if is_temporal:
                # Temporal: each observation gets indicator in its temporal position
                # Assumes observations are ordered temporally within each group
                for t_idx, obs_idx in enumerate(obs_indices):
                    Z_block[obs_idx, col_start + t_idx] = 1.0
            else:
                # Standard: intercept + slopes
                col_idx = 0

                # Random intercept
                if re.include_intercept:
                    Z_block[obs_indices, col_start + col_idx] = 1.0
                    col_idx += 1

                # Random slopes
                for var_idx in re.variables:
                    # Extract variable values from X
                    if isinstance(var_idx, int):
                        if var_idx >= X.shape[1]:
                            raise ValueError(
                                f"Variable index {var_idx} out of bounds for X with {X.shape[1]} columns"
                            )
                        var_values = X[:, var_idx]
                    else:
                        raise TypeError(
                            f"Variable index must be int when using design matrix X, got {type(var_idx)}"
                        )

                    Z_block[obs_indices, col_start + col_idx] = var_values[obs_indices]
                    col_idx += 1

        # Store info
        Z_blocks.append(Z_block)
        Z_info.append({
            'grouping': re.grouping,
            'n_effects': n_effects,
            'n_groups': n_groups,
            'groups': unique_groups,
            'start_col': current_col,
            'end_col': current_col + n_groups * n_effects,
            'covariance': re.covariance,  # Store covariance structure type
        })
        current_col += n_groups * n_effects

    # Concatenate all Z blocks horizontally
    if len(Z_blocks) == 0:
        # No random effects
        Z = np.zeros((n, 0))
    elif len(Z_blocks) == 1:
        Z = Z_blocks[0]
    else:
        Z = np.concatenate(Z_blocks, axis=1)

    return Z, Z_info


def extract_random_effects(
    b: np.ndarray,
    Z_info: list[dict],
) -> dict[str | int, dict[int, np.ndarray]]:
    """Extract random effect coefficients by group.

    Parameters
    ----------
    b : ndarray, shape (q,)
        Random effect coefficients (BLUPs) from model fitting
    Z_info : list of dict
        Information about Z matrix structure (from construct_Z_matrix)

    Returns
    -------
    random_effects : dict
        Dictionary mapping grouping_variable -> {group_id -> coefficients}.
        For each group, coefficients are [intercept, slope1, slope2, ...]
        depending on the random effect specification.

    Examples
    --------
    >>> # For (1 | subject) with 3 groups
    >>> b = np.array([0.5, -0.3, 0.2])
    >>> Z_info = [{
    ...     'grouping': 'subject',
    ...     'n_effects': 1,
    ...     'n_groups': 3,
    ...     'groups': np.array([1, 2, 3]),
    ...     'start_col': 0,
    ...     'end_col': 3,
    ... }]
    >>> random_effects = extract_random_effects(b, Z_info)
    >>> random_effects['subject'][1]
    array([0.5])
    >>> random_effects['subject'][2]
    array([-0.3])

    Notes
    -----
    This function is the inverse of construct_Z_matrix: it takes the
    vector of random effect coefficients and organizes them by group
    for interpretation and prediction.
    """
    random_effects = {}

    for info in Z_info:
        grouping = info['grouping']
        n_effects = info['n_effects']
        n_groups = info['n_groups']
        groups = info['groups']
        start_col = info['start_col']

        # Extract coefficients for this random effect term
        b_term = b[start_col:start_col + n_groups * n_effects]

        # Reshape to (n_groups, n_effects)
        b_reshaped = b_term.reshape(n_groups, n_effects)

        # Create dictionary for this grouping variable
        random_effects[grouping] = {}
        for g_idx, group_val in enumerate(groups):
            random_effects[grouping][group_val] = b_reshaped[g_idx, :]

    return random_effects


__all__ = [
    'construct_Z_matrix',
    'extract_random_effects',
]
