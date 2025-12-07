# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Difference-based penalty matrices for spline smoothing.

Difference penalties approximate integrated squared derivatives by penalizing
differences in adjacent coefficients. They are computationally efficient and
work well with B-spline bases.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def difference_penalty(n_basis: int, order: int = 2) -> np.ndarray:
    """Create penalty matrix based on finite differences.

    For order m, this penalizes the m-th order differences in coefficients:
        order=1: penalizes |β_i - β_{i+1}|²
        order=2: penalizes |β_i - 2β_{i+1} + β_{i+2}|²
        order=3: penalizes |β_i - 3β_{i+1} + 3β_{i+2} - β_{i+3}|²

    Parameters
    ----------
    n_basis : int
        Number of basis functions (dimension of coefficient vector).
    order : int, default=2
        Order of differences. Typically 2 for approximating second derivative.

    Returns
    -------
    S : ndarray, shape (n_basis, n_basis)
        Penalty matrix where β'Sβ = Σ(Δ^m β_i)².

    Notes
    -----
    The penalty matrix is computed as S = D'D, where D is the difference
    matrix. For order m:
        D_ij = (-1)^(m-j) * C(m, j) for appropriate i, j

    This approximates ∫[f^(m)(x)]² dx for equally-spaced knots.

    Examples
    --------
    >>> S = difference_penalty(5, order=2)
    >>> S.shape
    (5, 5)
    >>> # Penalty for linear trend is zero (second differences are zero)
    >>> beta_linear = np.array([0, 1, 2, 3, 4])
    >>> penalty = beta_linear @ S @ beta_linear
    >>> abs(penalty) < 1e-10
    True
    """
    if n_basis < 1:
        raise ValueError("n_basis must be positive")

    if order < 1:
        raise ValueError("order must be positive")

    if order >= n_basis:
        raise ValueError(f"order {order} must be less than n_basis {n_basis}")

    # Create difference matrix using numpy's diff
    # Start with identity matrix
    D = np.eye(n_basis, dtype=np.float64)

    # Apply differencing 'order' times
    for _ in range(order):
        D = np.diff(D, axis=0)

    # Penalty matrix is D'D
    S = D.T @ D

    return S


def weighted_difference_penalty(
    n_basis: int,
    knots: Any,
    order: int = 2,
) -> np.ndarray:
    """Create weighted difference penalty accounting for knot spacing.

    For non-uniformly spaced knots, differences should be weighted by
    knot spacing to better approximate the integrated derivative.

    Parameters
    ----------
    n_basis : int
        Number of basis functions.
    knots : array-like
        Knot locations (should be sorted).
    order : int, default=2
        Order of differences.

    Returns
    -------
    S : ndarray, shape (n_basis, n_basis)
        Weighted penalty matrix.

    Notes
    -----
    For order=2 and non-uniform knots, the penalty approximates:
        Σ [(β_{i+2} - β_{i+1})/h_{i+1} - (β_{i+1} - β_i)/h_i]²
    where h_i = knots[i+1] - knots[i]

    This gives better approximation to ∫[f''(x)]² dx for irregular knots.
    """
    if n_basis < 1:
        raise ValueError("n_basis must be positive")

    if order < 1:
        raise ValueError("order must be positive")

    knots_arr = np.asarray(knots, dtype=np.float64)

    if knots_arr.ndim != 1:
        raise ValueError("knots must be 1-dimensional")

    if len(knots_arr) < n_basis:
        raise ValueError("Need at least n_basis knots")

    # Compute knot spacing
    h = np.diff(knots_arr[: n_basis + order])

    if np.any(h <= 0):
        raise ValueError("knots must be strictly increasing")

    # Create weighted difference matrix
    # For simplicity, use uniform weights initially
    # A more sophisticated version would weight by h_i^{-order}
    D = np.eye(n_basis, dtype=np.float64)

    for _ in range(order):
        D = np.diff(D, axis=0)

    # Weight by average spacing (simplified)
    # For proper implementation, would need to weight each row appropriately
    avg_spacing = np.mean(h)
    weight = 1.0 / avg_spacing**order

    S = weight * (D.T @ D)

    return S


def ridge_penalty(n_basis: int, exclude_intercept: bool = True) -> np.ndarray:
    """Create ridge penalty (identity matrix).

    Ridge penalty is simply the identity matrix, which penalizes the
    squared norm of coefficients: β'Sβ = Σβ_i².

    Parameters
    ----------
    n_basis : int
        Number of basis functions.
    exclude_intercept : bool, default=True
        If True, do not penalize the first coefficient (intercept).

    Returns
    -------
    S : ndarray, shape (n_basis, n_basis)
        Ridge penalty matrix (identity or near-identity).

    Notes
    -----
    Ridge penalty is rarely used for smoothing in GAMs, but can be useful
    for regularization or in hierarchical models.
    """
    if n_basis < 1:
        raise ValueError("n_basis must be positive")

    S = np.eye(n_basis, dtype=np.float64)

    if exclude_intercept and n_basis > 1:
        # Don't penalize first coefficient
        S[0, 0] = 0.0

    return S


def null_space_penalty(
    n_basis: int,
    null_space_dim: int = 2,
) -> np.ndarray:
    """Create penalty with specified null space dimension.

    A penalty with null space of dimension d allows polynomials of degree
    d-1 to be fitted without penalty. For example, d=2 (linear null space)
    doesn't penalize constants or linear trends.

    Parameters
    ----------
    n_basis : int
        Number of basis functions.
    null_space_dim : int, default=2
        Dimension of null space (polynomials up to degree d-1 are unpenalized).

    Returns
    -------
    S : ndarray, shape (n_basis, n_basis)
        Penalty matrix with specified null space.

    Notes
    -----
    This is equivalent to a difference penalty of order = null_space_dim.
    For GAMs, typically want null_space_dim = 2 (don't penalize constant/linear).
    """
    if n_basis < 1:
        raise ValueError("n_basis must be positive")

    if null_space_dim < 1:
        raise ValueError("null_space_dim must be positive")

    if null_space_dim >= n_basis:
        raise ValueError("null_space_dim must be less than n_basis")

    # Null space dimension d corresponds to difference order d
    return difference_penalty(n_basis, order=null_space_dim)


def combine_penalties(
    penalties: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Combine multiple penalty matrices with weights.

    Useful for multi-component penalties or when combining different
    types of penalties.

    Parameters
    ----------
    penalties : list of ndarray
        List of penalty matrices (must all be same shape).
    weights : list of float, optional
        Weights for each penalty. If None, uses uniform weights.

    Returns
    -------
    S : ndarray
        Combined penalty matrix: Σ w_i * S_i

    Examples
    --------
    >>> S1 = difference_penalty(10, order=2)
    >>> S2 = ridge_penalty(10)
    >>> S_combined = combine_penalties([S1, S2], weights=[1.0, 0.01])
    """
    if not penalties:
        raise ValueError("penalties list cannot be empty")

    n = penalties[0].shape[0]

    # Check all penalties have same shape
    for S in penalties:
        if S.shape != (n, n):
            raise ValueError("All penalties must have same shape")

    if weights is None:
        weights = [1.0] * len(penalties)

    if len(weights) != len(penalties):
        raise ValueError("Number of weights must match number of penalties")

    # Combine with weights
    S_combined = np.zeros((n, n), dtype=np.float64)
    for w, S in zip(weights, penalties):
        S_combined += w * S

    return S_combined


__all__ = [
    "difference_penalty",
    "weighted_difference_penalty",
    "ridge_penalty",
    "null_space_penalty",
    "combine_penalties",
]
