# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Tensor product smooths for modeling interactions between variables.

Tensor product smooths allow for flexible modeling of interactions between
two or more variables. They construct a smooth surface by taking the tensor
product of marginal basis functions.

References
----------
Wood, S.N. (2017). Generalized Additive Models: An Introduction with R.
    Chapman and Hall/CRC, 2nd edition. Chapter 5.
"""

from __future__ import annotations

import numpy as np


def tensor_product_basis(
    X1: np.ndarray,
    X2: np.ndarray,
    basis1: object,
    basis2: object,
) -> np.ndarray:
    """Compute tensor product basis matrix.

    Creates a tensor product basis by multiplying basis functions from
    two marginal bases: B(x₁, x₂) = B₁(x₁) ⊗ B₂(x₂)

    Parameters
    ----------
    X1 : ndarray, shape (n,)
        First variable values.
    X2 : ndarray, shape (n,)
        Second variable values.
    basis1 : object
        First marginal basis (must have basis_matrix method).
    basis2 : object
        Second marginal basis (must have basis_matrix method).

    Returns
    -------
    B : ndarray, shape (n, n_basis1 * n_basis2)
        Tensor product basis matrix.

    Notes
    -----
    For each observation i, the tensor product basis is:
        B[i, :] = vec(B₁[i, :] ⊗ B₂[i, :])

    where ⊗ denotes the outer product.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.smoothing.splines.bspline import BSplineBasis
    >>> from aurora.smoothing.tensor import tensor_product_basis
    >>> # Create marginal bases
    >>> x1 = np.linspace(0, 1, 100)
    >>> x2 = np.linspace(0, 1, 100)
    >>> knots1 = BSplineBasis.create_knots(x1, n_basis=8, degree=3)
    >>> knots2 = BSplineBasis.create_knots(x2, n_basis=8, degree=3)
    >>> basis1 = BSplineBasis(knots1, degree=3)
    >>> basis2 = BSplineBasis(knots2, degree=3)
    >>> # Compute tensor product
    >>> B_tensor = tensor_product_basis(x1, x2, basis1, basis2)
    >>> B_tensor.shape
    (100, 64)
    """
    # Compute marginal basis matrices
    B1 = basis1.basis_matrix(X1)  # (n, p1)
    B2 = basis2.basis_matrix(X2)  # (n, p2)

    n = B1.shape[0]
    p1 = B1.shape[1]
    p2 = B2.shape[1]

    # Compute tensor product
    # For each row i: B_tensor[i, :] = vec(B1[i, :] ⊗ B2[i, :])
    B_tensor = np.zeros((n, p1 * p2))

    for i in range(n):
        # Outer product of basis functions
        outer = np.outer(B1[i, :], B2[i, :])
        # Flatten to vector
        B_tensor[i, :] = outer.ravel()

    return B_tensor


def tensor_product_penalty(
    S1: np.ndarray,
    S2: np.ndarray,
    p1: int,
    p2: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute tensor product penalty matrices.

    For tensor products, we need two penalty matrices:
    - S_x1: penalizes variation in x₁ direction
    - S_x2: penalizes variation in x₂ direction

    Parameters
    ----------
    S1 : ndarray, shape (p1, p1)
        Penalty matrix for first marginal basis.
    S2 : ndarray, shape (p2, p2)
        Penalty matrix for second marginal basis.
    p1 : int
        Number of basis functions in first marginal basis.
    p2 : int
        Number of basis functions in second marginal basis.

    Returns
    -------
    S_x1 : ndarray, shape (p1*p2, p1*p2)
        Penalty matrix for variation in x₁ direction.
    S_x2 : ndarray, shape (p1*p2, p1*p2)
        Penalty matrix for variation in x₂ direction.

    Notes
    -----
    The tensor product penalty is:
        S_x1 = S1 ⊗ I2  (penalizes wiggliness in x₁)
        S_x2 = I1 ⊗ S2  (penalizes wiggliness in x₂)

    where ⊗ is the Kronecker product.

    The full penalty is: λ₁ β' S_x1 β + λ₂ β' S_x2 β

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.smoothing.tensor import tensor_product_penalty
    >>> # Create marginal penalties
    >>> S1 = np.eye(5)
    >>> S2 = np.eye(6)
    >>> # Compute tensor penalties
    >>> S_x1, S_x2 = tensor_product_penalty(S1, S2, 5, 6)
    >>> S_x1.shape
    (30, 30)
    >>> S_x2.shape
    (30, 30)
    """
    # Identity matrices
    I1 = np.eye(p1)
    I2 = np.eye(p2)

    # Kronecker products
    # S_x1 = S1 ⊗ I2: penalizes variation in x₁
    S_x1 = np.kron(S1, I2)

    # S_x2 = I1 ⊗ S2: penalizes variation in x₂
    S_x2 = np.kron(I1, S2)

    return S_x1, S_x2


def fit_tensor_product(
    X1: np.ndarray,
    X2: np.ndarray,
    y: np.ndarray,
    basis1: object,
    basis2: object,
    S1: np.ndarray,
    S2: np.ndarray,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Fit a tensor product smooth.

    Parameters
    ----------
    X1 : ndarray, shape (n,)
        First variable values.
    X2 : ndarray, shape (n,)
        Second variable values.
    y : ndarray, shape (n,)
        Response variable.
    basis1 : object
        First marginal basis.
    basis2 : object
        Second marginal basis.
    S1 : ndarray, shape (p1, p1)
        Penalty matrix for first marginal basis.
    S2 : ndarray, shape (p2, p2)
        Penalty matrix for second marginal basis.
    lambda1 : float, default=1.0
        Smoothing parameter for x₁ direction.
    lambda2 : float, default=1.0
        Smoothing parameter for x₂ direction.
    weights : ndarray, shape (n,), optional
        Observation weights.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'coefficients': Coefficient estimates
        - 'fitted_values': Fitted values
        - 'basis_matrix': Tensor product basis matrix
        - 'edf': Effective degrees of freedom

    Notes
    -----
    Minimizes: ||W^(1/2)(y - Bβ)||² + λ₁ β' S_x1 β + λ₂ β' S_x2 β

    where B is the tensor product basis matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.smoothing.splines.bspline import BSplineBasis
    >>> from aurora.smoothing.tensor import fit_tensor_product
    >>> # Generate data
    >>> n = 200
    >>> x1 = np.random.uniform(0, 1, n)
    >>> x2 = np.random.uniform(0, 1, n)
    >>> y = np.sin(2*np.pi*x1) * np.cos(2*np.pi*x2) + 0.1*np.random.randn(n)
    >>> # Create bases
    >>> knots1 = BSplineBasis.create_knots(x1, n_basis=10, degree=3)
    >>> knots2 = BSplineBasis.create_knots(x2, n_basis=10, degree=3)
    >>> basis1 = BSplineBasis(knots1, degree=3)
    >>> basis2 = BSplineBasis(knots2, degree=3)
    >>> S1 = basis1.penalty_matrix(order=2)
    >>> S2 = basis2.penalty_matrix(order=2)
    >>> # Fit tensor product
    >>> result = fit_tensor_product(x1, x2, y, basis1, basis2, S1, S2)
    >>> result['fitted_values'].shape
    (200,)
    """
    n = len(y)

    # Compute tensor product basis
    B = tensor_product_basis(X1, X2, basis1, basis2)

    # Get dimensions
    p = B.shape[1]
    p1 = basis1.basis_matrix(X1[:1]).shape[1]
    p2 = basis2.basis_matrix(X2[:1]).shape[1]

    # Compute tensor product penalties
    S_x1, S_x2 = tensor_product_penalty(S1, S2, p1, p2)

    # Combined penalty
    S_total = lambda1 * S_x1 + lambda2 * S_x2

    # Weight matrix
    if weights is None:
        W = np.eye(n)
    else:
        W = np.diag(weights)

    # Solve penalized least squares
    # (B'WB + S_total) β = B'Wy
    BtWB = B.T @ W @ B
    BtWy = B.T @ W @ y
    A = BtWB + S_total

    coefficients = np.linalg.solve(A, BtWy)
    fitted_values = B @ coefficients

    # Compute effective degrees of freedom
    # EDF = trace(B (B'WB + S)^(-1) B'W)
    try:
        A_inv = np.linalg.inv(A)
        H = B @ A_inv @ B.T @ W
        edf = float(np.trace(H))
    except np.linalg.LinAlgError:
        edf = float("nan")

    return {
        "coefficients": coefficients,
        "fitted_values": fitted_values,
        "basis_matrix": B,
        "edf": edf,
    }


__all__ = [
    "tensor_product_basis",
    "tensor_product_penalty",
    "fit_tensor_product",
]
