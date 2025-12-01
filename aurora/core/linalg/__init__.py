"""Linear algebra primitives with multi-backend support.

This module provides backend-agnostic linear algebra operations
for NumPy, PyTorch, and JAX, with robust fallbacks.

Decompositions
--------------
- QR decomposition (with column pivoting option)
- Cholesky decomposition (with safe fallback)
- SVD (singular value decomposition)
- Eigendecomposition for symmetric matrices

Solvers
-------
- solve_triangular: Efficient triangular system solver
- solve_cholesky: Solve via Cholesky factorization
- solve_qr: Solve via QR decomposition
- lstsq: Least squares solution

Matrix Operations
-----------------
- safe_inverse: Matrix inverse with regularization
- woodbury_inverse: Efficient update of inverse
- quadratic_form: x'Ax computation
- log_determinant: Log of matrix determinant

Examples
--------
>>> from aurora.core.linalg import safe_cholesky, solve_cholesky
>>> 
>>> # Robust Cholesky with fallback
>>> L = safe_cholesky(A)
>>> x = solve_cholesky(L, b)
>>> 
>>> # QR for least squares
>>> Q, R = qr_decomposition(X)
>>> beta = solve_qr(Q, R, y)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Decompositions
# =============================================================================

def qr_decomposition(
    A: np.ndarray,
    *,
    mode: Literal['reduced', 'complete', 'r', 'raw'] = 'reduced',
    pivoting: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """QR decomposition with optional column pivoting.
    
    Decomposes A = QR where Q is orthogonal and R is upper triangular.
    With pivoting, AP = QR where P is a permutation matrix.
    
    Parameters
    ----------
    A : ndarray of shape (m, n)
        Matrix to decompose.
    mode : {'reduced', 'complete', 'r', 'raw'}, default='reduced'
        - 'reduced': Return Q (m, k) and R (k, n), where k = min(m, n)
        - 'complete': Return Q (m, m) and R (m, n)
        - 'r': Return only R
        - 'raw': Return (H, tau) for advanced use
    pivoting : bool, default=False
        If True, use column pivoting and return permutation indices.
        
    Returns
    -------
    Q : ndarray
        Orthogonal matrix (unless mode='r' or 'raw').
    R : ndarray
        Upper triangular matrix.
    P : ndarray (only if pivoting=True)
        Permutation indices such that A[:, P] = Q @ R.
        
    Examples
    --------
    >>> A = np.array([[1, 2], [3, 4], [5, 6]])
    >>> Q, R = qr_decomposition(A)
    >>> np.allclose(A, Q @ R)
    True
    
    >>> # With pivoting for rank-deficient matrices
    >>> Q, R, P = qr_decomposition(A, pivoting=True)
    >>> np.allclose(A[:, P], Q @ R)
    True
    """
    if pivoting:
        from scipy.linalg import qr as scipy_qr
        Q, R, P = scipy_qr(A, mode='economic', pivoting=True)
        return Q, R, P
    else:
        return np.linalg.qr(A, mode=mode)


def safe_cholesky(
    A: np.ndarray,
    *,
    lower: bool = True,
    max_tries: int = 5,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Cholesky decomposition with robustness to near-singular matrices.
    
    Attempts standard Cholesky, and if it fails, adds progressively
    larger diagonal jitter until successful.
    
    Parameters
    ----------
    A : ndarray of shape (n, n)
        Symmetric positive semi-definite matrix.
    lower : bool, default=True
        If True, return lower triangular L. If False, return upper.
    max_tries : int, default=5
        Maximum number of attempts with increasing jitter.
    jitter : float, default=1e-6
        Initial diagonal jitter to add on failure.
        
    Returns
    -------
    L : ndarray of shape (n, n)
        Cholesky factor such that A ≈ L @ L.T.
        
    Raises
    ------
    np.linalg.LinAlgError
        If decomposition fails after max_tries attempts.
        
    Examples
    --------
    >>> A = np.array([[4, 2], [2, 1.001]])  # Nearly singular
    >>> L = safe_cholesky(A)
    >>> np.allclose(A, L @ L.T, atol=1e-5)
    True
    """
    A = np.asarray(A)
    
    for i in range(max_tries):
        try:
            if i > 0:
                # Add jitter to diagonal
                A = A + (jitter * (10 ** (i - 1))) * np.eye(A.shape[0])
            
            L = np.linalg.cholesky(A)
            
            if not lower:
                L = L.T
            
            return L
            
        except np.linalg.LinAlgError:
            if i == max_tries - 1:
                raise np.linalg.LinAlgError(
                    f"Cholesky decomposition failed after {max_tries} attempts "
                    f"with jitter up to {jitter * (10 ** (i - 1)):.2e}"
                )
    
    # Should not reach here
    raise np.linalg.LinAlgError("Cholesky decomposition failed")


def svd(
    A: np.ndarray,
    *,
    full_matrices: bool = False,
    compute_uv: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    """Singular Value Decomposition.
    
    Decomposes A = U @ S @ V.T where U and V are orthogonal and S is diagonal.
    
    Parameters
    ----------
    A : ndarray of shape (m, n)
        Matrix to decompose.
    full_matrices : bool, default=False
        If True, return full U (m, m) and Vh (n, n).
        If False, return reduced U (m, k) and Vh (k, n).
    compute_uv : bool, default=True
        If True, return U, S, Vh. If False, return only S.
        
    Returns
    -------
    U : ndarray
        Left singular vectors (if compute_uv=True).
    S : ndarray
        Singular values (sorted in descending order).
    Vh : ndarray
        Right singular vectors transposed (if compute_uv=True).
        
    Examples
    --------
    >>> A = np.random.randn(5, 3)
    >>> U, S, Vh = svd(A)
    >>> np.allclose(A, U @ np.diag(S) @ Vh)
    True
    """
    return np.linalg.svd(A, full_matrices=full_matrices, compute_uv=compute_uv)


def eigh(
    A: np.ndarray,
    *,
    subset_by_index: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition for symmetric/Hermitian matrices.
    
    Parameters
    ----------
    A : ndarray of shape (n, n)
        Symmetric matrix.
    subset_by_index : tuple of (lo, hi), optional
        Only compute eigenvalues/vectors in index range [lo, hi].
        
    Returns
    -------
    eigenvalues : ndarray of shape (n,) or (hi-lo+1,)
        Eigenvalues in ascending order.
    eigenvectors : ndarray of shape (n, n) or (n, hi-lo+1)
        Eigenvectors as columns.
        
    Examples
    --------
    >>> A = np.array([[2, 1], [1, 2]])
    >>> vals, vecs = eigh(A)
    >>> np.allclose(A @ vecs, vecs @ np.diag(vals))
    True
    """
    if subset_by_index is not None:
        from scipy.linalg import eigh as scipy_eigh
        return scipy_eigh(A, subset_by_index=subset_by_index)
    else:
        return np.linalg.eigh(A)


# =============================================================================
# Solvers
# =============================================================================

def solve_triangular(
    A: np.ndarray,
    b: np.ndarray,
    *,
    lower: bool = True,
    trans: bool = False,
) -> np.ndarray:
    """Solve a triangular system Ax = b.
    
    Parameters
    ----------
    A : ndarray of shape (n, n)
        Triangular matrix.
    b : ndarray of shape (n,) or (n, k)
        Right-hand side.
    lower : bool, default=True
        True if A is lower triangular, False if upper.
    trans : bool, default=False
        If True, solve A.T @ x = b instead.
        
    Returns
    -------
    x : ndarray
        Solution to the system.
    """
    from scipy.linalg import solve_triangular as scipy_solve_tri
    
    trans_arg = 'T' if trans else 'N'
    return scipy_solve_tri(A, b, lower=lower, trans=trans_arg)


def solve_cholesky(
    L: np.ndarray,
    b: np.ndarray,
    *,
    lower: bool = True,
) -> np.ndarray:
    """Solve a system using pre-computed Cholesky factor.
    
    Solves A x = b where A = L @ L.T (lower=True) or A = U.T @ U (lower=False).
    
    Parameters
    ----------
    L : ndarray of shape (n, n)
        Cholesky factor.
    b : ndarray of shape (n,) or (n, k)
        Right-hand side.
    lower : bool, default=True
        True if L is lower triangular.
        
    Returns
    -------
    x : ndarray
        Solution to the system.
        
    Examples
    --------
    >>> A = np.array([[4, 2], [2, 5]])
    >>> L = safe_cholesky(A)
    >>> b = np.array([1, 2])
    >>> x = solve_cholesky(L, b)
    >>> np.allclose(A @ x, b)
    True
    """
    # Solve L @ y = b
    y = solve_triangular(L, b, lower=lower, trans=False)
    
    # Solve L.T @ x = y
    x = solve_triangular(L, y, lower=lower, trans=True)
    
    return x


def solve_qr(
    Q: np.ndarray,
    R: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Solve a least squares problem using QR factors.
    
    Solves min ||Ax - b||² where A = QR.
    
    Parameters
    ----------
    Q : ndarray of shape (m, k)
        Orthogonal factor.
    R : ndarray of shape (k, n)
        Upper triangular factor.
    b : ndarray of shape (m,)
        Right-hand side.
        
    Returns
    -------
    x : ndarray of shape (n,)
        Least squares solution.
    """
    # x = R^{-1} @ Q.T @ b
    Qtb = Q.T @ b
    x = solve_triangular(R, Qtb, lower=False)
    return x


def lstsq(
    A: np.ndarray,
    b: np.ndarray,
    *,
    rcond: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Least squares solution to Ax = b.
    
    Parameters
    ----------
    A : ndarray of shape (m, n)
        Design matrix.
    b : ndarray of shape (m,) or (m, k)
        Right-hand side.
    rcond : float, optional
        Cutoff ratio for small singular values.
        
    Returns
    -------
    x : ndarray
        Least squares solution.
    residuals : ndarray
        Sum of squared residuals (if m > n).
    rank : int
        Effective rank of A.
    s : ndarray
        Singular values of A.
    """
    return np.linalg.lstsq(A, b, rcond=rcond)


# =============================================================================
# Matrix Operations
# =============================================================================

def safe_inverse(
    A: np.ndarray,
    *,
    rcond: float = 1e-15,
) -> np.ndarray:
    """Compute matrix inverse with regularization for near-singular matrices.
    
    Uses SVD-based pseudoinverse with condition number thresholding.
    
    Parameters
    ----------
    A : ndarray of shape (n, n)
        Matrix to invert.
    rcond : float, default=1e-15
        Cutoff for small singular values. Singular values smaller than
        rcond * largest_singular_value are set to zero.
        
    Returns
    -------
    A_inv : ndarray of shape (n, n)
        (Pseudo)inverse of A.
        
    Examples
    --------
    >>> A = np.array([[1, 2], [2, 4.001]])  # Nearly singular
    >>> A_inv = safe_inverse(A)
    """
    return np.linalg.pinv(A, rcond=rcond)


def woodbury_inverse(
    A_inv: np.ndarray,
    U: np.ndarray,
    C: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """Woodbury matrix identity for efficient inverse updates.
    
    Computes (A + UCV)^{-1} given A^{-1} efficiently.
    
    The Woodbury identity states:
    (A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}
    
    Parameters
    ----------
    A_inv : ndarray of shape (n, n)
        Inverse of A.
    U : ndarray of shape (n, k)
        Update matrix.
    C : ndarray of shape (k, k)
        Core matrix.
    V : ndarray of shape (k, n)
        Update matrix.
        
    Returns
    -------
    result : ndarray of shape (n, n)
        (A + UCV)^{-1}
        
    Examples
    --------
    >>> A = np.eye(3)
    >>> U = np.array([[1], [0], [0]])
    >>> C = np.array([[1]])
    >>> V = np.array([[1, 0, 0]])
    >>> new_inv = woodbury_inverse(np.eye(3), U, C, V)
    """
    # A^{-1}U
    AiU = A_inv @ U
    
    # VA^{-1}
    VAi = V @ A_inv
    
    # C^{-1} + VA^{-1}U
    inner = np.linalg.inv(C) + VAi @ U
    
    # (C^{-1} + VA^{-1}U)^{-1}
    inner_inv = np.linalg.inv(inner)
    
    # A^{-1} - A^{-1}U(...)VA^{-1}
    return A_inv - AiU @ inner_inv @ VAi


def quadratic_form(
    x: np.ndarray,
    A: np.ndarray,
    y: np.ndarray | None = None,
) -> float | np.ndarray:
    """Compute quadratic form x'Ay or x'Ax.
    
    Parameters
    ----------
    x : ndarray of shape (n,)
        Left vector.
    A : ndarray of shape (n, n)
        Matrix.
    y : ndarray of shape (n,), optional
        Right vector. If None, computes x'Ax.
        
    Returns
    -------
    result : float
        The quadratic form value.
    """
    if y is None:
        y = x
    return x @ A @ y


def log_determinant(
    A: np.ndarray,
    *,
    method: Literal['cholesky', 'svd', 'auto'] = 'auto',
) -> float:
    """Compute log of matrix determinant in a numerically stable way.
    
    Parameters
    ----------
    A : ndarray of shape (n, n)
        Positive definite matrix.
    method : {'cholesky', 'svd', 'auto'}, default='auto'
        Method to use. 'auto' tries Cholesky first, falls back to SVD.
        
    Returns
    -------
    logdet : float
        Log of determinant of A.
        
    Examples
    --------
    >>> A = np.array([[2, 1], [1, 2]])
    >>> logdet = log_determinant(A)
    >>> np.isclose(logdet, np.log(np.linalg.det(A)))
    True
    """
    if method == 'cholesky' or method == 'auto':
        try:
            L = np.linalg.cholesky(A)
            return 2 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            if method == 'cholesky':
                raise
            # Fall through to SVD
    
    # SVD method
    _, s, _ = np.linalg.svd(A)
    return np.sum(np.log(s))


def matrix_rank(
    A: np.ndarray,
    *,
    tol: float | None = None,
) -> int:
    """Compute matrix rank.
    
    Parameters
    ----------
    A : ndarray
        Input matrix.
    tol : float, optional
        Threshold below which singular values are considered zero.
        
    Returns
    -------
    rank : int
        Numerical rank of the matrix.
    """
    return int(np.linalg.matrix_rank(A, tol=tol))


def condition_number(
    A: np.ndarray,
    *,
    p: int | float | str = 2,
) -> float:
    """Compute matrix condition number.
    
    Parameters
    ----------
    A : ndarray
        Input matrix.
    p : {1, 2, inf, 'fro'}, default=2
        Order of the norm.
        
    Returns
    -------
    cond : float
        Condition number of A.
    """
    return np.linalg.cond(A, p=p)


__all__ = [
    # Decompositions
    "qr_decomposition",
    "safe_cholesky",
    "svd",
    "eigh",
    # Solvers
    "solve_triangular",
    "solve_cholesky",
    "solve_qr",
    "lstsq",
    # Matrix operations
    "safe_inverse",
    "woodbury_inverse",
    "quadratic_form",
    "log_determinant",
    "matrix_rank",
    "condition_number",
]