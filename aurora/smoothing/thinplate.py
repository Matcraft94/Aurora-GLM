# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Thin Plate Spline implementation for multidimensional smoothing.

Thin plate splines provide a smooth interpolation/regression surface in
multiple dimensions by minimizing a measure of bending energy.

References
----------
Wood, S.N. (2003). Thin plate regression splines. Journal of the Royal
    Statistical Society: Series B, 65(1), 95-114.

Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in
    Sobolev spaces. Constructive Theory of Functions of Several Variables,
    85-100.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
from scipy.spatial.distance import cdist


def _resolve_m(d: int, m: int | None) -> int:
    """Resolve the Duchon smoothness order m for dimension d.

    Duchon (1977) requires 2m > d.  The default keeps the classical
    thin plate spline (m = 2) for d <= 3 and uses the minimal valid
    order for d >= 4.
    """
    if m is None:
        m = 2 if d <= 3 else d // 2 + 1
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}")
    if 2 * m <= d:
        raise ValueError(
            f"Duchon (1977) requires 2m > d; got m={m}, d={d}. "
            f"Use m >= {d // 2 + 1} for dimension {d}."
        )
    return m


def _tps_radial(distances: np.ndarray, d: int, m: int) -> np.ndarray:
    """Duchon (1977) radial function eta(r) evaluated on a distance matrix.

    With smoothness order m (2m > d):

    * d even: eta(r) = c r^(2m-d) log(r)   (0 at r = 0)
    * d odd:  eta(r) = c r^(2m-d)

    where the parity is on the *dimension* d, not on the exponent, and the
    sign c = (-1)^(m - ceil(d/2) + 1) makes the radial energy matrix
    conditionally positive semi-definite of order m (equivalently, the sign
    of the Gamma(d/2 - m) factor in Duchon's formula; positive constants
    are absorbed into the smoothing parameter).  For example: +r^3 (d=1,
    m=2), +r^2 log(r) (d=2, m=2), -r (d=3, m=2), -r^4 log(r) (d=2, m=3).
    """
    exponent = 2 * m - d
    sign = -1.0 if (m - (d + 1) // 2 + 1) % 2 == 1 else 1.0
    if d % 2 == 0:
        eta = np.zeros_like(distances)
        nonzero = distances > 0
        r_pow = distances[nonzero] ** exponent
        eta[nonzero] = r_pow * np.log(distances[nonzero])
        return sign * eta
    return sign * distances**exponent


def _tps_polynomial(X: np.ndarray, d: int, m: int) -> np.ndarray:
    """Polynomial terms of total degree <= m - 1 (the penalty null space).

    Returns an (n, M) matrix with M = C(m - 1 + d, d) columns: the constant,
    then all monomials of total degree 1, 2, ..., m-1.
    """
    n = X.shape[0]
    columns = [np.ones(n)]
    for degree in range(1, m):
        for multi_index in combinations_with_replacement(range(d), degree):
            term = np.ones(n)
            for idx in multi_index:
                term = term * X[:, idx]
            columns.append(term)
    return np.column_stack(columns)


def tps_basis(
    X: np.ndarray,
    knots: np.ndarray,
    d: int = 2,
    m: int | None = None,
) -> np.ndarray:
    """Compute thin plate spline basis functions.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Data points where basis is evaluated.
    knots : ndarray, shape (k, d)
        Knot locations (typically a subset of data points).
    d : int, default=2
        Dimensionality (number of variables).
    m : int, optional
        Duchon smoothness order. Must satisfy 2m > d. Default is m = 2
        for d <= 3 and the minimal valid order (d//2 + 1) for d >= 4.

    Returns
    -------
    B : ndarray, shape (n, k + M)
        Basis matrix with k radial basis functions followed by the M
        polynomial terms spanning the penalty null space, where
        M = C(m - 1 + d, d) (for m = 2: M = d + 1, i.e. 1, x_1, ..., x_d).

    Notes
    -----
    The thin plate spline basis consists of:
    - k radial basis functions: eta(||x - x_j||) with the Duchon (1977)
      radial function (c = (-1)^(m - ceil(d/2) + 1) is the sign that makes
      the radial energy matrix conditionally positive semi-definite):
        * d even: eta(r) = c r^(2m-d) log(r)
        * d odd:  eta(r) = c r^(2m-d)
    - M polynomial terms of total degree <= m - 1.

    Classical choices (m = 2): eta(r) = r^3 for d = 1, r^2 log(r) for
    d = 2, -r for d = 3.  For d = 4 the minimal valid order is m = 3,
    giving eta(r) = r^2 log(r).

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.smoothing.thinplate import tps_basis
    >>> # 2D example
    >>> X = np.random.randn(100, 2)
    >>> knots = X[:20]  # Use subset as knots
    >>> B = tps_basis(X, knots, d=2)
    >>> B.shape
    (100, 23)  # 20 radial + 3 polynomial (1, x, y)
    """
    m = _resolve_m(d, m)

    # Compute pairwise distances
    distances = cdist(X, knots, metric="euclidean")

    # Radial basis functions (Duchon 1977)
    eta = _tps_radial(distances, d, m)

    # Polynomial terms of total degree <= m - 1
    polynomial = _tps_polynomial(X, d, m)

    # Combine: [eta_1, ..., eta_k, polynomials]
    B = np.column_stack([eta, polynomial])

    return B


def _null_space_of_transpose(T: np.ndarray) -> np.ndarray:
    """Orthonormal basis Z for null(T') via QR of T.

    T : (k, M).  Returns Z : (k, k - rank(T)) with Z' T = 0, so that
    delta = Z gamma exactly satisfies the constraint T' delta = 0.
    """
    k, _ = T.shape
    Q, R = np.linalg.qr(T, mode="complete")
    diag_R = np.abs(np.diag(R))
    tol = max(T.shape) * np.finfo(float).eps * (diag_R[0] if diag_R.size else 0.0)
    rank_T = int(np.sum(diag_R > tol))
    return Q[:, rank_T:]


def tps_penalty(
    knots: np.ndarray,
    d: int = 2,
    m: int | None = None,
) -> np.ndarray:
    """Compute thin plate spline penalty matrix.

    Parameters
    ----------
    knots : ndarray, shape (k, d)
        Knot locations.
    d : int, default=2
        Dimensionality.
    m : int, optional
        Duchon smoothness order (2m > d). See :func:`tps_basis`.

    Returns
    -------
    S : ndarray, shape (k + M, k + M)
        Penalty matrix. Only the first k x k block (radial part) is
        non-zero.  M = C(m - 1 + d, d) is the number of polynomial terms.

    Notes
    -----
    The raw radial energy matrix E with E[i,j] = eta(||x_i - x_j||) is only
    *conditionally* positive semi-definite: beta' E beta >= 0 solely for
    coefficient vectors satisfying the constraints T' delta = 0, where T is
    the matrix of polynomial terms evaluated at the knots (Wood 2003).

    This function absorbs the constraints by projection.  With Z an
    orthonormal basis for null(T'), the returned radial block is

        S_radial = Z (Z' E Z) Z'

    which is positive semi-definite (all eigenvalues >= 0) and equals E on
    the feasible subspace: for any delta with T' delta = 0,
    delta' S_radial delta = delta' E delta.  This is the standard
    reparametrization used for thin plate regression splines (Wood 2003).

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.smoothing.thinplate import tps_penalty
    >>> knots = np.random.randn(20, 2)
    >>> S = tps_penalty(knots, d=2)
    >>> S.shape
    (23, 23)  # 20 + 2 + 1
    """
    k = knots.shape[0]
    m = _resolve_m(d, m)

    # Polynomial terms at the knots define the constraints T' delta = 0
    T = _tps_polynomial(knots, d, m)
    M = T.shape[1]

    if k <= M:
        raise ValueError(
            f"Need more knots than polynomial terms for a TPS penalty: "
            f"got k={k} knots and M={M} polynomial terms (d={d}, m={m})."
        )

    # Radial energy matrix between knots
    distances = cdist(knots, knots, metric="euclidean")
    E = _tps_radial(distances, d, m)

    # Absorb the constraints T' delta = 0: project E onto null(T')
    Z = _null_space_of_transpose(T)
    S_radial = Z @ (Z.T @ E @ Z) @ Z.T

    # Full penalty matrix (only radial part is penalized)
    p_total = k + M
    S = np.zeros((p_total, p_total))
    S[:k, :k] = S_radial

    return S


def fit_tps(
    X: np.ndarray,
    y: np.ndarray,
    knots: np.ndarray | None = None,
    lambda_: float = 1.0,
    weights: np.ndarray | None = None,
    m: int | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Fit a thin plate spline.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Predictor matrix (d-dimensional).
    y : ndarray, shape (n,)
        Response variable.
    knots : ndarray, shape (k, d), optional
        Knot locations. If None, uses all data points (can be slow for large n).
    lambda_ : float, default=1.0
        Smoothing parameter.
    weights : ndarray, shape (n,), optional
        Observation weights.
    m : int, optional
        Duchon smoothness order (2m > d). See :func:`tps_basis`.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'coefficients': Coefficient estimates (radial then polynomial)
        - 'fitted_values': Fitted values
        - 'knots': Knot locations used
        - 'edf': Effective degrees of freedom
        - 'm': Duchon smoothness order used

    Notes
    -----
    Minimizes: ||W^(1/2)(y - Bβ)||² + λ β' S β

    subject to the exact thin plate constraints T'δ = 0 on the radial
    coefficients, where T is the matrix of polynomial terms at the knots
    (Duchon 1977; Wood 2003).  The constraints are absorbed by
    reparametrization: with Z an orthonormal basis for null(T'), write
    δ = Zγ and solve the unconstrained penalized system in
    (γ, α).  The effective penalty Z'EZ is positive semi-definite, unlike
    the raw radial energy matrix E.

    The effective degrees of freedom are computed as
    ``tr((X_c'WX_c + λS_c)^{-1} X_c'WX_c)`` in the constrained
    parametrization, without materializing the n x n hat matrix.

    For computational efficiency with large datasets, use a subset of
    data points as knots (k << n).

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.smoothing.thinplate import fit_tps
    >>> # 2D spatial data
    >>> n = 200
    >>> X = np.random.uniform(-1, 1, (n, 2))
    >>> y = np.sin(3*X[:, 0]) * np.cos(3*X[:, 1]) + 0.1*np.random.randn(n)
    >>> # Fit with subset of knots for efficiency
    >>> knots = X[::10]  # Every 10th point
    >>> result = fit_tps(X, y, knots=knots, lambda_=0.01)
    >>> result['fitted_values'].shape
    (200,)
    """
    n, d = X.shape
    m = _resolve_m(d, m)

    # Use all points as knots if not specified (can be slow)
    if knots is None:
        knots = X.copy()

    k = knots.shape[0]

    # Full basis: radial part K (n x k) and polynomial part T_n (n x M)
    B = tps_basis(X, knots, d=d, m=m)
    M = B.shape[1] - k
    K = B[:, :k]
    T_n = B[:, k:]

    if k <= M:
        raise ValueError(
            f"Need more knots than polynomial terms to fit a TPS: "
            f"got k={k} knots and M={M} polynomial terms (d={d}, m={m})."
        )

    # Radial energy matrix and constraint matrix at the knots
    E = _tps_radial(cdist(knots, knots, metric="euclidean"), d, m)
    T = _tps_polynomial(knots, d, m)

    # Absorb the constraints T'δ = 0: δ = Zγ with Z spanning null(T')
    Z = _null_space_of_transpose(T)

    # Constrained design: X_c = [K Z | T_n], shape (n, k)
    X_c = np.column_stack([K @ Z, T_n])

    # Constrained penalty: only the γ block is penalized, with the
    # positive semi-definite matrix Z'EZ
    S_c = np.zeros((k, k))
    S_c[: Z.shape[1], : Z.shape[1]] = Z.T @ E @ Z

    # Weight matrix
    if weights is None:
        W = np.eye(n)
    else:
        W = np.diag(weights)

    # Penalized least squares in the constrained parametrization
    XtWX = X_c.T @ W @ X_c
    XtWy = X_c.T @ W @ y
    A = XtWX + lambda_ * S_c

    try:
        theta = np.linalg.solve(A, XtWy)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        theta = np.linalg.lstsq(A, XtWy, rcond=None)[0]

    # Map back to the original parametrization: δ = Zγ
    gamma = theta[: Z.shape[1]]
    alpha = theta[Z.shape[1] :]
    coefficients = np.concatenate([Z @ gamma, alpha])

    fitted_values = B @ coefficients

    # Effective degrees of freedom without materializing the hat matrix:
    # tr(H) = tr((X_c'WX_c + λS_c)^{-1} X_c'WX_c)
    try:
        edf = float(np.trace(np.linalg.solve(A, XtWX)))
    except np.linalg.LinAlgError:
        edf = float("nan")

    return {
        "coefficients": coefficients,
        "fitted_values": fitted_values,
        "knots": knots,
        "edf": edf,
        "m": m,
    }


def select_knots(
    X: np.ndarray,
    n_knots: int | None = None,
    method: str = "uniform",
) -> np.ndarray:
    """Select knot locations for thin plate splines.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Data points.
    n_knots : int, optional
        Number of knots to select. If None, uses min(n, 100).
    method : {'uniform', 'random', 'kmeans'}, default='uniform'
        Method for selecting knots:
        - 'uniform': Every n/k-th point
        - 'random': Random subset
        - 'kmeans': K-means clustering via scipy.cluster.vq.kmeans2; knots
          are cluster centroids, giving density-aware coverage of X.

    Returns
    -------
    knots : ndarray, shape (n_knots, d)
        Selected knot locations.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.smoothing.thinplate import select_knots
    >>> X = np.random.randn(1000, 2)
    >>> knots = select_knots(X, n_knots=50, method='uniform')
    >>> knots.shape
    (50, 2)
    """
    n = X.shape[0]

    if n_knots is None:
        n_knots = min(n, 100)

    if n_knots > n:
        n_knots = n

    if method == "uniform":
        # Every k-th point
        step = max(1, n // n_knots)
        indices = np.arange(0, n, step)[:n_knots]
        knots = X[indices]
    elif method == "random":
        # Random subset
        indices = np.random.choice(n, size=n_knots, replace=False)
        knots = X[indices]
    elif method == "kmeans":
        # K-means clustering - uses scipy.cluster.vq.kmeans2 (Lloyd's algorithm).
        # Cluster centroids serve as knot locations, giving representative
        # coverage of the data distribution. Seeded from random data points
        # for reproducibility via the fixed seed.
        from scipy.cluster.vq import kmeans2

        n_clusters = min(n_knots, n)
        centroids, _ = kmeans2(
            X.astype(np.float64, copy=False),
            k=n_clusters,
            minit="points",
            seed=42,
            missing="warn",
        )
        knots = centroids[:n_knots] if n_clusters >= n_knots else centroids
    else:
        raise ValueError(f"Unknown method: {method}")

    return knots


__all__ = [
    "tps_basis",
    "tps_penalty",
    "fit_tps",
    "select_knots",
]
