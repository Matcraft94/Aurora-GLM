"""Generalized Cross-Validation (GCV) for smoothing parameter selection.

GCV provides a computationally efficient approximation to leave-one-out
cross-validation without actually fitting n different models.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize


def gcv_score(
    y: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    lambda_: float,
    weights: np.ndarray | None = None,
) -> float:
    """Compute GCV score for given smoothing parameter.

    The GCV score is:
        GCV(λ) = (n * RSS) / (n - tr(H))²

    where RSS is residual sum of squares and H is the hat matrix.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response variable.
    X : ndarray, shape (n, p)
        Design matrix (basis matrix).
    S : ndarray, shape (p, p)
        Penalty matrix.
    lambda_ : float
        Smoothing parameter.
    weights : ndarray, shape (n,), optional
        Observation weights (for weighted least squares).

    Returns
    -------
    score : float
        GCV score (lower is better).

    Notes
    -----
    The penalized least squares problem is:
        min_β ||y - Xβ||² + λ β'Sβ

    Solution: β̂ = (X'WX + λS)⁻¹ X'Wy
    Hat matrix: H = X(X'WX + λS)⁻¹X'W

    GCV approximates leave-one-out CV by:
        GCV = RSS / (1 - tr(H)/n)²

    References
    ----------
    Craven, P. & Wahba, G. (1978). Smoothing noisy data with spline functions.
        Numerische Mathematik, 31, 377-403.
        https://doi.org/10.1007/BF01404567
    Wood, S.N. (2017). Generalized Additive Models: An Introduction with R.
    """
    n, p = X.shape

    if weights is None:
        W = np.eye(n)
        w = np.ones(n)
    else:
        w = np.asarray(weights)
        if w.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {w.shape}")
        W = np.diag(w)

    # Compute penalized solution
    # β̂ = (X'WX + λS)⁻¹ X'Wy
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y

    # Add penalty
    A = XtWX + lambda_ * S

    # Solve for coefficients
    try:
        beta = np.linalg.solve(A, XtWy)
    except np.linalg.LinAlgError:
        # Singular matrix - return large penalty
        return 1e10

    # Fitted values
    y_hat = X @ beta

    # Residual sum of squares
    residuals = y - y_hat
    rss = np.sum(w * residuals**2)

    # Effective degrees of freedom (trace of hat matrix)
    # H = X(X'WX + λS)⁻¹X'W
    # tr(H) = tr(X(X'WX + λS)⁻¹X'W)
    #       = tr((X'WX + λS)⁻¹ X'WX)
    try:
        A_inv = np.linalg.inv(A)
        edf = np.trace(A_inv @ XtWX)
    except np.linalg.LinAlgError:
        return 1e10

    # GCV score
    if n - edf <= 0:
        # Model is too complex
        return 1e10

    gcv = (n * rss) / (n - edf) ** 2

    return float(gcv)


def select_smoothing_parameter(
    y: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    weights: np.ndarray | None = None,
    lambda_min: float = 1e-6,
    lambda_max: float = 1e6,
    method: str = "golden",
) -> dict[str, Any]:
    """Select optimal smoothing parameter using GCV.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response variable.
    X : ndarray, shape (n, p)
        Design matrix (basis matrix).
    S : ndarray, shape (p, p)
        Penalty matrix.
    weights : ndarray, shape (n,), optional
        Observation weights.
    lambda_min : float, default=1e-6
        Minimum smoothing parameter to consider.
    lambda_max : float, default=1e6
        Maximum smoothing parameter to consider.
    method : str, default='golden'
        Optimization method:
        - 'golden': Golden section search (robust, no derivatives)
        - 'brent': Brent's method (faster, still no derivatives)

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'lambda_opt': Optimal smoothing parameter
        - 'gcv_score': GCV score at optimum
        - 'edf': Effective degrees of freedom at optimum
        - 'coefficients': Fitted coefficients at optimum
        - 'fitted_values': Fitted values at optimum

    Notes
    -----
    GCV tends to undersmooth slightly compared to REML/ML, but is
    computationally simpler and doesn't require distributional assumptions.

    For multiple smoothing parameters (multi-dimensional λ), use REML instead.

    Examples
    --------
    >>> from aurora.smoothing.splines.bspline import BSplineBasis
    >>> x = np.linspace(0, 1, 100)
    >>> y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(100)
    >>> knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    >>> basis = BSplineBasis(knots, degree=3)
    >>> X = basis.basis_matrix(x)
    >>> S = basis.penalty_matrix(order=2)
    >>> result = select_smoothing_parameter(y, X, S)
    >>> print(f"Optimal λ: {result['lambda_opt']:.4f}")
    >>> print(f"Effective DoF: {result['edf']:.2f}")
    """
    n, p = X.shape

    # Validate inputs
    if y.shape != (n,):
        raise ValueError(f"y must have shape ({n},), got {y.shape}")

    if S.shape != (p, p):
        raise ValueError(f"S must have shape ({p}, {p}), got {S.shape}")

    # Define objective function (on log scale for better optimization)
    def objective(log_lambda: float) -> float:
        lambda_ = np.exp(log_lambda)
        return gcv_score(y, X, S, lambda_, weights)

    # Optimize on log scale
    log_lambda_min = np.log(lambda_min)
    log_lambda_max = np.log(lambda_max)

    if method == "golden":
        result = optimize.minimize_scalar(
            objective,
            bounds=(log_lambda_min, log_lambda_max),
            method="bounded",
        )
    elif method == "brent":
        result = optimize.minimize_scalar(
            objective,
            bracket=(log_lambda_min, log_lambda_max),
            method="brent",
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'golden' or 'brent'.")

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    # Extract optimal lambda
    lambda_opt = np.exp(result.x)
    gcv_opt = result.fun

    # Compute solution at optimal lambda
    if weights is None:
        W = np.eye(n)
        w = np.ones(n)
    else:
        w = np.asarray(weights)
        W = np.diag(w)

    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    A = XtWX + lambda_opt * S

    beta_opt = np.linalg.solve(A, XtWy)
    y_hat = X @ beta_opt

    # Compute effective degrees of freedom
    A_inv = np.linalg.inv(A)
    edf = np.trace(A_inv @ XtWX)

    return {
        "lambda_opt": lambda_opt,
        "gcv_score": gcv_opt,
        "edf": float(edf),
        "coefficients": beta_opt,
        "fitted_values": y_hat,
    }


__all__ = [
    "gcv_score",
    "select_smoothing_parameter",
]
