"""GAM fitting using penalized regression splines.

This module implements univariate Generalized Additive Models using:
- B-spline or cubic spline bases
- Difference penalties for smoothness
- GCV for automatic smoothing parameter selection
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from aurora.models.gam.result import GAMResult
from aurora.smoothing.penalties.difference import difference_penalty
from aurora.smoothing.selection.gcv import select_smoothing_parameter
from aurora.smoothing.splines.bspline import BSplineBasis
from aurora.smoothing.splines.cubic import CubicSplineBasis


def fit_gam(
    x: np.ndarray,
    y: np.ndarray,
    n_basis: int = 10,
    basis_type: Literal["bspline", "cubic"] = "bspline",
    degree: int = 3,
    penalty_order: int = 2,
    lambda_: float | None = None,
    lambda_min: float = 1e-6,
    lambda_max: float = 1e6,
    knot_method: Literal["quantile", "uniform"] = "quantile",
    weights: np.ndarray | None = None,
) -> GAMResult:
    """Fit univariate Generalized Additive Model using penalized splines.

    This function fits a smooth function f(x) to data (x, y) using:
        y = f(x) + ε
    where f is represented as a linear combination of basis functions with
    a roughness penalty.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Predictor variable.
    y : ndarray, shape (n,)
        Response variable.
    n_basis : int, default=10
        Number of basis functions to use.
    basis_type : {'bspline', 'cubic'}, default='bspline'
        Type of spline basis:
        - 'bspline': B-spline basis (local support, partition of unity)
        - 'cubic': Natural cubic spline basis (global support)
    degree : int, default=3
        Degree of splines (3 for cubic splines).
    penalty_order : int, default=2
        Order of difference penalty (2 approximates second derivative).
    lambda_ : float, optional
        Smoothing parameter. If None, selected automatically via GCV.
    lambda_min : float, default=1e-6
        Minimum lambda for GCV search.
    lambda_max : float, default=1e6
        Maximum lambda for GCV search.
    knot_method : {'quantile', 'uniform'}, default='quantile'
        Method for placing knots:
        - 'quantile': Place knots at quantiles of x
        - 'uniform': Space knots uniformly over range of x
    weights : ndarray, shape (n,), optional
        Observation weights for weighted least squares.

    Returns
    -------
    result : GAMResult
        Fitted GAM result containing:
        - coefficients: Spline coefficients
        - fitted_values: Predicted values at x
        - lambda_: Smoothing parameter used
        - edf: Effective degrees of freedom
        - basis: Basis object for prediction
        - residuals: y - fitted_values

    Notes
    -----
    The model minimizes the penalized least squares criterion:
        ||y - Xβ||² + λ β'Sβ

    where X is the basis matrix and S is the penalty matrix.

    For automatic smoothing parameter selection (lambda_=None), uses GCV:
        GCV(λ) = (n * RSS) / (n - tr(H))²

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.gam import fit_gam
    >>> # Generate noisy data
    >>> x = np.linspace(0, 1, 100)
    >>> y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(100)
    >>> # Fit GAM with automatic smoothing
    >>> result = fit_gam(x, y, n_basis=12)
    >>> print(f"EDF: {result.edf:.2f}")
    >>> print(f"Lambda: {result.lambda_:.4f}")
    >>> # Make predictions
    >>> x_new = np.linspace(0, 1, 200)
    >>> y_pred = result.predict(x_new)

    References
    ----------
    Wood, S.N. (2017). Generalized Additive Models: An Introduction with R.
        Chapman and Hall/CRC.
    Eilers, P.H.C. & Marx, B.D. (1996). Flexible smoothing with B-splines and
        penalties. Statistical Science, 11(2), 89-121.
    """
    # Validate inputs
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if x_arr.ndim != 1:
        raise ValueError("x must be 1-dimensional")

    if y_arr.ndim != 1:
        raise ValueError("y must be 1-dimensional")

    n = len(x_arr)
    if len(y_arr) != n:
        raise ValueError(f"x and y must have same length, got {n} and {len(y_arr)}")

    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {weights_arr.shape}")
    else:
        weights_arr = None

    if n_basis < 3:
        raise ValueError("n_basis must be at least 3")

    # Create basis
    if basis_type == "bspline":
        knots = BSplineBasis.create_knots(
            x_arr, n_basis=n_basis, degree=degree, method=knot_method
        )
        basis = BSplineBasis(knots, degree=degree)
    elif basis_type == "cubic":
        knots_interior = CubicSplineBasis.create_knots(
            x_arr, n_knots=n_basis - 2, method=knot_method
        )
        basis = CubicSplineBasis(knots_interior)
    else:
        raise ValueError(f"Unknown basis_type: {basis_type}")

    # Compute basis matrix
    X = basis.basis_matrix(x_arr)

    # Create penalty matrix
    if basis_type == "bspline":
        S = basis.penalty_matrix(order=penalty_order)
    else:  # cubic
        # Cubic splines use analytical integrated squared second derivative
        # (no order parameter)
        S = basis.penalty_matrix()

    # Select or use provided smoothing parameter
    if lambda_ is None:
        # Automatic selection via GCV
        gcv_result = select_smoothing_parameter(
            y_arr,
            X,
            S,
            weights=weights_arr,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )
        lambda_used = gcv_result["lambda_opt"]
        coefficients = gcv_result["coefficients"]
        fitted_values = gcv_result["fitted_values"]
        edf = gcv_result["edf"]
        gcv_score = gcv_result["gcv_score"]
    else:
        # Use provided lambda
        lambda_used = float(lambda_)

        # Solve penalized least squares
        if weights_arr is None:
            W = np.eye(n)
        else:
            W = np.diag(weights_arr)

        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y_arr
        A = XtWX + lambda_used * S

        coefficients = np.linalg.solve(A, XtWy)
        fitted_values = X @ coefficients

        # Compute EDF
        A_inv = np.linalg.inv(A)
        edf = float(np.trace(A_inv @ XtWX))

        gcv_score = None

    # Compute residuals
    residuals = y_arr - fitted_values

    # Create result object
    result = GAMResult(
        coefficients=coefficients,
        fitted_values=fitted_values,
        residuals=residuals,
        lambda_=lambda_used,
        edf=edf,
        basis=basis,
        x=x_arr,
        y=y_arr,
        weights=weights_arr,
        gcv_score=gcv_score,
    )

    return result


__all__ = ["fit_gam"]
