# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""REML (Restricted Maximum Likelihood) for smoothing parameter selection.

REML provides an alternative to GCV that is often more stable and
theoretically justified, especially for models with multiple smoothing
parameters.

References
----------
Wood, S.N. (2011). Fast stable restricted maximum likelihood and marginal
    likelihood estimation of semiparametric generalized linear models.
    Journal of the Royal Statistical Society: Series B, 73(1), 3-36.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def reml_score(
    y: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    lambda_: float,
    weights: np.ndarray | None = None,
) -> float:
    """Compute REML score for given smoothing parameter.

    The REML criterion is:
        -2 log(REML) = log|X'WX + λS| + log|X'WX| + (n-p) log(RSS)

    where RSS is the penalized residual sum of squares.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response variable.
    X : ndarray, shape (n, p)
        Design matrix (basis functions).
    S : ndarray, shape (p, p)
        Penalty matrix.
    lambda_ : float
        Smoothing parameter (must be positive).
    weights : ndarray, shape (n,), optional
        Observation weights.

    Returns
    -------
    reml : float
        REML score (lower is better).

    Notes
    -----
    The REML criterion penalizes both overfitting (via RSS) and model
    complexity (via determinant terms). It is invariant to fixed effects
    and provides unbiased estimates of variance components.

    REML is generally preferred over GCV for:
    - Multiple smoothing parameters
    - Small sample sizes
    - Models with fixed effects
    """
    n, p = X.shape

    if lambda_ <= 0:
        return np.inf

    # Weight matrix
    if weights is None:
        W = np.eye(n)
        sqrt_W = np.eye(n)
    else:
        W = np.diag(weights)
        sqrt_W = np.diag(np.sqrt(weights))

    # Penalized precision matrix
    XtWX = X.T @ W @ X
    A = XtWX + lambda_ * S

    try:
        # Solve penalized least squares
        XtWy = X.T @ W @ y
        coefficients = np.linalg.solve(A, XtWy)

        # Compute residual sum of squares
        fitted_values = X @ coefficients
        residuals = y - fitted_values
        RSS = np.sum(weights * residuals**2 if weights is not None else residuals**2)

        # Log determinants
        # log|A| = log|X'WX + λS|
        sign_A, logdet_A = np.linalg.slogdet(A)
        if sign_A <= 0:
            return np.inf

        # log|X'WX|
        sign_XtWX, logdet_XtWX = np.linalg.slogdet(XtWX)
        if sign_XtWX <= 0:
            return np.inf

        # REML score
        # -2 log(REML) = log|A| + log|X'WX| + (n-p) log(RSS/(n-p))
        # Simplified: log|A| + log|X'WX| + (n-p) log(RSS)
        reml = logdet_A - logdet_XtWX + (n - p) * np.log(RSS)

        return float(reml)

    except np.linalg.LinAlgError:
        return np.inf


def select_smoothing_parameter_reml(
    y: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    weights: np.ndarray | None = None,
    lambda_min: float = 1e-6,
    lambda_max: float = 1e6,
    method: str = "brent",
) -> dict[str, float | np.ndarray]:
    """Select smoothing parameter via REML optimization.

    Minimizes the REML criterion over the smoothing parameter λ.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response variable.
    X : ndarray, shape (n, p)
        Design matrix (basis functions).
    S : ndarray, shape (p, p)
        Penalty matrix.
    weights : ndarray, shape (n,), optional
        Observation weights.
    lambda_min : float, default=1e-6
        Minimum smoothing parameter to search.
    lambda_max : float, default=1e6
        Maximum smoothing parameter to search.
    method : {'brent', 'golden'}, default='brent'
        Optimization method:
        - 'brent': Brent's method (recommended)
        - 'golden': Golden section search

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'lambda_opt': Optimal smoothing parameter
        - 'reml_score': REML score at optimum
        - 'coefficients': Coefficient estimates
        - 'fitted_values': Fitted values
        - 'edf': Effective degrees of freedom

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.smoothing.selection.reml import select_smoothing_parameter_reml
    >>> from aurora.smoothing.splines.bspline import BSplineBasis
    >>> # Generate data
    >>> x = np.linspace(0, 1, 100)
    >>> y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(100)
    >>> # Create basis
    >>> knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    >>> basis = BSplineBasis(knots, degree=3)
    >>> X = basis.basis_matrix(x)
    >>> S = basis.penalty_matrix(order=2)
    >>> # Select lambda via REML
    >>> result = select_smoothing_parameter_reml(y, X, S)
    >>> print(f"Optimal lambda: {result['lambda_opt']:.4f}")
    >>> print(f"EDF: {result['edf']:.2f}")

    Notes
    -----
    REML is preferred over GCV when:
    - You have multiple smoothing parameters
    - Sample size is small
    - You want theoretical justification (ML-based)

    The optimization is performed on log(λ) scale for numerical stability.

    References
    ----------
    Wood, S.N. (2017). Generalized Additive Models: An Introduction with R.
        Chapman and Hall/CRC, 2nd edition.
    """
    n, p = X.shape

    # Validate inputs
    if n != len(y):
        raise ValueError(f"X and y must have same number of rows, got {n} and {len(y)}")

    if S.shape != (p, p):
        raise ValueError(f"S must have shape ({p}, {p}), got {S.shape}")

    if lambda_min <= 0 or lambda_max <= lambda_min:
        raise ValueError("Must have 0 < lambda_min < lambda_max")

    # Objective function (on log scale for numerical stability)
    def objective(log_lambda: float) -> float:
        lambda_val = np.exp(log_lambda)
        return reml_score(y, X, S, lambda_val, weights=weights)

    # Optimize on log scale
    log_lambda_min = np.log(lambda_min)
    log_lambda_max = np.log(lambda_max)

    # Use 'bounded' method when bounds are specified
    if method == "golden":
        # Golden section needs bracket, not bounds
        result = minimize_scalar(
            objective,
            bracket=(
                log_lambda_min,
                (log_lambda_min + log_lambda_max) / 2,
                log_lambda_max,
            ),
            method="golden",
        )
    else:
        # Use bounded method for Brent
        result = minimize_scalar(
            objective,
            bounds=(log_lambda_min, log_lambda_max),
            method="bounded",
        )

    if not result.success:
        raise RuntimeError(f"REML optimization failed: {result.message}")

    # Extract optimal lambda
    lambda_opt = np.exp(result.x)
    reml_opt = result.fun

    # Compute final solution at optimal lambda
    if weights is None:
        W = np.eye(n)
    else:
        W = np.diag(weights)

    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    A = XtWX + lambda_opt * S

    # Solve with ridge regularization fallback for singular matrices
    try:
        coefficients = np.linalg.solve(A, XtWy)
    except np.linalg.LinAlgError:
        # Add small ridge regularization for numerical stability
        ridge = 1e-8 * np.eye(A.shape[0])
        coefficients = np.linalg.solve(A + ridge, XtWy)

    fitted_values = X @ coefficients

    # Compute effective degrees of freedom
    # EDF = trace(X (X'WX + λS)^(-1) X'W)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse for singular matrices
        A_inv = np.linalg.pinv(A)
    edf = float(np.trace(A_inv @ XtWX))

    return {
        "lambda_opt": float(lambda_opt),
        "reml_score": float(reml_opt),
        "coefficients": coefficients,
        "fitted_values": fitted_values,
        "edf": edf,
    }


def select_multiple_smoothing_parameters_reml(
    y: np.ndarray,
    X_list: list[np.ndarray],
    S_list: list[np.ndarray],
    weights: np.ndarray | None = None,
    lambda_init: list[float] | None = None,
    lambda_min: float = 1e-6,
    lambda_max: float = 1e6,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> dict[str, list[float] | np.ndarray | float]:
    """Select multiple smoothing parameters via REML optimization.

    Uses alternating optimization: fix all λⱼ except one, optimize that one,
    repeat until convergence.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response variable.
    X_list : list of ndarray
        List of design matrices, one per smooth term.
    S_list : list of ndarray
        List of penalty matrices, one per smooth term.
    weights : ndarray, shape (n,), optional
        Observation weights.
    lambda_init : list of float, optional
        Initial smoothing parameters. If None, starts with all 1.0.
    lambda_min : float, default=1e-6
        Minimum smoothing parameter.
    lambda_max : float, default=1e6
        Maximum smoothing parameter.
    max_iter : int, default=20
        Maximum number of alternating iterations.
    tol : float, default=1e-4
        Convergence tolerance for relative change in λ.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'lambda_opt': List of optimal smoothing parameters
        - 'reml_score': REML score at optimum
        - 'coefficients': Full coefficient vector
        - 'fitted_values': Fitted values
        - 'edf_values': List of EDF per smooth term
        - 'converged': Whether optimization converged
        - 'n_iter': Number of iterations

    Notes
    -----
    This implements a simple alternating optimization scheme. More sophisticated
    methods exist (e.g., Newton-Raphson on all λ simultaneously) but are more
    complex to implement.

    The alternating scheme is:
    1. Fix λ₁, ..., λⱼ₋₁, λⱼ₊₁, ..., λₘ
    2. Optimize λⱼ via REML
    3. Cycle through j = 1, ..., m
    4. Repeat until convergence

    References
    ----------
    Wood, S.N. (2011). Fast stable restricted maximum likelihood and marginal
        likelihood estimation of semiparametric generalized linear models.
    """
    n = len(y)
    m = len(X_list)  # Number of smooth terms

    if len(S_list) != m:
        raise ValueError(
            f"X_list and S_list must have same length, got {m} and {len(S_list)}"
        )

    # Initialize lambdas
    if lambda_init is None:
        lambdas = [1.0] * m
    else:
        if len(lambda_init) != m:
            raise ValueError(
                f"lambda_init must have length {m}, got {len(lambda_init)}"
            )
        lambdas = list(lambda_init)

    # Build full design matrix
    X_full = np.column_stack(X_list)
    p_list = [X.shape[1] for X in X_list]  # Basis sizes per term

    converged = False
    for iteration in range(max_iter):
        lambdas_old = lambdas.copy()

        # Optimize each lambda in turn
        for j in range(m):
            # Build full penalty matrix with current lambdas
            # S_full = block_diag(λ₁S₁, λ₂S₂, ..., λₘSₘ)
            from scipy.linalg import block_diag

            S_blocks = [
                lambdas[i] * S_list[i] if i != j else S_list[i] for i in range(m)
            ]
            S_full = block_diag(*S_blocks)

            # Optimize λⱼ
            def objective(log_lambda: float) -> float:
                lambda_j = np.exp(log_lambda)
                # Update jth block
                S_blocks_temp = [
                    lambdas[i] * S_list[i] if i != j else lambda_j * S_list[i]
                    for i in range(m)
                ]
                S_temp = block_diag(*S_blocks_temp)
                return reml_score(y, X_full, S_temp, 1.0, weights=weights)

            log_lambda_min = np.log(lambda_min)
            log_lambda_max = np.log(lambda_max)

            opt_result = minimize_scalar(
                objective,
                bounds=(log_lambda_min, log_lambda_max),
                method="bounded",
            )

            lambdas[j] = np.exp(opt_result.x)

        # Check convergence
        rel_change = np.max(
            np.abs(np.array(lambdas) - np.array(lambdas_old))
            / (np.array(lambdas_old) + 1e-10)
        )
        if rel_change < tol:
            converged = True
            break

    # Compute final solution
    from scipy.linalg import block_diag

    S_blocks = [lambdas[i] * S_list[i] for i in range(m)]
    S_full = block_diag(*S_blocks)

    if weights is None:
        W = np.eye(n)
    else:
        W = np.diag(weights)

    XtWX = X_full.T @ W @ X_full
    XtWy = X_full.T @ W @ y
    A = XtWX + S_full

    coefficients = np.linalg.solve(A, XtWy)
    fitted_values = X_full @ coefficients

    # Compute REML score
    final_reml = reml_score(y, X_full, S_full, 1.0, weights=weights)

    # Compute EDF per term
    A_inv = np.linalg.inv(A)
    edf_values = []
    idx = 0
    for p_j in p_list:
        # EDF for term j = trace(X_j (X'WX + λS)^(-1) X_j' W)
        X_j = X_full[:, idx : idx + p_j]
        H_j = X_j @ A_inv[idx : idx + p_j, :] @ X_full.T @ W
        edf_j = float(np.trace(H_j))
        edf_values.append(edf_j)
        idx += p_j

    return {
        "lambda_opt": lambdas,
        "reml_score": float(final_reml),
        "coefficients": coefficients,
        "fitted_values": fitted_values,
        "edf_values": edf_values,
        "converged": converged,
        "n_iter": iteration + 1,
    }


__all__ = [
    "reml_score",
    "select_smoothing_parameter_reml",
    "select_multiple_smoothing_parameters_reml",
]
