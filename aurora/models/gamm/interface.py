"""High-level interface for GAMM fitting.

This module provides user-friendly functions for fitting GAMMs,
automatically handling design matrix construction and integration
of smooth terms with random effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from aurora.models.gamm.design import construct_Z_matrix
from aurora.models.gamm.fitting import GAMMResult, fit_gamm_gaussian, predict_gamm
from aurora.models.gamm.random_effects import RandomEffect

if TYPE_CHECKING:
    from numpy.typing import NDArray


def fit_gamm(
    y: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame | None = None,
    random_effects: list[RandomEffect] | None = None,
    groups_data: dict[str, np.ndarray] | pd.DataFrame | None = None,
    family: str = "gaussian",
    covariance: str = "unstructured",
    maxiter: int = 100,
    tol: float = 1e-6,
) -> GAMMResult:
    """Fit a Generalized Additive Mixed Model.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response variable.
    X : array-like, shape (n, p), optional
        Design matrix for parametric fixed effects.
        If None, uses intercept-only model.
    random_effects : list of RandomEffect, optional
        Random effect specifications.
    groups_data : dict or DataFrame, optional
        Grouping variables for random effects.
        Keys should match RandomEffect.grouping names.
    family : str, default='gaussian'
        Distribution family. Currently only 'gaussian' supported.
    covariance : str, default='unstructured'
        Covariance structure for random effects:
        - 'unstructured': Full covariance matrix
        - 'diagonal': Independent random effects
        - 'identity': Equal variances, no correlation
    maxiter : int, default=100
        Maximum iterations for optimization.
    tol : float, default=1e-6
        Convergence tolerance.

    Returns
    -------
    result : GAMMResult
        Fitted GAMM result with coefficients, variance components,
        and diagnostics.

    Raises
    ------
    ValueError
        If family is not 'gaussian'.
        If random_effects provided but groups_data is None.
        If group variables not found in groups_data.

    Examples
    --------
    >>> # Random intercept model
    >>> import numpy as np
    >>> from aurora.models.gamm import fit_gamm, RandomEffect
    >>>
    >>> # Generate data
    >>> n_groups, n_per_group = 10, 20
    >>> n = n_groups * n_per_group
    >>> groups = np.repeat(np.arange(n_groups), n_per_group)
    >>> x = np.random.randn(n)
    >>> X = np.column_stack([np.ones(n), x])
    >>>
    >>> # Random intercepts
    >>> b = np.random.randn(n_groups)
    >>> y = 2.0 + 0.5*x + b[groups] + np.random.randn(n)*0.5
    >>>
    >>> # Fit model
    >>> re = RandomEffect(grouping='subject')
    >>> result = fit_gamm(
    ...     y=y,
    ...     X=X,
    ...     random_effects=[re],
    ...     groups_data={'subject': groups},
    ...     covariance='identity'
    ... )
    >>>
    >>> # Access results
    >>> print(result.beta_parametric)  # Fixed effects
    >>> print(result.variance_components)  # Random effect variance
    >>> print(result.residual_variance)  # Residual variance

    >>> # Random intercept + slope
    >>> re_slope = RandomEffect(grouping='subject', variables=(1,))
    >>> result = fit_gamm(
    ...     y=y,
    ...     X=X,
    ...     random_effects=[re_slope],
    ...     groups_data={'subject': groups},
    ...     covariance='unstructured'
    ... )
    >>> print(result.variance_components)  # 2x2 covariance matrix

    Notes
    -----
    This is a high-level interface that automatically constructs
    design matrices for random effects and calls the appropriate
    fitting function based on the family.

    For Gaussian family, uses exact REML estimation.
    For other families (future implementation), will use PQL or Laplace.

    Currently supports:
    - Single random effect term
    - Gaussian family only
    - No smooth terms (use fit_gamm_gaussian directly for smooth terms)
    """
    # Input validation
    if family != "gaussian":
        raise ValueError(
            f"Only 'gaussian' family currently supported, got '{family}'. "
            "Other families (Poisson, Binomial) will be implemented in Milestone 4."
        )

    # Convert inputs to numpy arrays
    if isinstance(y, pd.Series):
        y = y.values
    y = np.asarray(y, dtype=float)

    if X is None:
        # Intercept-only model
        X = np.ones((len(y), 1))
    elif isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X, dtype=float)

    # Validate dimensions
    if len(y) != X.shape[0]:
        raise ValueError(
            f"Length of y ({len(y)}) must match number of rows in X ({X.shape[0]})"
        )

    # Handle random effects
    if random_effects is None:
        random_effects = []

    if len(random_effects) > 0:
        if groups_data is None:
            raise ValueError(
                "groups_data must be provided when random_effects are specified"
            )

        # Convert groups_data if DataFrame
        if isinstance(groups_data, pd.DataFrame):
            groups_dict = {col: groups_data[col].values for col in groups_data.columns}
        else:
            groups_dict = groups_data

        # Construct Z matrix
        Z, Z_info = construct_Z_matrix(X, random_effects, groups_dict)

        if Z.shape[0] != len(y):
            raise ValueError(
                f"Z matrix has {Z.shape[0]} rows but y has {len(y)} elements"
            )

    else:
        # No random effects - use dummy Z matrix
        Z = np.zeros((len(y), 0))
        Z_info = []

    # Fit model based on family
    if family == "gaussian":
        if len(random_effects) == 0:
            raise ValueError(
                "At least one random effect required for GAMM. "
                "For models without random effects, use fit_glm or fit_gam instead."
            )

        result = fit_gamm_gaussian(
            X_parametric=X,
            X_smooth=None,
            Z=Z,
            Z_info=Z_info,
            y=y,
            S_smooth=None,
            lambda_smooth=None,
            covariance=covariance,
            maxiter=maxiter,
            tol=tol,
        )

        return result
    else:
        # Future: PQL/Laplace for GLM families
        raise NotImplementedError(
            f"Family '{family}' not yet implemented. "
            "Will be available in Milestone 4."
        )


def fit_gamm_with_smooth(
    y: np.ndarray | pd.Series,
    X_parametric: np.ndarray | pd.DataFrame,
    X_smooth: dict[str, np.ndarray],
    S_smooth: dict[str, np.ndarray],
    random_effects: list[RandomEffect],
    groups_data: dict[str, np.ndarray] | pd.DataFrame,
    lambda_smooth: dict[str, float] | None = None,
    family: str = "gaussian",
    covariance: str = "unstructured",
    maxiter: int = 100,
    tol: float = 1e-6,
) -> GAMMResult:
    """Fit GAMM with smooth terms (advanced interface).

    Parameters
    ----------
    y : array-like, shape (n,)
        Response variable.
    X_parametric : array-like, shape (n, p)
        Design matrix for parametric fixed effects.
    X_smooth : dict of str -> ndarray
        Smooth term basis matrices by term name.
    S_smooth : dict of str -> ndarray
        Penalty matrices by term name.
    random_effects : list of RandomEffect
        Random effect specifications.
    groups_data : dict or DataFrame
        Grouping variables for random effects.
    lambda_smooth : dict of str -> float, optional
        Smoothing parameters by term name. If None, uses λ=1.0 for all.
    family : str, default='gaussian'
        Distribution family.
    covariance : str, default='unstructured'
        Covariance structure for random effects.
    maxiter : int, default=100
        Maximum iterations.
    tol : float, default=1e-6
        Convergence tolerance.

    Returns
    -------
    result : GAMMResult
        Fitted GAMM result.

    Examples
    --------
    >>> # GAMM with smooth term + random intercept
    >>> from aurora.smoothing import fit_cubic_spline
    >>> from aurora.models.gamm import fit_gamm_with_smooth, RandomEffect
    >>>
    >>> # Data
    >>> n = 100
    >>> x_smooth = np.linspace(0, 2*np.pi, n)
    >>> groups = np.repeat(np.arange(10), 10)
    >>>
    >>> # Build smooth basis
    >>> import scipy.interpolate as interp
    >>> k = 10
    >>> knots = np.linspace(0, 2*np.pi, k-2)
    >>> tck = interp.splrep(x_smooth, np.zeros(n), t=knots[1:-1], k=3)
    >>> X_smooth_basis = interp.BSpline.design_matrix(x_smooth, tck[0], 3)
    >>>
    >>> # Parametric design
    >>> X_para = np.ones((n, 1))
    >>>
    >>> # Random effects
    >>> re = RandomEffect(grouping='subject')
    >>>
    >>> # Generate response
    >>> y = 2.0 + np.sin(x_smooth) + np.random.randn(10)[groups] + np.random.randn(n)*0.3
    >>>
    >>> # Penalty matrix (second derivative)
    >>> S = np.diag([0]*2 + [1]*(k-2))  # Penalize non-linear components
    >>>
    >>> result = fit_gamm_with_smooth(
    ...     y=y,
    ...     X_parametric=X_para,
    ...     X_smooth={'s(x)': X_smooth_basis},
    ...     S_smooth={'s(x)': S},
    ...     random_effects=[re],
    ...     groups_data={'subject': groups},
    ...     lambda_smooth={'s(x)': 0.1},
    ...     covariance='identity'
    ... )

    Notes
    -----
    This is the advanced interface for GAMMs with smooth terms.
    For basic usage without smooth terms, use fit_gamm() instead.

    Smooth term construction (basis matrices and penalties) should be
    done using the smoothing module from Phase 3.
    """
    # Input validation
    if family != "gaussian":
        raise ValueError(f"Only 'gaussian' family currently supported, got '{family}'")

    # Convert inputs
    if isinstance(y, pd.Series):
        y = y.values
    y = np.asarray(y, dtype=float)

    if isinstance(X_parametric, pd.DataFrame):
        X_parametric = X_parametric.values
    X_parametric = np.asarray(X_parametric, dtype=float)

    # Convert groups_data if DataFrame
    if isinstance(groups_data, pd.DataFrame):
        groups_dict = {col: groups_data[col].values for col in groups_data.columns}
    else:
        groups_dict = groups_data

    # Construct Z matrix
    Z, Z_info = construct_Z_matrix(X_parametric, random_effects, groups_dict)

    # Call fitting function
    result = fit_gamm_gaussian(
        X_parametric=X_parametric,
        X_smooth=X_smooth,
        Z=Z,
        Z_info=Z_info,
        y=y,
        S_smooth=S_smooth,
        lambda_smooth=lambda_smooth,
        covariance=covariance,
        maxiter=maxiter,
        tol=tol,
    )

    return result


def predict_from_gamm(
    result: GAMMResult,
    X_new: np.ndarray | pd.DataFrame,
    groups_new: np.ndarray | pd.Series | None = None,
    X_smooth_new: dict[str, np.ndarray] | None = None,
    include_random: bool = False,
) -> np.ndarray:
    """Make predictions from fitted GAMM.

    Parameters
    ----------
    result : GAMMResult
        Fitted GAMM result.
    X_new : array-like, shape (n_new, p)
        New parametric design matrix.
    groups_new : array-like, shape (n_new,), optional
        Group indicators for new observations.
        Required if include_random=True.
    X_smooth_new : dict of str -> ndarray, optional
        New smooth term basis matrices.
    include_random : bool, default=False
        Whether to include random effects in predictions.

    Returns
    -------
    predictions : ndarray, shape (n_new,)
        Predicted values.

    Examples
    --------
    >>> # Population-level predictions (no random effects)
    >>> X_new = np.column_stack([np.ones(20), np.random.randn(20)])
    >>> pred_pop = predict_from_gamm(result, X_new)
    >>>
    >>> # Conditional predictions (with random effects)
    >>> groups_new = np.array([0, 0, 1, 1, 2, 2, ...])  # Existing groups
    >>> pred_cond = predict_from_gamm(
    ...     result, X_new,
    ...     groups_new=groups_new,
    ...     include_random=True
    ... )

    Notes
    -----
    - Population-level predictions are appropriate for new, unobserved groups
    - Conditional predictions are appropriate for predicting within observed groups
    - For new groups, random effects default to 0 (population mean)
    """
    # Convert inputs
    if isinstance(X_new, pd.DataFrame):
        X_new = X_new.values
    X_new = np.asarray(X_new, dtype=float)

    # Construct Z_new if needed
    Z_new = None
    if include_random:
        if groups_new is None:
            raise ValueError("groups_new required when include_random=True")

        if isinstance(groups_new, pd.Series):
            groups_new = groups_new.values
        groups_new = np.asarray(groups_new)

        if result._Z_info is None or len(result._Z_info) == 0:
            raise ValueError("Result does not contain random effects information")

        # Need to reconstruct Z with proper structure for random slopes
        # We need the random_effects specification from the original fit
        # For now, reconstruct based on Z_info
        Z_info = result._Z_info[0]  # Assuming single random effect term
        n_effects = Z_info['n_effects']
        grouping_var = Z_info['grouping']

        # Reconstruct Z for new data using construct_Z_matrix
        # But we need the original random_effects specification...
        # Simplification: build Z manually based on Z_info
        n_new = len(X_new)
        q = Z_info['end_col'] - Z_info['start_col']
        Z_new = np.zeros((n_new, q))

        # If n_effects == 1, it's just indicator matrix
        # If n_effects > 1, we need variables from X_new
        if n_effects == 1:
            # Random intercept only
            for i, group_id in enumerate(groups_new):
                if 0 <= group_id < result.n_groups:
                    Z_new[i, group_id] = 1
        else:
            # Random intercept + slopes
            # Columns of Z are organized as: [intercept_g0, slope1_g0, ..., intercept_g1, slope1_g1, ...]
            n_groups = result.n_groups
            for i, group_id in enumerate(groups_new):
                if 0 <= group_id < n_groups:
                    # Set intercept
                    col_base = group_id * n_effects
                    Z_new[i, col_base] = 1
                    # Set slopes (use variables from X_new)
                    for effect_idx in range(1, n_effects):
                        # Assume variables are columns 1, 2, ... in X_new
                        Z_new[i, col_base + effect_idx] = X_new[i, effect_idx]

    # Call predict_gamm
    predictions = predict_gamm(
        result=result,
        X_parametric_new=X_new,
        X_smooth_new=X_smooth_new,
        Z_new=Z_new,
        include_random=include_random,
    )

    return predictions
