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
    formula: str | None = None,
    data: pd.DataFrame | dict | None = None,
    y: np.ndarray | pd.Series | None = None,
    X: np.ndarray | pd.DataFrame | None = None,
    random_effects: list[RandomEffect] | None = None,
    groups_data: dict[str, np.ndarray] | pd.DataFrame | None = None,
    family: str = "gaussian",
    covariance: str = "unstructured",
    maxiter: int = 100,
    tol: float = 1e-6,
    use_sparse: bool = False,
    backend: str = "numpy",
    device: str | None = None,
) -> GAMMResult:
    """Fit a Generalized Additive Mixed Model.

    Can be called in two modes:
    1. **Formula mode**: Pass `formula` and `data`
    2. **Matrix mode**: Pass `y`, `X`, `random_effects`, and `groups_data`

    Parameters
    ----------
    formula : str, optional
        R-style formula string (formula mode).
        Example: "y ~ x1 + (1 | subject)" or "y ~ x1 + (1 + time | subject)"
    data : DataFrame or dict, optional
        Data containing all variables referenced in formula (formula mode).
    y : array-like, shape (n,), optional
        Response variable (matrix mode).
    X : array-like, shape (n, p), optional
        Design matrix for parametric fixed effects (matrix mode).
        If None, uses intercept-only model.
    random_effects : list of RandomEffect, optional
        Random effect specifications (matrix mode).
    groups_data : dict or DataFrame, optional
        Grouping variables for random effects (matrix mode).
        Keys should match RandomEffect.grouping names.
    family : str, default='gaussian'
        Distribution family. Currently only 'gaussian' supported.
    covariance : str, default='unstructured'
        Covariance structure for random effects. Options:
        - 'identity': Independent random effects (default)
        - 'diagonal': Heterogeneous variances, no correlation
        - 'unstructured': Full covariance matrix
        - 'ar1': Autoregressive AR(1) for temporal correlation
        - 'compound_symmetry' or 'cs': Exchangeable correlation
        - 'exponential': Spatial decay (requires coordinates)
        - 'matern': Matérn spatial covariance (requires coordinates)

        For temporal/longitudinal data, use 'ar1'. For clustered data
        with equal correlations, use 'compound_symmetry'.
    maxiter : int, default=100
        Maximum iterations for optimization.
    tol : float, default=1e-6
        Convergence tolerance.
    use_sparse : bool, default=False
        Whether to use sparse matrix operations for smooth terms.
        When True, uses sparse CSR format for basis matrices and sparse
        linear solvers. Provides significant speedup (10-100×) and memory
        reduction (6-8×) for large problems with B-spline basis functions.
        Currently only supported when smooth terms use B-spline basis.
    backend : str, default='numpy'
        Computational backend: 'numpy', 'torch', or 'jax'.
    device : str, optional
        Device for computation (for torch backend): 'cpu', 'cuda', 'cuda:0', etc.
        If None, uses CUDA if available, else CPU.

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
    **Formula mode (recommended):**

    >>> # Random intercept model
    >>> import pandas as pd
    >>> import numpy as np
    >>> from aurora.models.gamm import fit_gamm
    >>>
    >>> # Generate data
    >>> n_groups, n_per_group = 10, 20
    >>> n = n_groups * n_per_group
    >>> data = pd.DataFrame({
    ...     'y': np.random.randn(n),
    ...     'x1': np.random.randn(n),
    ...     'subject': np.repeat(np.arange(n_groups), n_per_group)
    ... })
    >>>
    >>> # Fit model with formula
    >>> result = fit_gamm(
    ...     formula="y ~ x1 + (1 | subject)",
    ...     data=data,
    ...     covariance='identity'
    ... )
    >>>
    >>> # Access results
    >>> print(result.beta_parametric)  # Fixed effects
    >>> print(result.variance_components)  # Random effect variance

    >>> # Random intercept + slope
    >>> result = fit_gamm(
    ...     formula="y ~ x1 + (1 + x1 | subject)",
    ...     data=data,
    ...     covariance='unstructured'
    ... )
    >>> print(result.variance_components)  # 2x2 covariance matrix

    **Matrix mode (advanced):**

    >>> from aurora.models.gamm import RandomEffect
    >>>
    >>> # Manually construct matrices
    >>> groups = data['subject'].values
    >>> x = data['x1'].values
    >>> y = data['y'].values
    >>> X = np.column_stack([np.ones(n), x])
    >>>
    >>> # Specify random effects manually
    >>> re = RandomEffect(grouping='subject')
    >>> result = fit_gamm(
    ...     y=y,
    ...     X=X,
    ...     random_effects=[re],
    ...     groups_data={'subject': groups},
    ...     covariance='identity'
    ... )
    >>> print(result.beta_parametric)

    Notes
    -----
    This is a high-level interface that automatically constructs
    design matrices for random effects and calls the appropriate
    fitting function based on the family.

    For Gaussian family, uses exact REML estimation.
    For other families (future implementation), will use PQL or Laplace.

    Formula mode supports:
    - R-style formula syntax with lme4-style random effects
    - Automatic design matrix construction from DataFrames
    - Random intercepts: (1 | group)
    - Random slopes: (1 + x | group)
    - Nested effects: (1 | a/b)
    - Crossed effects: (1 | a) + (1 | b)
    """
    # Mode detection
    if formula is not None:
        # Formula mode
        if data is None:
            raise ValueError("data must be provided when using formula mode")
        if y is not None or X is not None or random_effects is not None:
            raise ValueError(
                "Cannot mix formula mode (formula, data) with matrix mode "
                "(y, X, random_effects). Use one or the other."
            )

        # Parse formula
        from aurora.models.gam.formula import parse_formula

        spec = parse_formula(formula)

        # Convert data to DataFrame if dict
        if isinstance(data, dict):
            data = pd.DataFrame(data)

        # Extract response
        if spec.response not in data.columns:
            raise ValueError(f"Response variable '{spec.response}' not found in data")
        y = data[spec.response].values

        # Build design matrix from parametric terms
        # Always include intercept
        X_cols = [np.ones(len(y))]
        var_to_col_idx = {'intercept': 0}  # Map variable names to X column indices

        for term in spec.parametric_terms:
            var_name = term.variable
            if isinstance(var_name, int):
                # Column index in data
                if var_name >= len(data.columns):
                    raise ValueError(f"Column index {var_name} out of range")
                col_name = data.columns[var_name]
                X_cols.append(data.iloc[:, var_name].values)
                var_to_col_idx[col_name] = len(X_cols) - 1
                var_to_col_idx[var_name] = len(X_cols) - 1  # Also map integer index
            elif var_name in data.columns:
                X_cols.append(data[var_name].values)
                var_to_col_idx[var_name] = len(X_cols) - 1
            else:
                raise ValueError(f"Variable '{var_name}' not found in data")

        # Add variables referenced in random slopes to X if not already there
        for re in spec.random_effects:
            for var_name in re.variables:
                if var_name not in var_to_col_idx:
                    # Add this variable to X
                    if isinstance(var_name, int):
                        if var_name >= len(data.columns):
                            raise ValueError(f"Variable index {var_name} out of range")
                        X_cols.append(data.iloc[:, var_name].values)
                        var_to_col_idx[var_name] = len(X_cols) - 1
                        var_to_col_idx[data.columns[var_name]] = len(X_cols) - 1
                    elif var_name in data.columns:
                        X_cols.append(data[var_name].values)
                        var_to_col_idx[var_name] = len(X_cols) - 1
                    else:
                        raise ValueError(f"Variable '{var_name}' not found in data")

        X = np.column_stack(X_cols) if X_cols else np.ones((len(y), 1))

        # Convert random effects variable names to column indices in X
        random_effects_converted = []
        for re in spec.random_effects:
            # Map variable names/indices to X column indices
            var_indices = tuple(var_to_col_idx[v] for v in re.variables)

            # Create new RandomEffect with column indices
            re_converted = RandomEffect(
                grouping=re.grouping,
                variables=var_indices,
                include_intercept=re.include_intercept,
                covariance=re.covariance,
            )
            random_effects_converted.append(re_converted)

        random_effects = random_effects_converted

        # Build groups_data dict
        groups_data = {}
        for re in spec.random_effects:  # Use original spec for grouping names
            group_var = re.grouping
            if isinstance(group_var, int):
                # Column index
                if group_var >= len(data.columns):
                    raise ValueError(f"Grouping column index {group_var} out of range")
                groups_data[group_var] = data.iloc[:, group_var].values
            elif group_var in data.columns:
                groups_data[group_var] = data[group_var].values
            else:
                raise ValueError(f"Grouping variable '{group_var}' not found in data")

        # Handle smooth terms from spec.smooth_terms
        X_smooth_dict = {}
        S_smooth_dict = {}
        lambda_smooth_dict = {}

        if len(spec.smooth_terms) > 0:
            from aurora.smoothing.splines.bspline import BSplineBasis

            for smooth_term in spec.smooth_terms:
                # Extract smooth variable and parameters
                var_name = smooth_term.variable
                term_name = f"s({var_name})"

                # Get the data for this variable
                if isinstance(var_name, int):
                    if var_name >= len(data.columns):
                        raise ValueError(f"Smooth variable index {var_name} out of range")
                    x_smooth = data.iloc[:, var_name].values
                elif var_name in data.columns:
                    x_smooth = data[var_name].values
                else:
                    raise ValueError(f"Smooth variable '{var_name}' not found in data")

                # Get parameters from SmoothTerm attributes
                n_basis = smooth_term.n_basis  # Default 10 in SmoothTerm
                degree = 3  # Default cubic splines for BSplineBasis
                penalty_order = smooth_term.penalty_order  # Default 2 in SmoothTerm
                lambda_val = smooth_term.lambda_  # May be None for automatic selection

                # Create B-spline basis
                knots = BSplineBasis.create_knots(
                    x_smooth, n_basis=n_basis, degree=degree, method=smooth_term.knot_method
                )
                basis = BSplineBasis(knots, degree=degree)

                # Build basis matrix (sparse if requested)
                if use_sparse:
                    X_smooth_dict[term_name] = basis.basis_matrix(x_smooth, sparse=True)
                else:
                    X_smooth_dict[term_name] = basis.basis_matrix(x_smooth)

                # Build penalty matrix
                S_smooth_dict[term_name] = basis.penalty_matrix(order=penalty_order)

                # Store smoothing parameter if provided
                if lambda_val is not None:
                    lambda_smooth_dict[term_name] = lambda_val

        # Use lambda_smooth_dict only if some values were specified
        if len(lambda_smooth_dict) > 0 and len(lambda_smooth_dict) == len(spec.smooth_terms):
            lambda_smooth_final = lambda_smooth_dict
        else:
            lambda_smooth_final = None  # Will use automatic selection

    else:
        # Matrix mode - require y
        if y is None:
            raise ValueError(
                "Either formula+data (formula mode) or y (matrix mode) must be provided"
            )

        # Initialize smooth term dictionaries (empty for matrix mode without formula)
        X_smooth_dict = {}
        S_smooth_dict = {}
        lambda_smooth_final = None

    # Input validation
    valid_families = ["gaussian", "poisson", "binomial", "gamma"]
    if family not in valid_families:
        raise ValueError(
            f"Family '{family}' not supported. "
            f"Valid families: {valid_families}"
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
            X_smooth=X_smooth_dict if len(X_smooth_dict) > 0 else None,
            Z=Z,
            Z_info=Z_info,
            y=y,
            S_smooth=S_smooth_dict if len(S_smooth_dict) > 0 else None,
            lambda_smooth=lambda_smooth_final,
            covariance=covariance,
            maxiter=maxiter,
            tol=tol,
            backend=backend,
            device=device,
        )

        return result
    else:
        # Non-Gaussian families: use PQL approximation
        if len(random_effects) == 0:
            raise ValueError(
                "At least one random effect required for GAMM. "
                "For models without random effects, use fit_glm instead."
            )

        # Check if smooth terms are present
        if len(X_smooth_dict) > 0:
            # Use PQL with smooth terms (Phase 5.1)
            from aurora.models.gamm.pql_smooth import fit_pql_with_smooth

            result_dict = fit_pql_with_smooth(
                X_parametric=X,
                X_smooth_dict=X_smooth_dict,
                Z=Z,
                Z_info=Z_info,
                y=y,
                family=family,
                S_smooth_dict=S_smooth_dict,
                lambda_smooth=lambda_smooth_final,
                maxiter_outer=maxiter,
                tol_outer=tol,
                verbose=False,
            )

            # Convert to GAMMResult format
            # Calculate residuals and other diagnostics
            mu = result_dict['fitted_values']
            residuals = y - mu

            result = GAMMResult(
                coefficients=np.concatenate([
                    result_dict['beta_parametric'],
                    np.concatenate([result_dict['beta_smooth'][name]
                                   for name in sorted(result_dict['beta_smooth'].keys())]),
                    result_dict['random_effects']
                ]),
                beta_parametric=result_dict['beta_parametric'],
                beta_smooth=result_dict['beta_smooth'],
                random_effects={f"re_{i}": result_dict['random_effects'][i:i+1]
                               for i in range(len(result_dict['random_effects']))},
                variance_components=result_dict['variance_components'],
                covariance_params=result_dict.get('covariance_params'),  # May be None for PQL smooth
                residual_variance=np.var(residuals),  # Approximate for non-Gaussian
                smoothing_parameters=result_dict['smoothing_parameters'],
                edf_total=sum(result_dict['edf_smooth'].values()) + len(result_dict['beta_parametric']),
                edf_parametric=float(len(result_dict['beta_parametric'])),
                edf_smooth=result_dict['edf_smooth'],
                fitted_values=result_dict['fitted_values'],
                residuals=residuals,
                log_likelihood=0.0,  # TODO: compute proper log-likelihood
                aic=0.0,
                bic=0.0,
                converged=result_dict['converged'],
                n_iterations=result_dict['n_iterations_outer'],
                n_obs=len(y),
                n_groups=len(set(result_dict['random_effects'])),
                family=family,
            )

            return result
        else:
            # Use PQL without smooth terms
            from aurora.models.gamm.pql import fit_pql_gamm

            result = fit_pql_gamm(
                X_parametric=X,
                X_smooth=None,
                Z=Z,
                Z_info=Z_info,
                y=y,
                family=family,
                covariance=covariance,
                maxiter_outer=maxiter,
                tol_outer=tol,
                backend=backend,
                device=device,
            )

            return result


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

        # Check if temporal covariance structure
        cov_type = Z_info.get('covariance', 'unstructured')
        is_temporal = cov_type in ('ar1', 'compound_symmetry', 'cs')

        # If n_effects == 1, it's just indicator matrix
        # If n_effects > 1, check if temporal or random slopes
        if n_effects == 1:
            # Random intercept only
            for i, group_id in enumerate(groups_new):
                if 0 <= group_id < result.n_groups:
                    Z_new[i, group_id] = 1
        elif is_temporal:
            # Temporal covariance: n_effects = n_times
            # Each observation has one random effect corresponding to its time index
            # Assume time index is in X_new[:, 1]
            n_groups = result.n_groups
            for i, group_id in enumerate(groups_new):
                if 0 <= group_id < n_groups:
                    # Extract time index from X_new (assuming it's in column 1)
                    time_idx = int(X_new[i, 1])  # Convert time to integer index
                    if 0 <= time_idx < n_effects:
                        col_idx = group_id * n_effects + time_idx
                        Z_new[i, col_idx] = 1
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
