# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Additive GAM fitting with multiple smooth terms.

This module implements multivariate Generalized Additive Models with
multiple smooth and parametric terms:
    y = β₀ + f₁(x₁) + f₂(x₂) + ... + βₖxₖ + ε
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aurora.models.gam.terms import ParametricTerm, SmoothTerm
from aurora.smoothing.selection.gcv import select_smoothing_parameter
from aurora.smoothing.splines.bspline import BSplineBasis
from aurora.smoothing.splines.cubic import CubicSplineBasis


class AdditiveGAMResult:
    """Result of fitting an additive GAM with multiple terms.

    Parameters
    ----------
    parametric_coef : ndarray
        Coefficients for parametric terms (including intercept).
    smooth_coef : dict
        Dictionary mapping smooth term names to their coefficients.
    smooth_bases : dict
        Dictionary mapping smooth term names to their basis objects.
    lambda_values : dict
        Dictionary mapping smooth term names to their smoothing parameters.
    edf_values : dict
        Dictionary mapping smooth term names to their effective degrees of freedom.
    fitted_values : ndarray
        Fitted values at training data.
    residuals : ndarray
        Residuals (y - fitted_values).
    X_parametric : ndarray
        Parametric design matrix.
    X_train : ndarray
        Full design matrix used for training.
    y : ndarray
        Training response values.
    smooth_terms : list of SmoothTerm
        Smooth term specifications.
    parametric_terms : list of ParametricTerm
        Parametric term specifications.
    weights : ndarray, optional
        Observation weights.
    gcv_score : float, optional
        GCV score if lambdas were selected automatically.
    """

    def __init__(
        self,
        parametric_coef: np.ndarray,
        smooth_coef: dict[str, np.ndarray],
        smooth_bases: dict[str, Any],
        lambda_values: dict[str, float],
        edf_values: dict[str, float],
        fitted_values: np.ndarray,
        residuals: np.ndarray,
        X_parametric: np.ndarray,
        X_train: np.ndarray,
        y: np.ndarray,
        smooth_terms: list[SmoothTerm],
        parametric_terms: list[ParametricTerm],
        weights: np.ndarray | None = None,
        gcv_score: float | None = None,
    ):
        self.parametric_coef = parametric_coef
        self.smooth_coef = smooth_coef
        self.smooth_bases = smooth_bases
        self.lambda_values = lambda_values
        self.edf_values = edf_values
        self.fitted_values = fitted_values
        self.residuals = residuals
        self.X_parametric = X_parametric
        self.X_train = X_train
        self.y = y
        self.smooth_terms = smooth_terms
        self.parametric_terms = parametric_terms
        self.weights = weights
        self.gcv_score = gcv_score

        self.n_obs_ = len(y)
        self.n_smooth_terms_ = len(smooth_terms)
        self.n_parametric_terms_ = len(parametric_terms)
        self.total_edf_ = sum(edf_values.values()) + len(parametric_coef)

    @property
    def r_squared(self) -> float:
        """Coefficient of determination (R²).

        Returns
        -------
        r_squared : float
            R² value, proportion of variance explained by the model.
        """
        rss = np.sum(self.residuals**2)
        if self.weights is not None:
            rss = np.sum(self.weights * self.residuals**2)

        tss = np.sum((self.y - np.mean(self.y)) ** 2)
        return 1 - rss / tss

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict at new data points.

        Parameters
        ----------
        X_new : ndarray, shape (m, n_features)
            New predictor values. Must have same number of columns as
            training data.

        Returns
        -------
        y_pred : ndarray, shape (m,)
            Predicted values.

        Notes
        -----
        Predictions are computed as:
            ŷ = X_param @ β_param + Σ X_smooth_j @ β_smooth_j
        """
        X_new_arr = np.asarray(X_new, dtype=np.float64)

        if X_new_arr.ndim == 1:
            X_new_arr = X_new_arr.reshape(1, -1)

        n_new = X_new_arr.shape[0]

        # Start with parametric terms
        if len(self.parametric_terms) > 0:
            # Extract parametric columns (including intercept)
            X_param_new = np.column_stack(
                [
                    np.ones(n_new),  # Intercept
                    *[
                        X_new_arr[:, t.variable]
                        if isinstance(t.variable, int)
                        else X_new_arr[:, t.variable]
                        for t in self.parametric_terms
                    ],
                ]
            )
            y_pred = X_param_new @ self.parametric_coef
        else:
            y_pred = np.zeros(n_new)

        # Add smooth terms
        for i, term in enumerate(self.smooth_terms):
            term_name = f"s({term.variable})"

            # Get predictor values
            if isinstance(term.variable, int):
                x_smooth = X_new_arr[:, term.variable]
            else:
                x_smooth = X_new_arr[:, term.variable]

            # Evaluate basis
            basis = self.smooth_bases[term_name]
            X_smooth_new = basis.basis_matrix(x_smooth)

            # Add contribution
            y_pred += X_smooth_new @ self.smooth_coef[term_name]

        return y_pred

    def summary(self) -> str:
        """Generate summary string of additive GAM fit.

        Returns
        -------
        summary : str
            Formatted summary with parametric and smooth terms.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("Additive Generalized Additive Model (GAM) - Fitted Summary")
        lines.append("=" * 70)
        lines.append("")

        # Model information
        lines.append("Model Structure:")
        lines.append(f"  Smooth terms:        {self.n_smooth_terms_}")
        lines.append(f"  Parametric terms:    {self.n_parametric_terms_}")
        lines.append(f"  Observations:        {self.n_obs_}")
        lines.append(f"  Total EDF:           {self.total_edf_:.2f}")
        if self.weights is not None:
            lines.append("  Weighted fit:        Yes")
        lines.append("")

        # Parametric terms
        if len(self.parametric_terms) > 0:
            lines.append("Parametric Coefficients:")
            lines.append(f"  Intercept:           {self.parametric_coef[0]:.4f}")
            for i, term in enumerate(self.parametric_terms):
                lines.append(
                    f"  {term.variable}:             {self.parametric_coef[i + 1]:.4f}"
                )
            lines.append("")

        # Smooth terms
        lines.append("Smooth Terms:")
        for term in self.smooth_terms:
            term_name = f"s({term.variable})"
            lines.append(f"  {term_name}:")
            lines.append(f"    Basis:             {term.basis_type}")
            lines.append(f"    n_basis:           {term.n_basis}")
            lines.append(f"    Lambda:            {self.lambda_values[term_name]:.6e}")
            lines.append(f"    EDF:               {self.edf_values[term_name]:.2f}")
        lines.append("")

        # Fit statistics
        rss = np.sum(self.residuals**2)
        if self.weights is not None:
            rss = np.sum(self.weights * self.residuals**2)

        tss = np.sum((self.y - np.mean(self.y)) ** 2)
        r_squared = 1 - rss / tss

        lines.append("Fit Statistics:")
        lines.append(f"  Residual sum sq:     {rss:.4f}")
        lines.append(f"  R-squared:           {r_squared:.4f}")
        lines.append(f"  Residual std:        {np.std(self.residuals):.4f}")
        if self.gcv_score is not None:
            lines.append(f"  GCV score:           {self.gcv_score:.6e}")
        lines.append("")

        # Residual diagnostics
        lines.append("Residuals:")
        lines.append(f"  Min:                 {np.min(self.residuals):.4f}")
        lines.append(f"  Q1:                  {np.quantile(self.residuals, 0.25):.4f}")
        lines.append(f"  Median:              {np.median(self.residuals):.4f}")
        lines.append(f"  Q3:                  {np.quantile(self.residuals, 0.75):.4f}")
        lines.append(f"  Max:                 {np.max(self.residuals):.4f}")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"AdditiveGAMResult(n_obs={self.n_obs_}, "
            f"n_smooth={self.n_smooth_terms_}, "
            f"n_parametric={self.n_parametric_terms_}, "
            f"total_edf={self.total_edf_:.2f})"
        )


def fit_additive_gam(
    X: np.ndarray,
    y: np.ndarray,
    smooth_terms: list[SmoothTerm],
    parametric_terms: list[ParametricTerm] | None = None,
    weights: np.ndarray | None = None,
    method: str = "GCV",
) -> AdditiveGAMResult:
    """Fit additive GAM with multiple smooth and parametric terms.

    This function fits a model of the form:
        y = β₀ + f₁(x₁) + f₂(x₂) + ... + βₖxₖ + ε

    where f_j are smooth functions represented as penalized splines,
    and the last terms are linear (parametric).

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Predictor matrix.
    y : ndarray, shape (n,)
        Response variable.
    smooth_terms : list of SmoothTerm
        Specifications for smooth terms. Each defines which column of X
        to smooth and how.
    parametric_terms : list of ParametricTerm, optional
        Specifications for parametric (linear) terms.
    weights : ndarray, shape (n,), optional
        Observation weights for weighted least squares.
    method : {'GCV', 'REML'}, default='GCV'
        Method for smoothing parameter selection:
        - 'GCV': Generalized Cross-Validation
        - 'REML': Restricted Maximum Likelihood
        Note: Currently both methods use a single λ for all smooth terms.

    Returns
    -------
    result : AdditiveGAMResult
        Fitted additive GAM result.

    Notes
    -----
    The model minimizes the penalized least squares criterion:
        ||y - Xβ||² + Σⱼ λⱼ βⱼ'Sⱼβⱼ

    Each smooth term has its own smoothing parameter λⱼ selected via GCV.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.gam import fit_additive_gam, SmoothTerm
    >>> # Generate data with two nonlinear effects
    >>> n = 200
    >>> X = np.random.randn(n, 2)
    >>> y = np.sin(2 * X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(n)
    >>> # Fit additive GAM
    >>> result = fit_additive_gam(
    ...     X, y,
    ...     smooth_terms=[
    ...         SmoothTerm(variable=0, n_basis=10),
    ...         SmoothTerm(variable=1, n_basis=10)
    ...     ]
    ... )
    >>> print(result.summary())

    References
    ----------
    Wood, S.N. (2017). Generalized Additive Models: An Introduction with R.
    """
    # Validate inputs
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if X_arr.ndim != 2:
        raise ValueError("X must be 2-dimensional")

    if y_arr.ndim != 1:
        raise ValueError("y must be 1-dimensional")

    n, p = X_arr.shape
    if len(y_arr) != n:
        raise ValueError(
            f"X and y must have same number of rows, got {n} and {len(y_arr)}"
        )

    if len(smooth_terms) == 0:
        raise ValueError("Must specify at least one smooth term")

    if parametric_terms is None:
        parametric_terms = []

    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {weights_arr.shape}")
        W = np.diag(weights_arr)
    else:
        weights_arr = None
        W = np.eye(n)

    # Validate method
    if method.upper() not in ("GCV", "REML"):
        raise ValueError(f"method must be 'GCV' or 'REML', got '{method}'")

    # Build design matrix and penalty matrix for each smooth term
    smooth_bases = {}
    smooth_design_matrices = {}
    smooth_penalties = {}

    for term in smooth_terms:
        term_name = f"s({term.variable})"

        # Extract predictor
        if isinstance(term.variable, int):
            if term.variable >= p:
                raise ValueError(
                    f"Variable index {term.variable} out of range (0-{p - 1})"
                )
            x_smooth = X_arr[:, term.variable]
        else:
            raise NotImplementedError("Named variables require DataFrame support")

        # Create basis
        if term.basis_type == "bspline":
            knots = BSplineBasis.create_knots(
                x_smooth, n_basis=term.n_basis, degree=3, method=term.knot_method
            )
            basis = BSplineBasis(knots, degree=3)
        elif term.basis_type == "cubic":
            knots_interior = CubicSplineBasis.create_knots(
                x_smooth, n_knots=term.n_basis - 2, method=term.knot_method
            )
            basis = CubicSplineBasis(knots_interior)
        else:
            raise NotImplementedError(
                f"basis_type='{term.basis_type}' not yet implemented"
            )

        # Compute basis matrix and penalty
        X_smooth = basis.basis_matrix(x_smooth)

        if term.basis_type == "bspline":
            S_smooth = basis.penalty_matrix(order=term.penalty_order)
        else:  # cubic
            S_smooth = basis.penalty_matrix()

        smooth_bases[term_name] = basis
        smooth_design_matrices[term_name] = X_smooth
        smooth_penalties[term_name] = S_smooth

    # Build parametric design matrix (intercept + parametric terms)
    X_parametric_list = [np.ones(n)]  # Intercept

    for term in parametric_terms:
        if isinstance(term.variable, int):
            if term.variable >= p:
                raise ValueError(
                    f"Variable index {term.variable} out of range (0-{p - 1})"
                )
            X_parametric_list.append(X_arr[:, term.variable])
        else:
            raise NotImplementedError("Named variables require DataFrame support")

    X_parametric = np.column_stack(X_parametric_list)

    # Assemble full design matrix: [X_parametric | X_smooth1 | X_smooth2 | ...]
    design_matrices = [X_parametric] + list(smooth_design_matrices.values())
    X_full = np.column_stack(design_matrices)

    # Build block-diagonal penalty matrix
    # Parametric terms have zero penalty
    n_parametric = X_parametric.shape[1]
    penalty_blocks = [np.zeros((n_parametric, n_parametric))]

    for term in smooth_terms:
        term_name = f"s({term.variable})"
        penalty_blocks.append(smooth_penalties[term_name])

    # Create block-diagonal penalty matrix
    from scipy.linalg import block_diag

    S_full = block_diag(*penalty_blocks)

    # Select smoothing parameters
    # For now, use single lambda for all terms (both GCV and REML)
    # Per-term lambda optimization with REML is complex and requires more sophisticated
    # optimization than alternating optimization provides
    if method.upper() == "REML":
        from aurora.smoothing.selection.reml import select_smoothing_parameter_reml

        selection_result = select_smoothing_parameter_reml(
            y_arr,
            X_full,
            S_full,
            weights=weights_arr,
            lambda_min=1e-6,
            lambda_max=1e6,
        )
        gcv_score = selection_result.get("reml_score")
    else:
        # GCV selection
        selection_result = select_smoothing_parameter(
            y_arr,
            X_full,
            S_full,
            weights=weights_arr,
            lambda_min=1e-6,
            lambda_max=1e6,
        )
        gcv_score = selection_result["gcv_score"]

    lambda_opt = selection_result["lambda_opt"]
    coefficients = selection_result["coefficients"]
    fitted_values = selection_result["fitted_values"]

    # All terms use same lambda (no per-term optimization yet)
    use_per_term_lambda = False

    # Split coefficients back into parametric and smooth components
    idx = 0

    # Parametric coefficients
    parametric_coef = coefficients[idx : idx + n_parametric]
    idx += n_parametric

    # Smooth coefficients
    smooth_coef = {}

    # Compute EDF for each term with single lambda
    # Compute influence matrix for EDF calculation
    XtWX = X_full.T @ W @ X_full
    A = XtWX + lambda_opt * S_full

    try:
        A_inv = np.linalg.inv(A)
        H = X_full @ A_inv @ X_full.T @ W
    except np.linalg.LinAlgError:
        # Fallback if singular
        H = None

    lambda_values = {}
    edf_values = {}
    idx = n_parametric  # Reset to start of smooth terms

    for term in smooth_terms:
        term_name = f"s({term.variable})"
        n_basis = smooth_design_matrices[term_name].shape[1]

        smooth_coef[term_name] = coefficients[idx : idx + n_basis]

        # All smooths use same lambda
        lambda_values[term_name] = lambda_opt

        # Compute EDF for this smooth term
        # EDF_j = trace(X_j (X'WX + λS)^(-1) X_j' W)
        try:
            if H is not None:
                # Get columns corresponding to this smooth term
                X_j = X_full[:, idx : idx + n_basis]
                # EDF for this term is trace of its influence
                H_j = X_j @ A_inv[idx : idx + n_basis, :] @ X_full.T @ W
                edf_j = float(np.trace(H_j))

                # Sanity check: EDF should be between 0 and n_basis
                if not (0 <= edf_j <= n_basis + 1):
                    # Fall back to simple division
                    edf_j = max(
                        0.0,
                        (selection_result.get("edf", n_basis) - n_parametric)
                        / len(smooth_terms),
                    )

                edf_values[term_name] = edf_j
            else:
                # Fallback: equal division (subtract parametric)
                edf_values[term_name] = max(
                    0.0,
                    (selection_result.get("edf", n_basis) - n_parametric)
                    / len(smooth_terms),
                )
        except (np.linalg.LinAlgError, ValueError):
            # Numerical issues - use fallback
            edf_values[term_name] = max(
                0.0,
                (selection_result.get("edf", n_basis) - n_parametric)
                / len(smooth_terms),
            )

        idx += n_basis

    # Compute residuals
    residuals = y_arr - fitted_values

    # Create result object
    result = AdditiveGAMResult(
        parametric_coef=parametric_coef,
        smooth_coef=smooth_coef,
        smooth_bases=smooth_bases,
        lambda_values=lambda_values,
        edf_values=edf_values,
        fitted_values=fitted_values,
        residuals=residuals,
        X_parametric=X_parametric,
        X_train=X_full,
        y=y_arr,
        smooth_terms=smooth_terms,
        parametric_terms=parametric_terms,
        weights=weights_arr,
        gcv_score=gcv_score,
    )

    return result


def fit_gam_formula(
    formula: str,
    data: dict[str, np.ndarray] | np.ndarray,
    weights: np.ndarray | None = None,
    method: str = "GCV",
) -> AdditiveGAMResult:
    """Fit additive GAM using R-style formula syntax.

    Convenience function that parses a formula string and fits a GAM.

    Parameters
    ----------
    formula : str
        Formula string in R-style syntax, e.g.:
        - "y ~ s(x1) + s(x2)"
        - "y ~ s(x1, n_basis=15) + s(x2, basis='cubic') + x3"
    data : dict or ndarray
        If dict: mapping from variable names to arrays
        If ndarray: shape (n, p) array where formula uses column indices
    weights : ndarray, shape (n,), optional
        Observation weights.
    method : {'GCV', 'REML'}, default='GCV'
        Method for smoothing parameter selection.

    Returns
    -------
    result : AdditiveGAMResult
        Fitted additive GAM result.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.gam import fit_gam_formula
    >>> # Using column indices with array
    >>> n = 200
    >>> X = np.random.randn(n, 3)
    >>> y = np.sin(2 * X[:, 0]) + np.cos(X[:, 1]) + 0.5 * X[:, 2] + 0.1 * np.random.randn(n)
    >>> result = fit_gam_formula("0 ~ s(1) + s(2) + 3", np.column_stack([y, X]))
    >>> print(result.summary())

    >>> # Using dict with named variables
    >>> data = {
    ...     'response': y,
    ...     'temp': X[:, 0],
    ...     'pressure': X[:, 1],
    ...     'humidity': X[:, 2]
    ... }
    >>> result = fit_gam_formula("response ~ s(temp) + s(pressure) + humidity", data)

    Notes
    -----
    When using column indices (integers), the data should be a 2D array where
    the first column can be the response variable.

    When using variable names (strings), data should be a dict mapping names
    to 1D arrays.
    """
    from aurora.models.gam.formula import parse_formula

    # Parse formula
    spec = parse_formula(formula)

    # Extract data based on type
    if isinstance(data, dict):
        # Dict mode: extract by variable name
        y = data[spec.response]

        # Collect all unique variable names
        variable_names = set()
        for term in spec.smooth_terms:
            if isinstance(term.variable, str):
                variable_names.add(term.variable)
        for term in spec.parametric_terms:
            if isinstance(term.variable, str):
                variable_names.add(term.variable)

        # Build mapping from name to column index
        var_to_idx = {name: i for i, name in enumerate(sorted(variable_names))}

        # Build X matrix
        X_columns = [data[name] for name in sorted(variable_names)]
        X = np.column_stack(X_columns)

        # Update term variables to use indices
        updated_smooth_terms = []
        for term in spec.smooth_terms:
            if isinstance(term.variable, str):
                new_var = var_to_idx[term.variable]
                updated_smooth_terms.append(
                    SmoothTerm(
                        variable=new_var,
                        basis_type=term.basis_type,
                        n_basis=term.n_basis,
                        penalty_order=term.penalty_order,
                        lambda_=term.lambda_,
                        knot_method=term.knot_method,
                    )
                )
            else:
                updated_smooth_terms.append(term)

        updated_parametric_terms = []
        for term in spec.parametric_terms:
            if isinstance(term.variable, str):
                new_var = var_to_idx[term.variable]
                updated_parametric_terms.append(ParametricTerm(variable=new_var))
            else:
                updated_parametric_terms.append(term)

        smooth_terms = updated_smooth_terms
        parametric_terms = updated_parametric_terms

    else:
        # Array mode: use column indices directly
        data_arr = np.asarray(data)
        if data_arr.ndim != 2:
            raise ValueError("data array must be 2-dimensional")

        # Extract response
        try:
            response_idx = int(spec.response)
        except ValueError:
            raise ValueError(
                f"When using array data, response must be a column index, got '{spec.response}'"
            )

        y = data_arr[:, response_idx]

        # Build X from remaining columns
        all_indices = set(range(data_arr.shape[1]))
        all_indices.discard(response_idx)

        # Get predictor indices from terms
        predictor_indices = set()
        for term in spec.smooth_terms:
            if not isinstance(term.variable, int):
                raise ValueError(
                    f"When using array data, all variables must be column indices, "
                    f"got '{term.variable}'"
                )
            predictor_indices.add(term.variable)

        for term in spec.parametric_terms:
            if not isinstance(term.variable, int):
                raise ValueError(
                    f"When using array data, all variables must be column indices, "
                    f"got '{term.variable}'"
                )
            predictor_indices.add(term.variable)

        # Extract predictor columns
        predictor_indices_sorted = sorted(predictor_indices)
        X = data_arr[:, predictor_indices_sorted]

        # Remap variable indices in terms to new X matrix
        idx_map = {
            old_idx: new_idx for new_idx, old_idx in enumerate(predictor_indices_sorted)
        }

        smooth_terms = [
            SmoothTerm(
                variable=idx_map[term.variable],
                basis_type=term.basis_type,
                n_basis=term.n_basis,
                penalty_order=term.penalty_order,
                lambda_=term.lambda_,
                knot_method=term.knot_method,
            )
            for term in spec.smooth_terms
        ]

        parametric_terms = [
            ParametricTerm(variable=idx_map[term.variable])
            for term in spec.parametric_terms
        ]

    # Fit GAM
    return fit_additive_gam(
        X=X,
        y=y,
        smooth_terms=smooth_terms,
        parametric_terms=parametric_terms if parametric_terms else None,
        weights=weights,
        method=method,
    )


__all__ = ["fit_additive_gam", "fit_gam_formula", "AdditiveGAMResult"]
