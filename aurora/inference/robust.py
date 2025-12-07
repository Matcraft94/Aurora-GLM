# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Robust inference methods for GLM results.

This module provides robust standard errors and confidence intervals that are
valid under heteroscedasticity and potential model misspecification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

from ..models.base import GLMResult


class HCType(str, Enum):
    """Types of heteroscedasticity-consistent (HC) standard errors.

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator
        and a direct test for heteroskedasticity. Econometrica, 48(4), 817-838.
    MacKinnon, J.G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    Cribari-Neto, F. (2004). Asymptotic inference under heteroskedasticity of
        unknown form. Computational Statistics & Data Analysis, 45(2), 215-233.
    """

    HC0 = "HC0"  # White (1980) - basic sandwich estimator
    HC1 = "HC1"  # Degrees of freedom correction
    HC2 = "HC2"  # Leverage correction
    HC3 = "HC3"  # MacKinnon & White (1985) - better for small samples
    HC4 = "HC4"  # Cribari-Neto (2004) - best for influential points


@dataclass(frozen=True)
class RobustInferenceResult:
    """Container for robust inference results.

    Attributes
    ----------
    std_errors : np.ndarray
        Robust standard errors for coefficients (excluding intercept)
    coef_cov : np.ndarray
        Robust covariance matrix for all parameters (including intercept if fitted)
    intercept_std_error : float | None
        Robust standard error for intercept, if fitted
    hc_type : str
        Type of HC correction used
    """

    std_errors: np.ndarray
    coef_cov: np.ndarray
    intercept_std_error: float | None = None
    hc_type: str = "HC0"


def robust_covariance(
    result: GLMResult,
    *,
    hc_type: Literal["HC0", "HC1", "HC2", "HC3", "HC4"] = "HC3",
) -> RobustInferenceResult:
    """Compute heteroscedasticity-consistent (sandwich) covariance matrix.

    Computes robust standard errors that are valid under heteroscedasticity
    and potential model misspecification. The sandwich estimator has the form:

        Var(β) = (X'X)⁻¹ V (X'X)⁻¹

    where V depends on the type of HC correction.

    Parameters
    ----------
    result : GLMResult
        Fitted GLM result object
    hc_type : {'HC0', 'HC1', 'HC2', 'HC3', 'HC4'}, default='HC3'
        Type of heteroscedasticity-consistent standard errors:

        - 'HC0': White (1980) - basic sandwich estimator, no correction
        - 'HC1': Degrees of freedom correction (n/(n-p))
        - 'HC2': Leverage correction (1/(1-h_i))
        - 'HC3': MacKinnon & White (1985) - 1/(1-h_i)², better for small samples
        - 'HC4': Cribari-Neto (2004) - best for influential points

        HC3 is recommended for general use.

    Returns
    -------
    RobustInferenceResult
        Container with robust standard errors and covariance matrix

    Raises
    ------
    RuntimeError
        If design matrix or response are not available in result
    ValueError
        If hc_type is not recognized

    Notes
    -----
    The robust covariance matrix is computed using the sandwich estimator,
    which provides asymptotically valid inference even when the variance
    assumption is violated (heteroscedasticity).

    For GLMs with non-Gaussian families, this provides robustness against
    misspecification of the variance function.

    Examples
    --------
    >>> from aurora.models import fit_glm
    >>> from aurora.inference import robust_covariance
    >>> import numpy as np
    >>>
    >>> # Fit a model with heteroscedastic errors
    >>> x = np.random.randn(100)
    >>> y = 2 * x + np.random.randn(100) * (1 + 0.5 * np.abs(x))
    >>> result = fit_glm(x.reshape(-1, 1), y, family='gaussian')
    >>>
    >>> # Get robust standard errors
    >>> robust_result = robust_covariance(result, hc_type='HC3')
    >>> print(robust_result.std_errors)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator
        and a direct test for heteroskedasticity. Econometrica, 48(4), 817-838.
    MacKinnon, J.G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    if result._X is None or result._y is None:
        raise RuntimeError(
            "Design matrix and response are required for robust inference. "
            "These should be stored automatically during model fitting."
        )

    if hc_type not in HCType.__members__:
        raise ValueError(
            f"Unknown HC type: {hc_type!r}. "
            f"Valid options are: {', '.join(HCType.__members__.keys())}"
        )

    # Get design matrix (may not include intercept column)
    X_no_intercept = np.asarray(result._X, dtype=float)
    y = np.asarray(result._y, dtype=float)
    n = len(y)

    # Build full design matrix with intercept if needed
    if result._fit_intercept and result.intercept_ is not None:
        X = np.column_stack([np.ones(n), X_no_intercept])
    else:
        X = X_no_intercept

    n, p = X.shape

    # Get residuals
    y_pred = result.predict(X_no_intercept)
    resid = y - np.asarray(y_pred, dtype=float)

    # Compute (X'X)⁻¹
    XtX_inv = np.linalg.inv(X.T @ X)

    # Compute leverage (hat values) for HC2, HC3, HC4
    if hc_type in ["HC2", "HC3", "HC4"]:
        H = X @ XtX_inv @ X.T
        h = np.diag(H)

    # Compute weights based on HC type
    if hc_type == "HC0":
        # Original White (1980) - no correction
        weights = resid**2
    elif hc_type == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - p)) * resid**2
    elif hc_type == "HC2":
        # Leverage correction
        weights = resid**2 / (1 - h)
    elif hc_type == "HC3":
        # MacKinnon & White (1985) - better for small samples
        weights = resid**2 / (1 - h) ** 2
    elif hc_type == "HC4":
        # Cribari-Neto (2004) - even better for influential points
        delta = np.minimum(4, n * h / p)
        weights = resid**2 / (1 - h) ** delta

    # Sandwich estimator: (X'X)⁻¹ V (X'X)⁻¹
    V_meat = X.T @ np.diag(weights) @ X
    V_robust = XtX_inv @ V_meat @ XtX_inv

    # Extract standard errors
    se_full = np.sqrt(np.clip(np.diag(V_robust), 1e-12, None))

    if result._fit_intercept and result.intercept_ is not None:
        intercept_se = float(se_full[0])
        coef_se = se_full[1:]
    else:
        intercept_se = None
        coef_se = se_full

    return RobustInferenceResult(
        std_errors=coef_se,
        coef_cov=V_robust,
        intercept_std_error=intercept_se,
        hc_type=hc_type,
    )


def bootstrap_inference(
    result: GLMResult,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute bootstrap standard errors and confidence intervals.

    Uses case resampling (pairs bootstrap) to estimate the sampling
    distribution of model parameters without parametric assumptions.

    Parameters
    ----------
    result : GLMResult
        Fitted GLM result object
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level for confidence intervals (e.g., 0.05 for 95% CI)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing:

        - 'std_errors': Bootstrap standard errors (excluding intercept)
        - 'intercept_std_error': Bootstrap SE for intercept, if fitted
        - 'ci_lower': Lower confidence interval bounds
        - 'ci_upper': Upper confidence interval bounds
        - 'intercept_ci': Tuple of (lower, upper) for intercept, if fitted
        - 'boot_coefs': Array of bootstrap coefficient samples (n_bootstrap, p)

    Raises
    ------
    RuntimeError
        If design matrix or response are not available in result

    Notes
    -----
    This function uses the pairs bootstrap (case resampling), which:

    - Makes minimal assumptions about the data generating process
    - Is valid for heteroscedastic and non-normal errors
    - Can be computationally intensive for large samples
    - Provides percentile confidence intervals

    The bootstrap is particularly useful when:

    - You suspect the asymptotic approximation may be poor
    - The sample size is small to moderate
    - You want inference for non-linear functions of parameters

    Examples
    --------
    >>> from aurora.models import fit_glm
    >>> from aurora.inference import bootstrap_inference
    >>> import numpy as np
    >>>
    >>> x = np.random.randn(100)
    >>> y = 2 * x + np.random.randn(100)
    >>> result = fit_glm(x.reshape(-1, 1), y, family='gaussian')
    >>>
    >>> boot_result = bootstrap_inference(result, n_bootstrap=1000, seed=42)
    >>> print(f"Bootstrap SE: {boot_result['std_errors']}")
    >>> print(f"95% CI: [{boot_result['ci_lower']}, {boot_result['ci_upper']}]")

    References
    ----------
    Efron, B., & Tibshirani, R.J. (1993). An Introduction to the Bootstrap.
        Chapman and Hall/CRC.
    Davison, A.C., & Hinkley, D.V. (1997). Bootstrap Methods and their Application.
        Cambridge University Press.
    """
    if result._X is None or result._y is None:
        raise RuntimeError(
            "Design matrix and response are required for bootstrap inference. "
            "These should be stored automatically during model fitting."
        )

    from ..models import fit_glm

    if seed is not None:
        np.random.seed(seed)

    X_no_intercept = np.asarray(result._X, dtype=float)
    y = np.asarray(result._y, dtype=float)
    n = len(y)

    # Determine number of parameters
    if result._fit_intercept:
        n_params = 2  # intercept + 1 coefficient (assuming univariate for now)
        if X_no_intercept.ndim == 2:
            n_params = X_no_intercept.shape[1] + 1
    else:
        n_params = X_no_intercept.shape[1] if X_no_intercept.ndim == 2 else 1

    boot_coefs = np.zeros((n_bootstrap, n_params))

    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = (
            X_no_intercept[idx]
            if X_no_intercept.ndim == 2
            else X_no_intercept[idx].reshape(-1, 1)
        )
        y_boot = y[idx]

        # Fit model
        try:
            result_boot = fit_glm(
                X_boot,
                y_boot,
                family=result.family,
                link=result.link,
                fit_intercept=result._fit_intercept,
            )

            if result._fit_intercept:
                boot_coefs[i, 0] = result_boot.intercept_
                boot_coefs[i, 1:] = np.asarray(result_boot.coef_).ravel()
            else:
                boot_coefs[i] = np.asarray(result_boot.coef_).ravel()
        except Exception:
            # If bootstrap sample causes numerical issues, use NaN
            boot_coefs[i] = np.nan

    # Remove any failed bootstrap samples
    boot_coefs = boot_coefs[~np.isnan(boot_coefs).any(axis=1)]

    if len(boot_coefs) < n_bootstrap * 0.9:
        import warnings

        warnings.warn(
            f"Only {len(boot_coefs)}/{n_bootstrap} bootstrap samples succeeded. "
            "Results may be unreliable.",
            RuntimeWarning,
        )

    # Compute standard errors
    boot_se_full = np.std(boot_coefs, axis=0)

    # Compute percentile confidence intervals
    lower = np.percentile(boot_coefs, 100 * alpha / 2, axis=0)
    upper = np.percentile(boot_coefs, 100 * (1 - alpha / 2), axis=0)

    if result._fit_intercept:
        return {
            "std_errors": boot_se_full[1:],
            "intercept_std_error": float(boot_se_full[0]),
            "ci_lower": lower[1:],
            "ci_upper": upper[1:],
            "intercept_ci": (float(lower[0]), float(upper[0])),
            "boot_coefs": boot_coefs,
        }
    else:
        return {
            "std_errors": boot_se_full,
            "intercept_std_error": None,
            "ci_lower": lower,
            "ci_upper": upper,
            "intercept_ci": None,
            "boot_coefs": boot_coefs,
        }
