"""Sensitivity analysis and influence diagnostics.

This module provides tools for identifying influential observations
and assessing model stability.

Influence Diagnostics
---------------------
- Cook's Distance: Overall influence on fitted values
- Leverage (hat values): Influence on own fitted value
- DFBETAS: Influence on each coefficient
- DFFITS: Influence on own predicted value

Leave-One-Out Analysis
----------------------
- LOO residuals: Residuals from models fit without each observation
- LOO predictions: Predictions from leave-one-out models

Examples
--------
>>> from aurora.validation.sensitivity import cooks_distance, leverage
>>> 
>>> # Compute influence measures
>>> cooks_d = cooks_distance(result)
>>> lev = leverage(result)
>>> 
>>> # Find influential points
>>> influential = cooks_d > 4 / n
>>> high_leverage = lev > 2 * p / n

References
----------
.. [1] Cook, R.D. (1977). Detection of Influential Observation in Linear Regression.
.. [2] Belsley, D.A., Kuh, E., & Welsch, R.E. (1980). Regression Diagnostics.
.. [3] Pregibon, D. (1981). Logistic Regression Diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class InfluenceResult:
    """Result of influence diagnostics.
    
    Attributes
    ----------
    cooks_distance : ndarray of shape (n_samples,)
        Cook's distance for each observation.
    leverage : ndarray of shape (n_samples,)
        Hat values (leverage) for each observation.
    studentized_residuals : ndarray of shape (n_samples,)
        Externally studentized residuals.
    dffits : ndarray of shape (n_samples,)
        DFFITS values.
    dfbetas : ndarray of shape (n_samples, n_params)
        DFBETAS for each coefficient.
    n_obs : int
        Number of observations.
    n_params : int
        Number of parameters.
    """
    
    cooks_distance: np.ndarray
    leverage: np.ndarray
    studentized_residuals: np.ndarray
    dffits: np.ndarray
    dfbetas: np.ndarray
    n_obs: int
    n_params: int
    
    @property
    def influential_cooks(self) -> np.ndarray:
        """Indices of influential points by Cook's distance (> 4/n)."""
        threshold = 4 / self.n_obs
        return np.where(self.cooks_distance > threshold)[0]
    
    @property
    def high_leverage(self) -> np.ndarray:
        """Indices of high leverage points (> 2p/n)."""
        threshold = 2 * self.n_params / self.n_obs
        return np.where(self.leverage > threshold)[0]
    
    @property
    def outliers(self) -> np.ndarray:
        """Indices of outliers by studentized residuals (|t| > 2)."""
        return np.where(np.abs(self.studentized_residuals) > 2)[0]
    
    def summary(self) -> str:
        """Return formatted influence summary."""
        lines = []
        sep = "=" * 65
        
        lines.append(sep)
        lines.append(f"{'Influence Diagnostics Summary':^65}")
        lines.append(sep)
        lines.append(f"Number of observations:        {self.n_obs:>8}")
        lines.append(f"Number of parameters:          {self.n_params:>8}")
        lines.append("-" * 65)
        
        # Thresholds
        cooks_thresh = 4 / self.n_obs
        lev_thresh = 2 * self.n_params / self.n_obs
        
        lines.append("Thresholds:")
        lines.append(f"  Cook's D threshold (4/n):    {cooks_thresh:>8.4f}")
        lines.append(f"  Leverage threshold (2p/n):   {lev_thresh:>8.4f}")
        lines.append("")
        
        # Summary statistics
        lines.append("Influence Measure Statistics:")
        lines.append(f"{'Measure':>25} {'Min':>10} {'Max':>10} {'Mean':>10}")
        lines.append("-" * 65)
        
        for name, values in [
            ("Cook's Distance", self.cooks_distance),
            ("Leverage", self.leverage),
            ("Studentized Resid", self.studentized_residuals),
            ("DFFITS", self.dffits),
        ]:
            lines.append(
                f"{name:>25} {np.min(values):>10.4f} "
                f"{np.max(values):>10.4f} {np.mean(values):>10.4f}"
            )
        
        lines.append("")
        lines.append("Influential Observations:")
        lines.append(f"  By Cook's D (> {cooks_thresh:.4f}):  {len(self.influential_cooks):>5} obs")
        lines.append(f"  By Leverage (> {lev_thresh:.4f}):   {len(self.high_leverage):>5} obs")
        lines.append(f"  By Stud. Resid (|t| > 2):   {len(self.outliers):>5} obs")
        lines.append(sep)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"InfluenceResult(n_obs={self.n_obs}, n_params={self.n_params}, "
            f"n_influential={len(self.influential_cooks)})"
        )


def influence_measures(
    result: Any,
    *,
    X: np.ndarray | None = None,
) -> InfluenceResult:
    """Compute all influence diagnostics for a fitted model.
    
    Parameters
    ----------
    result : ModelResult
        A fitted model result.
    X : ndarray, optional
        Design matrix. If not provided, attempts to extract from result.
        
    Returns
    -------
    influence : InfluenceResult
        Influence diagnostics for all observations.
        
    Examples
    --------
    >>> from aurora.validation.sensitivity import influence_measures
    >>> 
    >>> influence = influence_measures(result)
    >>> print(influence.summary())
    >>> 
    >>> # Get influential observations
    >>> influential_idx = influence.influential_cooks
    """
    # Get design matrix
    if X is None:
        if hasattr(result, '_X') and result._X is not None:
            X = result._X
        else:
            raise ValueError("Design matrix X must be provided.")
    
    # Get residuals and fitted values
    residuals = result.residuals
    n = len(residuals)
    
    # Number of parameters
    if hasattr(result, 'coef_'):
        p = len(result.coef_)
        if getattr(result, 'intercept_', None) is not None:
            p += 1
    elif hasattr(result, 'fixed_effects_'):
        p = len(result.fixed_effects_)
    else:
        p = X.shape[1]
    
    # Compute hat matrix diagonal (leverage)
    lev = leverage(result, X=X)
    
    # Residual variance
    if hasattr(result, 'residual_variance_'):
        mse = result.residual_variance_
    else:
        mse = np.sum(residuals ** 2) / (n - p)
    
    # Studentized residuals (externally studentized)
    stud_resid = studentized_residuals(result, X=X)
    
    # Cook's distance
    cooks_d = cooks_distance(result, X=X)
    
    # DFFITS
    dff = dffits(result, X=X)
    
    # DFBETAS (simplified computation)
    dfb = dfbetas(result, X=X)
    
    return InfluenceResult(
        cooks_distance=cooks_d,
        leverage=lev,
        studentized_residuals=stud_resid,
        dffits=dff,
        dfbetas=dfb,
        n_obs=n,
        n_params=p,
    )


def leverage(
    result: Any,
    *,
    X: np.ndarray | None = None,
) -> np.ndarray:
    """Compute leverage (hat values) for each observation.
    
    The leverage h_ii is the i-th diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'
    
    Parameters
    ----------
    result : ModelResult
        A fitted model result.
    X : ndarray, optional
        Design matrix.
        
    Returns
    -------
    h : ndarray of shape (n_samples,)
        Leverage values.
        
    Notes
    -----
    High leverage points have unusual predictor values (far from center).
    A common threshold is h_ii > 2p/n.
    
    .. math::
        h_{ii} = x_i' (X'X)^{-1} x_i
    """
    if X is None:
        if hasattr(result, '_X') and result._X is not None:
            X = result._X
        else:
            raise ValueError("Design matrix X must be provided.")
    
    # Compute hat matrix diagonal efficiently
    # H = X @ (X'X)^{-1} @ X'
    # h_ii = X[i, :] @ (X'X)^{-1} @ X[i, :].T
    
    try:
        # Use QR decomposition for numerical stability
        Q, R = np.linalg.qr(X)
        h = np.sum(Q ** 2, axis=1)
    except np.linalg.LinAlgError:
        # Fallback to direct computation
        XtX_inv = np.linalg.pinv(X.T @ X)
        h = np.sum((X @ XtX_inv) * X, axis=1)
    
    return h


def cooks_distance(
    result: Any,
    *,
    X: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Cook's distance for each observation.
    
    Cook's distance measures the influence of observation i on all
    fitted values.
    
    Parameters
    ----------
    result : ModelResult
        A fitted model result.
    X : ndarray, optional
        Design matrix.
        
    Returns
    -------
    D : ndarray of shape (n_samples,)
        Cook's distance values.
        
    Notes
    -----
    Large values indicate influential observations. Common thresholds:
    - D_i > 4/n (conservative)
    - D_i > 1 (very influential)
    
    .. math::
        D_i = \\frac{e_i^2}{p \\cdot MSE} \\cdot \\frac{h_{ii}}{(1 - h_{ii})^2}
    """
    if X is None:
        if hasattr(result, '_X') and result._X is not None:
            X = result._X
        else:
            raise ValueError("Design matrix X must be provided.")
    
    residuals = result.residuals
    n = len(residuals)
    p = X.shape[1]
    
    h = leverage(result, X=X)
    
    # MSE
    if hasattr(result, 'residual_variance_'):
        mse = result.residual_variance_
    else:
        mse = np.sum(residuals ** 2) / (n - p)
    
    # Cook's distance
    D = (residuals ** 2 / (p * mse)) * (h / (1 - h) ** 2)
    
    return D


def studentized_residuals(
    result: Any,
    *,
    X: np.ndarray | None = None,
    external: bool = True,
) -> np.ndarray:
    """Compute studentized residuals.
    
    Parameters
    ----------
    result : ModelResult
        A fitted model result.
    X : ndarray, optional
        Design matrix.
    external : bool, default=True
        If True, compute externally studentized residuals (using
        leave-one-out variance). If False, use internally studentized.
        
    Returns
    -------
    r : ndarray of shape (n_samples,)
        Studentized residuals.
        
    Notes
    -----
    Externally studentized residuals follow a t-distribution with
    n-p-1 degrees of freedom under the null hypothesis.
    
    .. math::
        r_i = \\frac{e_i}{\\hat{\\sigma}_{(i)} \\sqrt{1 - h_{ii}}}
    """
    if X is None:
        if hasattr(result, '_X') and result._X is not None:
            X = result._X
        else:
            raise ValueError("Design matrix X must be provided.")
    
    residuals = result.residuals
    n = len(residuals)
    p = X.shape[1]
    
    h = leverage(result, X=X)
    
    if hasattr(result, 'residual_variance_'):
        mse = result.residual_variance_
    else:
        mse = np.sum(residuals ** 2) / (n - p)
    
    if external:
        # Leave-one-out variance estimate
        # sigma2_i = ((n-p)*mse - e_i^2/(1-h_ii)) / (n-p-1)
        sigma2_i = ((n - p) * mse - residuals ** 2 / (1 - h)) / (n - p - 1)
        sigma2_i = np.maximum(sigma2_i, 1e-10)  # Avoid division by zero
        r = residuals / (np.sqrt(sigma2_i) * np.sqrt(1 - h))
    else:
        # Internal studentization
        r = residuals / (np.sqrt(mse) * np.sqrt(1 - h))
    
    return r


def dffits(
    result: Any,
    *,
    X: np.ndarray | None = None,
) -> np.ndarray:
    """Compute DFFITS for each observation.
    
    DFFITS measures the influence of observation i on its own
    fitted value.
    
    Parameters
    ----------
    result : ModelResult
        A fitted model result.
    X : ndarray, optional
        Design matrix.
        
    Returns
    -------
    dffits : ndarray of shape (n_samples,)
        DFFITS values.
        
    Notes
    -----
    A common threshold is |DFFITS| > 2 * sqrt(p/n).
    
    .. math::
        DFFITS_i = r_i \\sqrt{\\frac{h_{ii}}{1 - h_{ii}}}
    """
    if X is None:
        if hasattr(result, '_X') and result._X is not None:
            X = result._X
        else:
            raise ValueError("Design matrix X must be provided.")
    
    h = leverage(result, X=X)
    r = studentized_residuals(result, X=X, external=True)
    
    return r * np.sqrt(h / (1 - h))


def dfbetas(
    result: Any,
    *,
    X: np.ndarray | None = None,
) -> np.ndarray:
    """Compute DFBETAS for each observation and coefficient.
    
    DFBETAS_ij measures the influence of observation i on
    coefficient j.
    
    Parameters
    ----------
    result : ModelResult
        A fitted model result.
    X : ndarray, optional
        Design matrix.
        
    Returns
    -------
    dfbetas : ndarray of shape (n_samples, n_params)
        DFBETAS matrix.
        
    Notes
    -----
    A common threshold is |DFBETAS| > 2/sqrt(n).
    
    .. math::
        DFBETAS_{ij} = \\frac{\\hat{\\beta}_j - \\hat{\\beta}_{j(i)}}{\\hat{\\sigma}_{(i)} \\sqrt{(X'X)^{-1}_{jj}}}
    """
    if X is None:
        if hasattr(result, '_X') and result._X is not None:
            X = result._X
        else:
            raise ValueError("Design matrix X must be provided.")
    
    residuals = result.residuals
    n, p = X.shape
    
    h = leverage(result, X=X)
    r = studentized_residuals(result, X=X, external=True)
    
    # Compute (X'X)^{-1}
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    
    # DFBETAS formula (approximation)
    # dfbetas[i, j] = r[i] * (XtX_inv @ X[i, :]) / sqrt((1 - h[i]) * XtX_inv[j, j])
    
    dfb = np.zeros((n, p))
    for i in range(n):
        xi = X[i, :]
        c = XtX_inv @ xi
        for j in range(p):
            if XtX_inv[j, j] > 0:
                dfb[i, j] = r[i] * c[j] / np.sqrt((1 - h[i]) * XtX_inv[j, j])
    
    return dfb


def loo_residuals(
    result: Any,
    *,
    X: np.ndarray | None = None,
) -> np.ndarray:
    """Compute leave-one-out (PRESS) residuals.
    
    These are the residuals from predictions made by models
    fit without each observation.
    
    Parameters
    ----------
    result : ModelResult
        A fitted model result.
    X : ndarray, optional
        Design matrix.
        
    Returns
    -------
    press : ndarray of shape (n_samples,)
        Leave-one-out residuals.
        
    Notes
    -----
    .. math::
        e_{(i)} = \\frac{e_i}{1 - h_{ii}}
    
    The sum of squared LOO residuals is the PRESS statistic.
    """
    if X is None:
        if hasattr(result, '_X') and result._X is not None:
            X = result._X
        else:
            raise ValueError("Design matrix X must be provided.")
    
    residuals = result.residuals
    h = leverage(result, X=X)
    
    return residuals / (1 - h)


def press_statistic(
    result: Any,
    *,
    X: np.ndarray | None = None,
) -> float:
    """Compute PRESS (Predicted Residual Error Sum of Squares).
    
    Parameters
    ----------
    result : ModelResult
        A fitted model result.
    X : ndarray, optional
        Design matrix.
        
    Returns
    -------
    press : float
        PRESS statistic.
        
    Notes
    -----
    PRESS is useful for model selection and cross-validation.
    Lower values indicate better predictive ability.
    """
    loo = loo_residuals(result, X=X)
    return np.sum(loo ** 2)


__all__ = [
    "InfluenceResult",
    "influence_measures",
    "leverage",
    "cooks_distance",
    "studentized_residuals",
    "dffits",
    "dfbetas",
    "loo_residuals",
    "press_statistic",
]