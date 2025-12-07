# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""LOESS (Locally Estimated Scatterplot Smoothing).

LOESS (also known as LOWESS - Locally Weighted Scatterplot Smoothing) is a
non-parametric regression method that fits a smooth curve through data by
performing local polynomial regression at each target point.

Mathematical Framework
----------------------
At each point x₀, LOESS:

1. Selects the k = span × n nearest neighbors
2. Computes weights using a tricube kernel:
   w_i = (1 - |d_i/h|³)³ for |d_i| < h, else 0
   where d_i = |x_i - x₀| and h = max(|d_i|) for neighbors
3. Fits weighted least squares:
   - Local linear: y = a + b(x - x₀)
   - Local quadratic: y = a + b(x - x₀) + c(x - x₀)²
4. Returns the fitted value at x₀ (the intercept a)

Key Properties
--------------
- **Bandwidth (span)**: Controls smoothness (smaller = more wiggly)
- **Degree**: Local polynomial degree (1 = linear, 2 = quadratic)
- **Robustness**: Optional iterative reweighting for outlier resistance
- **No assumptions**: Does not assume a global functional form

Computational Complexity
------------------------
- Fitting: O(n²) for full LOESS (local fit at each data point)
- Prediction at new points: O(nm) where m = number of new points
- Memory: O(n) for data storage

Practical Guidelines
--------------------
- **Span**: Start with 0.5-0.75; lower for more complex curves
- **Degree**: 1 (linear) often sufficient; 2 for curves with inflection points
- **Robustness**: Enable for data with outliers

References
----------
.. [1] Cleveland, W. S. (1979).
       "Robust locally weighted regression and smoothing scatterplots."
       Journal of the American Statistical Association, 74(368), 829-836.
.. [2] Cleveland, W. S., & Devlin, S. J. (1988).
       "Locally weighted regression: An approach to regression analysis
       by local fitting."
       Journal of the American Statistical Association, 83(403), 596-610.
.. [3] Cleveland, W. S., & Grosse, E. (1991).
       "Computational methods for local regression."
       Statistics and Computing, 1(1), 47-62.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["LOESSSmoother", "LOESSResult", "loess"]


class LOESSSmoother:
    """LOESS (Locally Estimated Scatterplot Smoothing) smoother.

    Fits a smooth curve by performing weighted local polynomial regression
    at each target point.

    Parameters
    ----------
    span : float, default=0.75
        Fraction of data to use in each local regression (bandwidth).
        Must be in (0, 1]. Smaller values give more flexible fits.
    degree : int, default=1
        Degree of local polynomial:
        - 1: Local linear regression (faster, usually sufficient)
        - 2: Local quadratic regression (better for curves with inflections)
    robust : bool, default=False
        If True, use iterative reweighting for robustness to outliers.
        Applies bisquare weights based on residuals.
    n_robust_iter : int, default=3
        Number of robustness iterations if robust=True

    Attributes
    ----------
    span : float
        Bandwidth parameter
    degree : int
        Local polynomial degree
    robust : bool
        Whether robustness is enabled
    n_robust_iter : int
        Number of robustness iterations

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.smoothing.local import LOESSSmoother
    >>>
    >>> # Generate noisy data with outliers
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> y = np.sin(x) + np.random.normal(0, 0.2, 100)
    >>> y[50] = 5  # Add outlier
    >>>
    >>> # Fit robust LOESS
    >>> smoother = LOESSSmoother(span=0.3, degree=1, robust=True)
    >>> result = smoother.fit(x, y)
    >>> y_smooth = result.fitted_values_

    Notes
    -----
    LOESS has several advantages over parametric smoothers:
    - No global functional form assumption
    - Automatically adapts to local structure
    - Robust variant handles outliers well

    Disadvantages:
    - Slower than parametric methods (O(n²) fitting)
    - Extrapolation outside data range is unreliable
    - No simple formula for fitted function

    For large datasets, consider using P-splines or other parametric
    smoothers which are much faster.
    """

    def __init__(
        self,
        span: float = 0.75,
        degree: int = 1,
        robust: bool = False,
        n_robust_iter: int = 3,
    ):
        if not 0 < span <= 1:
            raise ValueError(f"span must be in (0, 1], got {span}")
        if degree not in (1, 2):
            raise ValueError(f"degree must be 1 or 2, got {degree}")
        if n_robust_iter < 1:
            raise ValueError("n_robust_iter must be at least 1")

        self.span = span
        self.degree = degree
        self.robust = robust
        self.n_robust_iter = n_robust_iter

    def fit(self, x: NDArray, y: NDArray) -> "LOESSResult":
        """Fit LOESS smoother to data.

        Parameters
        ----------
        x : array-like (n,)
            Predictor values
        y : array-like (n,)
            Response values

        Returns
        -------
        LOESSResult
            Fitted smoother with fitted values and methods for prediction

        Notes
        -----
        Fitting computes the smoothed value at each data point, which is
        O(n × k) where k = span × n is the neighborhood size.
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        n = len(x)
        if n < 3:
            raise ValueError("Need at least 3 data points")

        # Number of points in each neighborhood
        k = max(3, int(np.ceil(self.span * n)))
        k = min(k, n)  # Can't use more points than available

        # Sort data by x for efficient neighbor finding
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        fitted = np.zeros(n)
        robustness_weights = np.ones(n)

        # Robustness iterations
        n_iter = self.n_robust_iter if self.robust else 1

        for iteration in range(n_iter):
            # Fit at each point
            for i in range(n):
                fitted[i] = self._local_regression(
                    x_sorted, y_sorted, x_sorted[i], k, robustness_weights
                )

            if self.robust and iteration < n_iter - 1:
                # Update robustness weights based on residuals
                residuals = y_sorted - fitted
                # MAD scale estimate
                s = 6.0 * np.median(np.abs(residuals))
                if s > 0:
                    u = residuals / s
                    robustness_weights = self._bisquare_weights(u)
                else:
                    robustness_weights = np.ones(n)

        # Unsort to original order
        unsort_idx = np.argsort(sort_idx)
        fitted_original = fitted[unsort_idx]

        return LOESSResult(
            x_=x,
            y_=y,
            fitted_values_=fitted_original,
            span=self.span,
            degree=self.degree,
            robust=self.robust,
            _smoother=self,
        )

    def _local_regression(
        self,
        x: NDArray,
        y: NDArray,
        x0: float,
        k: int,
        robustness_weights: NDArray,
    ) -> float:
        """Perform local polynomial regression at point x0.

        Parameters
        ----------
        x : ndarray
            Sorted predictor values
        y : ndarray
            Response values (sorted with x)
        x0 : float
            Point at which to fit
        k : int
            Number of neighbors to use
        robustness_weights : ndarray
            Robustness weights (1 if not robust)

        Returns
        -------
        float
            Fitted value at x0
        """
        n = len(x)

        # Find k nearest neighbors efficiently
        # Since x is sorted, use binary search approach
        distances = np.abs(x - x0)
        neighbor_idx = np.argsort(distances)[:k]

        x_local = x[neighbor_idx]
        y_local = y[neighbor_idx]
        d_local = distances[neighbor_idx]
        rw_local = robustness_weights[neighbor_idx]

        # Bandwidth is distance to furthest neighbor
        h = d_local.max()
        if h == 0:
            # All points at same location, return mean
            return np.average(y_local, weights=rw_local)

        # Tricube kernel weights
        u = d_local / h
        kernel_weights = self._tricube_weights(u)
        w = kernel_weights * rw_local

        # Check for degenerate case
        if np.sum(w) == 0:
            return np.mean(y_local)

        # Design matrix for local polynomial
        dx = x_local - x0
        if self.degree == 1:
            # Local linear: y = a + b(x - x0)
            X = np.column_stack([np.ones(k), dx])
        else:
            # Local quadratic: y = a + b(x - x0) + c(x - x0)^2
            X = np.column_stack([np.ones(k), dx, dx**2])

        # Weighted least squares
        W = np.diag(w)
        try:
            XtW = X.T @ W
            XtWX = XtW @ X
            XtWy = XtW @ y_local

            # Add small ridge for numerical stability
            XtWX += 1e-10 * np.eye(XtWX.shape[0])

            beta = np.linalg.solve(XtWX, XtWy)
            return beta[0]  # Intercept = fitted value at x0
        except np.linalg.LinAlgError:
            # Fallback to weighted mean
            return np.average(y_local, weights=w)

    def _tricube_weights(self, u: NDArray) -> NDArray:
        """Tricube kernel: (1 - |u|³)³ for |u| < 1, else 0.

        The tricube kernel is the standard kernel for LOESS:
        - Smooth (twice differentiable)
        - Compact support (exactly zero outside [-1, 1])
        - Gives more weight to nearby points
        """
        return np.where(np.abs(u) < 1, (1 - np.abs(u) ** 3) ** 3, 0.0)

    def _bisquare_weights(self, u: NDArray) -> NDArray:
        """Bisquare (Tukey's biweight) function for robustness.

        w(u) = (1 - u²)² for |u| < 1, else 0

        Points with large standardized residuals (|u| >= 1) get zero weight,
        removing their influence in subsequent iterations.
        """
        return np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)


@dataclass
class LOESSResult:
    """Results from LOESS fitting.

    Attributes
    ----------
    x_ : ndarray
        Training predictor values
    y_ : ndarray
        Training response values
    fitted_values_ : ndarray
        Smoothed values at training points
    span : float
        Bandwidth parameter used
    degree : int
        Local polynomial degree used
    robust : bool
        Whether robustness was used
    """

    x_: NDArray
    y_: NDArray
    fitted_values_: NDArray
    span: float
    degree: int
    robust: bool
    _smoother: LOESSSmoother

    def predict(self, x_new: NDArray) -> NDArray:
        """Predict at new points.

        Parameters
        ----------
        x_new : array-like
            Points at which to predict

        Returns
        -------
        y_pred : ndarray
            Predicted (smoothed) values

        Notes
        -----
        For efficiency with many new points, this uses linear interpolation
        between fitted values. For exact LOESS predictions at arbitrary points,
        use predict_exact().
        """
        x_new = np.asarray(x_new).ravel()

        # Use linear interpolation for efficiency
        from scipy.interpolate import interp1d

        # Sort training data for interpolation
        sort_idx = np.argsort(self.x_)
        x_sorted = self.x_[sort_idx]
        y_sorted = self.fitted_values_[sort_idx]

        f = interp1d(
            x_sorted,
            y_sorted,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )
        return f(x_new)

    def predict_exact(self, x_new: NDArray) -> NDArray:
        """Predict at new points using exact LOESS computation.

        This performs full local regression at each new point, giving
        exact LOESS predictions but at higher computational cost.

        Parameters
        ----------
        x_new : array-like
            Points at which to predict

        Returns
        -------
        y_pred : ndarray
            Predicted values
        """
        x_new = np.asarray(x_new).ravel()
        n = len(self.x_)
        k = max(3, int(np.ceil(self.span * n)))
        k = min(k, n)

        y_pred = np.zeros(len(x_new))
        robustness_weights = np.ones(n)

        for i, x0 in enumerate(x_new):
            y_pred[i] = self._smoother._local_regression(
                self.x_, self.y_, x0, k, robustness_weights
            )

        return y_pred

    def residuals(self) -> NDArray:
        """Compute residuals (y - fitted).

        Returns
        -------
        resid : ndarray
            Residuals at training points
        """
        return self.y_ - self.fitted_values_

    def summary(self) -> dict:
        """Return summary statistics.

        Returns
        -------
        dict
            Summary including span, degree, R², residual stats
        """
        residuals = self.residuals()
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.y_ - np.mean(self.y_)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "n_obs": len(self.y_),
            "span": self.span,
            "degree": self.degree,
            "robust": self.robust,
            "r_squared": r_squared,
            "rss": ss_res,
            "residual_std": np.std(residuals),
        }


def loess(
    x: NDArray,
    y: NDArray,
    span: float = 0.75,
    degree: int = 1,
    robust: bool = False,
    **kwargs,
) -> LOESSResult:
    """Fit LOESS smoother to data.

    Convenience function that creates a LOESSSmoother and fits it.

    Parameters
    ----------
    x : array-like
        Predictor values
    y : array-like
        Response values
    span : float, default=0.75
        Fraction of data used in each local fit (bandwidth)
    degree : int, default=1
        Local polynomial degree (1 or 2)
    robust : bool, default=False
        Use iterative reweighting for outlier robustness
    **kwargs
        Additional arguments passed to LOESSSmoother

    Returns
    -------
    LOESSResult
        Fitted smoother

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.smoothing.local import loess
    >>>
    >>> # Generate data
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x) + np.random.normal(0, 0.3, 100)
    >>>
    >>> # Fit LOESS
    >>> result = loess(x, y, span=0.3)
    >>> y_smooth = result.fitted_values_
    >>>
    >>> # Predict at new points
    >>> x_new = np.linspace(0, 10, 200)
    >>> y_new = result.predict(x_new)
    """
    smoother = LOESSSmoother(span=span, degree=degree, robust=robust, **kwargs)
    return smoother.fit(x, y)
