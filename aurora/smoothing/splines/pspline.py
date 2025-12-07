# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""P-Splines: B-splines with Difference Penalty.

P-Splines (Penalized B-splines) combine a large number of equally-spaced B-spline
basis functions with a difference penalty on coefficients. This approach was
introduced by Eilers & Marx (1996) as a flexible and computationally efficient
alternative to smoothing splines and knot-selection methods.

Mathematical Framework
----------------------
The P-spline smoother minimizes:

    ||y - Bβ||² + λ β'Sβ

where:
- B is the B-spline basis matrix (n × k)
- β is the coefficient vector (k × 1)
- S = D'D is the difference penalty matrix (k × k)
- D is the d-th order difference matrix
- λ is the smoothing parameter

For d = 2 (second-order differences):

    D₂β = [β₃ - 2β₂ + β₁, β₄ - 2β₃ + β₂, ...]

This penalizes curvature in the coefficient sequence, which approximates
∫[f''(x)]² dx for the fitted function.

Advantages
----------
1. **No knot selection**: Use many equally-spaced knots; smoothness controlled by λ
2. **Single tuning parameter**: Just λ (compared to knot locations + number)
3. **Computational efficiency**: Banded matrices, O(nk²) fitting
4. **Flexibility**: Can fit any smooth function given enough basis functions
5. **Simple derivatives**: Derivative of fitted function easily computed

Practical Guidelines
--------------------
- **Number of basis functions**: k = 20-40 typically sufficient
- **Degree**: Cubic (degree=3) is standard; quadratic works too
- **Penalty order**: d = 2 (curvature) most common; d = 1 for monotone-like fits
- **Smoothing selection**: GCV, AIC, or REML

References
----------
.. [1] Eilers, P. H. C., & Marx, B. D. (1996).
       "Flexible smoothing with B-splines and penalties."
       Statistical Science, 11(2), 89-121.
.. [2] Eilers, P. H. C., & Marx, B. D. (2010).
       "Splines, knots, and penalties."
       Wiley Interdisciplinary Reviews: Computational Statistics, 2(6), 637-653.
.. [3] Ruppert, D., Wand, M. P., & Carroll, R. J. (2003).
       Semiparametric Regression. Cambridge University Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar

from .bspline import BSplineBasis
from ..penalties.difference import difference_penalty

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["PSplineBasis", "PSplineResult", "fit_pspline"]


class PSplineBasis:
    """P-Spline basis: B-splines with equally-spaced knots and difference penalty.

    P-Splines use a large number of B-spline basis functions with a difference
    penalty on coefficients to achieve smooth fits. The amount of smoothing is
    controlled by a single parameter λ.

    Parameters
    ----------
    n_basis : int, default=20
        Number of B-spline basis functions. More basis functions allow more
        flexibility but require larger penalty. Typically 15-40 is sufficient.
    degree : int, default=3
        B-spline degree. Cubic (degree=3) is standard and provides C² continuity.
    penalty_order : int, default=2
        Order of difference penalty:
        - 1: Penalizes first differences (roughness of coefficients)
        - 2: Penalizes second differences (curvature, most common)
        - 3: Penalizes third differences (rate of curvature change)
    domain : tuple of float, optional
        (min, max) domain for the basis. If None, determined from data.

    Attributes
    ----------
    n_basis : int
        Number of basis functions
    degree : int
        B-spline degree
    penalty_order : int
        Order of difference penalty
    knots_ : ndarray
        Knot locations (set after fitting)

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.smoothing.splines import PSplineBasis
    >>>
    >>> # Create basis
    >>> basis = PSplineBasis(n_basis=25, degree=3, penalty_order=2)
    >>>
    >>> # Generate data
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> y = np.sin(x) + np.random.normal(0, 0.2, 100)
    >>>
    >>> # Fit with automatic smoothing selection
    >>> result = basis.fit(x, y, lambda_='gcv')
    >>> y_smooth = result.fitted_values_

    Notes
    -----
    The P-spline approach separates the basis complexity from smoothness control:
    - Use enough basis functions (n_basis) to capture the true function
    - Control smoothness entirely via the penalty parameter λ

    For most applications, 20-30 basis functions with cubic degree and
    second-order penalty works well. The optimal λ is typically selected
    via GCV, AIC, or REML.
    """

    def __init__(
        self,
        n_basis: int = 20,
        degree: int = 3,
        penalty_order: int = 2,
        domain: tuple[float, float] | None = None,
    ):
        if n_basis < 4:
            raise ValueError("n_basis must be at least 4")
        if degree < 1:
            raise ValueError("degree must be at least 1")
        if penalty_order < 1:
            raise ValueError("penalty_order must be at least 1")
        if penalty_order >= n_basis:
            raise ValueError(
                f"penalty_order ({penalty_order}) must be less than n_basis ({n_basis})"
            )

        self.n_basis = n_basis
        self.degree = degree
        self.penalty_order = penalty_order
        self._domain = domain

        # Set after fitting
        self.knots_: NDArray | None = None
        self._bspline_basis: BSplineBasis | None = None
        self._penalty_matrix: NDArray | None = None

    def _setup(self, x: NDArray) -> None:
        """Setup basis and penalty matrix for given data range."""
        x = np.asarray(x).ravel()

        # Determine domain
        if self._domain is not None:
            x_min, x_max = self._domain
        else:
            x_min, x_max = x.min(), x.max()

        # Add small buffer to avoid boundary issues
        margin = 0.001 * (x_max - x_min)
        x_min -= margin
        x_max += margin

        # Create equally-spaced interior knots
        # For n_basis B-splines of degree d, we need:
        # n_knots = n_basis + degree + 1 total knots
        # With (degree + 1) repeated at each boundary
        # So n_interior = n_basis - degree - 1 interior knots
        n_interior = self.n_basis - self.degree - 1

        if n_interior < 0:
            raise ValueError(
                f"Not enough basis functions ({self.n_basis}) "
                f"for degree {self.degree}"
            )

        # Interior knots equally spaced
        if n_interior > 0:
            interior_knots = np.linspace(x_min, x_max, n_interior + 2)[1:-1]
        else:
            interior_knots = np.array([])

        # Full knot vector with repeated boundary knots
        boundary_left = np.repeat(x_min, self.degree + 1)
        boundary_right = np.repeat(x_max, self.degree + 1)
        self.knots_ = np.concatenate([boundary_left, interior_knots, boundary_right])

        # Create B-spline basis (BSplineBasis computes n_basis from knots)
        self._bspline_basis = BSplineBasis(
            knots=self.knots_, degree=self.degree
        )

        # Create difference penalty matrix
        self._penalty_matrix = difference_penalty(self.n_basis, order=self.penalty_order)

    def basis_matrix(
        self, x: NDArray, sparse: bool = False
    ) -> NDArray | sparse.csr_matrix:
        """Evaluate B-spline basis at given points.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate basis functions
        sparse : bool, default=False
            Return sparse matrix (efficient for large n_basis)

        Returns
        -------
        B : ndarray or sparse matrix, shape (len(x), n_basis)
            Basis matrix where B[i,j] = B_j(x[i])
        """
        x = np.asarray(x).ravel()

        if self._bspline_basis is None:
            self._setup(x)

        return self._bspline_basis.basis_matrix(x, sparse=sparse)

    def penalty_matrix(self) -> NDArray:
        """Return the difference penalty matrix S = D'D.

        Returns
        -------
        S : ndarray, shape (n_basis, n_basis)
            Penalty matrix for penalized regression

        Raises
        ------
        ValueError
            If basis has not been set up yet (call basis_matrix first)
        """
        if self._penalty_matrix is None:
            raise ValueError(
                "Penalty matrix not set up. Call basis_matrix() first."
            )
        return self._penalty_matrix

    def fit(
        self,
        x: NDArray,
        y: NDArray,
        lambda_: float | str = "gcv",
        weights: NDArray | None = None,
        lambda_range: tuple[float, float] = (1e-6, 1e6),
    ) -> "PSplineResult":
        """Fit P-spline to data.

        Parameters
        ----------
        x : array-like
            Predictor values
        y : array-like
            Response values
        lambda_ : float or {'gcv', 'aic', 'reml'}
            Smoothing parameter. If string, automatically select via:
            - 'gcv': Generalized Cross-Validation (default, fast)
            - 'aic': Akaike Information Criterion
            - 'reml': Restricted Maximum Likelihood (better for small samples)
        weights : array-like, optional
            Observation weights. If None, uniform weights.
        lambda_range : tuple of float
            Search range for automatic lambda selection

        Returns
        -------
        PSplineResult
            Fitted P-spline with coefficients, fitted values, etc.

        Examples
        --------
        >>> # Automatic smoothing selection
        >>> result = basis.fit(x, y, lambda_='gcv')
        >>>
        >>> # Fixed smoothing parameter
        >>> result = basis.fit(x, y, lambda_=100.0)
        >>>
        >>> # With observation weights
        >>> result = basis.fit(x, y, weights=w, lambda_='gcv')
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        n = len(x)

        if len(y) != n:
            raise ValueError("x and y must have same length")

        if weights is None:
            weights = np.ones(n)
        else:
            weights = np.asarray(weights).ravel()
            if len(weights) != n:
                raise ValueError("weights must have same length as x")

        # Build basis and penalty
        B = self.basis_matrix(x)
        S = self.penalty_matrix()
        W = np.diag(weights)

        # Precompute weighted quantities
        BtW = B.T @ W
        BtWB = BtW @ B
        BtWy = BtW @ y

        # Select or use given lambda
        if isinstance(lambda_, str):
            if lambda_.lower() == "gcv":
                lambda_opt = self._select_lambda_gcv(B, y, S, weights, lambda_range)
            elif lambda_.lower() == "aic":
                lambda_opt = self._select_lambda_aic(B, y, S, weights, lambda_range)
            elif lambda_.lower() == "reml":
                lambda_opt = self._select_lambda_reml(B, y, S, weights, lambda_range)
            else:
                raise ValueError(f"Unknown selection method: {lambda_}")
        else:
            lambda_opt = float(lambda_)

        # Solve penalized least squares: (B'WB + λS)β = B'Wy
        coef = np.linalg.solve(BtWB + lambda_opt * S, BtWy)
        fitted = B @ coef

        # Compute effective degrees of freedom
        # edf = trace(H) where H = B(B'WB + λS)^{-1}B'W
        # BtW is (k, n), so we need (B'WB + λS)^{-1} @ B' @ W
        # H_factor = (B'WB + λS)^{-1} @ B' is (k, n)
        P_inv = np.linalg.inv(BtWB + lambda_opt * S)  # (k, k)
        H = B @ P_inv @ BtW  # (n, k) @ (k, k) @ (k, n) = (n, n)
        edf = np.trace(H)

        # Residual variance estimate
        residuals = y - fitted
        rss = np.sum(weights * residuals**2)
        sigma2 = rss / (n - edf) if n > edf else rss / n

        # GCV score for reference
        gcv = n * rss / (n - edf) ** 2 if n > edf else np.inf

        return PSplineResult(
            coef_=coef,
            fitted_values_=fitted,
            lambda_=lambda_opt,
            basis=self,
            edf_=edf,
            gcv_=gcv,
            sigma2_=sigma2,
            x_=x,
            y_=y,
            weights_=weights,
        )

    def _select_lambda_gcv(
        self,
        B: NDArray,
        y: NDArray,
        S: NDArray,
        weights: NDArray,
        lambda_range: tuple[float, float],
    ) -> float:
        """Select λ by minimizing Generalized Cross-Validation score.

        GCV(λ) = n × RSS / (n - edf)²

        GCV is an approximation to leave-one-out cross-validation that
        avoids n separate fits.
        """
        n = len(y)
        W = np.diag(weights)
        BtW = B.T @ W
        BtWB = BtW @ B
        BtWy = BtW @ y

        def gcv_score(log_lambda: float) -> float:
            lam = np.exp(log_lambda)
            try:
                coef = np.linalg.solve(BtWB + lam * S, BtWy)
                fitted = B @ coef

                # Hat matrix trace: trace(B @ (B'WB + λS)^{-1} @ B' @ W)
                P_inv = np.linalg.inv(BtWB + lam * S)
                H = B @ P_inv @ BtW
                edf = np.trace(H)

                if edf >= n:
                    return np.inf

                rss = np.sum(weights * (y - fitted) ** 2)
                return n * rss / (n - edf) ** 2
            except np.linalg.LinAlgError:
                return np.inf

        # Optimize over log(lambda) for numerical stability
        log_range = (np.log(lambda_range[0]), np.log(lambda_range[1]))
        result = minimize_scalar(gcv_score, bounds=log_range, method="bounded")

        return np.exp(result.x)

    def _select_lambda_aic(
        self,
        B: NDArray,
        y: NDArray,
        S: NDArray,
        weights: NDArray,
        lambda_range: tuple[float, float],
    ) -> float:
        """Select λ by minimizing AIC.

        AIC = n × log(RSS/n) + 2 × edf
        """
        n = len(y)
        W = np.diag(weights)
        BtW = B.T @ W
        BtWB = BtW @ B
        BtWy = BtW @ y

        def aic_score(log_lambda: float) -> float:
            lam = np.exp(log_lambda)
            try:
                coef = np.linalg.solve(BtWB + lam * S, BtWy)
                fitted = B @ coef

                # Hat matrix trace
                P_inv = np.linalg.inv(BtWB + lam * S)
                H = B @ P_inv @ BtW
                edf = np.trace(H)

                rss = np.sum(weights * (y - fitted) ** 2)
                if rss <= 0:
                    return np.inf
                return n * np.log(rss / n) + 2 * edf
            except np.linalg.LinAlgError:
                return np.inf

        log_range = (np.log(lambda_range[0]), np.log(lambda_range[1]))
        result = minimize_scalar(aic_score, bounds=log_range, method="bounded")

        return np.exp(result.x)

    def _select_lambda_reml(
        self,
        B: NDArray,
        y: NDArray,
        S: NDArray,
        weights: NDArray,
        lambda_range: tuple[float, float],
    ) -> float:
        """Select λ by maximizing REML (Restricted Maximum Likelihood).

        REML provides unbiased variance estimation and often works better
        than GCV for small samples or when edf is large relative to n.
        """
        n, k = B.shape
        W = np.diag(weights)
        BtW = B.T @ W
        BtWB = BtW @ B
        BtWy = BtW @ y

        def neg_reml(log_lambda: float) -> float:
            lam = np.exp(log_lambda)
            try:
                # Precision matrix
                P = BtWB + lam * S

                # Coefficients
                coef = np.linalg.solve(P, BtWy)
                fitted = B @ coef

                # RSS
                rss = np.sum(weights * (y - fitted) ** 2)

                # Log determinants
                sign_P, logdet_P = np.linalg.slogdet(P)
                sign_BtWB, logdet_BtWB = np.linalg.slogdet(BtWB + 1e-10 * np.eye(k))

                if sign_P <= 0 or sign_BtWB <= 0:
                    return np.inf

                # REML log-likelihood (up to constants)
                # -2 * REML ∝ (n-k)*log(σ²) + log|P| - log|B'WB| + RSS/σ²
                # For fixed σ², minimize: log|P| + (n-k)*log(RSS)
                reml_crit = logdet_P + (n - k) * np.log(rss + 1e-10)

                return reml_crit
            except np.linalg.LinAlgError:
                return np.inf

        log_range = (np.log(lambda_range[0]), np.log(lambda_range[1]))
        result = minimize_scalar(neg_reml, bounds=log_range, method="bounded")

        return np.exp(result.x)

    def derivative_matrix(self, x: NDArray, order: int = 1) -> NDArray:
        """Compute derivative basis matrix.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate derivatives
        order : int
            Order of derivative (1, 2, ...)

        Returns
        -------
        dB : ndarray, shape (len(x), n_basis)
            Derivative basis matrix
        """
        if self._bspline_basis is None:
            self._setup(x)

        return self._bspline_basis.derivative_basis_matrix(x, order=order)


@dataclass
class PSplineResult:
    """Results from P-spline fitting.

    Attributes
    ----------
    coef_ : ndarray
        Fitted spline coefficients
    fitted_values_ : ndarray
        Fitted values at training points
    lambda_ : float
        Smoothing parameter used
    basis : PSplineBasis
        The basis object used for fitting
    edf_ : float
        Effective degrees of freedom (trace of hat matrix)
    gcv_ : float
        Generalized Cross-Validation score
    sigma2_ : float
        Estimated residual variance
    x_ : ndarray
        Training predictor values
    y_ : ndarray
        Training response values
    weights_ : ndarray
        Observation weights used
    """

    coef_: NDArray
    fitted_values_: NDArray
    lambda_: float
    basis: PSplineBasis
    edf_: float
    gcv_: float
    sigma2_: float
    x_: NDArray
    y_: NDArray
    weights_: NDArray

    def predict(self, x_new: NDArray) -> NDArray:
        """Predict at new points.

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
        B_new = self.basis.basis_matrix(x_new)
        return B_new @ self.coef_

    def derivative(self, x: NDArray | None = None, order: int = 1) -> NDArray:
        """Evaluate derivative of fitted spline.

        Parameters
        ----------
        x : array-like, optional
            Points at which to evaluate. If None, uses training points.
        order : int
            Order of derivative (1, 2, ...)

        Returns
        -------
        deriv : ndarray
            Derivative values
        """
        if x is None:
            x = self.x_
        x = np.asarray(x).ravel()
        dB = self.basis.derivative_matrix(x, order=order)
        return dB @ self.coef_

    def confidence_band(
        self, x: NDArray | None = None, level: float = 0.95
    ) -> tuple[NDArray, NDArray]:
        """Compute pointwise confidence band.

        Parameters
        ----------
        x : array-like, optional
            Points at which to compute band. If None, uses training points.
        level : float
            Confidence level (e.g., 0.95 for 95%)

        Returns
        -------
        lower, upper : ndarray
            Lower and upper confidence bounds
        """
        from scipy import stats

        if x is None:
            x = self.x_
        x = np.asarray(x).ravel()

        B = self.basis.basis_matrix(x)
        y_pred = B @ self.coef_

        # Covariance of coefficients: σ² (B'WB + λS)^{-1}
        W = np.diag(self.weights_)
        S = self.basis.penalty_matrix()
        BtWB = self.basis.basis_matrix(self.x_).T @ W @ self.basis.basis_matrix(self.x_)
        cov_coef = self.sigma2_ * np.linalg.inv(BtWB + self.lambda_ * S)

        # Variance of predictions: diag(B @ cov_coef @ B')
        var_pred = np.sum((B @ cov_coef) * B, axis=1)
        se_pred = np.sqrt(var_pred)

        # Critical value
        alpha = 1 - level
        z = stats.norm.ppf(1 - alpha / 2)

        return y_pred - z * se_pred, y_pred + z * se_pred

    def summary(self) -> dict:
        """Return summary statistics.

        Returns
        -------
        dict
            Summary including lambda, edf, GCV, R², etc.
        """
        residuals = self.y_ - self.fitted_values_
        ss_res = np.sum(self.weights_ * residuals**2)
        ss_tot = np.sum(self.weights_ * (self.y_ - np.average(self.y_, weights=self.weights_))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "n_obs": len(self.y_),
            "n_basis": self.basis.n_basis,
            "degree": self.basis.degree,
            "penalty_order": self.basis.penalty_order,
            "lambda": self.lambda_,
            "edf": self.edf_,
            "gcv": self.gcv_,
            "sigma2": self.sigma2_,
            "r_squared": r_squared,
            "rss": ss_res,
        }


def fit_pspline(
    x: NDArray,
    y: NDArray,
    n_basis: int = 20,
    degree: int = 3,
    penalty_order: int = 2,
    lambda_: float | str = "gcv",
    weights: NDArray | None = None,
    **kwargs,
) -> PSplineResult:
    """Fit P-spline smoother to data.

    Convenience function that creates a PSplineBasis and fits it to data.

    Parameters
    ----------
    x : array-like
        Predictor values
    y : array-like
        Response values
    n_basis : int, default=20
        Number of B-spline basis functions
    degree : int, default=3
        B-spline degree (cubic by default)
    penalty_order : int, default=2
        Order of difference penalty
    lambda_ : float or str, default='gcv'
        Smoothing parameter or selection method ('gcv', 'aic', 'reml')
    weights : array-like, optional
        Observation weights
    **kwargs
        Additional arguments passed to PSplineBasis.fit()

    Returns
    -------
    PSplineResult
        Fitted P-spline

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.smoothing.splines import fit_pspline
    >>>
    >>> # Generate noisy sine data
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> y = np.sin(x) + np.random.normal(0, 0.2, 100)
    >>>
    >>> # Fit P-spline with automatic smoothing
    >>> result = fit_pspline(x, y, n_basis=25)
    >>> print(f"EDF: {result.edf_:.2f}, λ: {result.lambda_:.2f}")
    >>>
    >>> # Predict at new points
    >>> x_new = np.linspace(0, 2*np.pi, 200)
    >>> y_smooth = result.predict(x_new)
    """
    basis = PSplineBasis(
        n_basis=n_basis,
        degree=degree,
        penalty_order=penalty_order,
    )
    return basis.fit(x, y, lambda_=lambda_, weights=weights, **kwargs)
