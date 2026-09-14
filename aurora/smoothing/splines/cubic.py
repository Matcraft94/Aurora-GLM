# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Natural cubic spline basis functions.

Natural cubic splines are piecewise cubic polynomials that:
- Are continuous and have continuous first and second derivatives at knots
- Have zero second derivative at boundaries (natural boundary condition)
- Minimize integrated squared second derivative among interpolating functions

This implementation follows the approach in Wood (2017) and R's splines package.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aurora.distributions._utils import as_namespace_array, namespace


class CubicSplineBasis:
    """Natural cubic spline basis for smooth function representation.

    Natural cubic splines provide a flexible basis for representing smooth functions.
    They are especially useful in GAMs for modeling non-linear relationships.

    Parameters
    ----------
    knots : array-like
        Interior knot locations. Should be sorted in ascending order.
    boundary_knots : tuple of float, optional
        (lower, upper) boundary knots. If None, uses min and max of knots.

    Attributes
    ----------
    knots_ : ndarray
        Interior knots
    boundary_knots_ : tuple
        Boundary knots
    n_basis_ : int
        Number of basis functions (len(knots) + 2 for natural splines)

    References
    ----------
    Wood, S.N. (2017). Generalized Additive Models: An Introduction with R (2nd ed.)
    """

    def __init__(
        self,
        knots: Any,
        boundary_knots: tuple[float, float] | None = None,
    ):
        knots_arr = np.asarray(knots, dtype=np.float64)

        if knots_arr.ndim != 1:
            raise ValueError("knots must be a 1-dimensional array")
        if len(knots_arr) < 1:
            raise ValueError("At least one interior knot is required")
        if not np.all(np.diff(knots_arr) > 0):
            raise ValueError("knots must be strictly increasing")

        self.knots_ = knots_arr

        if boundary_knots is None:
            # Add some padding beyond the knot range
            knot_range = knots_arr[-1] - knots_arr[0]
            self.boundary_knots_ = (
                float(knots_arr[0] - 0.1 * knot_range),
                float(knots_arr[-1] + 0.1 * knot_range),
            )
        else:
            if len(boundary_knots) != 2:
                raise ValueError("boundary_knots must be a tuple of (lower, upper)")
            if boundary_knots[0] >= boundary_knots[1]:
                raise ValueError("Lower boundary must be less than upper boundary")
            if boundary_knots[0] > knots_arr[0] or boundary_knots[1] < knots_arr[-1]:
                raise ValueError("Boundary knots must enclose all interior knots")
            self.boundary_knots_ = boundary_knots

        # Natural cubic splines have k+2 basis functions for k interior knots
        self.n_basis_ = len(self.knots_) + 2

    def basis_matrix(self, x: Any) -> Any:
        """Compute the basis matrix B where B[i,j] = b_j(x_i).

        Parameters
        ----------
        x : array-like, shape (n_samples,)
            Points at which to evaluate the basis functions.

        Returns
        -------
        B : array, shape (n_samples, n_basis)
            Basis matrix. Uses the same array backend as input x.

        Notes
        -----
        The basis functions are constructed to satisfy:
        - Cubic polynomials between knots
        - Continuous second derivatives at knots
        - Zero second derivatives at boundaries (natural condition)
        """
        xp = namespace(x)
        x_arr = as_namespace_array(x, xp)

        if x_arr.ndim == 0:
            x_arr = xp.reshape(x_arr, (1,))
        elif x_arr.ndim != 1:
            raise ValueError("x must be 1-dimensional")

        n = x_arr.shape[0]

        # All knots including boundaries
        # Convert to numpy first, then to target backend
        all_knots_np = np.concatenate(
            [[self.boundary_knots_[0]], self.knots_, [self.boundary_knots_[1]]]
        )
        all_knots = as_namespace_array(all_knots_np, xp, like=x_arr)

        # Initialize basis matrix
        B = xp.zeros((n, self.n_basis_), dtype=x_arr.dtype)

        # Natural cubic spline basis (Hastie, Tibshirani & Friedman 2009,
        # ESL eq. 5.4-5.5).  With all K knots xi_1 < ... < xi_K (boundary
        # knots included):
        #   N_1(x) = 1,   N_2(x) = x,
        #   N_{j+2}(x) = d_j(x) - d_{K-1}(x),   j = 1, ..., K-2,
        #   d_j(x) = [(x - xi_j)_+^3 - (x - xi_K)_+^3] / (xi_K - xi_j).
        # Every N_j has zero second derivative at both boundary knots, so
        # the natural boundary conditions f''(xi_1) = f''(xi_K) = 0 hold
        # exactly for any coefficient vector.

        # Linear terms (first two basis functions)
        B[:, 0] = xp.ones(n)
        B[:, 1] = x_arr

        n_total = len(all_knots_np)  # K: interior knots + 2 boundary knots
        xi_upper = all_knots[n_total - 1]  # xi_K

        def _truncated_cube(knot: Any) -> Any:
            diff = x_arr - knot
            return xp.where(diff > 0, diff**3, xp.zeros_like(diff))

        cube_upper = _truncated_cube(xi_upper)

        def _d(knot: Any) -> Any:
            return (_truncated_cube(knot) - cube_upper) / (xi_upper - knot)

        d_penultimate = _d(all_knots[n_total - 2])  # d_{K-1}

        for j in range(n_total - 2):
            B[:, j + 2] = _d(all_knots[j]) - d_penultimate

        return B

    def penalty_matrix(self) -> np.ndarray:
        """Compute the penalty matrix S for integrated squared second derivative.

        The penalty matrix satisfies: β'Sβ = ∫ [f''(x)]² dx

        Returns
        -------
        S : ndarray, shape (n_basis, n_basis)
            Penalty matrix (always NumPy array).

        Notes
        -----
        For natural cubic splines, the penalty can be computed analytically.
        The first two basis functions (constant and linear) receive zero penalty
        since their second derivatives are zero.

        For the ESL (5.4-5.5) basis N_{j+2} = d_j - d_{K-1} with
        d_j(x) = [(x - xi_j)_+^3 - (x - xi_K)_+^3] / (xi_K - xi_j), the second
        derivative is d_j''(x) = 6[(x - xi_j)_+ - (x - xi_K)_+]/(xi_K - xi_j).
        Since (x - xi_K)_+ = 0 on [xi_1, xi_K] and every N_j'' vanishes
        outside the boundary knots, the integral over the real line reduces to

            ∫ d_i'' d_j'' dx = J(xi_i, xi_j) / [(xi_K - xi_i)(xi_K - xi_j)]

        with J(t, s) = ∫ 6(x - t)_+ 6(x - s)_+ dx, and

            S[i+2, j+2] = D(i,j) - D(i,K-1) - D(K-1,j) + D(K-1,K-1).
        """
        n_basis = self.n_basis_

        # Initialize penalty matrix
        S = np.zeros((n_basis, n_basis), dtype=np.float64)

        # The constant and linear terms have zero penalty (second derivative is zero)
        # Penalty only on cubic terms

        # Compute penalties using integrated second derivatives
        all_knots = np.concatenate(
            [[self.boundary_knots_[0]], self.knots_, [self.boundary_knots_[1]]]
        )
        n_total = len(all_knots)  # K
        upper = all_knots[-1]  # xi_K
        km1 = n_total - 2  # Index of xi_{K-1}

        def _D(p: int, q: int) -> float:
            """Integral of d_p''(x) d_q''(x) over the real line."""
            return float(
                self._integrate_second_derivatives(all_knots[p], all_knots[q], upper)
                / ((upper - all_knots[p]) * (upper - all_knots[q]))
            )

        # For cubic terms: S[i+2, j+2] = D(i,j) - D(i,K-1) - D(K-1,j) + D(K-1,K-1)
        for i in range(n_total - 2):
            for j in range(i, n_total - 2):
                penalty_ij = _D(i, j) - _D(i, km1) - _D(km1, j) + _D(km1, km1)
                S[i + 2, j + 2] = penalty_ij
                S[j + 2, i + 2] = penalty_ij  # Symmetric

        return S

    def _integrate_second_derivatives(self, t_i: float, t_j: float, upper: float) -> float:
        """Integrate the product of two truncated-power second derivatives.

        Evaluates  J(t_i, t_j) = ∫ 6(x - t_i)_+ 6(x - t_j)_+ dx  analytically
        over [max(t_i, t_j), upper] (the integrand is zero below both knots).

        Parameters
        ----------
        t_i : float
            Knot of the first truncated power term b(x) = (x - t_i)_+^3,
            whose second derivative is 6(x - t_i)_+.
        t_j : float
            Knot of the second truncated power term.
        upper : float
            Upper integration limit (upper boundary knot).

        Returns
        -------
        penalty : float
            Value of the integrated product of second derivatives.
            Returns 0.0 when the integration interval is degenerate.
        """
        # Integrate from max(t_i, t_j) to the upper boundary
        lower = max(t_i, t_j)

        if lower >= upper:
            return 0.0

        # For (x - t_i)_+ and (x - t_j)_+, second derivatives are 6(x-t_i) and 6(x-t_j)
        # Integral of 36(x-t_i)(x-t_j) from lower to upper

        # Expand: (x-t_i)(x-t_j) = x² - (t_i+t_j)x + t_i*t_j
        # Integral: x³/3 - (t_i+t_j)x²/2 + t_i*t_j*x

        def antiderivative(x: float) -> float:
            return x**3 / 3.0 - (t_i + t_j) * x**2 / 2.0 + t_i * t_j * x

        integral = antiderivative(upper) - antiderivative(lower)
        return 36.0 * integral

    @staticmethod
    def create_knots(x: Any, n_knots: int = 10, method: str = "quantile") -> np.ndarray:
        """Create knot locations from data.

        Parameters
        ----------
        x : array-like
            Data values used to determine knot positions.
        n_knots : int, default=10
            Number of interior knots to place.
        method : str, default='quantile'
            Knot placement method:

            - ``'quantile'``: Place knots at quantiles of *x*, adapting to the
              empirical data distribution.
            - ``'uniform'``: Place knots uniformly between min(x) and max(x).

        Returns
        -------
        knots : ndarray, shape (n_knots,) or fewer
            Sorted interior knot locations.  May contain fewer than *n_knots*
            entries if duplicate quantiles are merged.

        Raises
        ------
        ValueError
            If *n_knots* is less than 1 or *method* is unknown.

        Examples
        --------
        >>> x = np.random.randn(500)
        >>> knots = CubicSplineBasis.create_knots(x, n_knots=8)
        >>> len(knots) <= 8
        True
        """
        x_np = np.asarray(x)

        if n_knots < 1:
            raise ValueError("n_knots must be at least 1")

        if method == "quantile":
            # Place knots at quantiles
            probs = np.linspace(0, 1, n_knots + 2)[1:-1]  # Exclude 0 and 1
            knots = np.quantile(x_np, probs)
        elif method == "uniform":
            # Place knots uniformly
            x_min, x_max = x_np.min(), x_np.max()
            knots = np.linspace(x_min, x_max, n_knots + 2)[1:-1]
        else:
            raise ValueError(f"Unknown method: {method}")

        # Ensure unique knots
        knots = np.unique(knots)

        return knots


__all__ = ["CubicSplineBasis"]
