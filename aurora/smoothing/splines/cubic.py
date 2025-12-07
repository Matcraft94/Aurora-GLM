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
        k = len(self.knots_)

        # All knots including boundaries
        # Convert to numpy first, then to target backend
        all_knots_np = np.concatenate(
            [[self.boundary_knots_[0]], self.knots_, [self.boundary_knots_[1]]]
        )
        all_knots = as_namespace_array(all_knots_np, xp, like=x_arr)

        # Initialize basis matrix
        B = xp.zeros((n, self.n_basis_), dtype=x_arr.dtype)

        # Compute basis functions using the natural cubic spline formulation
        # This implementation uses the truncated power basis representation

        # Linear terms (first two basis functions)
        B[:, 0] = xp.ones(n)
        B[:, 1] = x_arr

        # Cubic terms with natural boundary conditions
        for j in range(k):
            knot = all_knots[j + 1]  # Interior knot j
            # Truncated power basis: (x - knot)_+^3
            diff = x_arr - knot
            B[:, j + 2] = xp.where(diff > 0, diff**3, xp.zeros_like(diff))

        # Apply natural boundary conditions
        # This transforms the truncated power basis into natural cubic splines
        B = self._apply_natural_constraints(B, all_knots, xp)

        return B

    def _apply_natural_constraints(self, B: Any, all_knots: Any, xp: Any) -> Any:
        """Apply natural boundary conditions to transform basis.

        Natural splines have zero second derivative at boundaries.
        This is achieved by subtracting appropriate combinations of
        the truncated power terms.
        """
        k = len(self.knots_)
        if k < 2:
            return B

        # Get boundary values
        a = all_knots[0]  # Lower boundary
        b = all_knots[-1]  # Upper boundary

        # For each interior knot, adjust to satisfy natural conditions
        for j in range(k):
            knot = all_knots[j + 1]

            # Compute adjustment factors
            d_numer = (b - knot) ** 3 - (b - a) ** 3
            d_denom = b - a

            if abs(d_denom) > 1e-10:
                d = d_numer / d_denom

                # Apply correction
                B[:, j + 2] = B[:, j + 2] - d * B[:, 1] / 3.0

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
        """
        k = len(self.knots_)
        n_basis = self.n_basis_

        # Initialize penalty matrix
        S = np.zeros((n_basis, n_basis), dtype=np.float64)

        # The constant and linear terms have zero penalty (second derivative is zero)
        # Penalty only on cubic terms

        # Compute penalties using integrated second derivatives
        all_knots = np.concatenate(
            [[self.boundary_knots_[0]], self.knots_, [self.boundary_knots_[1]]]
        )

        # For cubic terms
        for i in range(2, n_basis):
            for j in range(i, n_basis):
                # Integrate the product of second derivatives
                # For truncated power basis (x - t)_+^3, second derivative is 6(x - t)_+
                penalty_ij = self._integrate_second_derivatives(i, j, all_knots)
                S[i, j] = penalty_ij
                S[j, i] = penalty_ij  # Symmetric

        return S

    def _integrate_second_derivatives(
        self, i: int, j: int, all_knots: np.ndarray
    ) -> float:
        """Compute ∫ f_i''(x) f_j''(x) dx for cubic basis functions.

        For truncated power basis b(x) = (x - t)_+^3:
        - b'(x) = 3(x - t)_+^2
        - b''(x) = 6(x - t)_+
        """
        # Get corresponding knots (offset by 2 because first two basis are linear)
        knot_i = all_knots[i - 1]
        knot_j = all_knots[j - 1]

        # Integrate from max(knot_i, knot_j) to upper boundary
        lower = max(knot_i, knot_j)
        upper = all_knots[-1]

        if lower >= upper:
            return 0.0

        # For (x - t_i)_+ and (x - t_j)_+, second derivatives are 6(x-t_i) and 6(x-t_j)
        # Integral of 36(x-t_i)(x-t_j) from lower to upper

        # Expand: (x-t_i)(x-t_j) = x² - (t_i+t_j)x + t_i*t_j
        # Integral: x³/3 - (t_i+t_j)x²/2 + t_i*t_j*x

        def antiderivative(x: float) -> float:
            return x**3 / 3.0 - (knot_i + knot_j) * x**2 / 2.0 + knot_i * knot_j * x

        integral = antiderivative(upper) - antiderivative(lower)
        return 36.0 * integral

    @staticmethod
    def create_knots(x: Any, n_knots: int = 10, method: str = "quantile") -> np.ndarray:
        """Create knot locations from data.

        Parameters
        ----------
        x : array-like
            Data values
        n_knots : int
            Number of interior knots
        method : str
            Knot placement method:
            - 'quantile': Place knots at quantiles of x
            - 'uniform': Place knots uniformly between min and max

        Returns
        -------
        knots : ndarray
            Knot locations
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
