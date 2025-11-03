"""B-spline basis functions using Cox-de Boor recursion algorithm.

B-splines have several advantages over other spline representations:
- Local support: changing one coefficient only affects a local region
- Stable numerical computation via de Boor algorithm
- Efficient evaluation (only need to evaluate non-zero basis functions)
- Natural generalization to higher dimensions (tensor products)

This implementation follows de Boor (1978) and is compatible with scipy.interpolate.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from aurora.distributions._utils import as_namespace_array, namespace


class BSplineBasis:
    """B-spline basis functions via Cox-de Boor recursion.

    B-splines are piecewise polynomials with compact support, defined recursively
    using the Cox-de Boor formula. They form a basis for the space of splines of
    a given degree with specified knots.

    Parameters
    ----------
    knots : array-like
        Knot vector (should include boundary knots with multiplicity = degree + 1).
        For open B-splines, repeat boundary knots degree+1 times.
    degree : int, default=3
        Degree of the B-spline basis (3 for cubic B-splines).
    cyclic : bool, default=False
        Whether to use cyclic (periodic) boundary conditions.

    Attributes
    ----------
    knots_ : ndarray
        Full knot vector including repeated boundary knots.
    degree_ : int
        Degree of the splines.
    n_basis_ : int
        Number of basis functions.
    cyclic_ : bool
        Whether cyclic boundary conditions are used.

    Notes
    -----
    For degree p and n_basis basis functions, the knot vector must have
    length n_basis + p + 1.

    References
    ----------
    de Boor, C. (1978). A Practical Guide to Splines. Springer-Verlag.
    Eilers, P.H.C. & Marx, B.D. (1996). Flexible smoothing with B-splines and
        penalties. Statistical Science, 11(2), 89-121.
    """

    def __init__(
        self,
        knots: Any,
        degree: int = 3,
        cyclic: bool = False,
    ):
        if degree < 0:
            raise ValueError("degree must be non-negative")

        knots_arr = np.asarray(knots, dtype=np.float64)

        if knots_arr.ndim != 1:
            raise ValueError("knots must be 1-dimensional")

        if not np.all(np.diff(knots_arr) >= 0):
            raise ValueError("knots must be non-decreasing")

        self.knots_ = knots_arr
        self.degree_ = degree
        self.cyclic_ = cyclic

        # Number of basis functions
        self.n_basis_ = len(knots_arr) - degree - 1

        if self.n_basis_ < 1:
            raise ValueError(
                f"Need at least {degree + 2} knots for degree {degree} B-splines"
            )

    def basis_matrix(self, x: Any) -> Any:
        """Compute B-spline basis matrix using Cox-de Boor recursion.

        Parameters
        ----------
        x : array-like, shape (n_samples,)
            Points at which to evaluate basis functions.

        Returns
        -------
        B : array, shape (n_samples, n_basis)
            Basis matrix where B[i,j] = B_j^p(x_i), the j-th B-spline
            of degree p evaluated at x_i.

        Notes
        -----
        Uses the Cox-de Boor recursion formula:
            B_i^0(x) = 1 if t_i ≤ x < t_{i+1}, else 0
            B_i^p(x) = w_i^p(x) B_i^{p-1}(x) + (1 - w_{i+1}^p(x)) B_{i+1}^{p-1}(x)
        where w_i^p(x) = (x - t_i) / (t_{i+p} - t_i)

        Only evaluates non-zero basis functions for efficiency.
        """
        xp = namespace(x)
        x_arr = as_namespace_array(x, xp)

        if x_arr.ndim == 0:
            x_arr = xp.reshape(x_arr, (1,))
        elif x_arr.ndim != 1:
            raise ValueError("x must be 1-dimensional")

        n = x_arr.shape[0]

        # Convert knots to target backend
        knots = as_namespace_array(self.knots_, xp, like=x_arr)

        # Initialize basis matrix
        B = xp.zeros((n, self.n_basis_), dtype=x_arr.dtype)

        # For each evaluation point
        for idx in range(n):
            x_val = x_arr[idx]

            # Find knot interval containing x_val
            # For each basis function that could be non-zero at x_val
            for i in range(self.n_basis_):
                # Basis function i has support [knots[i], knots[i+degree+1]]
                B_val = self._evaluate_basis(x_val, i, self.degree_, knots, xp)
                B[idx, i] = B_val

        return B

    def _evaluate_basis(
        self, x: Any, i: int, p: int, knots: Any, xp: Any
    ) -> Any:
        """Evaluate single B-spline basis function using Cox-de Boor recursion.

        Parameters
        ----------
        x : scalar
            Evaluation point.
        i : int
            Basis function index.
        p : int
            Current degree in recursion.
        knots : array
            Knot vector.
        xp : module
            Array namespace.

        Returns
        -------
        value : scalar
            B_i^p(x)
        """
        # Base case: degree 0 (piecewise constant)
        if p == 0:
            # B_i^0(x) = 1 if t_i ≤ x < t_{i+1}, else 0
            # Use slightly loose comparison to handle boundary
            in_interval = (knots[i] <= x) & (x <= knots[i + 1])
            return xp.where(in_interval, xp.ones_like(x), xp.zeros_like(x))

        # Recursive case
        # Left term: w_i^p(x) * B_i^{p-1}(x)
        denom_left = knots[i + p] - knots[i]
        if float(denom_left) > 1e-10:  # Avoid division by zero
            w_left = (x - knots[i]) / denom_left
            left_term = w_left * self._evaluate_basis(x, i, p - 1, knots, xp)
        else:
            left_term = xp.zeros_like(x)

        # Right term: (1 - w_{i+1}^p(x)) * B_{i+1}^{p-1}(x)
        denom_right = knots[i + p + 1] - knots[i + 1]
        if float(denom_right) > 1e-10:
            w_right = (x - knots[i + 1]) / denom_right
            right_term = (1.0 - w_right) * self._evaluate_basis(
                x, i + 1, p - 1, knots, xp
            )
        else:
            right_term = xp.zeros_like(x)

        return left_term + right_term

    def penalty_matrix(self, order: int = 2) -> np.ndarray:
        """Compute difference penalty matrix.

        For B-splines, we typically use a difference penalty that approximates
        the integrated squared derivative penalty.

        Parameters
        ----------
        order : int, default=2
            Order of differences (2 for approximating second derivative).

        Returns
        -------
        S : ndarray, shape (n_basis, n_basis)
            Penalty matrix where β'Sβ approximates ∫ [f^(m)(x)]^2 dx.

        Notes
        -----
        The difference penalty is:
            S = D'D
        where D is the order-th difference matrix.

        For order=2: D_ij penalizes (β_i - 2β_{i+1} + β_{i+2})²
        This approximates the integrated squared second derivative.
        """
        if order < 1:
            raise ValueError("order must be positive")

        if order > self.n_basis_:
            raise ValueError(f"order {order} too large for {self.n_basis_} basis functions")

        # Create difference matrix
        D = np.diff(np.eye(self.n_basis_), n=order, axis=0)

        # Penalty matrix is D'D
        S = D.T @ D

        return S

    @staticmethod
    def create_knots(
        x: Any,
        n_basis: int = 10,
        degree: int = 3,
        method: str = "quantile",
    ) -> np.ndarray:
        """Create knot vector for B-splines from data.

        Parameters
        ----------
        x : array-like
            Data values.
        n_basis : int
            Number of basis functions desired.
        degree : int
            Degree of B-splines.
        method : str
            Knot placement method:
            - 'quantile': Place interior knots at quantiles
            - 'uniform': Place interior knots uniformly

        Returns
        -------
        knots : ndarray
            Full knot vector with repeated boundary knots.

        Notes
        -----
        For open B-splines, boundary knots are repeated (degree + 1) times.
        This ensures the spline interpolates at boundaries.
        """
        x_np = np.asarray(x)

        if n_basis < 1:
            raise ValueError("n_basis must be at least 1")

        # Number of interior knots
        n_interior = n_basis - degree - 1

        if n_interior < 0:
            raise ValueError(
                f"n_basis={n_basis} too small for degree={degree}. "
                f"Need at least {degree + 1} basis functions."
            )

        x_min, x_max = x_np.min(), x_np.max()

        if n_interior == 0:
            # No interior knots, just boundaries
            interior_knots = np.array([])
        elif method == "quantile":
            # Place interior knots at quantiles
            probs = np.linspace(0, 1, n_interior + 2)[1:-1]
            interior_knots = np.quantile(x_np, probs)
        elif method == "uniform":
            # Place interior knots uniformly
            interior_knots = np.linspace(x_min, x_max, n_interior + 2)[1:-1]
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create full knot vector with repeated boundaries
        # For open B-splines: repeat each boundary (degree + 1) times
        knots = np.concatenate([
            np.repeat(x_min, degree + 1),
            interior_knots,
            np.repeat(x_max, degree + 1),
        ])

        return knots

    def derivative_basis_matrix(self, x: Any, order: int = 1) -> Any:
        """Compute derivative of B-spline basis functions.

        Parameters
        ----------
        x : array-like, shape (n_samples,)
            Evaluation points.
        order : int, default=1
            Order of derivative (1 for first derivative, etc.).

        Returns
        -------
        dB : array, shape (n_samples, n_basis)
            Matrix of derivative values.

        Notes
        -----
        Uses the derivative formula:
            dB_i^p(x)/dx = p * [B_i^{p-1}(x)/(t_{i+p}-t_i) -
                                 B_{i+1}^{p-1}(x)/(t_{i+p+1}-t_{i+1})]
        """
        if order < 1:
            raise ValueError("order must be positive")

        if order > self.degree_:
            # Derivative of polynomial of degree p is zero for order > p
            xp = namespace(x)
            x_arr = as_namespace_array(x, xp)
            if x_arr.ndim == 0:
                x_arr = xp.reshape(x_arr, (1,))
            n = x_arr.shape[0]
            return xp.zeros((n, self.n_basis_), dtype=x_arr.dtype)

        # For first derivative, use recursive formula
        # For higher derivatives, recursively call this function
        if order > 1:
            # d^n f / dx^n = d/dx (d^{n-1} f / dx^{n-1})
            # Create basis with degree-1 to compute derivative
            reduced_basis = BSplineBasis(self.knots_, degree=self.degree_ - 1)
            return reduced_basis.derivative_basis_matrix(x, order - 1)

        # First derivative using de Boor formula
        xp = namespace(x)
        x_arr = as_namespace_array(x, xp)

        if x_arr.ndim == 0:
            x_arr = xp.reshape(x_arr, (1,))

        n = x_arr.shape[0]
        knots = as_namespace_array(self.knots_, xp, like=x_arr)

        dB = xp.zeros((n, self.n_basis_), dtype=x_arr.dtype)

        # Create basis of degree p-1 for derivative computation
        if self.degree_ > 0:
            reduced_basis = BSplineBasis(self.knots_, degree=self.degree_ - 1)
            B_reduced = reduced_basis.basis_matrix(x_arr)

            for i in range(self.n_basis_):
                # First term
                denom1 = knots[i + self.degree_] - knots[i]
                if float(denom1) > 1e-10:
                    term1 = self.degree_ * B_reduced[:, i] / denom1
                else:
                    term1 = xp.zeros(n, dtype=x_arr.dtype)

                # Second term
                if i + 1 < reduced_basis.n_basis_:
                    denom2 = knots[i + self.degree_ + 1] - knots[i + 1]
                    if float(denom2) > 1e-10:
                        term2 = self.degree_ * B_reduced[:, i + 1] / denom2
                    else:
                        term2 = xp.zeros(n, dtype=x_arr.dtype)
                else:
                    term2 = xp.zeros(n, dtype=x_arr.dtype)

                dB[:, i] = term1 - term2

        return dB


__all__ = ["BSplineBasis"]
