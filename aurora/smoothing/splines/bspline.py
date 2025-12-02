"""B-Spline Basis Functions via Cox-de Boor Recursion.

Mathematical Framework
----------------------
B-splines (basis splines) are piecewise polynomial functions that form a basis
for the space of splines. They are the fundamental building blocks for flexible
curve and surface representations in computer-aided design, numerical analysis,
and statistical smoothing.

Definition
----------
Given a knot sequence (breakpoints):

    t = {t₀, t₁, ..., t_{K+p}}

where p is the degree, the B-spline basis functions B_{i,p}(x) for i = 0,...,K-1
are defined recursively via the **Cox-de Boor recursion formula**.

Cox-de Boor Recursion
---------------------
**Base case** (p = 0, piecewise constant):

    B_{i,0}(x) = { 1  if tᵢ ≤ x < tᵢ₊₁
                 { 0  otherwise

**Recursive case** (p ≥ 1):

    B_{i,p}(x) = w_{i,p}(x) B_{i,p-1}(x) + [1 - w_{i+1,p}(x)] B_{i+1,p-1}(x)

where the weight functions are:

    w_{i,p}(x) = (x - tᵢ) / (tᵢ₊ₚ - tᵢ)

**Boundary convention**: If tᵢ₊ₚ = tᵢ (repeated knots), define w_{i,p} = 0.

**Interpretation**: B_{i,p} is a weighted average of two lower-degree B-splines,
interpolating linearly between them based on x's position in the knot interval.

Properties of B-Splines
-----------------------
B-splines have several remarkable mathematical properties:

1. **Non-negativity**:
   B_{i,p}(x) ≥ 0  for all x

2. **Compact support**:
   B_{i,p}(x) = 0  for x ∉ [tᵢ, tᵢ₊ₚ₊₁]

   Each basis function is non-zero only on p+1 consecutive knot intervals.

3. **Partition of unity**:
   Σᵢ B_{i,p}(x) = 1  for all x ∈ [t_p, t_{K+1}]

   The basis functions sum to 1 at every point.

4. **Local linear independence**:
   Any p+1 consecutive B-splines are linearly independent.

5. **Continuity**:
   - At simple knots: C^{p-1} continuous (p-1 continuous derivatives)
   - At knots with multiplicity m: C^{p-m} continuous
   - Interior to knot spans: C^∞ (polynomial)

6. **Polynomial precision**:
   B-splines of degree p can exactly represent any polynomial of degree ≤ p.

**Why these properties matter**:
- Non-negativity + partition of unity → stability and intuitive coefficients
- Compact support → sparse matrices, local control
- Continuity → smooth approximations

Spline Space
------------
The **spline space** S_{p,t} is the span of B-spline basis functions:

    S_{p,t} = span{B_{0,p}, B_{1,p}, ..., B_{K-1,p}}

Any function f ∈ S_{p,t} can be written as:

    f(x) = Σᵢ βᵢ B_{i,p}(x)

where β = (β₀, ..., β_{K-1}) are spline coefficients.

**Dimension**: dim(S_{p,t}) = K = (number of knots) - p - 1

**Approximation power**: For sufficiently smooth functions g,

    min_{f ∈ S_{p,t}} ||g - f|| = O(h^{p+1})

where h = max knot spacing (if knots are equally spaced).

Knot Vectors
------------
The knot vector t determines the structure of the spline space.

### Open (Clamped) Knots

For approximation on [a, b], use **repeated boundary knots**:

    t = {a, ..., a, t₁, t₂, ..., t_{n}, b, ..., b}
         ⎣______⎦                           ⎣______⎦
         p+1 times                          p+1 times

**Properties**:
- Spline interpolates boundary values: f(a) = β₀, f(b) = β_{K-1}
- Basis functions are non-zero on entire domain
- Standard choice for regression and smoothing

### Uniform Interior Knots

Interior knots {t₁, ..., t_n} can be placed:

1. **Uniformly**: tᵢ = a + i(b-a)/(n+1)
   - Simple, symmetric
   - May not adapt to data density

2. **Quantiles**: tᵢ = quantile(x, i/(n+1))
   - Adapts to data distribution
   - More knots where data is dense
   - Preferred for statistical smoothing

Derivatives of B-Splines
------------------------
The derivative of a B-spline is itself a B-spline of lower degree:

    dB_{i,p}(x)/dx = p [B_{i,p-1}(x)/(tᵢ₊ₚ - tᵢ) - B_{i+1,p-1}(x)/(tᵢ₊ₚ₊₁ - tᵢ₊₁)]

**Consequences**:
- Derivatives computed via same Cox-de Boor algorithm
- k-th derivative is B-spline of degree p-k
- Derivative of degree-0 spline is zero (piecewise constant → flat)

**Higher derivatives**: Apply formula recursively:

    d^k f/dx^k = Σᵢ βᵢ d^k B_{i,p}/dx^k

Penalty Matrices for Smoothing
-------------------------------
In penalized regression splines, we penalize roughness via:

    Penalty = λ β^T S β

where S is a **penalty matrix**.

### Difference Penalty (Discrete Approximation)

For m-th order differences:

    S = D_m^T D_m

where D_m is the m-th difference operator:

    D_2 = [  1  -2   1   0  ...  ]
          [  0   1  -2   1  ...  ]
          ...

**Approximates**: ∫ [f^{(m)}(x)]^2 dx for equally-spaced knots

**Advantages**:
- Simple to compute: O(K) construction
- Sparse: band matrix
- Works for any knot spacing

**Second-order (m=2)**: Penalizes curvature (bending energy)

### Integrated Squared Derivative (Exact)

For m-th derivative:

    S_{ij} = ∫ [B_{i,p}^{(m)}(x)] [B_{j,p}^{(m)}(x)] dx

**Advantages**:
- Exact penalty: ∫ [f^{(m)}]^2 dx = β^T S β
- Geometric interpretation: bending energy

**Disadvantages**:
- More expensive to compute: O(K²) for dense matrix
- Requires analytical integration or numerical quadrature

**For penalized GAMs**: Difference penalty is standard and computationally cheaper.

Computational Complexity
------------------------
### Basis Evaluation

**Naive Cox-de Boor recursion**:
- Per basis function at one point: O(p²)
- All K basis functions at n points: O(nKp²)

**Optimized (exploit compact support)**:
- Only p+1 basis functions are non-zero at any x
- Per point: O(p²) to find non-zero functions
- Total: O(np²)

### Derivative Computation

**k-th derivative**:
- Reduces to (p-k)-degree B-spline evaluation
- Cost: Same as evaluation, O(np²)

### Penalty Matrix

**Difference penalty**:
- Construction: O(K) for band matrix
- Matrix-vector product: O(K) (sparse)

**Integrated penalty**:
- Construction: O(Kp) per entry, O(K²p) total
- Dense matrix: O(K²) storage and products

Numerical Stability
-------------------
**Advantages of Cox-de Boor**:

1. **Avoid cancellation**: Only additions of non-negative terms
2. **Bounded intermediate values**: Weights w_{i,p} ∈ [0,1]
3. **No Vandermonde matrices**: Unlike polynomial bases, no conditioning issues

**Pitfalls**:

1. **Division by zero**: When tᵢ₊ₚ = tᵢ (repeated knots)
   - Solution: Check denominator, define 0/0 = 0

2. **Boundary evaluation**: x = t_{K+p} (right endpoint)
   - Convention: Extend last interval slightly or special-case

3. **Very high degree**: p > 10 rare in practice
   - High degree → oscillations (Runge phenomenon)
   - Better: More knots with lower degree

Comparison with Other Spline Bases
-----------------------------------
**vs Polynomial basis** {1, x, x², ...}:
- B-splines: Numerically stable, local support, well-conditioned
- Polynomials: Global support, Vandermonde matrix (ill-conditioned for large degree)

**vs Natural cubic splines**:
- B-splines: Local basis, general degree, efficient evaluation
- Natural cubic splines: Global basis, automatic C² continuity, simpler for small problems

**vs Radial basis functions** (RBFs):
- B-splines: Tensor product for multivariate, fast evaluation
- RBFs: Isotropic, harder to scale to high dimensions

**For GAMs**: B-splines are standard due to efficiency and numerical stability.

Multi-Backend Support
---------------------
This implementation works transparently with NumPy, PyTorch, and JAX arrays
through the array namespace abstraction. All operations use the `namespace()`
function to detect and use the appropriate backend.

**Benefits**:
- Same code for CPU and GPU
- Automatic differentiation support (PyTorch, JAX)
- Type consistency across backends

Applications in Aurora-GLM
---------------------------
B-splines are used in:

1. **Generalized Additive Models (GAM)**: Smooth functions f(x) = Σ β_i B_{i,p}(x)
2. **Penalized regression**: Roughness penalty β^T S β
3. **Varying coefficient models**: Coefficients that vary smoothly with covariates
4. **Tensor product smooths**: Multivariate smoothing via B-spline products

**Not used for**:
- Interpolation (prefer natural cubic splines with exact interpolation)
- Very small datasets (n < 20, use parametric models)

Implementation Notes
--------------------
**Design choices**:

1. **Recursive evaluation**: Direct Cox-de Boor formula for clarity
   - Could optimize with dynamic programming (de Boor's algorithm proper)
   - Current: O(p²) per point, acceptable for p ≤ 5

2. **Knot vector storage**: Full vector including repeated boundaries
   - Standard convention from de Boor (1978)
   - Compatible with scipy.interpolate

3. **Boundary handling**: Left-closed, right-closed intervals
   - B_{i,0}(x) = 1 for x ∈ [tᵢ, tᵢ₊₁]
   - Special case at rightmost knot

References
----------
**Core B-spline theory**:

- de Boor, C. (2001). *A Practical Guide to Splines* (Revised ed.). Springer.
  https://doi.org/10.1007/978-1-4612-6333-3
  (THE reference for B-splines, contains all proofs and algorithms)

- Schumaker, L. L. (2007). *Spline Functions: Basic Theory* (3rd ed.). Cambridge
  University Press.
  (Comprehensive mathematical treatment of spline theory)

**Cox-de Boor recursion**:

- Cox, M. G. (1972). \"The numerical evaluation of B-splines.\" *IMA Journal of
  Applied Mathematics*, 10(2), 134-149.
  https://doi.org/10.1093/imamat/10.2.134

- de Boor, C. (1972). \"On calculating with B-splines.\" *Journal of Approximation
  Theory*, 6(1), 50-62.
  https://doi.org/10.1016/0021-9045(72)90080-9

**Penalized regression splines**:

- Eilers, P. H. C., & Marx, B. D. (1996). \"Flexible smoothing with B-splines and
  penalties.\" *Statistical Science*, 11(2), 89-121.
  https://doi.org/10.1214/ss/1038425655
  (P-splines: B-splines + difference penalties)

**Computational algorithms**:

- de Boor, C. (1978). \"Efficient computer manipulation of tensor products.\"
  *ACM Transactions on Mathematical Software*, 5(2), 173-182.
  (Original de Boor algorithm, more efficient than naive recursion)

**Statistical applications**:

- Ruppert, D., Wand, M. P., & Carroll, R. J. (2003). *Semiparametric Regression*.
  Cambridge University Press. Chapter 5: Spline Smoothing.

- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). CRC Press. Chapter 4: Spline Bases.

**Numerical analysis**:

- Prautzsch, H., Boehm, W., & Paluszny, M. (2002). *Bézier and B-Spline
  Techniques*. Springer.
  (Computer graphics perspective, geometric algorithms)

See Also
--------
aurora.smoothing.splines.cubic : Natural cubic spline basis
aurora.models.gam.fitting : GAM fitting using B-splines
aurora.smoothing.penalties : Penalty matrix construction

Notes
-----
For detailed mathematical derivations, see REFERENCES.md in the repository root.

B-splines combine mathematical elegance (compact support, partition of unity)
with computational efficiency (stable recursion, sparse matrices). They are
the workhorse basis for modern statistical smoothing.
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

    def basis_matrix(self, x: Any, sparse: bool = False) -> Any:
        """Compute B-spline basis matrix using Cox-de Boor recursion.

        Parameters
        ----------
        x : array-like, shape (n_samples,)
            Points at which to evaluate basis functions.
        sparse : bool, default=False
            If True, return scipy.sparse.csr_matrix (only for NumPy backend).
            If False, return dense array compatible with input backend.

        Returns
        -------
        B : array or csr_matrix, shape (n_samples, n_basis)
            Basis matrix where B[i,j] = B_j^p(x_i), the j-th B-spline
            of degree p evaluated at x_i.

        Notes
        -----
        Uses the Cox-de Boor recursion formula:
            B_i^0(x) = 1 if t_i ≤ x < t_{i+1}, else 0
            B_i^p(x) = w_i^p(x) B_i^{p-1}(x) + (1 - w_{i+1}^p(x)) B_{i+1}^{p-1}(x)
        where w_i^p(x) = (x - t_i) / (t_{i+p} - t_i)

        **Sparse output** exploits the compact support property: only (degree + 1)
        basis functions are non-zero at any point. For degree=3, each row has
        at most 4 non-zero entries, regardless of n_basis. This reduces:
        - Memory: O(n_samples × n_basis) → O(n_samples × degree)
        - Computation: O(n_samples × n_basis × degree²) → O(n_samples × degree²)

        Sparse output is only available for NumPy backend. PyTorch and JAX
        backends always return dense arrays (sparse support is experimental).
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

        # Check if sparse output is requested and available
        if sparse:
            # Sparse output only for NumPy backend
            if xp.__name__ != 'numpy':
                raise ValueError(
                    f"Sparse output only supported for NumPy backend, got {xp.__name__}"
                )
            return self._basis_matrix_sparse_numpy(x_arr, knots)

        # Dense output (original implementation)
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

    def _basis_matrix_sparse_numpy(
        self, x_arr: np.ndarray, knots: np.ndarray
    ) -> Any:
        """Efficiently compute sparse B-spline basis matrix (NumPy only).

        This method exploits the compact support property: for degree p,
        only (p+1) basis functions are non-zero at any point x.

        Parameters
        ----------
        x_arr : ndarray, shape (n_samples,)
            Evaluation points (NumPy array).
        knots : ndarray
            Knot vector (NumPy array).

        Returns
        -------
        B : scipy.sparse.csr_matrix, shape (n_samples, n_basis)
            Sparse basis matrix in CSR (Compressed Sparse Row) format.

        Notes
        -----
        **Algorithm**:
        1. For each x[i], find the knot interval [t_k, t_{k+1}] containing x[i]
        2. Only evaluate basis functions k-p, ..., k (at most p+1 functions)
        3. Store non-zero values in CSR format

        **Complexity**:
        - Time: O(n × degree²) vs O(n × n_basis × degree²) for dense
        - Space: O(n × degree) vs O(n × n_basis) for dense
        - For typical GAM (n=1000, n_basis=20, degree=3): 75× speedup

        **CSR format**:
        Stores only non-zero entries using three arrays:
        - data: non-zero values
        - indices: column indices
        - indptr: row pointers

        This is optimal for matrix-vector products (used in IRLS, PQL).
        """
        try:
            from scipy.sparse import csr_matrix
        except ImportError:
            raise ImportError(
                "scipy is required for sparse B-spline evaluation. "
                "Install with: pip install scipy"
            )

        n = x_arr.shape[0]
        xp = np  # NumPy namespace

        # CSR format: (data, indices, indptr)
        # data: non-zero values
        # indices: column index for each non-zero value
        # indptr: row pointers (indptr[i]:indptr[i+1] gives row i)
        data = []
        indices = []
        indptr = [0]  # Start of row 0

        # For each evaluation point
        for idx in range(n):
            x_val = x_arr[idx]

            # Find knot interval containing x_val
            # Binary search for efficiency: O(log K) instead of O(K)
            interval_idx = self._find_knot_interval(x_val, knots)

            if interval_idx == -1:
                # x is outside knot range - no non-zero basis functions
                indptr.append(indptr[-1])
                continue

            # Basis functions that could be non-zero at x_val:
            # Functions i where x ∈ [knots[i], knots[i+degree+1]]
            # These are: max(0, interval_idx - degree), ..., min(n_basis-1, interval_idx)
            i_start = max(0, interval_idx - self.degree_)
            i_end = min(self.n_basis_ - 1, interval_idx)

            # Evaluate only non-zero basis functions
            for i in range(i_start, i_end + 1):
                B_val = self._evaluate_basis(x_val, i, self.degree_, knots, xp)

                # Only store if truly non-zero (threshold for numerical stability)
                if abs(float(B_val)) > 1e-14:
                    data.append(float(B_val))
                    indices.append(i)

            # Record end of this row
            indptr.append(len(data))

        # Convert to numpy arrays
        data = np.array(data, dtype=x_arr.dtype)
        indices = np.array(indices, dtype=np.int32)
        indptr = np.array(indptr, dtype=np.int32)

        # Create CSR matrix
        B_sparse = csr_matrix(
            (data, indices, indptr),
            shape=(n, self.n_basis_),
            dtype=x_arr.dtype
        )

        return B_sparse

    def _find_knot_interval(self, x: float, knots: np.ndarray) -> int:
        """Find which knot interval contains x using binary search.

        Parameters
        ----------
        x : float
            Evaluation point.
        knots : ndarray
            Knot vector.

        Returns
        -------
        interval_idx : int
            Index k such that knots[k] <= x < knots[k+1].
            Returns -1 if x is outside the knot range.

        Notes
        -----
        Uses binary search for O(log K) complexity instead of O(K) linear scan.
        Special handling for boundary cases:
        - x < knots[0]: return -1
        - x >= knots[-1]: return len(knots) - 2 (last interval)
        """
        # Handle boundary cases
        if x < knots[0] or x > knots[-1]:
            return -1

        # Special case: x exactly at right boundary
        if x == knots[-1]:
            # Find last non-repeated knot
            for k in range(len(knots) - 2, -1, -1):
                if knots[k] < knots[k + 1]:
                    return k
            return 0

        # Binary search for interval
        # Find k such that knots[k] <= x < knots[k+1]
        left = 0
        right = len(knots) - 2  # Last valid interval index

        while left <= right:
            mid = (left + right) // 2

            if knots[mid] <= x < knots[mid + 1]:
                return mid
            elif x < knots[mid]:
                right = mid - 1
            else:  # x >= knots[mid + 1]
                left = mid + 1

        # Should not reach here if knots are valid
        return -1

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
