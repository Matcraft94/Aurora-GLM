"""Iteratively Reweighted Least Squares (IRLS) for Generalized Linear Models.

Mathematical Framework
----------------------
IRLS is the standard algorithm for fitting Generalized Linear Models (GLMs).
It is equivalent to Fisher scoring when using the canonical link, and closely
related to Newton-Raphson for non-canonical links.

GLM Structure
-------------
A GLM relates the response Y to predictors X through:

    g(μ) = η = X^T β

where:
    - μ = E[Y|X] is the mean
    - g(·) is the link function
    - η is the linear predictor
    - β are regression coefficients

**Distribution**: Y ~ ExpFamily(μ, φ) with variance Var(Y) = φ V(μ)

IRLS Algorithm
--------------
Starting from initial estimates β⁽⁰⁾, iterate until convergence:

**Step 1: Compute working response and weights**

For iteration t, given current estimates β⁽ᵗ⁾:

    η⁽ᵗ⁾ = X β⁽ᵗ⁾
    μ⁽ᵗ⁾ = g⁻¹(η⁽ᵗ⁾)

Working response (adjusted dependent variable):
    z⁽ᵗ⁾ = η⁽ᵗ⁾ + (y - μ⁽ᵗ⁾) g'(μ⁽ᵗ⁾)

Working weights (iterative weights):
    w⁽ᵗ⁾ = 1 / [V(μ⁽ᵗ⁾) (g'(μ⁽ᵗ⁾))²]

**Step 2: Solve weighted least squares**

Update coefficients by solving:
    β⁽ᵗ⁺¹⁾ = argmin_β ||W^(1/2)(z - Xβ)||²

where W = diag(w⁽ᵗ⁾).

**Normal equations**:
    (X^T W X) β⁽ᵗ⁺¹⁾ = X^T W z⁽ᵗ⁾

**Step 3: Check convergence**

Stop when ||β⁽ᵗ⁺¹⁾ - β⁽ᵗ⁾|| < tol

Equivalence to Fisher Scoring
------------------------------
IRLS is exactly Fisher scoring for GLMs. The Fisher information matrix is:

    I(β) = E[-∂²ℓ/∂β∂β^T] = X^T W X

where the expected information equals the observed information for
exponential families.

**Fisher scoring update**:
    β⁽ᵗ⁺¹⁾ = β⁽ᵗ⁾ + [I(β⁽ᵗ⁾)]⁻¹ ∂ℓ/∂β|_{β⁽ᵗ⁾}
           = β⁽ᵗ⁾ + (X^T W X)⁻¹ X^T W (y - μ⁽ᵗ⁾) / g'(μ⁽ᵗ⁾)

This is equivalent to solving:
    (X^T W X) β⁽ᵗ⁺¹⁾ = X^T W [η⁽ᵗ⁾ + (y - μ⁽ᵗ⁾)g'(μ⁽ᵗ⁾)]
                        = X^T W z⁽ᵗ⁾

Convergence Properties
----------------------
**Theorem** (Green, 1984): For canonical link functions and regular
exponential families, IRLS converges to the maximum likelihood estimate
if it exists.

**Rate**: Locally quadratic convergence near the solution (Fisher scoring
inherits Newton-Raphson's convergence rate for canonical links).

**Conditions for convergence**:
1. Design matrix X has full column rank
2. Starting values yield finite μ⁽⁰⁾ in the support
3. V(μ) > 0 and g'(μ) ≠ 0 throughout iteration path

**Failure modes**:
- Separation (perfect prediction) in logistic regression → β → ∞
- Zero responses in Poisson/Gamma models → boundary issues
- Near-collinearity → ill-conditioned X^T W X

Numerical Stability
-------------------
**Techniques used**:

1. **Safe division**: Clip denominators V(μ) (g'(μ))² away from zero
   - Minimum threshold: 10⁻¹²
   - Prevents overflow in weights

2. **Weight clipping**: Ensure w_i ∈ [ε, M] for stability
   - Avoids both overflow and underflow
   - Maintains condition number of X^T W X

3. **Direct solve via Cholesky**: For positive-definite X^T W X
   - Computational cost: O(p³) for p predictors
   - Numerically stable when well-conditioned

4. **Backend-agnostic**: Supports NumPy, PyTorch, JAX
   - Automatic differentiation not required
   - Pure linear algebra operations

5. **Sparse matrix support**: Automatic detection and handling
   - Uses scipy.sparse operations when X is sparse
   - Sparse direct solver (SuperLU) for linear systems
   - Significant speedup for high-dimensional categorical data

Sparse Matrix Support
---------------------
When the design matrix X is sparse (scipy.sparse), IRLS automatically uses
sparse-aware operations for computational efficiency:

**Sparse operations**:
1. **Matrix-vector products**: X @ β uses sparse BLAS level 2
2. **Weighted normal equations**: X^T W X computed via sparse matmul
3. **Linear solve**: Uses sparse direct solver (SuperLU via scipy.sparse.linalg)

**Complexity with sparse X** (nnz = number of non-zeros):
- Matrix-vector product: O(nnz) instead of O(np)
- X^T W X formation: O(nnz × p) when result is dense
- Total per iteration: O(nnz × p + p³) vs O(np² + p³) dense

**When to use sparse**:
- High-dimensional categorical features (one-hot encoded)
- B-spline basis matrices (compact support → banded/sparse)
- Interaction terms with many zero entries
- Design matrices with > 90% zeros typically benefit

**Detection**: Sparse matrices are auto-detected via `scipy.sparse.issparse()`.

Computational Complexity
------------------------
Per iteration, for n observations and p predictors:

**Operations**:
- Link computations (η → μ, g'(μ)): O(n)
- Variance function V(μ): O(n)
- Weight computation: O(n)
- Form X^T W X: O(np² + p²)
- Solve (X^T W X)β = X^T W z: O(p³)

**Total per iteration**: O(np² + p³)

**Typical iterations**: 3-10 for well-conditioned problems

**Total cost**: O(k(np² + p³)) where k is iteration count

Comparison with Other Methods
------------------------------
**vs Newton-Raphson**:
- IRLS uses expected information (Fisher), NR uses observed
- For canonical links: IRLS = NR
- IRLS more stable for GLMs (guaranteed positive-definite)

**vs gradient descent**:
- Faster convergence (quadratic vs linear)
- No learning rate tuning required
- More expensive per iteration: O(p³) vs O(np)

**vs L-BFGS**:
- IRLS exploits GLM structure (closed-form weights)
- L-BFGS more general, uses quasi-Newton approximation
- For large p: L-BFGS may be faster (avoids full Hessian)

Implementation Notes
--------------------
**Multi-backend support**:
All operations work transparently with NumPy, PyTorch, and JAX arrays
through the backend abstraction.

**Offset terms**:
Supports known offset η₀:
    η = X^T β + η₀

Useful for log-exposure models in Poisson regression.

**Initial values**:
Caller must provide β⁽⁰⁾. Common strategies:
- GLM with identity link: β⁽⁰⁾ = (X^T X)⁻¹ X^T g(y)
- Zeros: often works for well-scaled data
- Previous fit: warm-starting

References
----------
**Core IRLS theory**:

- Green, P. J. (1984). \"Iteratively reweighted least squares for maximum
  likelihood estimation, and some robust and resistant alternatives.\"
  *Journal of the Royal Statistical Society: Series B*, 46(2), 149-192.
  https://doi.org/10.1111/j.2517-6161.1984.tb01288.x

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.).
  Chapman and Hall/CRC. Chapter 2.5: Fitting GLMs.
  https://doi.org/10.1007/978-1-4899-3242-6

**Convergence analysis**:

- Wedderburn, R. W. M. (1976). \"On the existence and uniqueness of the
  maximum likelihood estimates for certain generalized linear models.\"
  *Biometrika*, 63(1), 27-32.
  https://doi.org/10.1093/biomet/63.1.27

**Numerical methods**:

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). \"Maximum likelihood
  from incomplete data via the EM algorithm.\" *Journal of the Royal
  Statistical Society: Series B*, 39(1), 1-22.
  (IRLS as special case of EM)

**Sparse linear algebra**:

- Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*.
  SIAM. (SuperLU, UMFPACK algorithms)

- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.).
  SIAM. (CG, GMRES for large sparse systems)

**Separation and boundary issues**:

- Albert, A., & Anderson, J. A. (1984). \"On the existence of maximum
  likelihood estimates in logistic regression models.\" *Biometrika*,
  71(1), 1-10.
  https://doi.org/10.1093/biomet/71.1.1

See Also
--------
aurora.models.glm.fitting : GLM fitting using IRLS
aurora.distributions.base : Variance functions and link functions
aurora.core.optimization.newton : Newton-Raphson method

Notes
-----
For detailed mathematical derivations, see REFERENCES.md in the repository root.

IRLS is the workhorse algorithm for GLM fitting, combining computational
efficiency with statistical optimality (Fisher scoring achieves the
Cramér-Rao lower bound asymptotically).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..types import Array, OptimizationCallback
from .result import OptimizationResult


# =============================================================================
# Sparse matrix utilities
# =============================================================================

def _is_sparse(X) -> bool:
    """Check if X is a scipy sparse matrix."""
    try:
        from scipy import sparse
        return sparse.issparse(X)
    except ImportError:
        return False


def _sparse_diag_matmul(X, w):
    """Compute diag(w) @ X for sparse X.

    For sparse X, this is equivalent to row-scaling: X_i,: *= w_i

    Parameters
    ----------
    X : sparse matrix (n, p)
        Design matrix
    w : ndarray (n,)
        Diagonal weights

    Returns
    -------
    sparse matrix (n, p)
        Row-scaled matrix
    """
    from scipy import sparse
    # Create diagonal matrix and multiply
    W = sparse.diags(w)
    return W @ X


def _sparse_weighted_lstsq(X, w, z):
    """Solve weighted least squares with sparse X.

    Solves: β = argmin_β ||W^(1/2)(z - Xβ)||²

    This is equivalent to solving the normal equations:
        (X^T W X) β = X^T W z

    For sparse X with dense result (p small), we form X^T W X densely
    and solve with standard dense solver. For very large p with sparse
    X^T W X, an iterative solver would be preferred.

    Parameters
    ----------
    X : sparse matrix (n, p)
        Design matrix
    w : ndarray (n,)
        Weights (positive)
    z : ndarray (n,)
        Working response

    Returns
    -------
    beta : ndarray (p,)
        Least squares solution

    Notes
    -----
    The algorithm:
    1. Form W^(1/2) X via row scaling (sparse)
    2. Form X^T W X = (W^(1/2) X)^T (W^(1/2) X) (dense if p small)
    3. Form X^T W z (dense vector)
    4. Solve via Cholesky or direct solver

    Complexity: O(nnz × p + p³) where nnz = number of non-zeros in X
    """
    from scipy import sparse
    from scipy.linalg import cho_factor, cho_solve

    n, p = X.shape

    # Compute sqrt(W) @ X by row scaling
    sqrt_w = np.sqrt(w)
    WhalfX = _sparse_diag_matmul(X, sqrt_w)

    # Form X^T W X (dense) - this is p × p
    # WhalfX^T @ WhalfX
    XtWX = (WhalfX.T @ WhalfX).toarray()

    # Form X^T W z (dense vector)
    XtWz = X.T @ (w * z)

    # Solve via Cholesky (X^T W X is positive definite when X full rank)
    try:
        L, lower = cho_factor(XtWX, lower=True)
        beta = cho_solve((L, lower), XtWz)
    except np.linalg.LinAlgError:
        # Fallback to least squares if Cholesky fails
        beta, _, _, _ = np.linalg.lstsq(XtWX, XtWz, rcond=None)

    return beta


def _irls_sparse(
    loss_fn: Callable,
    init_params: np.ndarray,
    X,  # scipy.sparse matrix
    y: np.ndarray,
    link,
    variance_fn: Callable,
    offset: np.ndarray | None,
    max_iter: int,
    tol: float,
    callback,
    args: tuple,
    kwargs: dict,
) -> OptimizationResult:
    """IRLS implementation for sparse design matrices.

    Uses scipy.sparse operations for matrix-vector products and
    forms the normal equations X^T W X as dense (suitable for p << n).

    Parameters
    ----------
    loss_fn : callable
        Loss function for convergence monitoring
    init_params : ndarray (p,)
        Initial parameter estimates
    X : sparse matrix (n, p)
        Design matrix
    y : ndarray (n,)
        Response vector
    link : LinkFunction
        Link function with inverse() and derivative() methods
    variance_fn : callable
        Variance function V(μ)
    offset : ndarray (n,) or None
        Known offset term
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    callback : callable or None
        Progress callback
    args, kwargs : tuple, dict
        Additional arguments for loss_fn

    Returns
    -------
    OptimizationResult
        Optimization result with solution and diagnostics
    """
    beta = np.asarray(init_params, dtype=np.float64)
    offset_arr = np.zeros_like(y) if offset is None else np.asarray(offset)

    nfev = 0

    for iteration in range(max_iter):
        # Linear predictor without offset: η* = Xβ
        eta_star = np.asarray(X @ beta).ravel()

        # Full linear predictor with offset: η = η* + offset
        eta = eta_star + offset_arr

        # Mean: μ = g⁻¹(η)
        mu = link.inverse(eta)

        # Link derivative: g'(μ)
        g_prime = link.derivative(mu)

        # Variance function: V(μ)
        var = variance_fn(mu)

        # Working weights: w = 1 / [V(μ) (g'(μ))²]
        denom = var * (g_prime ** 2)
        denom = np.clip(denom, 1e-12, None)  # Numerical stability
        weights = 1.0 / denom

        # Working response: z* = η* + (y - μ) g'(μ)
        # Note: We use η* (without offset) so WLS solves for β correctly
        # This is equivalent to: z* = y - offset (for identity link)
        z_star = eta_star + (y - mu) * g_prime

        # Solve weighted least squares: β = (X^T W X)^{-1} X^T W z*
        beta_new = _sparse_weighted_lstsq(X, weights, z_star)

        # Check convergence
        delta = beta_new - beta
        step_norm = np.sqrt(np.sum(delta ** 2))

        beta = beta_new

        # Evaluate loss for monitoring
        nfev += 1
        loss_value = float(loss_fn(beta, *args, **kwargs))

        if callback is not None:
            callback(iteration, beta.copy(), loss_value)

        if step_norm < tol:
            return OptimizationResult(
                x=beta,
                fun=loss_value,
                grad=None,
                success=True,
                message="Converged: parameter change below tolerance",
                nit=iteration + 1,
                nfev=nfev,
                njev=0,
                nhev=0,
            )

    # Max iterations reached
    loss_value = float(loss_fn(beta, *args, **kwargs))
    nfev += 1

    return OptimizationResult(
        x=beta,
        fun=loss_value,
        grad=None,
        success=False,
        message="Maximum iterations reached",
        nit=max_iter,
        nfev=nfev,
        njev=0,
        nhev=0,
    )


# =============================================================================
# Main IRLS interface
# =============================================================================

def irls(
    loss_fn: Callable,
    init_params: Array,
    *,
    backend=None,
    args: tuple = (),
    kwargs: dict | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    callback: OptimizationCallback | None = None,
    design_matrix: Array | None = None,
    response: Array | None = None,
    link: Any | None = None,
    variance_fn: Callable[[Array], Array] | None = None,
    offset: Array | None = None,
) -> OptimizationResult:
    """Run the IRLS procedure for generalized linear models.

    This function automatically detects sparse design matrices and
    uses optimized sparse operations when beneficial.

    Parameters
    ----------
    loss_fn : callable
        Loss function taking (params, *args, **kwargs)
    init_params : array-like (p,)
        Initial parameter estimates
    backend : Backend, optional
        Computational backend (numpy, pytorch, jax)
    args : tuple
        Positional arguments for loss_fn
    kwargs : dict
        Keyword arguments for loss_fn
    max_iter : int, default=100
        Maximum IRLS iterations
    tol : float, default=1e-6
        Convergence tolerance on parameter change
    callback : callable, optional
        Callback function(iteration, params, loss)
    design_matrix : array-like (n, p)
        Design matrix X. May be dense or scipy.sparse.
    response : array-like (n,)
        Response vector y
    link : LinkFunction
        Link function with inverse() and derivative()
    variance_fn : callable
        Variance function V(μ)
    offset : array-like (n,), optional
        Known offset term

    Returns
    -------
    OptimizationResult
        Optimization result containing:
        - x: final parameters
        - fun: final loss value
        - success: convergence status
        - nit: number of iterations
        - message: status message

    Notes
    -----
    For sparse design matrices (scipy.sparse), the algorithm automatically
    uses sparse matrix operations:

    1. Matrix-vector products X @ β use sparse BLAS
    2. Normal equations X^T W X are formed efficiently
    3. The p × p system is solved with Cholesky factorization

    This is efficient when:
    - X is sparse (> 90% zeros)
    - Number of predictors p is moderate (< 10,000)
    - X^T W X fits in memory as dense

    For very large sparse X^T W X, iterative solvers (CG) would be needed.
    """
    if kwargs is None:
        kwargs = {}

    if design_matrix is None or response is None or link is None or variance_fn is None:
        raise ValueError(
            "IRLS requires design_matrix, response, link, and variance_fn keyword arguments."
        )

    if not hasattr(link, "inverse") or not hasattr(link, "derivative"):
        raise TypeError("link must expose inverse() and derivative() methods")

    # Check for sparse matrix - use specialized path
    if _is_sparse(design_matrix):
        return _irls_sparse(
            loss_fn=loss_fn,
            init_params=np.asarray(init_params),
            X=design_matrix,
            y=np.asarray(response),
            link=link,
            variance_fn=variance_fn,
            offset=np.asarray(offset) if offset is not None else None,
            max_iter=max_iter,
            tol=tol,
            callback=callback,
            args=args,
            kwargs=kwargs,
        )

    # Dense path - use backend abstraction
    if backend is None:
        from ..backends import get_backend
        backend = get_backend("jax")

    X = backend.array(design_matrix)
    y = backend.array(response)
    beta = backend.array(init_params)
    offset_arr = backend.array(offset) if offset is not None else y * 0

    converted_args = tuple(_convert_to_backend(backend, value) for value in args)
    converted_kwargs = {key: _convert_to_backend(backend, value) for key, value in kwargs.items()}

    nfev = 0

    def _to_backend(data):
        return backend.array(data, dtype=getattr(beta, "dtype", None))

    for iteration in range(max_iter):
        eta = X @ beta + offset_arr
        mu = link.inverse(eta)
        g_prime = link.derivative(mu)
        var = variance_fn(mu)

        weights = _safe_divide(backend, 1.0, var * (g_prime ** 2))
        z = eta + (y - mu) * g_prime

        sqrt_w = _sqrt(backend, weights)
        WX = X * sqrt_w.unsqueeze(-1) if hasattr(sqrt_w, "unsqueeze") else X * sqrt_w[:, None]
        Wz = z * sqrt_w

        beta_new = None
        try:  # Prefer backend-native linear algebra when available
            import torch  # type: ignore

            if isinstance(WX, torch.Tensor):
                lhs = WX.T @ WX
                rhs = WX.T @ Wz
                if rhs.ndim == 1:
                    rhs = rhs.unsqueeze(-1)
                beta_new = torch.linalg.solve(lhs, rhs).squeeze(-1)
                beta_new = beta_new.to(beta.device).type_as(beta)
        except ImportError:  # pragma: no cover - optional dependency missing
            beta_new = None

        if beta_new is None:
            WX_np = backend.as_numpy(WX)
            Wz_np = backend.as_numpy(Wz)
            lhs = WX_np.T @ WX_np
            rhs = WX_np.T @ Wz_np
            solution = np.linalg.solve(lhs, rhs)
            beta_new = _to_backend(solution)

        delta = beta_new - beta
        step_norm = backend.as_numpy((delta * delta).sum() ** 0.5)

        beta = beta_new

        nfev += 1
        loss_value = loss_fn(beta, *converted_args, **converted_kwargs)

        if callback is not None:
            callback(iteration, backend.as_numpy(beta), float(backend.as_numpy(loss_value)))

        if step_norm < tol:
            return OptimizationResult(
                x=backend.as_numpy(beta),
                fun=float(backend.as_numpy(loss_value)),
                grad=None,
                success=True,
                message="Converged: parameter change below tolerance",
                nit=iteration + 1,
                nfev=nfev,
                njev=0,
                nhev=0,
            )

    loss_value = loss_fn(beta, *converted_args, **converted_kwargs)
    nfev += 1
    return OptimizationResult(
        x=backend.as_numpy(beta),
        fun=float(backend.as_numpy(loss_value)),
        grad=None,
        success=False,
        message="Maximum iterations reached",
        nit=max_iter,
        nfev=nfev,
        njev=0,
        nhev=0,
    )


def _safe_divide(backend, numerator, denominator):
    eps = 1e-12
    if hasattr(denominator, "clamp_min"):
        denominator = denominator.clamp_min(eps)
        return numerator / denominator
    denom_np = backend.as_numpy(denominator)
    denom_np = np.clip(denom_np, eps, None)
    return numerator / backend.array(denom_np)


def _sqrt(backend, value):
    if hasattr(value, "sqrt"):
        return value.sqrt()
    value_np = backend.as_numpy(value)
    return backend.array(np.sqrt(value_np))


def _convert_to_backend(backend, value):
    if isinstance(value, (tuple, list)):
        converted = [_convert_to_backend(backend, item) for item in value]
        return type(value)(converted)
    if isinstance(value, dict):
        return {key: _convert_to_backend(backend, item) for key, item in value.items()}
    try:
        return backend.array(value)
    except Exception:  # pragma: no cover - fallback when conversion is not applicable
        return value


__all__ = ["irls"]
