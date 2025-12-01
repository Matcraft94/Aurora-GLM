"""Limited-Memory BFGS (L-BFGS) Quasi-Newton Optimization.

Mathematical Framework
----------------------
L-BFGS is a limited-memory variant of the Broyden-Fletcher-Goldfarb-Shanno
(BFGS) algorithm, a quasi-Newton method for unconstrained optimization. It
approximates Newton's method without explicitly computing or storing the
Hessian matrix, making it suitable for large-scale problems.

Problem Formulation
-------------------
Find x* that minimizes f(x):

    x* = argmin_{x ∈ ℝᵖ} f(x)

where f: ℝᵖ → ℝ is continuously differentiable.

**Optimality condition**: ∇f(x*) = 0

Quasi-Newton Methods
--------------------
Quasi-Newton methods approximate the Newton step:

    p = -H⁻¹ ∇f(x)

by building an approximation B ≈ H (or H ≈ H⁻¹) iteratively using gradient
information, avoiding expensive Hessian computations.

**Key idea**: Use secant equation to update approximation:

    B_{k+1} s_k = y_k

where:
    s_k = x_{k+1} - x_k     (step)
    y_k = ∇f(x_{k+1}) - ∇f(x_k)     (gradient change)

BFGS Update Formula
-------------------
Starting from B₀ (often B₀ = I), BFGS updates the Hessian approximation:

    B_{k+1} = B_k - (B_k s_k s_k^T B_k)/(s_k^T B_k s_k) + (y_k y_k^T)/(y_k^T s_k)

Or equivalently, updating the inverse Hessian approximation H_k ≈ (∇²f)⁻¹:

    H_{k+1} = (I - ρ_k s_k y_k^T) H_k (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T

where ρ_k = 1/(y_k^T s_k).

**Properties**:
- Maintains positive definiteness if y_k^T s_k > 0 (curvature condition)
- Symmetric updates
- Superlinear convergence rate (faster than linear, slower than quadratic)

Limited-Memory Variant
-----------------------
**Challenge**: Storing H_k requires O(p²) memory, prohibitive for large p.

**L-BFGS solution**: Don't store H_k explicitly. Instead, store m recent
vector pairs {s_i, y_i} and compute H_k ∇f(x) implicitly via two-loop recursion.

**Memory**: O(mp) instead of O(p²), where m ≈ 3-20 (typically m = 10)

**Trade-off**:
- BFGS: O(p²) memory, better approximation
- L-BFGS: O(mp) memory, coarser approximation
- Convergence rate similar for moderate m

Two-Loop Recursion Algorithm
-----------------------------
Compute search direction d = -H_k g without forming H_k explicitly.

**Input**: Current gradient g = ∇f(x_k), stored pairs {(s_i, y_i)}_{i=k-m}^{k-1}

**Algorithm**:

    q ← g
    for i = k-1, k-2, ..., k-m:
        α_i ← ρ_i (s_i^T q)
        q ← q - α_i y_i

    r ← H_0^k q     # Initial approximation, often r ← γ_k q

    for i = k-m, ..., k-2, k-1:
        β ← ρ_i (y_i^T r)
        r ← r + s_i (α_i - β)

    return -r

**Output**: Search direction d = -H_k g

**Initial approximation**:
    γ_k = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})

scales the identity to match recent curvature.

**Computational cost**: O(mp) per iteration (m vector operations, each O(p))

Curvature Condition
-------------------
The BFGS update requires the **curvature condition**:

    y_k^T s_k > 0

**Interpretation**: Change in gradient must have positive projection onto step.

**When violated**:
- Non-convex functions away from minimum
- Poor line search (insufficient descent)
- Numerical errors in gradient

**Implementation response**:
- Skip update: set H_{k+1} = H_0 (restart)
- Ensures positive definiteness maintained

**Guarantee**: Strong Wolfe line search ensures y_k^T s_k > 0

Line Search
-----------
L-BFGS requires a line search to determine step length α:

    x_{k+1} = x_k + α d_k

**Armijo condition** (sufficient decrease):

    f(x_k + α d_k) ≤ f(x_k) + c₁ α ∇f(x_k)^T d_k

where c₁ ∈ (0, 1), typically c₁ = 10⁻⁴.

**Strong Wolfe conditions** (adds curvature):

    |∇f(x_k + α d_k)^T d_k| ≤ c₂ |∇f(x_k)^T d_k|

where c₂ ∈ (c₁, 1), typically c₂ = 0.9.

**Backtracking**: Start with α = 1, multiply by ρ ∈ (0,1) until Armijo holds.

**Cost**: Typically 1-5 function/gradient evaluations per iteration

Convergence Theory
------------------
**Theorem** (Liu & Nocedal, 1989): For strongly convex functions with
Lipschitz continuous gradient, L-BFGS with strong Wolfe line search
converges globally.

**Rate**: Superlinear convergence (faster than linear, slower than quadratic)

    ||x_{k+1} - x*|| ≤ C ||x_k - x*||^α

where 1 < α < 2 depends on m and function properties.

**Comparison**:
- Gradient descent: Linear rate (error × constant)
- L-BFGS: Superlinear (error^1.5 approximately)
- Newton: Quadratic (error²)

**Global vs local**:
- With line search: Global convergence (from any x₀)
- Near minimum: Superlinear convergence rate

Numerical Stability
-------------------
**Challenges**:

1. **Curvature condition failure**: y_k^T s_k ≤ 0
   - Solution: Reset to steepest descent (H_k = I)
   - Loses accumulated curvature information

2. **Ill-conditioning**: Large/small eigenvalues in true Hessian
   - L-BFGS inherits conditioning from ∇²f
   - Preconditioning helps: solve for P⁻¹ H ∇f

3. **Gradient noise**: Stochastic/approximate gradients
   - L-BFGS sensitive to noise (unlike stochastic gradient descent)
   - Requires batch gradients for stability

4. **Memory management**: Deciding m
   - Small m (3-5): Less memory, faster per iteration, slower convergence
   - Large m (20+): More memory, better approximation, faster convergence
   - Default m = 10 balances trade-offs

**Numerical tricks**:
- Store s_k, y_k normalized to prevent over/underflow
- Reject updates with |ρ_k| > threshold (near-singular)
- Damped updates when curvature condition weak

Computational Complexity
------------------------
Per iteration, for p parameters and memory m:

**Gradient computation**:
- Automatic differentiation: O(p)
- Finite differences: O(p²) [not used here]

**Two-loop recursion**: O(mp)
- m vector operations, each O(p)

**Line search**: k_ls evaluations (typically 1-5)
- Each evaluation: O(p) for function + gradient

**Total per iteration**: O(mp + k_ls × p) = O((m + k_ls)p)

**Storage**: O(mp) for {s_i, y_i} pairs

**vs Newton**:
- Newton: O(p³) per iteration (Hessian solve)
- L-BFGS: O(mp) per iteration
- Crossover: L-BFGS preferred for p > 1000

**vs IRLS (GLM)**:
- IRLS: O(np² + p³), exploits GLM structure
- L-BFGS: O(mp), general-purpose
- For GLMs with small p: IRLS faster
- For GLMs with large p: L-BFGS competitive

Comparison with Other Methods
------------------------------
**vs Gradient Descent**:
- L-BFGS: Superlinear convergence, O(mp) per iteration
- GD: Linear convergence, O(p) per iteration
- L-BFGS typically 5-10× fewer iterations

**vs Conjugate Gradient**:
- Both: O(p) memory, linear-to-superlinear convergence
- L-BFGS: More robust, fewer tuning parameters
- CG: Better for very large p (billions)

**vs Newton**:
- Newton: Quadratic convergence, O(p³) per iteration, O(p²) memory
- L-BFGS: Superlinear convergence, O(mp) per iteration, O(mp) memory
- Crossover: p ≈ 1000

**vs Trust Region**:
- Both: Globalization strategies
- L-BFGS + line search: Simpler implementation
- Trust region: Better worst-case guarantees

Applications in Aurora-GLM
---------------------------
L-BFGS is used for:

1. **Large GLMs**: When p > 1000 and IRLS becomes expensive
2. **Non-standard likelihoods**: Custom log-likelihood optimization
3. **Penalized GLMs**: Ridge/LASSO where IRLS doesn't apply directly
4. **GAMM variance components**: When dimension of variance parameters is large

**Not used for**:
- Standard small/medium GLMs → use IRLS (more stable, exploits structure)
- Stochastic optimization → use SGD variants
- Non-smooth penalties (L1) → use proximal methods

Implementation Notes
--------------------
**Multi-backend support**:
Transparently works with NumPy, PyTorch, and JAX arrays through
automatic differentiation for gradient computation.

**Line search**:
- Implements backtracking Armijo condition
- Can be extended to strong Wolfe (not implemented)
- Configurable parameters: c₁, ρ, max iterations

**Memory parameter m**:
- Default: m = 10 (good balance)
- Increase m for better approximation (if memory permits)
- Decrease m for memory-constrained problems

**Restart strategy**:
- Automatic restart when curvature condition fails
- Periodic restart can improve robustness (not automatic)

References
----------
**Core L-BFGS theory**:

- Liu, D. C., & Nocedal, J. (1989). \"On the limited memory BFGS method for
  large scale optimization.\" *Mathematical Programming*, 45(1-3), 503-528.
  https://doi.org/10.1007/BF01589116

- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.).
  Springer. Chapter 7: Large-Scale Unconstrained Optimization.
  https://doi.org/10.1007/978-0-387-40065-5

**BFGS algorithm**:

- Broyden, C. G. (1970). \"The convergence of a class of double-rank
  minimization algorithms.\" *IMA Journal of Applied Mathematics*, 6(1), 76-90.

- Fletcher, R. (1970). \"A new approach to variable metric algorithms.\"
  *The Computer Journal*, 13(3), 317-322.

- Goldfarb, D. (1970). \"A family of variable-metric methods derived by
  variational means.\" *Mathematics of Computation*, 24(109), 23-26.

- Shanno, D. F. (1970). \"Conditioning of quasi-Newton methods for function
  minimization.\" *Mathematics of Computation*, 24(111), 647-656.

**Convergence analysis**:

- Dennis, J. E., & Moré, J. J. (1977). \"Quasi-Newton methods, motivation
  and theory.\" *SIAM Review*, 19(1), 46-89.
  https://doi.org/10.1137/1019005

**Line search theory**:

- Wolfe, P. (1969). \"Convergence conditions for ascent methods.\"
  *SIAM Review*, 11(2), 226-235.

- Wolfe, P. (1971). \"Convergence conditions for ascent methods II: Some
  corrections.\" *SIAM Review*, 13(2), 185-188.

**Implementation reference**:

- Zhu, C., Byrd, R. H., Lu, P., & Nocedal, J. (1997). \"Algorithm 778:
  L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization.\"
  *ACM Transactions on Mathematical Software*, 23(4), 550-560.
  (Classic FORTRAN implementation)

See Also
--------
aurora.core.optimization.newton : Newton-Raphson method
aurora.core.optimization.irls : IRLS for GLMs
aurora.models.glm.fitting : GLM fitting algorithms

Notes
-----
For detailed mathematical derivations, see REFERENCES.md in the repository root.

L-BFGS is the method of choice for large-scale smooth optimization problems
where computing the Hessian is infeasible. Its combination of low memory
requirements, superlinear convergence, and robustness makes it a workhorse
algorithm in machine learning and statistics.
"""
from __future__ import annotations

from typing import Any, Callable

from ..types import Array, OptimizationCallback
from .result import OptimizationResult


def lbfgs(
    loss_fn: Callable,
    init_params: Array,
    *,
    backend=None,
    args: tuple = (),
    kwargs: dict | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    m: int = 10,
    line_search: str = "strong-wolfe",
    callback: OptimizationCallback | None = None,
) -> OptimizationResult:
    """L-BFGS (Limited-memory BFGS) optimization."""

    kwargs = kwargs or {}

    if backend is None:
        from ..backends import get_backend

        backend = get_backend("jax")

    converted_args = tuple(_convert_to_backend(backend, value) for value in args)
    converted_kwargs = {key: _convert_to_backend(backend, value) for key, value in kwargs.items()}

    grad_fn = backend.grad(loss_fn)
    x = backend.array(init_params)

    s_history: list[Any] = []
    y_history: list[Any] = []

    g = grad_fn(x, *converted_args, **converted_kwargs)

    nfev = 1
    njev = 1

    for iteration in range(max_iter):
        d = _lbfgs_direction(g, s_history, y_history, backend)

        alpha, f_new, g_new, ls_fev = _line_search(
            loss_fn,
            grad_fn,
            x,
            d,
            g,
            backend,
            args=converted_args,
            kwargs=converted_kwargs,
            method=line_search,
        )

        if alpha <= 0:
            failure_fun = loss_fn(x, *converted_args, **converted_kwargs)
            return OptimizationResult(
                x=backend.as_numpy(x),
                fun=float(failure_fun),
                grad=backend.as_numpy(g),
                success=False,
                message="Line search failed to find a descent direction",
                nit=iteration,
                nfev=nfev + ls_fev,
                njev=njev,
            )

        nfev += ls_fev
        njev += 1

        x_new = x + alpha * d
        s = x_new - x

        y = g_new - g

        if hasattr(backend, "as_numpy"):
            sy = backend.as_numpy((s * y).sum())
        else:
            sy = float((s * y).sum())

        if sy <= 0:
            s_history.clear()
            y_history.clear()
        else:
            s_history.append(s)
            y_history.append(y)
            if len(s_history) > m:
                s_history.pop(0)
                y_history.pop(0)

        x = x_new
        g = g_new

        if callback is not None:
            f_val = loss_fn(x, *converted_args, **converted_kwargs)
            callback(iteration, backend.as_numpy(x), float(backend.as_numpy(f_val)))
            nfev += 1

        grad_norm = backend.as_numpy((g * g).sum() ** 0.5)
        if grad_norm < tol:
            f_final = loss_fn(x, *converted_args, **converted_kwargs)
            nfev += 1
            return OptimizationResult(
                x=backend.as_numpy(x),
                fun=float(f_final),
                grad=backend.as_numpy(g),
                success=True,
                message="Converged: gradient norm below tolerance",
                nit=iteration + 1,
                nfev=nfev,
                njev=njev,
            )

    f_final = loss_fn(x, *converted_args, **converted_kwargs)
    nfev += 1
    return OptimizationResult(
        x=backend.as_numpy(x),
        fun=float(f_final),
        grad=backend.as_numpy(g),
        success=False,
        message="Maximum iterations reached",
        nit=max_iter,
        nfev=nfev,
        njev=njev,
    )


def _lbfgs_direction(g, s_history, y_history, backend):
    """Compute the L-BFGS search direction using the two-loop recursion."""
    q = g
    m = len(s_history)

    if m == 0:
        return -g

    alphas: list[Any] = []
    rhos: list[float] = []

    for i in range(m - 1, -1, -1):
        s_i = s_history[i]
        y_i = y_history[i]
        rho_i = 1.0 / (y_i * s_i).sum()
        alpha_i = rho_i * (s_i * q).sum()
        alphas.append(alpha_i)
        rhos.append(rho_i)
        q = q - alpha_i * y_i

    s_m = s_history[-1]
    y_m = y_history[-1]
    gamma = (s_m * y_m).sum() / (y_m * y_m).sum()
    r = gamma * q

    alphas.reverse()
    rhos.reverse()
    for i, alpha_i in enumerate(alphas):
        s_i = s_history[i]
        y_i = y_history[i]
        rho_i = rhos[i]
        beta_i = rho_i * (y_i * r).sum()
        r = r + s_i * (alpha_i - beta_i)

    return -r


def _line_search(
    loss_fn,
    grad_fn,
    x,
    d,
    g,
    backend,
    *,
    args=(),
    kwargs=None,
    method="strong-wolfe",
):
    """Line search satisfying Wolfe conditions.

    Strong Wolfe Conditions
    -----------------------
    For step size α to be acceptable, it must satisfy:

    1. **Armijo condition** (sufficient decrease):
       f(x + αd) ≤ f(x) + c₁ α ∇f(x)ᵀd

    2. **Curvature condition** (strong Wolfe):
       |∇f(x + αd)ᵀd| ≤ c₂ |∇f(x)ᵀd|

    Parameters c₁ and c₂ satisfy 0 < c₁ < c₂ < 1.
    Typical values: c₁ = 1e-4, c₂ = 0.9

    Algorithm
    ---------
    This implements the line search algorithm from Nocedal & Wright (2006),
    Algorithm 3.5 (Line Search Algorithm) with Algorithm 3.6 (Zoom).

    The algorithm proceeds in two phases:
    1. **Bracketing**: Find an interval [α_lo, α_hi] containing a step
       satisfying the strong Wolfe conditions.
    2. **Zoom**: Bisection-like refinement to find acceptable α in the bracket.

    Convergence
    -----------
    **Theorem** (Wolfe, 1969): If f is bounded below along the ray x + αd
    and f is continuously differentiable, then there exist step lengths
    satisfying the strong Wolfe conditions.

    References
    ----------
    [1] Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.).
        Springer. Section 3.1: Step Length Selection.
    [2] Wolfe, P. (1969). "Convergence conditions for ascent methods."
        SIAM Review, 11(2), 226-235.
    [3] Moré, J. J., & Thuente, D. J. (1994). "Line search algorithms with
        guaranteed sufficient decrease." ACM TOMS, 20(3), 286-307.
    """
    if kwargs is None:
        kwargs = {}

    if method == "backtracking":
        return _backtracking_line_search(
            loss_fn, grad_fn, x, d, g, backend, args=args, kwargs=kwargs
        )
    else:  # strong-wolfe (default)
        return _strong_wolfe_line_search(
            loss_fn, grad_fn, x, d, g, backend, args=args, kwargs=kwargs
        )


def _strong_wolfe_line_search(
    loss_fn,
    grad_fn,
    x,
    d,
    g,
    backend,
    *,
    args=(),
    kwargs=None,
    c1=1e-4,
    c2=0.9,
    alpha_max=50.0,
    max_iter=25,
):
    """Strong Wolfe line search (Algorithm 3.5 from Nocedal & Wright).

    Mathematical Conditions
    -----------------------
    Find α satisfying:
        (W1) f(x + αd) ≤ f(x) + c₁ α φ'(0)        [Armijo]
        (W2) |φ'(α)| ≤ c₂ |φ'(0)|                  [Strong Wolfe curvature]

    where φ(α) = f(x + αd) and φ'(α) = ∇f(x + αd)ᵀd.

    Parameters
    ----------
    c1 : float
        Armijo constant, typically 1e-4
    c2 : float
        Wolfe curvature constant, typically 0.9 for quasi-Newton
        (use 0.1 for nonlinear CG)
    alpha_max : float
        Maximum step size to consider
    max_iter : int
        Maximum iterations in bracketing phase
    """
    if kwargs is None:
        kwargs = {}

    # Initial values
    f_0 = loss_fn(x, *args, **kwargs)
    phi_0 = f_0  # φ(0) = f(x)

    # Directional derivative at α=0: φ'(0) = ∇f(x)ᵀd
    dphi_0 = _to_scalar((g * d).sum(), backend)
    fev = 1

    # If not a descent direction, return failure
    if dphi_0 >= 0:
        g_new = grad_fn(x, *args, **kwargs)
        return 0.0, f_0, g_new, fev

    alpha_prev = 0.0
    phi_prev = phi_0
    dphi_prev = dphi_0

    alpha = 1.0  # Initial step size (Newton step)

    for i in range(max_iter):
        x_new = x + alpha * d
        phi = loss_fn(x_new, *args, **kwargs)
        fev += 1

        # Check Armijo condition (W1)
        armijo_threshold = phi_0 + c1 * alpha * dphi_0

        if phi > armijo_threshold or (i > 0 and phi >= phi_prev):
            # Need to zoom in [alpha_prev, alpha]
            result = _zoom(
                loss_fn, grad_fn, x, d, backend,
                alpha_prev, alpha,
                phi_prev, phi,
                dphi_prev,
                phi_0, dphi_0, c1, c2,
                args=args, kwargs=kwargs
            )
            return result[0], result[1], result[2], fev + result[3]

        # Compute gradient at new point
        g_new = grad_fn(x_new, *args, **kwargs)
        dphi = _to_scalar((g_new * d).sum(), backend)

        # Check strong Wolfe curvature condition (W2)
        if abs(dphi) <= c2 * abs(dphi_0):
            # Found acceptable step
            return alpha, phi, g_new, fev

        # If slope is non-negative, zoom in [alpha, alpha_prev]
        if dphi >= 0:
            result = _zoom(
                loss_fn, grad_fn, x, d, backend,
                alpha, alpha_prev,
                phi, phi_prev,
                dphi,
                phi_0, dphi_0, c1, c2,
                args=args, kwargs=kwargs
            )
            return result[0], result[1], result[2], fev + result[3]

        # Update for next iteration
        alpha_prev = alpha
        phi_prev = phi
        dphi_prev = dphi

        # Expand step (use golden ratio or simple doubling)
        alpha = min(2.0 * alpha, alpha_max)

    # Max iterations reached, return current best
    x_new = x + alpha * d
    f_new = loss_fn(x_new, *args, **kwargs)
    g_new = grad_fn(x_new, *args, **kwargs)
    return alpha, f_new, g_new, fev + 1


def _zoom(
    loss_fn, grad_fn, x, d, backend,
    alpha_lo, alpha_hi,
    phi_lo, phi_hi,
    dphi_lo,
    phi_0, dphi_0, c1, c2,
    *,
    args=(),
    kwargs=None,
    max_iter=10,
):
    """Zoom phase of line search (Algorithm 3.6 from Nocedal & Wright).

    Refines a bracket [α_lo, α_hi] to find a step satisfying strong Wolfe.

    The interval [α_lo, α_hi] satisfies:
    1. α_lo and α_hi bracket a point satisfying Wolfe conditions
    2. α_lo has lower function value
    3. φ'(α_lo)(α_hi - α_lo) < 0

    Uses bisection with optional quadratic interpolation for faster convergence.
    """
    if kwargs is None:
        kwargs = {}

    fev = 0

    for _ in range(max_iter):
        # Bisection (could use quadratic interpolation for faster convergence)
        alpha = 0.5 * (alpha_lo + alpha_hi)

        x_new = x + alpha * d
        phi = loss_fn(x_new, *args, **kwargs)
        fev += 1

        armijo_threshold = phi_0 + c1 * alpha * dphi_0

        if phi > armijo_threshold or phi >= phi_lo:
            # Shrink from above
            alpha_hi = alpha
            phi_hi = phi
        else:
            g_new = grad_fn(x_new, *args, **kwargs)
            dphi = _to_scalar((g_new * d).sum(), backend)

            # Check strong Wolfe curvature condition
            if abs(dphi) <= c2 * abs(dphi_0):
                return alpha, phi, g_new, fev

            # Update bracket
            if dphi * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
                phi_hi = phi_lo

            alpha_lo = alpha
            phi_lo = phi
            dphi_lo = dphi

        # Check for convergence (bracket too small)
        if abs(alpha_hi - alpha_lo) < 1e-12:
            break

    # Return best found
    x_new = x + alpha_lo * d
    f_new = loss_fn(x_new, *args, **kwargs)
    g_new = grad_fn(x_new, *args, **kwargs)
    return alpha_lo, f_new, g_new, fev + 1


def _backtracking_line_search(
    loss_fn,
    grad_fn,
    x,
    d,
    g,
    backend,
    *,
    args=(),
    kwargs=None,
    c1=1e-4,
    rho=0.5,
    max_iter=20,
):
    """Simple backtracking line search (Armijo condition only).

    Finds α satisfying:
        f(x + αd) ≤ f(x) + c₁ α ∇f(x)ᵀd

    This is faster but may not guarantee curvature condition needed
    for quasi-Newton methods.
    """
    if kwargs is None:
        kwargs = {}

    alpha = 1.0
    f_0 = loss_fn(x, *args, **kwargs)
    directional_derivative = _to_scalar((g * d).sum(), backend)

    fev = 1
    for _ in range(max_iter):
        x_new = x + alpha * d
        f_new = loss_fn(x_new, *args, **kwargs)
        fev += 1

        if f_new <= f_0 + c1 * alpha * directional_derivative:
            g_new = grad_fn(x_new, *args, **kwargs)
            return alpha, f_new, g_new, fev

        alpha *= rho

    x_new = x + alpha * d
    f_new = loss_fn(x_new, *args, **kwargs)
    g_new = grad_fn(x_new, *args, **kwargs)
    return alpha, f_new, g_new, fev


def _to_scalar(value, backend):
    """Convert array scalar to Python float."""
    if hasattr(backend, 'as_numpy'):
        return float(backend.as_numpy(value))
    return float(value)


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

__all__ = ["lbfgs"]
