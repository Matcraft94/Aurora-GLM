# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Newton-Raphson Method for Unconstrained Optimization.

Mathematical Framework
----------------------
The Newton-Raphson method (also called Newton's method) is a second-order
optimization algorithm that uses both gradient and Hessian information to
find stationary points of a function.

Problem Formulation
-------------------
Find x* that minimizes f(x):

    x* = argmin_{x ∈ ℝᵖ} f(x)

where f: ℝᵖ → ℝ is twice continuously differentiable.

**Optimality condition**: At a local minimum, ∇f(x*) = 0

Newton's Method Algorithm
-------------------------
Starting from initial guess x⁽⁰⁾, iterate until convergence:

**Step 1: Compute gradient and Hessian**

    g⁽ᵗ⁾ = ∇f(x⁽ᵗ⁾)          (gradient, p-vector)
    H⁽ᵗ⁾ = ∇²f(x⁽ᵗ⁾)         (Hessian, p×p matrix)

**Step 2: Solve Newton system**

Compute search direction p⁽ᵗ⁾ by solving:

    H⁽ᵗ⁾ p⁽ᵗ⁾ = -g⁽ᵗ⁾

**Step 3: Update parameters**

    x⁽ᵗ⁺¹⁾ = x⁽ᵗ⁾ + p⁽ᵗ⁾

**Step 4: Check convergence**

Stop when ||g⁽ᵗ⁾|| < tol or ||p⁽ᵗ⁾|| < tol

Derivation via Taylor Approximation
------------------------------------
The Newton step is derived from the second-order Taylor expansion:

    f(x) ≈ f(x⁽ᵗ⁾) + g⁽ᵗ⁾ᵀ(x - x⁽ᵗ⁾) + ½(x - x⁽ᵗ⁾)ᵀ H⁽ᵗ⁾ (x - x⁽ᵗ⁾)

Minimizing this quadratic approximation (setting derivative to zero):

    ∇[quadratic] = g⁽ᵗ⁾ + H⁽ᵗ⁾(x - x⁽ᵗ⁾) = 0

gives the Newton step:
    x - x⁽ᵗ⁾ = -(H⁽ᵗ⁾)⁻¹ g⁽ᵗ⁾

**Interpretation**: At each iteration, Newton's method finds the minimum
of the local quadratic approximation of f.

Convergence Theory
------------------
**Theorem** (Quadratic convergence, Dennis & Schnabel, 1996):

Suppose:
1. f is twice continuously differentiable
2. x* is a local minimum with ∇f(x*) = 0
3. H(x*) is positive definite (strict local minimum)
4. Starting point x⁽⁰⁾ is sufficiently close to x*

Then:
- Newton's method converges to x*
- Convergence is quadratic: ||x⁽ᵗ⁺¹⁾ - x*|| ≤ C ||x⁽ᵗ⁾ - x*||²

**Rate**: Number of correct digits approximately doubles per iteration.

**Comparison**:
- Linear convergence: error × constant (gradient descent)
- Superlinear: error^α, 1 < α < 2 (quasi-Newton)
- Quadratic: error² (Newton's method)

**Global convergence**: Newton's method is NOT globally convergent.
- May diverge from poor starting points
- Requires positive-definite Hessian (not guaranteed away from minimum)
- Often combined with line search or trust regions for globalization

Conditions for Positive Definiteness
-------------------------------------
Newton's method requires H⁽ᵗ⁾ to be positive definite at each iteration.

**When H is positive definite**:
- f is strictly convex (globally or locally)
- Near a strict local minimum
- For GLMs with canonical link: always (Fisher information)

**When H may be indefinite**:
- Saddle points: some eigenvalues negative
- Far from minimum
- Non-convex optimization landscapes

**Remedy**: Modified Newton with regularization:
    (H⁽ᵗ⁾ + λI) p⁽ᵗ⁾ = -g⁽ᵗ⁾
where λ > 0 ensures positive definiteness (not implemented here).

Hessian Computation
-------------------
This implementation computes Hessians using three methods:

### 1. Automatic Differentiation (Preferred)

**PyTorch**:
    Uses torch.autograd.grad twice (forward-mode AD)
    Cost: O(p²) forward passes

**JAX**:
    Uses jax.hessian (reverse-over-reverse AD)
    Cost: O(p) forward + O(p) backward passes
    Most efficient for small to medium p

### 2. Finite Differences (Fallback)

When AD not available, uses central differences:

    ∂²f/∂xᵢ∂xⱼ ≈ [f(x+eᵢ+eⱼ) - f(x+eᵢ-eⱼ) - f(x-eᵢ+eⱼ) + f(x-eᵢ-eⱼ)] / (4h²)

where eᵢ is the i-th unit vector and h = 10⁻⁴.

**Cost**: O(p²) function evaluations (4 per Hessian entry)

**Accuracy**: O(h²) truncation error, but subject to roundoff for small h

Numerical Stability
-------------------
**Challenges**:

1. **Singular Hessian**: When H is rank-deficient (parameter redundancy)
   - Solution fails
   - Returns error message
   - Suggests using L-BFGS or ridge regularization

2. **Ill-conditioned Hessian**: When condition number κ(H) is large
   - Numerical error in solution: O(ε × κ(H))
   - Error amplification in gradient
   - Common in overparameterized models

3. **Finite-difference errors**: Tradeoff between truncation and roundoff
   - Step size h = 10⁻⁴ balances errors
   - Can fail for very steep or flat functions

**Improvements** (not implemented):
- Cholesky decomposition with diagonal pivoting
- Condition number monitoring
- Iterative refinement
- Hessian-free methods (conjugate gradient on H·p = -g)

Computational Complexity
------------------------
Per iteration, for p parameters:

**Gradient computation**:
- AD: O(p) backward pass
- Finite differences: O(p) function evals

**Hessian computation**:
- AD (JAX): O(p) gradients = O(p²) total
- AD (PyTorch): O(p²) backward passes
- Finite differences: O(p²) function evals

**System solve** H·p = -g:
- Direct (Cholesky): O(p³)
- Iterative (CG): O(kp²) for k iterations

**Total per iteration**: O(p³) dominated by linear solve

**vs IRLS**: Same O(p³), but Newton computes full Hessian not just X^T W X

**vs L-BFGS**: L-BFGS avoids O(p³) by approximating H⁻¹, cost O(mp) where m ≈ 10

Comparison with Other Methods
------------------------------
**vs Gradient Descent**:
- Newton: Quadratic convergence, expensive per iteration
- GD: Linear convergence, cheap per iteration
- Crossover: Newton better for moderate p, high accuracy needs

**vs Fisher Scoring (IRLS for GLMs)**:
- Newton uses observed Hessian: ∇²ℓ
- Fisher uses expected Hessian: E[∇²ℓ]
- For exponential families: same for canonical link
- Fisher more stable (always positive-definite)

**vs Quasi-Newton (L-BFGS)**:
- Newton: O(p³) per iteration, fewer iterations
- Quasi-Newton: O(p²) per iteration, more iterations
- Quasi-Newton preferred for large p (p > 1000)

**vs Trust Region**:
- Newton: No globalization, may diverge
- Trust region: Guaranteed descent, slower per iteration
- Hybrid approaches combine both

Applications in Aurora-GLM
---------------------------
Newton-Raphson is used for:

1. **Non-canonical GLM links**: When IRLS not equivalent to Fisher scoring
2. **Dispersion parameter estimation**: Optimize profile likelihood
3. **Variance component estimation**: REML in mixed models (via PQL)
4. **General maximum likelihood**: When IRLS doesn't apply

**Not used for**:
- Standard GLM fitting → use IRLS instead (more stable)
- Large-scale problems → use L-BFGS
- Non-smooth objectives → use subgradient methods

Implementation Notes
--------------------
**Multi-backend support**:
Transparently works with NumPy, PyTorch, and JAX arrays through
automatic differentiation when available, falling back to finite
differences.

**Automatic differentiation**:
- Leverages backend AD for exact gradient and Hessian
- No manual derivative implementation required
- Enables rapid prototyping of new models

**Convergence criteria**:
- Gradient norm: ||g|| < tol (first-order optimality)
- Step size: ||p|| < tol (stationarity)
- Both checked for robustness

References
----------
**Core Newton method theory**:

- Dennis, J. E., & Schnabel, R. B. (1996). *Numerical Methods for Unconstrained
  Optimization and Nonlinear Equations*. SIAM.
  https://doi.org/10.1137/1.9781611971200

- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.).
  Springer. Chapter 3: Line Search Methods.
  https://doi.org/10.1007/978-0-387-40065-5

**Convergence analysis**:

- Ortega, J. M., & Rheinboldt, W. C. (2000). *Iterative Solution of Nonlinear
  Equations in Several Variables*. SIAM.
  (Classic convergence rate proofs)

**Automatic differentiation**:

- Griewank, A., & Walther, A. (2008). *Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation* (2nd ed.). SIAM.
  https://doi.org/10.1137/1.9780898717761

- Baydin, A. G., et al. (2018). \"Automatic differentiation in machine learning:
  A survey.\" *Journal of Machine Learning Research*, 18(153), 1-43.

**Numerical linear algebra**:

- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.).
  Johns Hopkins University Press. Chapter 4: Linear systems.

**Modified Newton methods**:

- Gill, P. E., Murray, W., & Wright, M. H. (1981). *Practical Optimization*.
  Academic Press. Chapter 4: Modifications for indefinite Hessians.

See Also
--------
aurora.core.optimization.irls : IRLS for GLMs (Fisher scoring)
aurora.core.optimization.lbfgs : Quasi-Newton method (Hessian-free)
aurora.models.glm.fitting : GLM fitting algorithms

Notes
-----
For detailed mathematical derivations, see REFERENCES.md in the repository root.

Newton's method is the gold standard for small to moderate-sized optimization
problems where Hessian computation is feasible. Its quadratic convergence rate
makes it highly efficient near the solution, though care must be taken with
initialization and Hessian conditioning.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..types import Array, OptimizationCallback
from .result import OptimizationResult


def newton_raphson(
    loss_fn: Callable,
    init_params: Array,
    *,
    backend=None,
    args: tuple = (),
    kwargs: dict | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    callback: OptimizationCallback | None = None,
) -> OptimizationResult:
    """Run the Newton-Raphson method with automatic differentiation support."""
    if kwargs is None:
        kwargs = {}

    if backend is None:
        from ..backends import get_backend

        backend = get_backend("jax")

    grad_fn = backend.grad(loss_fn)

    x = backend.array(init_params)
    nfev = 0
    njev = 0
    nhev = 0

    for iteration in range(max_iter):
        grad = grad_fn(x, *args, **kwargs)
        njev += 1

        grad_np = np.asarray(backend.as_numpy(grad), dtype=float)
        grad_norm = np.linalg.norm(grad_np)

        if grad_norm < tol:
            f_val = backend.as_numpy(loss_fn(x, *args, **kwargs))
            nfev += 1
            return OptimizationResult(
                x=backend.as_numpy(x),
                fun=float(f_val),
                grad=backend.as_numpy(grad),
                success=True,
                message="Converged: gradient norm below tolerance",
                nit=iteration,
                nfev=nfev,
                njev=njev,
                nhev=nhev,
            )

        hess_np, evals = _compute_hessian(loss_fn, x, backend, args, kwargs)
        nfev += evals
        nhev += 1

        try:
            step = np.linalg.solve(hess_np, grad_np)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - rare singular cases
            f_val = backend.as_numpy(loss_fn(x, *args, **kwargs))
            nfev += 1
            return OptimizationResult(
                x=backend.as_numpy(x),
                fun=float(f_val),
                grad=backend.as_numpy(grad),
                success=False,
                message=f"Hessian is singular: {exc}",
                nit=iteration,
                nfev=nfev,
                njev=njev,
                nhev=nhev,
            )

        x_np = np.asarray(backend.as_numpy(x), dtype=float)
        x_new_np = x_np - step
        x = backend.array(x_new_np, dtype=getattr(x, "dtype", None))

        f_val = backend.as_numpy(loss_fn(x, *args, **kwargs))
        nfev += 1

        if callback is not None:
            callback(iteration, backend.as_numpy(x), float(f_val))

        if np.linalg.norm(step) < tol:
            final_grad = backend.as_numpy(grad_fn(x, *args, **kwargs))
            njev += 1
            return OptimizationResult(
                x=backend.as_numpy(x),
                fun=float(f_val),
                grad=final_grad,
                success=True,
                message="Converged: step size below tolerance",
                nit=iteration + 1,
                nfev=nfev,
                njev=njev,
                nhev=nhev,
            )

    f_val = backend.as_numpy(loss_fn(x, *args, **kwargs))
    nfev += 1
    final_grad = backend.as_numpy(grad_fn(x, *args, **kwargs))
    njev += 1
    return OptimizationResult(
        x=backend.as_numpy(x),
        fun=float(f_val),
        grad=final_grad,
        success=False,
        message="Maximum iterations reached",
        nit=max_iter,
        nfev=nfev,
        njev=njev,
        nhev=nhev,
    )


def _compute_hessian(loss_fn, params, backend, args, kwargs):
    """Compute Hessian via autograd when available, otherwise finite differences."""
    params_np = np.asarray(backend.as_numpy(params), dtype=float)
    dim = params_np.size
    hessian = np.zeros((dim, dim), dtype=float)

    try:
        import torch  # type: ignore

        if isinstance(params, torch.Tensor):
            params_t = params.detach().clone().requires_grad_(True)
            result = loss_fn(params_t, *args, **kwargs)
            grad = torch.autograd.grad(result, params_t, create_graph=True)[0]
            rows = []
            for grad_component in grad:
                if not grad_component.requires_grad:
                    rows.append(torch.zeros_like(params_t))
                    continue
                second = torch.autograd.grad(
                    grad_component,
                    params_t,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if second is None:
                    second = torch.zeros_like(params_t)
                rows.append(second)
            hess_tensor = torch.stack(rows)
            return hess_tensor.detach().cpu().numpy(), 1
    except ImportError:  # pragma: no cover - optional dependency missing
        pass

    try:
        import jax
        import jax.numpy as jnp

        if isinstance(params, jnp.ndarray):  # type: ignore[attr-defined]
            hess = jax.hessian(lambda p: loss_fn(p, *args, **kwargs))(params)
            return np.asarray(hess), 1
    except ImportError:  # pragma: no cover - optional dependency missing
        pass

    eps = 1e-4
    evaluations = 0

    for i in range(dim):
        for j in range(i, dim):
            e_i = np.zeros(dim)
            e_j = np.zeros(dim)
            e_i[i] = eps
            e_j[j] = eps

            f_pp = backend.as_numpy(
                loss_fn(backend.array(params_np + e_i + e_j), *args, **kwargs)
            )
            f_pm = backend.as_numpy(
                loss_fn(backend.array(params_np + e_i - e_j), *args, **kwargs)
            )
            f_mp = backend.as_numpy(
                loss_fn(backend.array(params_np - e_i + e_j), *args, **kwargs)
            )
            f_mm = backend.as_numpy(
                loss_fn(backend.array(params_np - e_i - e_j), *args, **kwargs)
            )
            evaluations += 4

            value = float((f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps))
            hessian[i, j] = value
            hessian[j, i] = value

    return hessian, evaluations


def modified_newton(
    loss_fn: Callable,
    init_params: Array,
    *,
    backend=None,
    args: tuple = (),
    kwargs: dict | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    lambda_init: float = 1e-3,
    lambda_factor: float = 10.0,
    lambda_max: float = 1e8,
    callback: OptimizationCallback | None = None,
) -> OptimizationResult:
    """Modified Newton-Raphson with Levenberg-Marquardt regularization.

    Mathematical Framework
    ----------------------
    When the Hessian H is indefinite or ill-conditioned, the standard Newton
    step p = -H⁻¹g may not be a descent direction. Modified Newton adds
    regularization:

        (H + λI) p = -g

    where λ > 0 ensures positive definiteness.

    Levenberg-Marquardt Strategy
    ----------------------------
    The regularization parameter λ is adjusted adaptively:

    1. **Start with small λ**: Allow near-Newton steps when H is well-behaved
    2. **Increase λ if step rejected**: When function doesn't decrease sufficiently,
       multiply λ by `lambda_factor` to trust gradient more than curvature
    3. **Decrease λ if step accepted**: After successful steps, reduce λ
       to allow faster convergence via better Hessian approximation

    **Interpretation**:
    - λ → 0: Pure Newton (quadratic convergence near solution)
    - λ → ∞: Steepest descent (slow but robust)

    The algorithm smoothly interpolates between these extremes.

    Guaranteeing Positive Definiteness
    ----------------------------------
    For symmetric H with eigenvalues λ₁ ≤ λ₂ ≤ ... ≤ λₙ:
    - H + λI has eigenvalues λ₁ + λ, λ₂ + λ, ..., λₙ + λ
    - If λ > -λ₁ (where λ₁ is the smallest eigenvalue), H + λI is positive definite

    In practice, we use the Cholesky factorization attempt:
    - If Cholesky succeeds: H + λI is positive definite
    - If Cholesky fails: Increase λ and retry

    Convergence Properties
    ----------------------
    **Theorem** (Dennis & Schnabel, 1996): Modified Newton with adaptive λ
    converges globally for any starting point if:
    1. f is bounded below
    2. ∇f is Lipschitz continuous
    3. f is twice continuously differentiable

    **Local rate**: Near the solution where H is positive definite:
    - If λ → 0 sufficiently fast: Quadratic convergence
    - If λ bounded away from 0: Linear convergence

    Parameters
    ----------
    loss_fn : callable
        Objective function to minimize
    init_params : array
        Initial parameter values
    backend : object, optional
        Array backend (NumPy, PyTorch, JAX)
    args : tuple
        Additional positional arguments for loss_fn
    kwargs : dict, optional
        Additional keyword arguments for loss_fn
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-6
        Convergence tolerance for gradient norm
    lambda_init : float, default=1e-3
        Initial regularization parameter
    lambda_factor : float, default=10.0
        Factor to increase/decrease λ
    lambda_max : float, default=1e8
        Maximum allowed λ (switches to pure gradient descent)
    callback : callable, optional
        Function called after each iteration

    Returns
    -------
    OptimizationResult
        Contains solution, final function value, convergence status

    References
    ----------
    [1] Levenberg, K. (1944). "A method for the solution of certain non-linear
        problems in least squares." Quarterly of Applied Mathematics, 2(2), 164-168.
    [2] Marquardt, D. W. (1963). "An algorithm for least-squares estimation of
        nonlinear parameters." Journal of SIAM, 11(2), 431-441.
    [3] Dennis, J. E., & Schnabel, R. B. (1996). Numerical Methods for
        Unconstrained Optimization and Nonlinear Equations. SIAM.
    [4] Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.).
        Springer. Chapter 4: Trust-Region Methods.

    Examples
    --------
    >>> def rosenbrock(x):
    ...     return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    >>> result = modified_newton(rosenbrock, [-1.0, 1.0])
    >>> print(result.x)  # Should be close to [1, 1]
    """
    if kwargs is None:
        kwargs = {}

    if backend is None:
        from ..backends import get_backend

        backend = get_backend("jax")

    grad_fn = backend.grad(loss_fn)

    x = backend.array(init_params)
    x_np = np.asarray(backend.as_numpy(x), dtype=float)

    nfev = 0
    njev = 0
    nhev = 0

    # Current function value
    f_val = float(backend.as_numpy(loss_fn(x, *args, **kwargs)))
    nfev += 1

    lam = lambda_init

    for iteration in range(max_iter):
        # Compute gradient
        grad = grad_fn(x, *args, **kwargs)
        njev += 1
        grad_np = np.asarray(backend.as_numpy(grad), dtype=float)
        grad_norm = np.linalg.norm(grad_np)

        # Check convergence
        if grad_norm < tol:
            return OptimizationResult(
                x=x_np,
                fun=f_val,
                grad=grad_np,
                success=True,
                message="Converged: gradient norm below tolerance",
                nit=iteration,
                nfev=nfev,
                njev=njev,
                nhev=nhev,
            )

        # Compute Hessian
        hess_np, evals = _compute_hessian(loss_fn, x, backend, args, kwargs)
        nfev += evals
        nhev += 1

        # Modified Newton with adaptive λ
        step_found = False
        for _ in range(20):  # Max attempts to find good λ
            try:
                # Form H + λI
                H_mod = hess_np + lam * np.eye(len(x_np))

                # Attempt Cholesky factorization (tests positive definiteness)
                L = np.linalg.cholesky(H_mod)

                # Solve (H + λI)p = -g via Cholesky
                # L L^T p = -g  =>  L y = -g, then L^T p = y
                y = np.linalg.solve(L, -grad_np)
                step = np.linalg.solve(L.T, y)

                # Evaluate new point
                x_new_np = x_np + step
                x_new = backend.array(x_new_np, dtype=getattr(x, "dtype", None))
                f_new = float(backend.as_numpy(loss_fn(x_new, *args, **kwargs)))
                nfev += 1

                # Armijo condition: sufficient decrease
                # f(x + p) ≤ f(x) + c₁ ∇f(x)ᵀp
                c1 = 1e-4
                directional_deriv = np.dot(grad_np, step)
                if f_new <= f_val + c1 * directional_deriv:
                    # Accept step
                    x_np = x_new_np
                    x = x_new
                    f_val = f_new
                    step_found = True

                    # Decrease λ for next iteration (trust Hessian more)
                    lam = max(lam / lambda_factor, 1e-10)
                    break
                else:
                    # Increase λ (trust gradient more)
                    lam = min(lam * lambda_factor, lambda_max)

            except np.linalg.LinAlgError:
                # Cholesky failed - matrix not positive definite
                # Increase λ and retry
                lam = min(lam * lambda_factor, lambda_max)

        if not step_found:
            # Fall back to gradient descent step
            step = -grad_np * (1.0 / (grad_norm + 1e-8))
            x_new_np = x_np + step * 0.1  # Small step
            x_new = backend.array(x_new_np, dtype=getattr(x, "dtype", None))
            f_new = float(backend.as_numpy(loss_fn(x_new, *args, **kwargs)))
            nfev += 1

            if f_new < f_val:
                x_np = x_new_np
                x = x_new
                f_val = f_new

        if callback is not None:
            callback(iteration, x_np, f_val)

        # Check step size convergence
        if step_found and np.linalg.norm(step) < tol:
            final_grad = np.asarray(
                backend.as_numpy(grad_fn(x, *args, **kwargs)), dtype=float
            )
            njev += 1
            return OptimizationResult(
                x=x_np,
                fun=f_val,
                grad=final_grad,
                success=True,
                message="Converged: step size below tolerance",
                nit=iteration + 1,
                nfev=nfev,
                njev=njev,
                nhev=nhev,
            )

    # Maximum iterations reached
    final_grad = np.asarray(backend.as_numpy(grad_fn(x, *args, **kwargs)), dtype=float)
    njev += 1
    return OptimizationResult(
        x=x_np,
        fun=f_val,
        grad=final_grad,
        success=False,
        message="Maximum iterations reached",
        nit=max_iter,
        nfev=nfev,
        njev=njev,
        nhev=nhev,
    )


__all__ = ["newton_raphson", "modified_newton"]
