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

            f_pp = backend.as_numpy(loss_fn(backend.array(params_np + e_i + e_j), *args, **kwargs))
            f_pm = backend.as_numpy(loss_fn(backend.array(params_np + e_i - e_j), *args, **kwargs))
            f_mp = backend.as_numpy(loss_fn(backend.array(params_np - e_i + e_j), *args, **kwargs))
            f_mm = backend.as_numpy(loss_fn(backend.array(params_np - e_i - e_j), *args, **kwargs))
            evaluations += 4

            value = float((f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps))
            hessian[i, j] = value
            hessian[j, i] = value

    return hessian, evaluations


__all__ = ["newton_raphson"]