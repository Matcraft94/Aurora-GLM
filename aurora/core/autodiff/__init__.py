# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Automatic differentiation helpers for Aurora-GLM.

This module provides unified autodiff utilities across backends (NumPy, PyTorch, JAX),
enabling gradient-based optimization for GLM, GAM, and GAMM fitting.

Mathematical Background
-----------------------
Automatic differentiation (AD) computes exact derivatives of functions defined
by computer programs. Unlike symbolic differentiation (expression swell) or
numerical differentiation (truncation error), AD provides:

1. **Exact derivatives** up to machine precision
2. **Efficient computation** via the chain rule
3. **Scalability** to high-dimensional problems

**Forward mode AD** computes directional derivatives:
    Dv f(x) = lim_{ε→0} [f(x + εv) - f(x)] / ε

**Reverse mode AD** (backpropagation) computes gradients efficiently:
    ∇f: R^n → R  requires O(1) backward passes vs O(n) forward passes

Key Functions
-------------
gradient : Compute gradient of scalar-valued function
hessian : Compute Hessian matrix of scalar-valued function
jacobian : Compute Jacobian matrix of vector-valued function
hvp : Hessian-vector product (memory efficient)
jvp : Jacobian-vector product (forward mode)
vjp : Vector-Jacobian product (reverse mode)

Backend Support
---------------
The module automatically detects the computational backend and uses:

- **JAX**: `jax.grad`, `jax.hessian`, `jax.jacfwd`, `jax.jacrev`
- **PyTorch**: `torch.autograd.grad`, `torch.autograd.functional.jacobian`
- **NumPy**: Numerical differentiation via finite differences

Example Usage
-------------
>>> from aurora.core.autodiff import gradient, hessian
>>>
>>> def loss(params, X, y):
...     pred = X @ params
...     return 0.5 * np.sum((y - pred) ** 2)
>>>
>>> # Create gradient function
>>> grad_loss = gradient(loss)
>>>
>>> # Evaluate gradient at params
>>> g = grad_loss(params, X, y)
>>>
>>> # Compute full Hessian (for small problems)
>>> hess_loss = hessian(loss)
>>> H = hess_loss(params, X, y)

Performance Notes
-----------------
**Memory vs Speed tradeoffs**:

1. Full Hessian: O(p²) memory, useful for small p (< 1000)
2. HVP (Hessian-vector product): O(p) memory, use for large p
3. Numerical Hessian: O(p) function evaluations per column

**Backend recommendations**:

- JAX: Best for JIT compilation and vectorization (vmap)
- PyTorch: Good for GPU acceleration and dynamic graphs
- NumPy: Fallback with numerical differentiation

References
----------
- Griewank, A., & Walther, A. (2008). "Evaluating Derivatives:
  Principles and Techniques of Algorithmic Differentiation" (2nd ed.).
  SIAM. https://doi.org/10.1137/1.9780898717761

- Baydin, A. G., et al. (2018). "Automatic differentiation in machine
  learning: a survey." Journal of Machine Learning Research, 18(153), 1-43.
  http://jmlr.org/papers/v18/17-468.html

See Also
--------
aurora.core.optimization : Optimization algorithms using autodiff
aurora.core.backends : Backend abstraction layer
"""

from .gradient import gradient
from .hessian import hessian
from .jacobian import jacobian
from .products import hvp, jvp, vjp
from .utils import check_gradient

__all__ = [
    # Core functions
    "gradient",
    "hessian",
    "jacobian",
    # Efficient products
    "hvp",
    "jvp",
    "vjp",
    # Utilities
    "check_gradient",
]
