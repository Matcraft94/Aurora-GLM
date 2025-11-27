"""Automatic differentiation helpers.

This module provides unified autodiff utilities across backends.

Planned Features
----------------
- Gradient computation for custom loss functions
- Hessian computation for Newton-Raphson
- JIT compilation wrappers (JAX/PyTorch)
- Vectorized Jacobians (vmap)

Notes
-----
Currently, automatic differentiation is handled natively by:
- PyTorch: torch.autograd
- JAX: jax.grad, jax.jacobian

For NumPy backend, numerical differentiation is used in:
- aurora.core.optimization.newton (finite differences)

Example (planned):
>>> from aurora.core.autodiff import gradient, hessian
>>> grad_fn = gradient(loss_fn)
>>> hess_fn = hessian(loss_fn)
>>> g = grad_fn(params)
>>> H = hess_fn(params)
"""