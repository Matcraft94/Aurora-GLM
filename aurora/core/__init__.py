"""Core Numerical Primitives for Aurora-GLM.

This module provides the foundational numerical operations and backend
abstraction layer that powers all Aurora-GLM computations.

Architecture Overview
=====================

Multi-Backend Design
--------------------
Aurora-GLM supports three computational backends:

1. **NumPy** (default): Pure Python, CPU-based
   - Universal compatibility
   - Excellent for development and debugging
   - Good performance for small to medium problems

2. **PyTorch**: GPU-accelerated, automatic differentiation
   - CUDA/ROCm support for GPU acceleration
   - Automatic differentiation for gradient-based optimization
   - Seamless integration with deep learning workflows

3. **JAX**: XLA-compiled, functional transformations
   - Just-in-time (JIT) compilation for performance
   - Automatic vectorization (vmap)
   - Automatic differentiation (grad, jacfwd, jacrev)

Array Namespace Abstraction
----------------------------
The `namespace()` function provides a unified API across backends:

    xp = namespace(array)  # Infer backend from array type
    
    # Use xp for all operations
    result = xp.sum(xp.exp(array))

This pattern allows the same code to run on any backend without modification.

Mathematical Operations
------------------------
The core module provides robust implementations of:

**Linear Algebra** (aurora.core.linalg):
- QR decomposition with column pivoting
- Cholesky decomposition with positive-definite checks
- SVD for rank-deficient problems
- Woodbury identity for efficient inverse updates

**Optimization** (aurora.core.optimization):
- Newton-Raphson with line search
- IRLS (Iteratively Reweighted Least Squares)
- L-BFGS for large-scale optimization

**Automatic Differentiation** (aurora.core.autodiff):
- Gradient computation
- Hessian computation
- Jacobian computation

Numerical Stability
-------------------
All operations include safeguards for numerical stability:

1. **Condition number monitoring**:
   Check κ(A) before solving Ax = b

2. **Regularization fallbacks**:
   Add small diagonal perturbation if near-singular

3. **Log-space computation**:
   Use log(det(A)) instead of det(A) for large matrices

4. **Clipping and bounding**:
   Prevent log(0), exp(large), division by zero

Type System
-----------
Aurora-GLM uses a flexible type system for array compatibility:

- Array: Generic array type (np.ndarray, torch.Tensor, jnp.ndarray)
- ArrayLike: Array or convertible (list, tuple, scalar)
- Scalar: Single numerical value (int, float)

The type system is defined in `aurora.core.types` and used throughout
the library for type hints and runtime validation.

Submodules
==========

backends
    Multi-backend abstraction layer
    - Backend detection and selection
    - Array namespace functions
    - Device management (CPU/GPU)

types
    Type definitions and utilities
    - ArrayLike, Array type aliases
    - Type checking functions

linalg
    Linear algebra primitives
    - Matrix decompositions (QR, Cholesky, SVD, eigendecomposition)
    - Solving linear systems
    - Determinant and inverse computation

optimization
    Numerical optimization algorithms
    - Newton-Raphson
    - Line search methods
    - Convergence monitoring

autodiff
    Automatic differentiation utilities
    - Gradient computation
    - Hessian computation
    - Jacobian computation

See Also
--------
aurora.distributions._utils : Array namespace utilities for distributions
aurora.models.glm.fitting : IRLS implementation using core primitives
aurora.models.gamm.estimation : REML optimization

Notes
-----
For GPU acceleration with PyTorch, ensure CUDA is properly installed
and specify device='cuda' when creating arrays.

For JAX JIT compilation, ensure the code is free of Python-level
control flow that depends on array values.
"""
from __future__ import annotations

from . import backends, types

__all__ = ["backends", "types"]
