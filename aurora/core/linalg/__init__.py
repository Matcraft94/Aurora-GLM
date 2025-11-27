"""Linear algebra primitives with multi-backend support.

This module provides backend-agnostic linear algebra operations
for NumPy, PyTorch, and JAX.

Planned Features
----------------
- QR decomposition with pivoting
- SVD (singular value decomposition)
- Cholesky decomposition with fallback
- Woodbury matrix identity for efficient updates
- Sparse matrix operations

Notes
-----
Currently, linear algebra operations are handled by:
- NumPy's linalg module
- PyTorch's linalg module
- JAX's numpy.linalg module

The backend abstraction in aurora.core.backends handles most cases.
This module will provide additional specialized routines.

Example (planned):
>>> from aurora.core.linalg import safe_cholesky, qr_solve
>>> L = safe_cholesky(XtWX + lambda_S)  # Falls back to pivoted Cholesky
>>> beta = qr_solve(X, y, weights=W)
"""