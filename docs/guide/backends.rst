.. _guide-backends:

Multi-Backend Support
######################

Aurora-GLM supports three computational backends: NumPy (default),
PyTorch, and JAX. All numerical code works transparently across backends.

.. _backends-overview:

Backend overview
================

.. list-table::
   :header-rows: 1
   :widths: 15 25 30 30
   :stub-columns: 0

   * - Backend
     - Install
     - Strengths
     - When to use
   * - NumPy
     - Core (default)
     - Stable, no extra deps
     - Default, small to medium data
   * - PyTorch
     - ``pip install aurora-glm[torch]``
     - GPU (CUDA), up to 141x speedup
     - Large datasets, GPU available
   * - JAX
     - ``pip install aurora-glm[jax]``
     - JIT compilation, automatic differentiation
     - Research, custom gradients

.. _backends-auto:

Automatic backend detection
---------------------------

The backend is detected automatically from input types:

.. code-block:: python

   import numpy as np
   import torch
   from aurora.models.glm import fit_glm

   # NumPy arrays -> NumPy backend (automatic)
   X_np = np.random.randn(1000, 5)
   y_np = np.random.randn(1000)
   result = fit_glm(X_np, y_np)

   # PyTorch tensors -> PyTorch backend (automatic)
   X_t = torch.tensor(X_np)
   y_t = torch.tensor(y_np)
   result = fit_glm(X_t, y_t)

.. _backends-namespace:

The namespace pattern
---------------------

Internal numerical code uses the multi-backend namespace pattern:

.. code-block:: python

   from aurora.distributions._utils import namespace, as_namespace_array

   def my_function(x, y):
       # Detect backend from inputs
       xp = namespace(x, y)
       # Convert to appropriate array type
       x_arr = as_namespace_array(x, xp, like=y)
       # Use xp instead of np
       return xp.sum(x_arr * xp.exp(y))

This pattern ensures all operations go through ``xp`` (which is ``numpy``,
``torch``, or ``jax.numpy``) instead of calling ``np.*`` directly.

.. _backends-gpu:

GPU acceleration
----------------

PyTorch CUDA provides dramatic speedups for large problems:

.. code-block:: python

   import torch

   # Move data to GPU
   device = "cuda" if torch.cuda.is_available() else "cpu"
   X = torch.randn(100000, 50, device=device)
   y = torch.randn(100000, device=device)

   # Fit on GPU
   from aurora.models.glm import fit_glm
   result = fit_glm(X, y, family="gaussian")

.. _backends-when:

When to use which backend
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40
   :stub-columns: 0

   * - Scenario
     - Recommended
     - Reason
   * - Default analysis
     - NumPy
     - No setup required, stable for n < 10,000
   * - Large GLM (n > 10k)
     - PyTorch GPU
     - Matrix ops benefit from parallelism
   * - GAM with many smooth terms
     - PyTorch GPU
     - Basis matrix construction is parallelizable
   * - GAMM with many groups
     - PyTorch GPU
     - Random effects design matrix benefits from GPU
   * - Custom likelihood / gradient
     - JAX
     - JIT + autodiff for custom objectives

.. _backends-compat:

Compatibility notes
-------------------

- All backends produce numerically identical results (validated against R)
- PyTorch tensors are converted to NumPy internally where needed (e.g., for
  SciPy operations in GAMM)
- JAX arrays work similarly, with automatic JIT compilation where applicable
- Sparse matrix support is available on all backends via SciPy
