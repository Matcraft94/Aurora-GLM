.. _guide-installation:

Installation
############

Prerequisites
=============

- Python 3.10 or later
- pip package manager

Stable release
==============

Install the core package from PyPI:

.. code-block:: bash

   pip install aurora-glm

This installs Aurora-GLM with NumPy and SciPy -- sufficient for all GLM,
GAM, and Gaussian GAMM workflows.

Optional dependencies
=====================

Aurora-GLM supports multiple computational backends and additional
features through optional dependencies:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Extra
     - Install command
     - Provides
   * - PyTorch GPU
     - ``pip install aurora-glm[torch]``
     - GPU acceleration via CUDA (up to 141x speedup)
   * - JAX
     - ``pip install aurora-glm[jax]``
     - JAX arrays and JIT compilation
   * - Bayesian
     - ``pip install aurora-glm[bayesian]``
     - Bayesian inference module
   * - Development
     - ``pip install aurora-glm[dev]``
     - pytest, ruff, mypy
   * - Everything
     - ``pip install aurora-glm[all]``
     - All of the above

From source
===========

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/Matcraft94/Aurora-GLM.git
   cd Aurora-GLM
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"

For GPU support, add the torch extra:

.. code-block:: bash

   pip install -e ".[torch]"

Verifying the installation
===========================

.. code-block:: python

   import aurora
   print(aurora.__version__)   # 0.7.0

   # Quick smoke test
   import numpy as np
   from aurora.models.glm import fit_glm

   X = np.random.randn(100, 3)
   y = X @ [1.5, -0.8, 0.3] + 0.5 + np.random.randn(100)
   result = fit_glm(X, y, family="gaussian")
   print(result.converged_)    # True

See :doc:`quickstart` for a full walkthrough.
