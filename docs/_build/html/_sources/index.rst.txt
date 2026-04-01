Aurora-GLM Documentation
========================

.. raw:: html

   <p style="font-size:1.1em; margin-bottom:2rem;">
   A modular, extensible, and high-performance Python framework for
   Generalized Linear Models (GLM), Generalized Additive Models (GAM),
   and Generalized Additive Mixed Models (GAMM).
   </p>

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: User Guide
      :link: guide/index
      :link-type: ref
      :class-card: guide-card

      Step-by-step tutorials covering GLM, GAM, GAMM, formula syntax,
      multi-backend support, and inference.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: ref
      :class-card: api-card

      Complete reference for all public modules, classes, and functions.

   .. grid-item-card:: Examples
      :link: examples/index
      :link-type: ref
      :class-card: examples-card

      17 case studies including insurance pricing, species distribution,
      clinical trials, and more.

Features
--------

- **Multi-backend**: NumPy, PyTorch, and JAX with automatic detection
- **GPU acceleration**: Up to 141x speedup via PyTorch CUDA
- **10+ distribution families**: Gaussian, Poisson, Binomial, Gamma, Beta, and more
- **9 link functions**: Logit, Log, Probit, CLogLog, and more
- **R-compatible formulas**: ``y ~ s(x1) + x2 + (1 | group)``
- **Validated against R**: Results match mgcv, lme4 within 1e-6 tolerance
- **Sparse matrix support**: Efficient for large n or many smooth terms

Quick Start
-----------

.. code-block:: python

   import aurora
   from aurora import fit_glm, GaussianFamily, LogitLink

   # Fit a GLM
   result = fit_glm(X, y, family=GaussianFamily())

   # Inspect results
   result.summary()
   result.plot_diagnostics()

   # Use PyTorch backend for GPU acceleration
   import torch
   result = fit_glm(torch.tensor(X), torch.tensor(y), family=GaussianFamily())

.. toctree::
   :hidden:
   :maxdepth: 2

   guide/index
   api/index
   examples/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
