.. _guide-gam:

Generalized Additive Models (GAM)
##################################

GAMs extend GLMs by replacing the linear predictor with smooth functions:

.. math::

   \mathbb{E}[Y] = \beta_0 + f_1(x_1) + f_2(x_2) + \dots + \epsilon

This chapter covers spline types, smoothing parameter selection, the formula
interface, and visualization.

.. _gam-univariate:

Univariate GAM
===============

Use :func:`~aurora.models.gam.fitting.fit_gam` for a single smooth:

.. code-block:: python

   from aurora.models.gam import fit_gam

   # Generate nonlinear data
   np.random.seed(42)
   x = np.linspace(0, 10, 300)
   y = 2.0 + np.sin(x) + 0.3 * np.random.randn(300)

   # Fit GAM
   result = fit_gam(x, y, n_basis=15)

   print(f"EDF:    {result.edf:.2f}")      # Effective degrees of freedom
   print(f"Lambda: {result.lambda_:.4f}")   # Smoothing parameter
   print(f"GCV:   {result.gcv_score:.6f}")

   # Predict
   x_new = np.linspace(0, 10, 100)
   y_pred = result.predict(x_new)

Key parameters for ``fit_gam``:

.. list-table::
   :header-rows: 1
   :widths: 20 80
   :stub-columns: 0

   * - Parameter
     - Description
   * - ``n_basis``
     - Number of basis functions (default 10). More = more flexible.
   * - ``basis_type``
     - ``"bspline"`` (default) or ``"cubic"`` (natural cubic spline)
   * - ``degree``
     - Spline degree (default 3 = cubic)
   * - ``penalty_order``
     - Difference penalty order (default 2, approximates second derivative)
   * - ``lambda_``
     - Smoothing parameter. ``None`` (default) = automatic via GCV.
   * - ``use_sparse``
     - Use sparse matrices for large problems (n > 500, k > 20)

.. _gam-splines:

Spline types
------------

**B-splines** (default) offer local support and numerical stability via the
Cox-de Boor recursion. They are the recommended choice for most problems.

**Natural cubic splines** enforce f''(x) = 0 at boundaries and minimize
the integrated squared second derivative.

.. code-block:: python

   # B-spline (default)
   result_bs = fit_gam(x, y, n_basis=15, basis_type="bspline")

   # Natural cubic spline
   result_cs = fit_gam(x, y, n_basis=12, basis_type="cubic")

.. _gam-selection:

Smoothing parameter selection
-----------------------------

Two methods control the smoothness-accuracy tradeoff:

**GCV (Generalized Cross-Validation)** -- default, fast:

.. math::

   GCV(\lambda) = \frac{n \cdot RSS}{(n - edf)^2}

**REML (Restricted Maximum Likelihood)** -- more stable for small samples.

.. code-block:: python

   from aurora.models.gam import fit_additive_gam

   # GCV (default, faster)
   result_gcv = fit_additive_gam(X, y, smooth_terms=[...], method="GCV")

   # REML (more stable)
   result_reml = fit_additive_gam(X, y, smooth_terms=[...], method="REML")

.. _gam-additive:

Additive GAM with multiple terms
---------------------------------

Use :func:`~aurora.models.gam.additive.fit_additive_gam`
for models combining smooth and parametric terms:

.. code-block:: python

   from aurora.models.gam import fit_additive_gam
   from aurora.models.gam.terms import SmoothTerm, ParametricTerm

   # Generate data with two nonlinear effects and one linear effect
   np.random.seed(42)
   n = 300
   X = np.random.randn(n, 3)
   y = (np.sin(2 * X[:, 0]) + np.cos(X[:, 1])
         + 0.5 * X[:, 2] + np.random.randn(n) * 0.3)

   # Define terms
   smooth_terms = [
       SmoothTerm(variable=0, n_basis=12),   # s(x0) - nonlinear
       SmoothTerm(variable=1, n_basis=10),   # s(x1) - nonlinear
   ]
   parametric_terms = [
       ParametricTerm(variable=2),            # x2 - linear
   ]

   result = fit_additive_gam(X, y, smooth_terms, parametric_terms)
   print(result.summary())

.. _gam-formula:

Formula interface
-----------------

The easiest way to fit a GAM is with
:func:`~aurora.models.gam.formula.fit_gam_formula`:

.. code-block:: python

   from aurora.models.gam import fit_gam_formula

   # Using a dict with named variables
   data = {
       "y": y,
       "temperature": X[:, 0],
       "pressure": X[:, 1],
       "humidity": X[:, 2],
   }
   result = fit_gam_formula("y ~ s(temperature) + s(pressure) + humidity", data)
   print(result.summary())

   # Using a 2D array with column indices
   M = np.column_stack([y, X])
   result = fit_gam_formula("0 ~ s(1) + s(2) + 3", M)

See :doc:`/guide/formula` for the complete formula syntax reference.

.. _gam-tensor:

Tensor product smooths
----------------------

Tensor products model interactions between variables using ``te()``:

.. code-block:: text

   y ~ te(x1, x2) + x3

This creates a 2D smooth surface of ``x1`` and ``x2``, equivalent to
``te()`` in R's mgcv package.

.. _gam-visualization:

Visualization
-------------

Plot smooth functions and diagnostic plots:

.. code-block:: python

   from aurora.visualization import plot_smooth, plot_all_smooths

   # Plot a single smooth term
   fig = plot_smooth(result, term_index=0)

   # Plot all smooth terms
   figs = plot_all_smooths(result)

   # Diagnostic plots (residuals, QQ, scale-location)
   from aurora.visualization import plot_diagnostics
   fig = plot_diagnostics(result)

.. _gam-sparse:

Sparse matrices
---------------

For large problems (n > 500, n_basis > 20), enable sparse operations:

.. code-block:: python

   # Large dataset with many basis functions
   result = fit_gam(x, y, n_basis=30, use_sparse=True)

Sparse operations use CSR format internally and dispatch to efficient
solvers. Only available for ``"bspline"`` basis type.
