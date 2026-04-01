.. _guide-quickstart:

Quickstart
##########

Five minutes to a working model.

All examples use ``numpy`` arrays; for real projects you can also use
``pandas.DataFrame`` with named columns.

.. _quickstart-glm:

Gaussian GLM
============

A simple linear regression (ordinary least squares) is a special case of GLM:

.. code-block:: python

   import numpy as np
   from aurora.models.glm import fit_glm

   # Generate data
   np.random.seed(42)
   n = 200
   X = np.random.randn(n, 3)
   y = 1.5 + 0.8 * X[:, 0] - 1.2 * X[:, 1] + 0.3 * X[:, 2] + np.random.randn(n) * 0.5

   # Fit a Gaussian GLM
   result = fit_glm(X, y, family="gaussian")

   # Inspect results
   print(result.summary())
   print(f"Deviance: {result.deviance_:.4f}")
   print(f"AIC:     {result.aic_:.4f}")

The ``family`` parameter defaults to ``"gaussian"`` so the call can be
shortened to:

.. code-block:: python

   result = fit_glm(X, y)

.. _quickstart-poisson:

Poisson GLM
===========

Count data with a log link (Poisson regression):

.. code-block:: python

   from aurora.models.glm import fit_glm

   # Generate count data
   np.random.seed(42)
   n = 300
   X = np.random.randn(n, 2)
   lin_pred = 0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1]
   mu = np.exp(lin_pred)
   y = np.random.poisson(mu)

   # Fit Poisson GLM
   result = fit_glm(X, y, family="poisson")

   print(f"Deviance: {result.deviance_:.4f}")
   print(f"AIC:     {result.aic_:.4f}")

   # Predictions on new data
   X_new = np.random.randn(5, 2)
   y_pred = result.predict(X_new)

.. _quickstart-gam:

Basic GAM
=========

Fit a smooth curve with automatic smoothing parameter selection:

.. code-block:: python

   from aurora.models.gam import fit_gam

   # Generate data with a nonlinear trend
   np.random.seed(42)
   x = np.linspace(0, 2 * np.pi, 200)
   y = np.sin(x) + 0.2 * np.random.randn(200)

   # Fit GAM with automatic smoothing (GCV)
   result = fit_gam(x, y, n_basis=15)

   print(f"EDF:    {result.edf:.2f}")
   print(f"Lambda: {result.lambda_:.4f}")

   # Predict at new points
   x_new = np.linspace(0, 2 * np.pi, 50)
   y_new = result.predict(x_new)

.. _quickstart-results:

Result objects
==============

Every fitting function returns a result object with a ``.summary()`` method:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30
   :stub-columns: 0

   * - Result type
     - Key attributes
     - Methods
   * - :class:`~aurora.models.glm.fitting.GLMResult`
     - ``coef_``, ``intercept_``, ``deviance_``, ``aic_``, ``bic_``, ``mu_``
     - ``summary()``, ``predict()``
   * - :class:`~aurora.models.gam.result.GAMResult`
     - ``coefficients``, ``fitted_values``, ``edf``, ``lambda_``, ``residuals``
     - ``predict()``
   * - :class:`~aurora.models.gam.additive.AdditiveGAMResult`
     - ``parametric_coef``, ``smooth_coef``, ``edf_values``, ``r_squared``
     - ``summary()``, ``predict()``
   * - :class:`~aurora.models.gamm.fitting.GAMMResult`
     - ``beta_parametric``, ``variance_components``, ``residual_variance``
     - ``summary()``, ``predict()``

.. code-block:: python

   # GLMResult example
   result = fit_glm(X, y, family="gaussian")
   print(result.summary())           # Formatted summary
   print(result.coef_)               # Coefficients (array)
   print(result.intercept_)          # Intercept (float or None)
   print(result.deviance_)           # Deviance
   print(result.aic_)                # AIC
   print(result.bic_)                # BIC
   print(result.converged_)          # Did IRLS converge?

   # Predictions
   y_pred = result.predict(X_new)    # On response scale
   y_link = result.predict(X_new, type="link")  # On link scale

   # Confidence intervals
   mean, lower, upper = result.predict(X_new, interval="confidence", level=0.95)

.. _quickstart-next:

Where to go next
================

- :doc:`/guide/glm` -- Full GLM tutorial with all families, links, and diagnostics
- :doc:`/guide/gam` -- GAM with splines, formula interface, and visualization
- :doc:`/guide/gamm` -- GAMM with random effects and covariance structures
- :doc:`/guide/formula` -- Complete formula syntax reference
- :doc:`/guide/backends` -- GPU acceleration with PyTorch and JAX
- :doc:`/guide/migration` -- Migrating from statsmodels or R
