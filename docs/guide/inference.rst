.. _guide-inference:

Inference & Diagnostics
########################

Aurora-GLM provides Wald tests, likelihood ratio tests, confidence intervals,
diagnostic plots, and robust standard errors.

.. _inference-wald:

Wald tests
==========

.. code-block:: python

   import numpy as np
   from aurora.inference.hypothesis import wald_test

   # Test a single coefficient (contrast vector over [intercept, X0, X1])
   result = wald_test(glm_result, [0, 1, 0])

   # Joint test of all coefficients
   p = len(glm_result.coef_) + 1  # +1 for the intercept
   result = wald_test(glm_result, np.eye(p))

``wald_test`` takes a contrast vector (or matrix) and returns a dict with
``statistic``, ``p_value``, and ``df``.

.. _inference-ci:

Confidence intervals
====================

.. code-block:: python

   from aurora.inference.intervals import confidence_intervals

   ci = confidence_intervals(glm_result, level=0.95)
   print(ci.lower, ci.upper)

   # Also available via predict
   mean, lower, upper = glm_result.predict(X_new, interval="confidence", level=0.95)

.. note::

   Only **Wald** confidence intervals are implemented. Profile-likelihood
   and bootstrap (BCa) intervals are not currently available
   (``method="profile"`` raises ``NotImplementedError``).

.. _inference-robust:

Robust standard errors
======================

Sandwich (HC) estimators for heteroscedasticity-consistent inference,
using the GLM bread matrix and leverage values:

.. code-block:: python

   from aurora.inference.robust import robust_covariance

   # HC0 (White's estimator)
   cov_robust = robust_covariance(glm_result, hc_type="HC0")

   # HC3 (default, jackknife-like, recommended for small samples)
   cov_robust = robust_covariance(glm_result, hc_type="HC3")

Available types: ``HC0``, ``HC1``, ``HC2``, ``HC3``, ``HC4``.

.. _inference-diagnostics:

GLM diagnostics
===============

.. code-block:: python

   from aurora.inference.diagnostics import glm_diagnostics

   diag = glm_diagnostics(glm_result)

   # Available residuals and measures:
   # - Deviance residuals
   # - Pearson residuals
   # - Working residuals
   # - Leverage (hat values)
   # - Cook's distance
   # - DFBETAS

   print(diag.summary())

.. _inference-plots:

Diagnostic plots
================

.. code-block:: python

   # Full diagnostic panel (residuals, QQ, scale-location, leverage)
   # GLMResult carries its own plotting method:
   fig = glm_result.plot_diagnostics()

.. note::

   :func:`aurora.visualization.plot_diagnostics` and
   :func:`aurora.visualization.plot_diagnostics_panel` operate on
   ``GAMMResult`` objects; for GLM results use the
   ``glm_result.plot_diagnostics()`` method shown above.
