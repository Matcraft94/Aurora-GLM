.. _guide-inference:

Inference & Diagnostics
########################

Aurora-GLM provides Wald tests, likelihood ratio tests, confidence intervals,
diagnostic plots, and robust standard errors.

.. _inference-wald:

Wald tests
==========

.. code-block:: python

   from aurora.inference.hypothesis import wald_test

   # Test all coefficients (joint hypothesis)
   result = wald_test(glm_result)

   # Test specific coefficients
   result = wald_test(glm_result, indices=[1, 2])

.. _inference-ci:

Confidence intervals
====================

.. code-block:: python

   from aurora.inference.intervals import confidence_intervals

   ci = confidence_intervals(glm_result, level=0.95)
   print(ci)

   # Also available via predict
   mean, lower, upper = glm_result.predict(X_new, interval="confidence", level=0.95)

.. _inference-robust:

Robust standard errors
======================

Sandwich (HC) estimators for heteroscedasticity-consistent inference:

.. code-block:: python

   from aurora.inference.robust import robust_covariance

   # HC0 (White's estimator)
   cov_robust = robust_covariance(glm_result, type="HC0")

   # HC1 (default, small-sample correction)
   cov_robust = robust_covariance(glm_result, type="HC1")

   # HC3 (jackknife-like, most conservative)
   cov_robust = robust_covariance(glm_result, type="HC3")

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

   from aurora.visualization import plot_diagnostics, plot_diagnostics_panel

   # Single diagnostic plot
   fig = plot_diagnostics(glm_result)

   # Full diagnostic panel (residuals, QQ, scale-location, Cook's)
   figs = plot_diagnostics_panel(glm_result)
