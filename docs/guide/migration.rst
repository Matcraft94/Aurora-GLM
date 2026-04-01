.. _guide-migration:

Migration Guide
################

This guide helps users transition from statsmodels (Python) and R packages
(mgcv, lme4) to Aurora-GLM.

.. _migration-statsmodels:

From statsmodels
=================

API comparison:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45
   :stub-columns: 0

   * - Task
     - statsmodels
     - Aurora-GLM
   * - Gaussian GLM
     - ``sm.OLS(y, X).fit()``
     - ``fit_glm(X, y, family="gaussian")``
   * - Logistic regression
     - ``sm.Logit(y, X).fit()``
     - ``fit_glm(X, y, family="binomial", link="logit")``
   * - Poisson regression
     - ``sm.Poisson(y, X).fit()``
     - ``fit_glm(X, y, family="poisson")``
   * - Gamma GLM
     - ``sm.GLM(y, X, family=sm.families.Gamma()).fit()``
     - ``fit_glm(X, y, family="gamma")``
   * - Standard errors
     - ``result.bse``
     - ``result.std_errors_``
   * - P-values
     - ``result.pvalues``
     - ``result.p_values_``
   * - Predict
     - ``result.predict(X_new)``
     - ``result.predict(X_new)``
   * - AIC
     - ``result.aic``
     - ``result.aic_``

Key differences:

- Aurora-GLM uses ``fit_glm()`` for all families; statsmodels has separate
  classes (``sm.OLS``, ``sm.Logit``, etc.)
- Aurora result attributes use trailing underscores (``aic_`` vs ``aic``)
- Aurora uses ``family=`` and ``link=`` string shortcuts; statsmodels requires
  importing family objects

.. code-block:: python

   # statsmodels
   import statsmodels.api as sm
   model = sm.GLM(y, sm.add_constant(X), family=sm.families.Gamma(sm.families.links.log()))
   result_sm = model.fit()

   # Aurora-GLM
   from aurora.models.glm import fit_glm
   result = fit_glm(X, y, family="gamma", link="log")

.. _migration-mgcv:

From R mgcv
===========

GAM comparison:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45
   :stub-columns: 0

   * - Task
     - R mgcv
     - Aurora-GLM
   * - Single smooth
     - ``gam(y ~ s(x), data=df)``
     - ``fit_gam(x, y, n_basis=10)``
   * - Multiple smooths
     - ``gam(y ~ s(x1) + s(x2), data=df)``
     - ``fit_gam_formula("y ~ s(x1) + s(x2)", data)``
   * - Mixed terms
     - ``gam(y ~ s(x1) + x2, data=df)``
     - ``fit_gam_formula("y ~ s(x1) + x2", data)``
   * - Tensor product
     - ``gam(y ~ te(x1, x2), data=df)``
     - ``fit_gam_formula("y ~ te(x1, x2)", data)``
   * - REML selection
     - ``gam(..., method="REML")``
     - ``fit_additive_gam(..., method="REML")``
   * - Basis dimension
     - ``s(x, k=10)``
     - ``s(x, k=10)`` (identical syntax)
   * - Plot smooth
     - ``plot(model)``
     - ``plot_smooth(result)``

.. code-block:: r

   # R mgcv
   library(mgcv)
   m <- gam(y ~ s(x1, k=10) + s(x2) + x3, data = df, method = "REML")
   summary(m)

.. code-block:: python

   # Aurora-GLM
   from aurora.models.gam import fit_gam_formula
   result = fit_gam_formula("y ~ s(x1, k=10) + s(x2) + x3", df, method="REML")
   print(result.summary())

.. _migration-lme4:

From R lme4
===========

GAMM / mixed model comparison:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45
   :stub-columns: 0

   * - Task
     - R lme4
     - Aurora-GLM
   * - Random intercept
     - ``lmer(y ~ x + (1|subject))``
     - ``fit_gamm("y ~ x + (1 | subject)", data)``
   * - Random slope
     - ``lmer(y ~ x + (x|subject))``
     - ``fit_gamm("y ~ x + (x | subject)", data)``
   * - Intercept + slope
     - ``lmer(y ~ x + (1+x|subject))``
     - ``fit_gamm("y ~ x + (1 + x | subject)", data)``
   * - Nested effects
     - ``lmer(y ~ x + (1|a/b))``
     - ``fit_gamm("y ~ x + (1 | a/b)", data)``
   * - Crossed effects
     - ``lmer(y ~ x + (1|a) + (1|b))``
     - ``fit_gamm("y ~ x + (1 | a) + (1 | b)", data)``
   * - Variance
     - ``VarCorr(m)``
     - ``result.variance_components``
   * - Fixed effects
     - ``fixef(m)``
     - ``result.beta_parametric``
   * - Predict
     - ``predict(m, newdata)``
     - ``predict_from_gamm(result, X_new)``

.. code-block:: r

   # R lme4
   library(lme4)
   m <- lmer(y ~ x + (1 + time | subject), data = df, REML = TRUE)
   summary(m)

.. code-block:: python

   # Aurora-GLM
   from aurora.models.gamm import fit_gamm
   result = fit_gamm(
       formula="y ~ x + (1 + time | subject)",
       data=df,
       covariance="unstructured",
   )
   print(result.summary())

.. _migration-validation:

Numerical validation
=====================

Aurora-GLM is validated against R and statsmodels to within 1e-6 tolerance:

- GLM coefficients match ``statsmodels`` and R's ``glm()``
- GAM smooth terms match R's ``mgcv::gam()``
- GAMM variance components match R's ``lme4::lmer()``
- All 520+ tests in the test suite verify this agreement

If results differ between Aurora and R, check:

1. Convergence tolerance (try decreasing ``tol``)
2. Number of IRLS iterations (increase ``max_iter``)
3. Basis dimension (try increasing ``n_basis`` or ``k=``)
4. Smoothing parameter (compare ``lambda_`` vs R's ``sp``)
