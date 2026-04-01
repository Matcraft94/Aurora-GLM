.. _guide-glm:

Generalized Linear Models (GLM)
################################

This chapter covers GLM fitting in detail: distribution families, link functions,
diagnostics, prediction, and inference.

.. _glm-overview:

Overview
========

A Generalized Linear Model relates a response variable to predictors through
a link function:

.. math::

   g(\mathbb{E}[Y]) = X \beta

Aurora-GLM fits GLMs using Iteratively Reweighted Least Squares (IRLS).
The main entry point is :func:`~aurora.models.glm.fitting.fit_glm`:

.. code-block:: python

   from aurora.models.glm import fit_glm

   result = fit_glm(
       X,                     # Design matrix (n, p)
       y,                     # Response vector (n,)
       family="gaussian",     # Distribution family
       link=None,             # Link function (default: family canonical)
       weights=None,          # Sample weights
       offset=None,           # Offset term
       max_iter=25,           # Max IRLS iterations
       tol=1e-8,             # Convergence tolerance
       fit_intercept=True,    # Include intercept
   )

The ``family`` parameter accepts either a string or a :class:`~aurora.distributions.base.Family` instance.
The ``link`` parameter works the same way.

.. _glm-families:

Distribution families
=====================

Aurora-GLM supports 10+ exponential family distributions:

.. list-table::
   :header-rows: 1
   :widths: 20 30 30 20
   :stub-columns: 0

   * - Family
     - Use case
     - Variance function
     - Canonical link
   * - :class:`~aurora.distributions.families.gaussian.GaussianFamily`
     - Continuous outcomes
     - V(mu) = 1
     - Identity
   * - :class:`~aurora.distributions.families.poisson.PoissonFamily`
     - Count data
     - V(mu) = mu
     - Log
   * - :class:`~aurora.distributions.families.binomial.BinomialFamily`
     - Binary/proportion data
     - V(mu) = mu(1-mu)
     - Logit
   * - :class:`~aurora.distributions.families.gamma.GammaFamily`
     - Positive continuous
     - V(mu) = mu^2
     - Inverse
   * - :class:`~aurora.distributions.families.beta.BetaFamily`
     - Proportions in (0,1)
     - Inverse quadratic
     - Logit
   * - :class:`~aurora.distributions.families.inverse_gaussian.InverseGaussianFamily`
     - Positive durations
     - V(mu) = mu^3
     - InverseSquare
   * - :class:`~aurora.distributions.families.negative_binomial.NegativeBinomialFamily`
     - Overdispersed counts
     - V(mu) = mu + mu^2/k
     - Log
   * - :class:`~aurora.distributions.families.student_t.StudentTFamily`
     - Robust regression
     - Heavy-tailed
     - Identity
   * - :class:`~aurora.distributions.families.tweedie.TweedieFamily`
     - Zero-inflated continuous
     - Variance power law
     - Log

Families can be specified by string or instance:

.. code-block:: python

   from aurora.distributions.families import GaussianFamily, PoissonFamily

   # String form (uses default link)
   result = fit_glm(X, y, family="poisson")

   # Instance form (uses default link)
   result = fit_glm(X, y, family=PoissonFamily())

.. _glm-links:

Link functions
==============

Nine link functions connect the mean to the linear predictor:

.. list-table::
   :header-rows: 1
   :widths: 20 25 35 20
   :stub-columns: 0

   * - Link
     - Formula
     - Typical use
     - Inverse
   * - :class:`~aurora.distributions.links.common.IdentityLink`
     - g(mu) = mu
     - Gaussian (OLS)
     - g^-1(eta) = eta
   * - :class:`~aurora.distributions.links.common.LogLink`
     - g(mu) = log(mu)
     - Poisson, Gamma
     - g^-1(eta) = exp(eta)
   * - :class:`~aurora.distributions.links.common.LogitLink`
     - g(mu) = log(mu/(1-mu))
     - Binomial
     - g^-1(eta) = 1/(1+exp(-eta))
   * - :class:`~aurora.distributions.links.common.ProbitLink`
     - g(mu) = Phi^-1(mu)
     - Binomial (probit)
     - g^-1(eta) = Phi(eta)
   * - :class:`~aurora.distributions.links.common.CLogLogLink`
     - g(mu) = log(-log(1-mu))
     - Binomial (extreme)
     - g^-1(eta) = 1-exp(-exp(eta))
   * - :class:`~aurora.distributions.links.common.InverseLink`
     - g(mu) = 1/mu
     - Gamma (canonical)
     - g^-1(eta) = 1/eta
   * - :class:`~aurora.distributions.links.common.InverseSquareLink`
     - g(mu) = 1/mu^2
     - Inverse Gaussian
     - g^-1(eta) = 1/sqrt(eta)
   * - :class:`~aurora.distributions.links.common.SqrtLink`
     - g(mu) = sqrt(mu)
     - Count data alt
     - g^-1(eta) = eta^2
   * - :class:`~aurora.distributions.links.common.PowerLink`
     - g(mu) = mu^p
     - General purpose
     - g^-1(eta) = eta^(1/p)

Override the default link:

.. code-block:: python

   from aurora.distributions.links import LogLink, LogitLink

   # Poisson with identity link (instead of canonical log)
   result = fit_glm(X, y, family="poisson", link=LogLink())

   # Binomial with probit link (instead of canonical logit)
   result = fit_glm(X, y, family="binomial", link="probit")

.. _glm-inference:

Inference
=========

Standard errors, p-values, and confidence intervals are computed lazily
on first access:

.. code-block:: python

   result = fit_glm(X, y)

   # Standard errors (computed from Fisher information)
   print(result.std_errors_)

   # Wald p-values (two-sided)
   print(result.p_values_)

   # Coefficient covariance matrix
   print(result.coef_cov_)

   # Confidence intervals
   mean, lower, upper = result.predict(X_new, interval="confidence", level=0.95)

.. _glm-diagnostics:

Diagnostics
===========

The :func:`~aurora.inference.diagnostics.glm.glm_diagnostics` function
computes comprehensive residual diagnostics:

.. code-block:: python

   from aurora.inference.diagnostics import glm_diagnostics

   diag = glm_diagnostics(result)
   print(diag.summary())

Available diagnostics include deviance residuals, Pearson residuals,
leverage values, Cook's distance, and DFBETAS.

.. _glm-prediction:

Prediction
==========

The ``.predict()`` method supports both the response and link scale:

.. code-block:: python

   # Predict on response scale (default)
   y_pred = result.predict(X_new)

   # Predict on link scale
   eta_pred = result.predict(X_new, type="link")

   # With 95% Wald confidence intervals
   mean, lower, upper = result.predict(X_new, interval="confidence", level=0.95)

.. _glm-comparison:

Comparison with statsmodels
===========================

Aurora-GLM produces results validated against both statsmodels and R:

.. code-block:: python

   import numpy as np
   from aurora.models.glm import fit_glm

   # Aurora-GLM
   X = np.random.randn(200, 3)
   y = 1.5 + X @ [0.8, -0.5, 0.3] + np.random.randn(200) * 0.5
   result = fit_glm(X, y)

   # Results match statsmodels within 1e-6 tolerance
   # Coefficients, deviance, AIC all agree with R's glm()
