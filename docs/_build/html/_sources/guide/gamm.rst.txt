.. _guide-gamm:

Generalized Additive Mixed Models (GAMM)
#########################################

GAMMs extend GAMs by adding random effects for hierarchical, longitudinal,
or clustered data:

.. math::

   y = X\beta + Zb + \epsilon, \quad b \sim N(0, \Psi)

This chapter covers random effects, covariance structures, estimation methods,
and the formula interface.

.. _gamm-quickstart:

Quick start
===========

Use :func:`~aurora.models.gamm.interface.fit_gamm` with a
formula string and a DataFrame:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from aurora.models.gamm import fit_gamm

   # Generate hierarchical data: 10 subjects, 20 observations each
   np.random.seed(42)
   n_groups, n_per = 10, 20
   n = n_groups * n_per

   data = pd.DataFrame({
       "y": np.random.randn(n),
       "x1": np.random.randn(n),
       "subject": np.repeat(np.arange(n_groups), n_per),
   })

   # Random intercept model
   result = fit_gamm(
       formula="y ~ x1 + (1 | subject)",
       data=data,
       covariance="identity",
   )

   print(result.summary())

   # Access results
   print(result.beta_parametric)         # Fixed effects
   print(result.variance_components)     # Random effect variance
   print(result.residual_variance)       # sigma^2
   print(result.aic)                     # AIC

.. _gamm-random:

Random effects
==============

Random effects are specified using
:class:`~aurora.models.gamm.random_effects.RandomEffect`:

.. code-block:: python

   from aurora.models.gamm import fit_gamm, RandomEffect

   # Random intercept: (1 | subject)
   re1 = RandomEffect(grouping="subject")

   # Random intercept + slope: (1 + time | subject)
   re2 = RandomEffect(grouping="subject", variables=("time",))

   # In formula mode, random effects are parsed automatically
   result = fit_gamm(formula="y ~ x + (1 + time | subject)", data=data)

.. _gamm-covariance:

Covariance structures
---------------------

Control how random effects co-vary:

.. list-table::
   :header-rows: 1
   :widths: 25 20 25 30
   :stub-columns: 0

   * - Structure
     - Parameters
     - Use case
     - R equivalent
   * - ``"identity"``
     - 1
     - Equal variance, no correlation
     - ``pdIdent()``
   * - ``"diagonal"``
     - q
     - Independent heterogeneous variances
     - ``pdDiag()``
   * - ``"unstructured"``
     - q(q+1)/2
     - Full covariance matrix
     - ``pdSym()``
   * - ``"ar1"``
     - 2
     - Temporal correlation, rho^|t-s|
     - ``corAR1()``
   * - ``"compound_symmetry"``
     - 2
     - Equal within-cluster correlation
     - ``corCompSymm()``
   * - ``"toeplitz"``
     - band+1
     - Banded temporal correlations
     - ``corSym()``
   * - ``"exponential"``
     - 2
     - Spatial exponential decay
     - Exp Cov
   * - ``"matern"``
     - 2
     - Spatial Matern covariance
     - Matern Cov

.. code-block:: python

   # AR(1) for longitudinal data
   result = fit_gamm(
       formula="y ~ time + (1 | subject)",
       data=data,
       covariance="ar1",
   )

   # Compound symmetry for clustered data
   result = fit_gamm(
       formula="y ~ x + (1 | clinic)",
       data=data,
       covariance="compound_symmetry",
   )

   # Unstructured for random intercept + slope
   result = fit_gamm(
       formula="y ~ x + (1 + x | subject)",
       data=data,
       covariance="unstructured",
   )

.. _gamm-estimation:

Estimation methods
------------------

**Gaussian family**: Direct REML estimation via Henderson's mixed model
equations. This is the default and recommended method.

**Non-Gaussian families**: Two approximations are available:

- **PQL (Penalized Quasi-Likelihood)**: Iteratively fits a working
  model using a linearized approximation.
- **Laplace approximation**: More accurate for small cluster sizes or binary
  outcomes.

.. code-block:: python

   # Gaussian (direct REML) - default, most accurate
   result = fit_gamm(formula="y ~ x + (1 | subject)", data=data, family="gaussian")

   # Poisson GAMM (uses PQL automatically)
   result = fit_gamm(formula="y ~ x + (1 | subject)", data=data, family="poisson")

.. _gamm-prediction:

Prediction
==========

.. code-block:: python

   # Population-level predictions (fixed effects only)
   pred_pop = result.predict(include_random=False)

   # Conditional predictions (fixed + random effects)
   pred_cond = result.predict(include_random=True)

   # New data prediction
   from aurora.models.gamm import predict_from_gamm

   X_new = np.column_stack([np.ones(20), np.random.randn(20)])
   pred = predict_from_gamm(result, X_new)

.. _gamm-with-smooth:

GAMM with smooth terms
-----------------------

Combine smooth terms and random effects in one model:

.. code-block:: python

   from aurora.models.gamm import fit_gamm

   # Smooth time trend + subject random intercept
   result = fit_gamm(
       formula="y ~ s(time) + (1 | subject)",
       data=data,
   )

This fits a penalized spline for the time effect alongside a
random intercept for each subject. The smoothing parameter and variance
components are estimated jointly by REML.

For direct control, use
:func:`~aurora.models.gamm.interface.fit_gamm_with_smooth`:

.. code-block:: python

   from aurora.models.gamm import fit_gamm_with_smooth, RandomEffect

   result = fit_gamm_with_smooth(
       y=y,
       X_parametric=np.ones((n, 1)),
       X_smooth=X_smooth,
       S_smooth=S_smooth,
       random_effects=[RandomEffect(grouping="subject")],
       groups_data={"subject": groups},
       lambda_smooth={"s(time)": 0.1},
   )
