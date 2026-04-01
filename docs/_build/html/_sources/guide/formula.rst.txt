.. _guide-formula:

Formula Syntax Reference
#########################

Aurora-GLM supports R/mgcv-compatible formula syntax for GAM and GAMM models.
The formula parser lives in :mod:`aurora.models.gam.formula`.

.. _formula-basic:

Basic syntax
============

Formulas follow the pattern ``response ~ terms``:

.. code-block:: text

   y ~ s(x1) + s(x2) + x3

- Left of ``~``: response variable name or column index
- Right of ``~``: predictor terms separated by ``+``
- ``s(...)``: smooth term
- bare names: parametric (linear) term

.. _formula-smooth:

Smooth terms
============

.. list-table::
   :header-rows: 1
   :widths: 40 60
   :stub-columns: 0

   * - Syntax
     - Meaning
   * - ``s(x)``
     - Smooth of x with default settings (B-spline, k=10)
   * - ``s(x, k=15)``
     - Smooth with 15 basis functions
   * - ``s(x, n_basis=12)``
     - Explicit n_basis parameter
   * - ``s(x, basis='cubic')``
     - Natural cubic spline basis
   * - ``s(x, k=10, m=3)``
     - B-spline with penalty order 3
   * - ``s(x, sp=0.1)``
     - Fixed smoothing parameter

.. code-block:: python

   from aurora.models.gam import fit_gam_formula

   # Default B-spline
   result = fit_gam_formula("y ~ s(x1)", data)

   # More basis functions for complex patterns
   result = fit_gam_formula("y ~ s(x1, k=20)", data)

   # Natural cubic spline
   result = fit_gam_formula("y ~ s(x1, basis='cubic')", data)

.. _formula-parametric:

Parametric terms
================

Bare variable names are treated as linear (parametric) predictors:

.. code-block:: text

   y ~ s(x1) + x2 + x3

Here ``x2`` and ``x3`` enter linearly, while ``x1`` has a smooth effect.
The intercept is included automatically.

.. _formula-random:

Random effects
==============

lme4-style random effects use the ``(effects | group)`` syntax:

.. list-table::
   :header-rows: 1
   :widths: 40 60
   :stub-columns: 0

   * - Syntax
     - Meaning
   * - ``(1 | subject)``
     - Random intercept for each subject
   * - ``(x | subject)``
     - Random slope of x (no intercept)
   * - ``(1 + x | subject)``
     - Random intercept and slope
   * - ``(x + y | subject)``
     - Two random slopes (no intercept)
   * - ``(1 | clinic/subject)``
     - Nested: subjects within clinics
   * - ``(1 | subject) + (1 | clinic)``
     - Crossed: subjects and clinics

.. code-block:: python

   from aurora.models.gamm import fit_gamm

   # Random intercept
   result = fit_gamm("y ~ x + (1 | subject)", data)

   # Random intercept + slope
   result = fit_gamm("y ~ x + (1 + x | subject)", data)

   # Nested random effects
   result = fit_gamm("y ~ x + (1 | clinic/subject)", data)

   # Crossed random effects
   result = fit_gamm("y ~ x + (1 | subject) + (1 | clinic)", data)

.. _formula-tensor:

Tensor products
===============

Tensor products model smooth interactions between two or more variables:

.. code-block:: text

   y ~ te(x1, x2) + x3

This creates a 2D smooth surface of ``x1`` and ``x2``, equivalent to
``te()`` in R's mgcv package.

.. _formula-data:

Data input modes
================

The formula interface accepts two data formats:

**Dict with named arrays:**

.. code-block:: python

   data = {"y": y_array, "temp": temp_array, "pressure": pressure_array}
   result = fit_gam_formula("y ~ s(temp) + s(pressure)", data)

**2D array with column indices:**

.. code-block:: python

   # Column 0 = response, columns 1-3 = predictors
   M = np.column_stack([y, x1, x2, x3])
   result = fit_gam_formula("0 ~ s(1) + s(2) + 3", M)

**pandas DataFrame** (recommended for GAMM):

.. code-block:: python

   import pandas as pd
   df = pd.DataFrame({"y": y, "x1": x1, "subject": subject})
   result = fit_gamm("y ~ x1 + (1 | subject)", df)

.. _formula-full:

Combined example
================

A GAMM formula combining all features:

.. code-block:: text

   y ~ s(time, k=15) + s(age, basis='cubic') + treatment + bmi + (1 + time | subject)

This specifies:

- ``s(time, k=15)``: smooth time effect with 15 basis functions
- ``s(age, basis='cubic')``: smooth age effect with natural cubic spline
- ``treatment``: parametric (linear) treatment effect
- ``bmi``: parametric BMI effect
- ``(1 + time | subject)``: random intercept and slope per subject
