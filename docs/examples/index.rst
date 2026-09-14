.. SPDX-License-Identifier: MIT
.. Copyright (c) 2025 Lucy Eduardo Arias

Examples & Case Studies
========================

Sixteen executable notebooks demonstrating Aurora-GLM across GLM, GAM, and
GAMM workflows. Each case study walks through data preparation, model
specification, fitting, diagnostics, and interpretation. Every model
mentioned in a notebook is actually fitted in it. All notebooks are
available on GitHub and can be run locally.

.. _examples-glm:

Generalized Linear Models (GLM)
--------------------------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Insurance Pricing (Gamma GLM)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/01_insurance_gamma_glm.ipynb

      Medical insurance charges with a Gamma GLM: log vs identity link
      comparison, age × BMI interaction, and risk segmentation.

   .. grid-item-card:: French Motor Claims (Poisson / Negative Binomial)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/02_french_motor_poisson_nb.ipynb

      Claim frequency with an exposure offset: overdispersion detection
      and Negative Binomial with ML-estimated dispersion.

   .. grid-item-card:: Telco Customer Churn (Binomial GLM)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/03_telco_churn_binomial_glm.ipynb

      Logistic regression with odds ratios, ROC analysis, and
      calibration diagnostics.

   .. grid-item-card:: Restaurant Health Scores (Beta GLM)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/04_restaurant_health_beta_glm.ipynb

      Beta regression for bounded scores, with spline regression for the
      non-linear age effect.

   .. grid-item-card:: Species Distribution (Poisson GLM)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/09_species_distribution_glm.ipynb

      Multi-model Poisson regression with rate ratios and nested-model
      comparison.

   .. grid-item-card:: US Accidents (Binomial GLM at scale)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/10_us_accidents_binomial_glm.ipynb

      Large-n binary classification with calibration and class-imbalance
      analysis.

.. _examples-gam:

Generalized Additive Models (GAM)
----------------------------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Air Quality with GAM
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/05_air_quality_gam.ipynb

      Additive model with GCV smoothing-parameter selection and
      identifiability constraints.

   .. grid-item-card:: Bike-Sharing Demand
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/06_bike_sharing_gam.ipynb

      Hourly rental counts with smooths for temperature, humidity, and
      time-of-day effects.

   .. grid-item-card:: Wind Power Forecasting
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/07_wind_power_gam.ipynb

      Gamma GLM baseline vs GAM on log(power), with a power-curve
      analysis.

   .. grid-item-card:: E-Commerce Conversion (Binomial GAMM with smooths)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/08_ecommerce_conversion_gam.ipynb

      Conversion rates with smooth price, duration, and hour effects
      (PQL) plus a random intercept per day-of-week.

.. _examples-gamm:

Generalized Additive Mixed Models (GAMM)
-----------------------------------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Sleep Study (GAMM)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/11_sleep_study_gamm.ipynb

      Random intercepts and slopes on the classic sleep-study dataset,
      with marginal vs conditional R².

   .. grid-item-card:: Longitudinal Clinical Trial
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/12_clinical_trial_longitudinal_gamm.ipynb

      Repeated measures with an identity vs AR(1) covariance comparison
      and effect sizes.

   .. grid-item-card:: Clinical Trial (Multilevel)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/13_clinical_trial_multilevel.ipynb

      Patients nested in clinics; the LPM-with-random-effects vs PQL
      trade-off under separation.

   .. grid-item-card:: Psychometric Crossed Effects
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/14_psychometric_crossed_gamm.ipynb

      Crossed random effects for subjects and items — the large sparse
      mixed-model case.

   .. grid-item-card:: Educational Multilevel Models (PISA UK)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/15_educational_multilevel_gamm.ipynb

      Eight models from null to random slopes on real PISA UK data, with
      variance partitioning.

   .. grid-item-card:: Breast Cancer Outcomes (Binomial GAMM, PQL)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/16_breast_cancer_binomial_gamm.ipynb

      Multi-center binary outcome via Penalized Quasi-Likelihood: odds
      ratios, latent-scale ICC, and PQL limitations.
