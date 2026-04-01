.. SPDX-License-Identifier: MIT
.. Copyright (c) 2025 Lucy Eduardo Arias

Examples & Case Studies
========================

Real-world notebooks demonstrating Aurora-GLM across a range of statistical
modeling tasks. Each case study walks through data preparation, model fitting,
diagnostics, and interpretation using GLM, GAM, or GAMM workflows. All
notebooks are available on GitHub and can be run locally or in Google Colab.

.. _examples-glm:

Generalized Linear Models (GLM)
--------------------------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Insurance Pricing with Gamma GLM
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/01_insurance_pricing_gamma_glm.ipynb

      Model claim severity using a Gamma GLM with log link. Demonstrates
      how heavy-tailed insurance losses are handled by the Gamma family.

   .. grid-item-card:: Insurance Pricing (General)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/01_insurance_pricing.ipynb

      A broader look at insurance rating factors with GLM, comparing
      Gaussian, Gamma, and Tweedie families for premium estimation.

   .. grid-item-card:: French Motor Claims (Poisson / NegBin)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/09_french_motor_claims.ipynb

      Model claim frequency on the classic French motor dataset, comparing
      Poisson and Negative Binomial regressions and handling over-dispersion.

   .. grid-item-card:: Medical Insurance Costs
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/10_medical_insurance_costs.ipynb

      Predict individual medical charges using GLM with feature engineering,
      residual analysis, and model comparison.

   .. grid-item-card:: Telco Customer Churn
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/12_telco_customer_churn.ipynb

      Binary classification of customer churn using a Binomial GLM with
      logit link, including variable selection and ROC analysis.

   .. grid-item-card:: Restaurant Health Scores (Beta Regression)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/17_restaurant_health_beta_regression.ipynb

      Model bounded health-inspection scores on (0, 1) with a Beta
      regression, showcasing Aurora's support for non-standard families.

.. _examples-gam:

Generalized Additive Models (GAM)
----------------------------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Air Quality with GAM
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/02_air_quality_gam.ipynb

      Use smooth splines to capture non-linear relationships between air
      pollutants and health outcomes. Demonstrates basis selection and
      GCV smoothing-parameter estimation.

   .. grid-item-card:: Species Distribution Modeling
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/03_species_distribution.ipynb

      Predict species presence/absence with a Binomial GAM using thin-plate
      and tensor-product splines for spatial covariates.

   .. grid-item-card:: Bike-Sharing Demand Forecasting
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/11_bike_sharing_demand.ipynb

      Model hourly rental counts with GAM smooths for temperature, humidity,
      and time-of-day effects. Compares Poisson and Negative Binomial families.

   .. grid-item-card:: Breast Cancer Survival Analysis
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/13_breast_cancer_survival.ipynb

      Apply GAM to survival-style outcomes, using smooth terms to capture
      non-linear prognostic effects of clinical covariates.

   .. grid-item-card:: Wind Power Forecasting
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/14_wind_power_forecasting.ipynb

      Forecast wind turbine output with GAM smooths of wind speed and
      direction, demonstrating tensor-product interactions.

   .. grid-item-card:: US Accidents Severity
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/15_us_accidents_severity.ipynb

      Model accident severity levels using GAM with ordered-categorical or
      multinomial approaches and geographic smooths.

.. _examples-gamm:

Generalized Additive Mixed Models (GAMM)
-----------------------------------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Sleep Study (GAMM)
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/04_sleep_study_gamm.ipynb

      The classic sleep-study dataset with random intercepts and slopes for
      subjects. Demonstrates PQL estimation and variance-component inference.

   .. grid-item-card:: Clinical Trial Analysis
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/05_clinical_trial.ipynb

      Analyze a clinical trial with Gaussian mixed models, covering random
      treatment effects, BLUPs, and diagnostic caterpillar plots.

   .. grid-item-card:: Longitudinal Clinical Trial
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/06_clinical_trial_longitudinal.ipynb

      Repeated-measures analysis with AR(1) and unstructured covariance
      patterns, comparing REML fits and information criteria.

   .. grid-item-card:: Psychometric Crossed Effects
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/07_psychometric_crossed_effects.ipynb

      Fit crossed random effects for subjects and items in a psychometric
      experiment, illustrating large sparse mixed-model computation.

   .. grid-item-card:: Educational Multilevel Models
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/08_educational_multilevel.ipynb

      Three-level hierarchical model with students nested in classes nested
      in schools, demonstrating variance partitioning and random slopes.

.. _examples-applied:

Applied & Domain-Specific
--------------------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: E-Commerce Conversion Optimization
      :link: https://github.com/Matcraft94/Aurora-GLM/tree/main/examples/06_case_studies/16_ecommerce_conversion_optimization.ipynb

      Model conversion rates with Binomial GLM/GAM, using smooth seasonality
      terms and interaction effects to optimize marketing spend.
