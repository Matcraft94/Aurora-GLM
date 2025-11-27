"""Analysis of variance helpers.

This module provides ANOVA functionality for GLM and GAM models (planned).

Planned Features
----------------
- Type I, II, III sum of squares
- Sequential and partial tests for GAM smooth terms
- Likelihood ratio tests for comparing nested models
- F-tests and chi-squared tests

Notes
-----
Currently, model comparison can be done using:
- Deviance comparison from model results
- AIC/BIC from aurora.validation.metrics
- Wald tests from aurora.inference.hypothesis

Example (planned):
>>> from aurora.inference.anova import anova_glm
>>> result1 = fit_glm(X[:, :2], y)
>>> result2 = fit_glm(X, y)
>>> anova_table = anova_glm(result1, result2)
"""