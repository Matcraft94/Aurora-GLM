# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Model validation metrics and resampling strategies.

This module provides comprehensive tools for model evaluation:

Submodules
----------
cross_val
    Cross-validation utilities: KFold, StratifiedKFold, cross_val_score

metrics
    Evaluation metrics for regression, classification, and GLMs:
    - Regression: MSE, MAE, R², RMSE
    - Classification: accuracy, precision, recall, F1, AUC, log loss
    - GLM-specific: deviance, AIC, BIC, pseudo-R²

sensitivity
    Sensitivity analysis for model robustness (planned).

Examples
--------
>>> from aurora.validation import cross_val_score, KFold
>>> from aurora.validation.metrics import mean_squared_error, r_squared
>>>
>>> # Cross-validation
>>> cv = KFold(n_splits=5)
>>> scores = cross_val_score(fit_glm, X, y, cv=cv, scoring='mse')
>>>
>>> # Direct metric calculation
>>> mse = mean_squared_error(y_true, y_pred)
>>> r2 = r_squared(y_true, y_pred)
"""

from .cross_val import CrossValResult, KFold, StratifiedKFold, cross_val_score
from .metrics import mean_squared_error, accuracy_score, r_squared

__all__ = [
    "CrossValResult",
    "KFold",
    "StratifiedKFold",
    "cross_val_score",
    "mean_squared_error",
    "accuracy_score",
    "r_squared",
]
