# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Evaluation metrics for model assessment.

This sub-package provides a collection of metrics for evaluating statistical
models fitted with Aurora-GLM. Metrics are organised into three modules:

**Regression metrics** (``regression``):
- ``mean_squared_error`` -- Weighted MSE (optionally squared root).
- ``mean_absolute_error`` -- Weighted MAE.
- ``root_mean_squared_error`` -- Weighted RMSE.
- ``r_squared`` -- Coefficient of determination.

**Classification metrics** (``classification``):
- ``accuracy_score`` -- Classification accuracy.
- ``log_loss`` -- Negative log-likelihood for probabilistic predictions.
- ``brier_score_loss`` -- Brier score for binary probabilities.
- ``concordance_index`` -- Concordance index (c-statistic).
- ``roc_auc`` -- Area under the ROC curve.
- ``confusion_matrix`` -- Binary confusion matrix.
- ``precision``, ``recall``, ``f1_score`` -- Binary classification metrics.

**GLM-specific metrics** (``glm``):
- ``generalized_deviance`` -- Deviance under a specified family.
- ``aic`` -- Akaike Information Criterion.
- ``bic`` -- Bayesian Information Criterion.
- ``pseudo_r2`` -- Pseudo R-squared (McFadden, Cox & Snell, Nagelkerke).
"""

from .classification import (
    accuracy_score,
    brier_score_loss,
    concordance_index,
    confusion_matrix,
    f1_score,
    log_loss,
    precision,
    recall,
    roc_auc,
)
from .glm import aic, bic, generalized_deviance, pseudo_r2
from .regression import (
    mean_absolute_error,
    mean_squared_error,
    r_squared,
    root_mean_squared_error,
)

__all__ = [
    "accuracy_score",
    "brier_score_loss",
    "concordance_index",
    "confusion_matrix",
    "f1_score",
    "log_loss",
    "precision",
    "recall",
    "roc_auc",
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "r_squared",
    "generalized_deviance",
    "aic",
    "bic",
    "pseudo_r2",
]
