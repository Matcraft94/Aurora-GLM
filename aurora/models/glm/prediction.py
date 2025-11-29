"""Prediction utilities for fitted GLM models.

This module provides functions for generating predictions from fitted GLM models,
including both response-scale and linear predictor predictions.

Functions
---------
predict_glm : Generate predictions from a fitted GLM model

Examples
--------
>>> from aurora.models import fit_glm, predict_glm
>>> result = fit_glm(X, y, family='gaussian')
>>>
>>> # Predict on new data (response scale)
>>> y_pred = predict_glm(result, X_new)
>>>
>>> # Predict on linear predictor scale
>>> eta_pred = predict_glm(result, X_new, type='link')
"""
from __future__ import annotations

from ...core.types import ArrayLike
from ..base.result import GLMResult


def predict_glm(
    model: GLMResult,
    design_matrix: ArrayLike,
    *,
    backend: str | None = None,
    type: str = "response",
) -> ArrayLike:
    """Generate predictions from a fitted GLM model.

    This is a convenience wrapper around the GLMResult.predict() method,
    providing a functional interface for predictions.

    Parameters
    ----------
    model : GLMResult
        A fitted GLM model result from fit_glm().
    design_matrix : ArrayLike
        Design matrix for prediction. Shape (n_samples, n_features).
    backend : str, optional
        Backend to use for computation ('numpy', 'torch', 'jax').
        If None, uses the backend from the fitted model.
    type : {'response', 'link'}, default='response'
        Type of prediction:
        - 'response': Predictions on the response scale (μ = g⁻¹(Xβ))
        - 'link': Predictions on the linear predictor scale (η = Xβ)

    Returns
    -------
    predictions : ArrayLike
        Predicted values. Shape (n_samples,).

    Examples
    --------
    >>> from aurora.models import fit_glm, predict_glm
    >>> from aurora import GaussianFamily
    >>>
    >>> # Fit a linear model
    >>> result = fit_glm(X_train, y_train, family=GaussianFamily())
    >>>
    >>> # Predict on test data
    >>> y_pred = predict_glm(result, X_test)
    >>>
    >>> # Get linear predictor values
    >>> eta = predict_glm(result, X_test, type='link')

    See Also
    --------
    fit_glm : Fit a Generalized Linear Model
    GLMResult.predict : Instance method for predictions
    """
    return model.predict(design_matrix, backend=backend, type=type)


__all__ = ["predict_glm"]
