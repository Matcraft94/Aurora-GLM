# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Result class for fitted GAM models."""

from __future__ import annotations

from typing import Any

import numpy as np


class GAMResult:
    """Result of fitting a Generalized Additive Model.

    Parameters
    ----------
    coefficients : ndarray
        Spline coefficients.
    fitted_values : ndarray
        Fitted values at training data points.
    residuals : ndarray
        Residuals (y - fitted_values).
    lambda_ : float
        Smoothing parameter used.
    edf : float
        Effective degrees of freedom.
    basis : object
        Basis object (BSplineBasis or CubicSplineBasis).
    x : ndarray
        Training predictor values.
    y : ndarray
        Training response values.
    weights : ndarray, optional
        Observation weights.
    gcv_score : float, optional
        GCV score if lambda was selected automatically.

    Attributes
    ----------
    All parameters are stored as attributes.
    n_obs_ : int
        Number of observations.
    """

    def __init__(
        self,
        coefficients: np.ndarray,
        fitted_values: np.ndarray,
        residuals: np.ndarray,
        lambda_: float,
        edf: float,
        basis: Any,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
        gcv_score: float | None = None,
    ):
        self.coefficients = coefficients
        self.fitted_values = fitted_values
        self.residuals = residuals
        self.lambda_ = lambda_
        self.edf = edf
        self.basis = basis
        self.x = x
        self.y = y
        self.weights = weights
        self.gcv_score = gcv_score

        self.n_obs_ = len(x)

    @property
    def r_squared(self) -> float:
        """Coefficient of determination (R²).

        Returns
        -------
        r_squared : float
            R² value, proportion of variance explained by the model.
        """
        rss = np.sum(self.residuals**2)
        if self.weights is not None:
            rss = np.sum(self.weights * self.residuals**2)

        tss = np.sum((self.y - np.mean(self.y)) ** 2)
        return 1 - rss / tss

    @property
    def lambda_opt(self) -> float:
        """Alias for lambda_ (optimal smoothing parameter).

        Returns
        -------
        lambda_opt : float
            The smoothing parameter used in the fit.
        """
        return self.lambda_

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """Predict at new x values.

        Parameters
        ----------
        x_new : ndarray, shape (m,)
            New predictor values.

        Returns
        -------
        y_pred : ndarray, shape (m,)
            Predicted values.
        """
        x_new_arr = np.asarray(x_new, dtype=np.float64)

        if x_new_arr.ndim == 0:
            x_new_arr = x_new_arr.reshape(1)

        # Evaluate basis at new points
        X_new = self.basis.basis_matrix(x_new_arr)

        # Compute predictions
        y_pred = X_new @ self.coefficients

        return y_pred

    def summary(self) -> str:
        """Generate summary string of fit.

        Returns
        -------
        summary : str
            Formatted summary of model fit.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Generalized Additive Model (GAM) - Fitted Summary")
        lines.append("=" * 60)
        lines.append("")

        # Model information
        lines.append("Model Information:")
        lines.append(f"  Basis type:          {type(self.basis).__name__}")
        lines.append(f"  Number of basis:     {len(self.coefficients)}")
        lines.append(f"  Observations:        {self.n_obs_}")
        if self.weights is not None:
            lines.append("  Weighted fit:        Yes")
        lines.append("")

        # Smoothing information
        lines.append("Smoothing:")
        lines.append(f"  Lambda:              {self.lambda_:.6e}")
        lines.append(f"  Effective DoF:       {self.edf:.2f}")
        if self.gcv_score is not None:
            lines.append(f"  GCV score:           {self.gcv_score:.6e}")
        lines.append("")

        # Fit statistics
        rss = np.sum(self.residuals**2)
        if self.weights is not None:
            rss = np.sum(self.weights * self.residuals**2)

        tss = np.sum((self.y - np.mean(self.y)) ** 2)
        r_squared = 1 - rss / tss

        lines.append("Fit Statistics:")
        lines.append(f"  Residual sum sq:     {rss:.4f}")
        lines.append(f"  R-squared:           {r_squared:.4f}")
        lines.append(f"  Residual std:        {np.std(self.residuals):.4f}")
        lines.append("")

        # Residual diagnostics
        lines.append("Residuals:")
        lines.append(f"  Min:                 {np.min(self.residuals):.4f}")
        lines.append(f"  Q1:                  {np.quantile(self.residuals, 0.25):.4f}")
        lines.append(f"  Median:              {np.median(self.residuals):.4f}")
        lines.append(f"  Q3:                  {np.quantile(self.residuals, 0.75):.4f}")
        lines.append(f"  Max:                 {np.max(self.residuals):.4f}")
        lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"GAMResult(n_obs={self.n_obs_}, n_basis={len(self.coefficients)}, "
            f"edf={self.edf:.2f}, lambda={self.lambda_:.4e})"
        )


__all__ = ["GAMResult"]
