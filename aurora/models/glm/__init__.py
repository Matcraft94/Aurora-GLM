"""Generalized Linear Model fitting routines."""
from __future__ import annotations

from ...core.types import ArrayLike
from ..base.result import GLMResult
from .fitting import fit_glm


def predict_glm(
    model: GLMResult,
    design_matrix: ArrayLike,
    *,
    backend: str | None = None,
    type: str = "response",
) -> ArrayLike:
    """Generate predictions from a fitted GLM model."""

    return model.predict(design_matrix, backend=backend, type=type)


__all__ = ["fit_glm", "predict_glm", "GLMResult"]
