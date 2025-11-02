"""Generalized Linear Model fitting routines."""
from __future__ import annotations

from typing import Any

from ...core.types import ArrayLike
from ..base import ModelResult


def fit_glm(
    design_matrix: ArrayLike,
    response: ArrayLike,
    *,
    family: Any = "gaussian",
    link: Any = "identity",
    optimizer: str = "irls",
    backend: str = "jax",
    max_iter: int = 100,
    tol: float = 1e-6,
    weights: ArrayLike | None = None,
    offset: ArrayLike | None = None,
) -> ModelResult:
    """Fit a GLM to the provided design matrix and response vector.

    This placeholder highlights the intended public API; the underlying numerical
    routine will be implemented in future iterations of the project roadmap.
    """
    raise NotImplementedError("GLM fitting is not implemented yet.")


def predict_glm(model: ModelResult, design_matrix: ArrayLike, *, backend: str | None = None) -> ArrayLike:
    """Generate predictions from a fitted GLM model."""
    selected_backend = backend or "jax"
    return model.predict(design_matrix, backend=selected_backend)


__all__ = ["fit_glm", "predict_glm"]
