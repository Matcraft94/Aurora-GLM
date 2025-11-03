"""High-level modeling interface for Aurora-GLM."""
from __future__ import annotations

from .base import GLMResult, ModelResult
from .gam import fit_gam
from .gamm import fit_gamm
from .glm import fit_glm, predict_glm

__all__ = ["ModelResult", "GLMResult", "fit_gam", "fit_gamm", "fit_glm", "predict_glm"]
