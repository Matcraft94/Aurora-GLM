"""High-level modeling interface for Aurora-GLM."""
from __future__ import annotations

from .base import GLMResult, ModelResult
from .gam import fit_gam
from .glm import fit_glm, predict_glm

# GAMM components (fit_gamm not yet implemented)
# from .gamm import fit_gamm

__all__ = ["ModelResult", "GLMResult", "fit_gam", "fit_glm", "predict_glm"]
