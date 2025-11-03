"""Generalized Additive Model routines."""
from __future__ import annotations

from aurora.models.gam.fitting import fit_gam
from aurora.models.gam.result import GAMResult

__all__ = ["fit_gam", "GAMResult"]
