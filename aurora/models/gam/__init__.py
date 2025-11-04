"""Generalized Additive Model routines."""
from __future__ import annotations

from aurora.models.gam.additive import AdditiveGAMResult, fit_additive_gam, fit_gam_formula
from aurora.models.gam.fitting import fit_gam
from aurora.models.gam.formula import FormulaSpec, parse_formula
from aurora.models.gam.plotting import plot_all_smooths, plot_smooth
from aurora.models.gam.result import GAMResult
from aurora.models.gam.terms import ParametricTerm, SmoothTerm, TensorTerm

__all__ = [
    "fit_gam",
    "GAMResult",
    "fit_additive_gam",
    "fit_gam_formula",
    "AdditiveGAMResult",
    "SmoothTerm",
    "ParametricTerm",
    "TensorTerm",
    "plot_smooth",
    "plot_all_smooths",
    "parse_formula",
    "FormulaSpec",
]
