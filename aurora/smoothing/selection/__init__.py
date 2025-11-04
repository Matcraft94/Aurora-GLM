"""Smoothing parameter selection strategies."""
from __future__ import annotations

from aurora.smoothing.selection.gcv import gcv_score, select_smoothing_parameter
from aurora.smoothing.selection.reml import (
    reml_score,
    select_multiple_smoothing_parameters_reml,
    select_smoothing_parameter_reml,
)

__all__ = [
    "gcv_score",
    "select_smoothing_parameter",
    "reml_score",
    "select_smoothing_parameter_reml",
    "select_multiple_smoothing_parameters_reml",
]