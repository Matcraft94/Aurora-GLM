"""Generalized Additive Mixed Model routines."""
from __future__ import annotations

from typing import Any

from ..base import ModelResult


def fit_gamm(
    formula: str,
    *,
    data: Any,
    family: Any = "gaussian",
    link: Any = "identity",
    backend: str = "jax",
    method: str = "REML",
    random_structure: str | None = None,
) -> ModelResult:
    """Fit a GAMM with optional random effects specification."""
    raise NotImplementedError("GAMM fitting is not implemented yet.")


__all__ = ["fit_gamm"]
