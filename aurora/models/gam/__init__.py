"""Generalized Additive Model routines."""
from __future__ import annotations

from typing import Any

from ..base import ModelResult


def fit_gam(
    formula: str,
    *,
    data: Any,
    family: Any = "gaussian",
    link: Any = "identity",
    backend: str = "jax",
    method: str = "REML",
) -> ModelResult:
    """Fit a GAM based on a formula specification."""
    raise NotImplementedError("GAM fitting is not implemented yet.")


__all__ = ["fit_gam"]
