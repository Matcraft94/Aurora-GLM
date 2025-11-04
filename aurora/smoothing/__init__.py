"""Smoothing components for additive models."""
from __future__ import annotations

from aurora.smoothing.tensor import (
    fit_tensor_product,
    tensor_product_basis,
    tensor_product_penalty,
)
from aurora.smoothing.thinplate import (
    fit_tps,
    select_knots,
    tps_basis,
    tps_penalty,
)

__all__ = [
    "tensor_product_basis",
    "tensor_product_penalty",
    "fit_tensor_product",
    "tps_basis",
    "tps_penalty",
    "fit_tps",
    "select_knots",
]