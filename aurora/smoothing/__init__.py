"""Smoothing components for additive models."""
from __future__ import annotations

from aurora.smoothing.tensor import (
    fit_tensor_product,
    tensor_product_basis,
    tensor_product_penalty,
)

__all__ = [
    "tensor_product_basis",
    "tensor_product_penalty",
    "fit_tensor_product",
]