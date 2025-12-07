# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Base optimizer abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import Array, LossFunction, OptimizationCallback
from .result import OptimizationResult


class Optimizer(ABC):
    """Abstract base class for Aurora optimizers."""

    @abstractmethod
    def minimize(
        self,
        loss_fn: LossFunction,
        init_params: Array,
        *,
        callback: OptimizationCallback | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Execute the optimization routine."""


__all__ = ["Optimizer"]
