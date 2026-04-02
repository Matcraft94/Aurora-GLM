# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Base abstractions shared by Aurora-GLM models.

This module provides the unified result hierarchy for all model types:

    BaseResult (ABC)
    ├── LinearModelResult
    │   └── GLMResult
    │       └── GAMResult (in aurora.models.gam)
    └── MixedModelResultBase
        └── GAMMResult, PQLResult, LaplaceResult (in aurora.models.gamm)
"""

from __future__ import annotations

from .base_result import (
    BaseResult,
    LinearModelResult,
    MixedModelProtocol,
    MixedModelResultBase,
    ResultProtocol,
)
from .result import GLMResult, ModelResult

__all__ = [
    # Legacy (for backward compatibility)
    "ModelResult",
    "GLMResult",
    # New unified hierarchy
    "BaseResult",
    "LinearModelResult",
    "MixedModelResultBase",
    # Protocols
    "ResultProtocol",
    "MixedModelProtocol",
]
