# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Local regression smoothing methods.

This module provides local regression methods including LOESS (Locally Estimated
Scatterplot Smoothing) and kernel regression.
"""

from __future__ import annotations

from .loess import LOESSSmoother, LOESSResult, loess

__all__ = [
    "LOESSSmoother",
    "LOESSResult",
    "loess",
]
