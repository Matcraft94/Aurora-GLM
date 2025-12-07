# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Custom exception hierarchy for Aurora-GLM."""

from __future__ import annotations


class AuroraError(RuntimeError):
    """Base exception for all Aurora-GLM errors."""


class BackendNotAvailableError(AuroraError):
    """Raised when a requested numerical backend is not installed or cannot be initialized."""


class ConfigurationError(AuroraError):
    """Raised when user-provided configuration is invalid."""


__all__ = [
    "AuroraError",
    "BackendNotAvailableError",
    "ConfigurationError",
]
