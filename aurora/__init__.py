"""Aurora-GLM: A modular framework for generalized linear modeling."""
from __future__ import annotations

from .core.backends import available_backends, get_backend, register_backend

__all__ = [
    "available_backends",
    "get_backend",
    "register_backend",
]

__version__ = "0.2.0-dev"
