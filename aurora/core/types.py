"""Common type aliases used across Aurora-GLM core modules."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ArrayLike(Protocol):
    """Runtime protocol for array-like objects."""

    def __array__(self, dtype: Any | None = None) -> Any:  # pragma: no cover - structural typing hook
        ...


Numeric = float | int


__all__ = ["ArrayLike", "Numeric"]
