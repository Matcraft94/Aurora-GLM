"""Dataclasses encapsulating fitted model state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.types import ArrayLike


@dataclass(frozen=True)
class ModelResult:
    """Immutable container storing the outcome of a model fitting routine."""

    params: ArrayLike
    fitted_values: ArrayLike
    converged: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary representation."""
        return {
            "params": self.params,
            "converged": self.converged,
            "diagnostics": self.diagnostics,
        }

    def predict(self, design_matrix: ArrayLike, *, backend: str = "jax") -> ArrayLike:
        """Dispatch to the selected backend to generate predictions."""
        from ...core.backends import get_backend

        backend_impl = get_backend(backend)
        return backend_impl.array(design_matrix) @ backend_impl.array(self.params)


__all__ = ["ModelResult"]
