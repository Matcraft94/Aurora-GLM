"""Optimization result container."""
from __future__ import annotations

from dataclasses import dataclass

from ..types import Array, Scalar


@dataclass(frozen=True)
class OptimizationResult:
    """Container for optimization results."""

    x: Array
    fun: Scalar
    grad: Array | None = None
    hess: Array | None = None
    success: bool = True
    message: str = "Optimization terminated successfully"
    nit: int = 0
    nfev: int = 0
    njev: int = 0
    nhev: int = 0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        status = "SUCCESS" if self.success else "FAILURE"
        return (
            "OptimizationResult(\n"
            f"  status={status},\n"
            f"  fun={self.fun:.6e},\n"
            f"  nit={self.nit},\n"
            f"  nfev={self.nfev},\n"
            f"  message='{self.message}'\n"
            ")"
        )


__all__ = ["OptimizationResult"]
