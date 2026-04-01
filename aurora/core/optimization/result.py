# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Optimization result container."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import Array, Scalar


@dataclass(frozen=True)
class OptimizationResult:
    """Container for optimization results.

    Attributes
    ----------
    x : Array
        Optimal parameter vector found by the optimiser.
    fun : Scalar
        Objective function value at ``x``.
    grad : Array or None
        Gradient of the objective at ``x`` (if available).
    hess : Array or None
        Hessian of the objective at ``x`` (if available).
    success : bool, default=True
        Whether the optimiser converged successfully.
    message : str
        Human-readable description of the termination reason.
    nit : int, default=0
        Number of iterations performed.
    nfev : int, default=0
        Number of objective function evaluations.
    njev : int, default=0
        Number of gradient evaluations.
    nhev : int, default=0
        Number of Hessian evaluations.
    backtrack_iterations : int, default=0
        Number of backtracking (step-halving) reductions performed.
    condition_number : float or None
        Maximum observed condition number of the weighted normal equations
        (IRLS) or the Hessian (Newton). ``None`` when not monitored.
    """

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
    backtrack_iterations: int = 0
    condition_number: float | None = None

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
