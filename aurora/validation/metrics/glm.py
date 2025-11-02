"""GLM-specific validation metrics."""
from __future__ import annotations

from ...models.base import GLMResult


def pseudo_r2(result: GLMResult, *, method: str = "mcfadden") -> float:
    """Compute pseudo :math:`R^2` scores from a fitted GLM result."""

    method_key = method.lower()
    deviance = float(result.deviance_)
    null_deviance = float(result.null_deviance_)

    if method_key == "mcfadden":
        if null_deviance <= 0.0:
            raise ValueError("null deviance must be positive for McFadden pseudo R^2")
        ratio = 1.0 - deviance / null_deviance
        return max(min(ratio, 1.0), 0.0)

    if method_key == "deviance":
        if null_deviance <= 0.0:
            raise ValueError("null deviance must be positive for deviance pseudo R^2")
        ratio = 1.0 - deviance / null_deviance
        return max(min(ratio, 1.0), 0.0)

    raise NotImplementedError(f"Unsupported pseudo R^2 method: {method!r}")


__all__ = ["pseudo_r2"]
