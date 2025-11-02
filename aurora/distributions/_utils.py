"""Utility helpers for distribution implementations."""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def is_torch(value) -> bool:
    """Return ``True`` when *value* is a torch tensor."""
    return torch is not None and isinstance(value, torch.Tensor)


def namespace(*values):
    """Return the numerical namespace (``torch`` or ``numpy``) for *values*."""
    for value in values:
        if is_torch(value):
            return torch  # type: ignore[return-value]
    return np


def as_namespace_array(value, xp, *, like=None):
    """Convert *value* to an array in the same namespace as *xp*."""
    if xp is torch:  # type: ignore[comparison-overlap]
        dtype = getattr(like, "dtype", torch.get_default_dtype())
        device = getattr(like, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return torch.as_tensor(value, dtype=dtype, device=device)
    dtype = getattr(like, "dtype", None)
    return np.asarray(value, dtype=dtype)


def ones_like(value):
    xp = namespace(value)
    if xp is torch:  # type: ignore[comparison-overlap]
        return torch.ones_like(value)
    return np.ones_like(value)


def clip_probability(prob, xp, eps: float = 1e-9):
    if xp is torch:  # type: ignore[comparison-overlap]
        return torch.clamp(prob, eps, 1.0 - eps)
    return np.clip(prob, eps, 1.0 - eps)


def log_factorial(value, xp):
    if xp is torch:  # type: ignore[comparison-overlap]
        return torch.lgamma(value + 1.0)
    return np.vectorize(lambda v: np.math.lgamma(v + 1.0))(value)
