# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Utility functions for autodiff."""

from __future__ import annotations

from typing import Callable, Any
import numpy as np

from .backends import detect_backend
from .gradient import gradient

ArrayLike = Any
ScalarFunc = Callable[..., float]


def check_gradient(
    func: ScalarFunc,
    x: ArrayLike,
    *args,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    **kwargs,
) -> dict:
    """Check gradient computation against numerical differentiation.

    Useful for debugging custom gradient implementations.

    Parameters
    ----------
    func : callable
        Scalar-valued function
    x : array-like
        Point at which to check
    *args, **kwargs : additional arguments to func
    rtol : float, default=1e-4
        Relative tolerance
    atol : float, default=1e-6
        Absolute tolerance

    Returns
    -------
    dict
        Contains 'analytic', 'numerical', 'max_abs_diff', 'max_rel_diff', 'passed'
    """
    backend = detect_backend(x)

    grad_fn = gradient(func, backend=backend)
    g_analytic = np.asarray(grad_fn(x, *args, **kwargs))

    grad_fn_num = gradient(func, backend="numpy")
    g_numerical = grad_fn_num(np.asarray(x), *args, **kwargs)

    abs_diff = np.abs(g_analytic - g_numerical)
    max_abs_diff = np.max(abs_diff)

    scale = np.maximum(np.abs(g_numerical), np.abs(g_analytic))
    scale = np.where(scale < atol, 1.0, scale)
    rel_diff = abs_diff / scale
    max_rel_diff = np.max(rel_diff)

    passed = max_abs_diff <= atol or max_rel_diff <= rtol

    return {
        "analytic": g_analytic,
        "numerical": g_numerical,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "passed": passed,
    }
