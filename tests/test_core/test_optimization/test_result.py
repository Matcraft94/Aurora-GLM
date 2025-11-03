"""Tests for the OptimizationResult container."""
from __future__ import annotations

import numpy as np

from aurora.core.optimization.result import OptimizationResult


def test_result_stores_attributes():
    params = np.array([1.0, 2.0])
    result = OptimizationResult(
        x=params,
        fun=0.5,
        grad=np.array([0.0, 0.0]),
        hess=np.eye(2),
        success=True,
        message="done",
        nit=3,
        nfev=5,
        njev=3,
        nhev=1,
    )

    assert result.x is params
    assert result.fun == 0.5
    assert result.success is True
    assert result.nit == 3
    assert "status=SUCCESS" in repr(result)
