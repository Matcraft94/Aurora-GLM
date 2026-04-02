# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""NumPy numerical backend implementation.

NumPy is the default, always-available backend. It provides basic array
operations but no automatic differentiation, JIT compilation, or GPU support.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np


class NumPyBackend:
    """Thin wrapper around NumPy exposing the Aurora backend protocol.

    NumPy has no autodiff, JIT, or GPU support. Methods like ``grad()``
    and ``jit()`` return the input function unchanged so that multi-backend
    code can call them without branching.
    """

    def array(self, data: Any, dtype: Any | None = None):
        return np.array(data, dtype=dtype)

    def as_numpy(self, data: Any):
        return np.asarray(data)

    def grad(self, func: Callable):
        raise NotImplementedError("NumPy backend does not support automatic differentiation")

    def jit(self, func: Callable):
        return func  # no-op

    def device_put(self, data: Any):
        return np.asarray(data)  # NumPy only has CPU

    def vmap(self, func: Callable, *, in_axes=0, out_axes=0):  # noqa: ANN001
        raise NotImplementedError("NumPy backend does not support vmap")

    def partial(self, func: Callable, *args: Any, **kwargs: Any) -> Callable:
        return partial(func, *args, **kwargs)


def create_backend() -> NumPyBackend:
    """Factory used by the backend registry."""
    return NumPyBackend()


__all__ = ["NumPyBackend", "create_backend"]
