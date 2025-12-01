"""Backend detection and utilities for autodiff."""
from __future__ import annotations

from typing import Any
import numpy as np

ArrayLike = Any


def detect_backend(x: ArrayLike) -> str:
    """Detect the computational backend from array type.

    Parameters
    ----------
    x : array-like
        Input array

    Returns
    -------
    str
        One of 'numpy', 'torch', 'jax'
    """
    type_name = type(x).__module__

    if 'torch' in type_name:
        return 'torch'
    elif 'jax' in type_name or 'jaxlib' in type_name:
        return 'jax'
    else:
        return 'numpy'


def get_backend_module(backend: str):
    """Get the backend module for array operations.

    Parameters
    ----------
    backend : str
        Backend name ('numpy', 'torch', 'jax')

    Returns
    -------
    module
        The backend module (numpy, torch, or jax.numpy)
    """
    if backend == 'torch':
        import torch
        return torch
    elif backend == 'jax':
        import jax.numpy as jnp
        return jnp
    else:
        return np
