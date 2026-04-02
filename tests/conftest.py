"""Shared pytest fixtures and utilities for multi-backend testing."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.core.backends import get_backend
from aurora.utils import BackendNotAvailableError

# Optional backend imports
try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False


# ============================================================================
# Array Conversion Utilities
# ============================================================================


def to_numpy(data):
    """Convert any backend array to NumPy.

    Parameters
    ----------
    data : array-like
        Array from any backend (NumPy, PyTorch, JAX)

    Returns
    -------
    ndarray
        NumPy array
    """
    if HAS_TORCH and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif HAS_JAX and isinstance(data, jnp.ndarray):
        return np.array(data)
    return np.asarray(data)


def as_backend_array(data, backend, device=None):
    """Convert data to specified backend array.

    Parameters
    ----------
    data : array-like
        Input data to convert
    backend : str
        Target backend: 'numpy', 'torch', or 'jax'
    device : str, optional
        Device for PyTorch (e.g., 'cpu', 'cuda')

    Returns
    -------
    array
        Array in the specified backend

    Raises
    ------
    pytest.skip
        If requested backend is not available
    ValueError
        If backend name is unknown
    """
    if backend == "numpy":
        return np.asarray(data, dtype=np.float64)
    elif backend == "torch":
        if not HAS_TORCH:
            pytest.skip("PyTorch not installed")
        tensor = torch.tensor(data, dtype=torch.float64)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    elif backend == "jax":
        if not HAS_JAX:
            pytest.skip("JAX not installed")
        # Enable float64 if needed
        if jax is not None:
            jax.config.update("jax_enable_x64", True)
        return jnp.array(data, dtype=jnp.float64)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def assert_arrays_close(actual, expected, backend=None, rtol=1e-5, atol=1e-8, err_msg=None):
    """Assert arrays are close, handling backend-specific conversion.

    Parameters
    ----------
    actual : array-like
        Actual values (any backend)
    expected : array-like
        Expected values (any backend)
    backend : str, optional
        Backend name for error message
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    err_msg : str, optional
        Custom error message
    """
    actual_np = to_numpy(actual)
    expected_np = to_numpy(expected) if not isinstance(expected, np.ndarray) else expected

    if err_msg is None and backend is not None:
        err_msg = f"Arrays not close on {backend} backend"

    np.testing.assert_allclose(actual_np, expected_np, rtol=rtol, atol=atol, err_msg=err_msg)


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture(params=["numpy", "torch", "jax"])
def backend(request):
    """Parametrize tests over available backends.

    Yields
    ------
    str
        Backend name: 'numpy', 'torch', or 'jax'

    Notes
    -----
    Automatically skips tests if the requested backend is not available.
    Enables JAX float64 mode when using JAX backend.
    """
    backend_name = request.param

    if backend_name == "torch" and not HAS_TORCH:
        pytest.skip("PyTorch not installed")
    if backend_name == "jax" and not HAS_JAX:
        pytest.skip("JAX not installed")

    # Enable JAX float64
    if backend_name == "jax" and jax is not None:
        jax.config.update("jax_enable_x64", True)

    return backend_name


@pytest.fixture(params=["numpy", "torch"])
def backend_no_jax(request):
    """Parametrize tests over NumPy and PyTorch only.

    Useful for tests that don't yet support JAX or have JAX-specific issues.

    Yields
    ------
    str
        Backend name: 'numpy' or 'torch'
    """
    backend_name = request.param

    if backend_name == "torch" and not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    return backend_name


def _require_pytorch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        raise BackendNotAvailableError("PyTorch not installed") from exc


@pytest.fixture(scope="session")
def pytorch_backend():
    """Provide the PyTorch backend instance or skip tests when unavailable."""
    try:
        _require_pytorch()
        return get_backend("pytorch")
    except BackendNotAvailableError:
        pytest.skip("PyTorch backend not available")
