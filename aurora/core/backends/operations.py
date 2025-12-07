# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Backend-agnostic numerical operations for Aurora-GLM.

This module provides unified interfaces for linear algebra and array operations
that work across NumPy, PyTorch, and JAX backends.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]


def get_namespace(backend: str = "numpy", device: str | None = None):
    """Get the numerical namespace for the specified backend.

    Parameters
    ----------
    backend : str
        Backend name: 'numpy', 'torch'/'pytorch', or 'jax'
    device : str, optional
        Device for torch backend: 'cpu', 'cuda', 'cuda:0', etc.

    Returns
    -------
    namespace
        The numerical namespace (np, torch, or jnp)
    device_info
        Device information for tensor creation
    """
    backend = backend.lower()

    if backend == "numpy":
        return np, None

    elif backend in ("torch", "pytorch"):
        if torch is None:
            raise ImportError(
                "PyTorch is not installed. Install with: pip install torch"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch, torch.device(device)

    elif backend == "jax":
        if jnp is None:
            raise ImportError(
                "JAX is not installed. Install with: pip install jax jaxlib"
            )
        return jnp, None

    else:
        raise ValueError(f"Unknown backend: {backend}. Choose from: numpy, torch, jax")


def to_backend_array(data: Any, xp, device=None, dtype=None):
    """Convert data to array in the specified backend.

    Parameters
    ----------
    data : array-like
        Input data (numpy array, list, pandas Series, etc.)
    xp : module
        Numerical namespace (np, torch, jnp)
    device : torch.device, optional
        Device for PyTorch tensors
    dtype : dtype, optional
        Data type for the output array

    Returns
    -------
    array
        Array in the target backend
    """
    # Handle pandas objects
    if hasattr(data, "values"):
        data = data.values

    # Convert to numpy first if needed
    if hasattr(data, "detach"):  # torch tensor
        data = data.detach().cpu().numpy()
    elif hasattr(data, "device_buffer"):  # jax array
        data = np.asarray(data)

    # Convert to target backend
    if xp is np:
        result = np.asarray(data, dtype=dtype)
    elif xp is torch:
        if dtype is None:
            dtype = torch.float64
        result = torch.tensor(data, dtype=dtype, device=device)
    elif xp is jnp:
        if dtype is None:
            dtype = jnp.float64
        result = jnp.array(data, dtype=dtype)
    else:
        result = np.asarray(data, dtype=dtype)

    return result


def to_numpy(data: Any) -> np.ndarray:
    """Convert any backend array to NumPy.

    Parameters
    ----------
    data : array-like
        Input array from any backend

    Returns
    -------
    np.ndarray
        NumPy array
    """
    if isinstance(data, np.ndarray):
        return data
    elif torch is not None and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif jax is not None and hasattr(data, "device_buffer"):
        return np.asarray(data)
    else:
        return np.asarray(data)


# Linear algebra operations


def solve(A, b, xp):
    """Solve linear system Ax = b.

    Parameters
    ----------
    A : array
        Coefficient matrix
    b : array
        Right-hand side
    xp : module
        Numerical namespace

    Returns
    -------
    x : array
        Solution vector
    """
    if xp is np:
        return np.linalg.solve(A, b)
    elif xp is torch:
        return torch.linalg.solve(A, b)
    elif xp is jnp:
        return jnp.linalg.solve(A, b)
    else:
        return np.linalg.solve(A, b)


def cholesky(A, xp):
    """Compute Cholesky decomposition.

    Parameters
    ----------
    A : array
        Positive definite matrix
    xp : module
        Numerical namespace

    Returns
    -------
    L : array
        Lower triangular Cholesky factor
    """
    if xp is np:
        return np.linalg.cholesky(A)
    elif xp is torch:
        return torch.linalg.cholesky(A)
    elif xp is jnp:
        return jnp.linalg.cholesky(A)
    else:
        return np.linalg.cholesky(A)


def inv(A, xp):
    """Compute matrix inverse.

    Parameters
    ----------
    A : array
        Square matrix
    xp : module
        Numerical namespace

    Returns
    -------
    A_inv : array
        Inverse matrix
    """
    if xp is np:
        return np.linalg.inv(A)
    elif xp is torch:
        return torch.linalg.inv(A)
    elif xp is jnp:
        return jnp.linalg.inv(A)
    else:
        return np.linalg.inv(A)


def det(A, xp):
    """Compute matrix determinant.

    Parameters
    ----------
    A : array
        Square matrix
    xp : module
        Numerical namespace

    Returns
    -------
    det : scalar
        Determinant
    """
    if xp is np:
        return np.linalg.det(A)
    elif xp is torch:
        return torch.linalg.det(A)
    elif xp is jnp:
        return jnp.linalg.det(A)
    else:
        return np.linalg.det(A)


def slogdet(A, xp):
    """Compute sign and log-determinant.

    Parameters
    ----------
    A : array
        Square matrix
    xp : module
        Numerical namespace

    Returns
    -------
    sign : scalar
        Sign of determinant
    logabsdet : scalar
        Log of absolute determinant
    """
    if xp is np:
        return np.linalg.slogdet(A)
    elif xp is torch:
        return torch.linalg.slogdet(A)
    elif xp is jnp:
        return jnp.linalg.slogdet(A)
    else:
        return np.linalg.slogdet(A)


def eigh(A, xp):
    """Compute eigenvalues and eigenvectors of symmetric matrix.

    Parameters
    ----------
    A : array
        Symmetric matrix
    xp : module
        Numerical namespace

    Returns
    -------
    eigenvalues : array
        Eigenvalues in ascending order
    eigenvectors : array
        Corresponding eigenvectors
    """
    if xp is np:
        return np.linalg.eigh(A)
    elif xp is torch:
        return torch.linalg.eigh(A)
    elif xp is jnp:
        return jnp.linalg.eigh(A)
    else:
        return np.linalg.eigh(A)


def qr(A, xp):
    """Compute QR decomposition.

    Parameters
    ----------
    A : array
        Input matrix
    xp : module
        Numerical namespace

    Returns
    -------
    Q : array
        Orthogonal matrix
    R : array
        Upper triangular matrix
    """
    if xp is np:
        return np.linalg.qr(A)
    elif xp is torch:
        return torch.linalg.qr(A)
    elif xp is jnp:
        return jnp.linalg.qr(A)
    else:
        return np.linalg.qr(A)


def lstsq(A, b, xp):
    """Least squares solution to Ax = b.

    Parameters
    ----------
    A : array
        Coefficient matrix
    b : array
        Right-hand side
    xp : module
        Numerical namespace

    Returns
    -------
    x : array
        Least squares solution
    """
    if xp is np:
        result = np.linalg.lstsq(A, b, rcond=None)
        return result[0]
    elif xp is torch:
        result = torch.linalg.lstsq(A, b)
        return result.solution
    elif xp is jnp:
        result = jnp.linalg.lstsq(A, b, rcond=None)
        return result[0]
    else:
        result = np.linalg.lstsq(A, b, rcond=None)
        return result[0]


# Array operations


def eye(n, xp, device=None, dtype=None):
    """Create identity matrix.

    Parameters
    ----------
    n : int
        Size of the matrix
    xp : module
        Numerical namespace
    device : torch.device, optional
        Device for PyTorch
    dtype : dtype, optional
        Data type

    Returns
    -------
    I : array
        Identity matrix
    """
    if xp is np:
        return np.eye(n, dtype=dtype)
    elif xp is torch:
        dtype = dtype or torch.float64
        return torch.eye(n, dtype=dtype, device=device)
    elif xp is jnp:
        dtype = dtype or jnp.float64
        return jnp.eye(n, dtype=dtype)
    else:
        return np.eye(n, dtype=dtype)


def zeros(shape, xp, device=None, dtype=None):
    """Create array of zeros.

    Parameters
    ----------
    shape : tuple
        Shape of the array
    xp : module
        Numerical namespace
    device : torch.device, optional
        Device for PyTorch
    dtype : dtype, optional
        Data type

    Returns
    -------
    arr : array
        Array of zeros
    """
    if xp is np:
        return np.zeros(shape, dtype=dtype)
    elif xp is torch:
        dtype = dtype or torch.float64
        return torch.zeros(shape, dtype=dtype, device=device)
    elif xp is jnp:
        dtype = dtype or jnp.float64
        return jnp.zeros(shape, dtype=dtype)
    else:
        return np.zeros(shape, dtype=dtype)


def ones(shape, xp, device=None, dtype=None):
    """Create array of ones.

    Parameters
    ----------
    shape : tuple
        Shape of the array
    xp : module
        Numerical namespace
    device : torch.device, optional
        Device for PyTorch
    dtype : dtype, optional
        Data type

    Returns
    -------
    arr : array
        Array of ones
    """
    if xp is np:
        return np.ones(shape, dtype=dtype)
    elif xp is torch:
        dtype = dtype or torch.float64
        return torch.ones(shape, dtype=dtype, device=device)
    elif xp is jnp:
        dtype = dtype or jnp.float64
        return jnp.ones(shape, dtype=dtype)
    else:
        return np.ones(shape, dtype=dtype)


def concatenate(arrays, axis=0, xp=None):
    """Concatenate arrays along an axis.

    Parameters
    ----------
    arrays : list of arrays
        Arrays to concatenate
    axis : int
        Axis along which to concatenate
    xp : module, optional
        Numerical namespace (inferred if not provided)

    Returns
    -------
    result : array
        Concatenated array
    """
    if xp is None:
        # Infer from first array
        first = arrays[0]
        if torch is not None and isinstance(first, torch.Tensor):
            xp = torch
        elif jnp is not None and hasattr(first, "device_buffer"):
            xp = jnp
        else:
            xp = np

    if xp is np:
        return np.concatenate(arrays, axis=axis)
    elif xp is torch:
        return torch.cat(arrays, dim=axis)
    elif xp is jnp:
        return jnp.concatenate(arrays, axis=axis)
    else:
        return np.concatenate(arrays, axis=axis)


def stack(arrays, axis=0, xp=None):
    """Stack arrays along a new axis.

    Parameters
    ----------
    arrays : list of arrays
        Arrays to stack
    axis : int
        Axis along which to stack
    xp : module, optional
        Numerical namespace

    Returns
    -------
    result : array
        Stacked array
    """
    if xp is None:
        first = arrays[0]
        if torch is not None and isinstance(first, torch.Tensor):
            xp = torch
        elif jnp is not None and hasattr(first, "device_buffer"):
            xp = jnp
        else:
            xp = np

    if xp is np:
        return np.stack(arrays, axis=axis)
    elif xp is torch:
        return torch.stack(arrays, dim=axis)
    elif xp is jnp:
        return jnp.stack(arrays, axis=axis)
    else:
        return np.stack(arrays, axis=axis)


def diag(v, xp):
    """Create diagonal matrix or extract diagonal.

    Parameters
    ----------
    v : array
        1D array for diagonal or 2D array to extract from
    xp : module
        Numerical namespace

    Returns
    -------
    result : array
        Diagonal matrix or extracted diagonal
    """
    if xp is np:
        return np.diag(v)
    elif xp is torch:
        return torch.diag(v)
    elif xp is jnp:
        return jnp.diag(v)
    else:
        return np.diag(v)


def trace(A, xp):
    """Compute matrix trace.

    Parameters
    ----------
    A : array
        Square matrix
    xp : module
        Numerical namespace

    Returns
    -------
    tr : scalar
        Trace of the matrix
    """
    if xp is np:
        return np.trace(A)
    elif xp is torch:
        return torch.trace(A)
    elif xp is jnp:
        return jnp.trace(A)
    else:
        return np.trace(A)


def matmul(A, B, xp):
    """Matrix multiplication.

    Parameters
    ----------
    A : array
        First matrix
    B : array
        Second matrix
    xp : module
        Numerical namespace

    Returns
    -------
    C : array
        Product A @ B
    """
    if xp is np:
        return np.matmul(A, B)
    elif xp is torch:
        return torch.matmul(A, B)
    elif xp is jnp:
        return jnp.matmul(A, B)
    else:
        return np.matmul(A, B)


def transpose(A, xp):
    """Matrix transpose.

    Parameters
    ----------
    A : array
        Input matrix
    xp : module
        Numerical namespace

    Returns
    -------
    A_T : array
        Transposed matrix
    """
    if xp is np:
        return A.T
    elif xp is torch:
        return A.T
    elif xp is jnp:
        return A.T
    else:
        return A.T


def sum(A, axis=None, xp=None):
    """Sum array elements.

    Parameters
    ----------
    A : array
        Input array
    axis : int, optional
        Axis along which to sum
    xp : module, optional
        Numerical namespace

    Returns
    -------
    result : array or scalar
        Sum
    """
    if xp is None:
        if torch is not None and isinstance(A, torch.Tensor):
            xp = torch
        elif jnp is not None and hasattr(A, "device_buffer"):
            xp = jnp
        else:
            xp = np

    if xp is np:
        return np.sum(A, axis=axis)
    elif xp is torch:
        if axis is None:
            return torch.sum(A)
        return torch.sum(A, dim=axis)
    elif xp is jnp:
        return jnp.sum(A, axis=axis)
    else:
        return np.sum(A, axis=axis)


def mean(A, axis=None, xp=None):
    """Compute mean of array elements.

    Parameters
    ----------
    A : array
        Input array
    axis : int, optional
        Axis along which to compute mean
    xp : module, optional
        Numerical namespace

    Returns
    -------
    result : array or scalar
        Mean
    """
    if xp is None:
        if torch is not None and isinstance(A, torch.Tensor):
            xp = torch
        elif jnp is not None and hasattr(A, "device_buffer"):
            xp = jnp
        else:
            xp = np

    if xp is np:
        return np.mean(A, axis=axis)
    elif xp is torch:
        if axis is None:
            return torch.mean(A)
        return torch.mean(A, dim=axis)
    elif xp is jnp:
        return jnp.mean(A, axis=axis)
    else:
        return np.mean(A, axis=axis)


def sqrt(x, xp):
    """Element-wise square root.

    Parameters
    ----------
    x : array
        Input array
    xp : module
        Numerical namespace

    Returns
    -------
    result : array
        Square root
    """
    if xp is np:
        return np.sqrt(x)
    elif xp is torch:
        return torch.sqrt(x)
    elif xp is jnp:
        return jnp.sqrt(x)
    else:
        return np.sqrt(x)


def exp(x, xp):
    """Element-wise exponential.

    Parameters
    ----------
    x : array
        Input array
    xp : module
        Numerical namespace

    Returns
    -------
    result : array
        Exponential
    """
    if xp is np:
        return np.exp(x)
    elif xp is torch:
        return torch.exp(x)
    elif xp is jnp:
        return jnp.exp(x)
    else:
        return np.exp(x)


def log(x, xp):
    """Element-wise natural logarithm.

    Parameters
    ----------
    x : array
        Input array
    xp : module
        Numerical namespace

    Returns
    -------
    result : array
        Natural logarithm
    """
    if xp is np:
        return np.log(x)
    elif xp is torch:
        return torch.log(x)
    elif xp is jnp:
        return jnp.log(x)
    else:
        return np.log(x)


def abs(x, xp):
    """Element-wise absolute value.

    Parameters
    ----------
    x : array
        Input array
    xp : module
        Numerical namespace

    Returns
    -------
    result : array
        Absolute value
    """
    if xp is np:
        return np.abs(x)
    elif xp is torch:
        return torch.abs(x)
    elif xp is jnp:
        return jnp.abs(x)
    else:
        return np.abs(x)


def max(x, axis=None, xp=None):
    """Maximum of array elements.

    Parameters
    ----------
    x : array
        Input array
    axis : int, optional
        Axis along which to find maximum
    xp : module, optional
        Numerical namespace

    Returns
    -------
    result : array or scalar
        Maximum value
    """
    if xp is None:
        if torch is not None and isinstance(x, torch.Tensor):
            xp = torch
        elif jnp is not None and hasattr(x, "device_buffer"):
            xp = jnp
        else:
            xp = np

    if xp is np:
        return np.max(x, axis=axis)
    elif xp is torch:
        if axis is None:
            return torch.max(x)
        return torch.max(x, dim=axis).values
    elif xp is jnp:
        return jnp.max(x, axis=axis)
    else:
        return np.max(x, axis=axis)


def clip(x, min_val, max_val, xp):
    """Clip array values to a range.

    Parameters
    ----------
    x : array
        Input array
    min_val : scalar
        Minimum value
    max_val : scalar
        Maximum value
    xp : module
        Numerical namespace

    Returns
    -------
    result : array
        Clipped array
    """
    if xp is np:
        return np.clip(x, min_val, max_val)
    elif xp is torch:
        return torch.clamp(x, min_val, max_val)
    elif xp is jnp:
        return jnp.clip(x, min_val, max_val)
    else:
        return np.clip(x, min_val, max_val)


__all__ = [
    # Namespace management
    "get_namespace",
    "to_backend_array",
    "to_numpy",
    # Linear algebra
    "solve",
    "cholesky",
    "inv",
    "det",
    "slogdet",
    "eigh",
    "qr",
    "lstsq",
    # Array creation
    "eye",
    "zeros",
    "ones",
    "concatenate",
    "stack",
    "diag",
    # Array operations
    "trace",
    "matmul",
    "transpose",
    "sum",
    "mean",
    "sqrt",
    "exp",
    "log",
    "abs",
    "max",
    "clip",
]
