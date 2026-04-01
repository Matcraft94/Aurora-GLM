# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Utility helpers for multi-backend distribution implementations.

Provides backend detection, array conversion, and numerical utility
functions that work transparently across NumPy, PyTorch, and JAX.
All distribution family code should use these helpers instead of
calling numpy, torch, or jax directly on user-provided arrays.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]


def is_torch(value) -> bool:
    """Return ``True`` when *value* is a torch tensor."""
    return torch is not None and isinstance(value, torch.Tensor)


def is_jax(value) -> bool:
    """Return ``True`` when *value* is a JAX array."""
    if jax is None:
        return False
    # Check for JAX array types
    return hasattr(value, "device_buffer") or (
        hasattr(jax, "Array") and isinstance(value, jax.Array)
    )


def namespace(*values):
    """Detect and return the numerical namespace for the given values.

    Inspects the types of the provided values and returns the corresponding
    array module:
    - If any value is a PyTorch tensor, returns ``torch``.
    - If any value is a JAX array, returns ``jax.numpy``.
    - Otherwise, returns ``numpy``.

    Parameters
    ----------
    *values : array_like
        One or more arrays or scalars to inspect. At least one value
        should be an array for meaningful backend detection.

    Returns
    -------
    module
        The numerical namespace (``numpy``, ``torch``, or ``jax.numpy``)
        matching the input types.

    See Also
    --------
    as_namespace_array : Convert a value to the array type for a given namespace
    namespace_from_backend : Get namespace from a string backend name

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions._utils import namespace
    >>> xp = namespace(np.array([1.0, 2.0]))
    >>> xp is np
    True

    >>> # With PyTorch tensor (if torch is installed)
    >>> # xp = namespace(torch.tensor([1.0, 2.0]))  # returns torch
    """
    for value in values:
        if is_torch(value):
            return torch  # type: ignore[return-value]
        if is_jax(value):
            return jnp  # type: ignore[return-value]
    return np


def namespace_from_backend(backend: str = "numpy", device: str | None = None):
    """Get the numerical namespace for a specified backend.

    Parameters
    ----------
    backend : str
        Backend name: 'numpy', 'torch'/'pytorch', or 'jax'
    device : str, optional
        Device for torch backend

    Returns
    -------
    xp : module
        Numerical namespace
    device_info : torch.device or None
        Device for tensor creation (only for torch)
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


def as_namespace_array(value, xp, *, like=None, device=None):
    """Convert a value to an array in the specified backend namespace.

    Handles dtype and device propagation for PyTorch tensors, dtype matching
    for JAX arrays, and standard NumPy conversion otherwise.

    Parameters
    ----------
    value : array_like
        The value to convert (scalar, list, or array).
    xp : module
        Target namespace: ``numpy``, ``torch``, or ``jax.numpy``.
    like : array, optional
        Reference array whose dtype (and device for PyTorch) to match.
        If None, uses the backend's default dtype.
    device : torch.device or str, optional
        Target device for PyTorch tensors. If None, infers from ``like``
        or defaults to CUDA if available, otherwise CPU.

    Returns
    -------
    array
        The value converted to an array in the target namespace.

    See Also
    --------
    namespace : Detect the namespace from existing arrays

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions._utils import as_namespace_array, namespace
    >>> xp = namespace(np.array([1.0]))
    >>> as_namespace_array(3.0, xp)
    array(3.)
    """
    if xp is torch:  # type: ignore[comparison-overlap]
        dtype = getattr(like, "dtype", torch.get_default_dtype())
        if device is None:
            device = getattr(
                like,
                "device",
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
        return torch.as_tensor(value, dtype=dtype, device=device)
    elif xp is jnp:  # type: ignore[comparison-overlap]
        dtype = getattr(like, "dtype", jnp.float64 if jnp is not None else None)
        return jnp.array(value, dtype=dtype)
    dtype = getattr(like, "dtype", None)
    return np.asarray(value, dtype=dtype)


def ones_like(value):
    """Return an array of ones with the same shape and type as *value*.

    Parameters
    ----------
    value : array
        Reference array whose shape and dtype to match.

    Returns
    -------
    array
        Array of ones with the same shape and backend type as *value*.
    """
    xp = namespace(value)
    if xp is torch:  # type: ignore[comparison-overlap]
        return torch.ones_like(value)
    return np.ones_like(value)


def clip_probability(prob, xp, eps: float = 1e-9):
    """Clip probability to [eps, 1-eps] with backend-aware epsilon.

    Parameters
    ----------
    prob : array
        Probability values to clip
    xp : module
        Array namespace (np, torch, or jnp)
    eps : float, default=1e-9
        Minimum distance from 0 and 1

    Returns
    -------
    array
        Clipped probability in range [eps, 1-eps]

    Notes
    -----
    Creates tensors with correct dtype and device for PyTorch/JAX compatibility.
    Ensures numerical stability across all backends.
    """
    if xp is torch:  # type: ignore[comparison-overlap]
        eps_tensor = torch.tensor(eps, dtype=prob.dtype, device=prob.device)
        one_tensor = torch.tensor(1.0, dtype=prob.dtype, device=prob.device)
        return torch.clamp(prob, eps_tensor, one_tensor - eps_tensor)
    elif xp is jnp:  # type: ignore[comparison-overlap]
        return jnp.clip(prob, eps, 1.0 - eps)
    return np.clip(prob, eps, 1.0 - eps)


def ensure_positive(value, xp, eps: float = 1e-12):
    """Ensure values are positive by clipping to [eps, inf).

    Parameters
    ----------
    value : array
        Input values to clip
    xp : module
        Array namespace (np, torch, or jnp)
    eps : float, default=1e-12
        Minimum value (must be positive)

    Returns
    -------
    array
        Clipped values with minimum eps

    Notes
    -----
    This is the centralized helper for ensuring positive values across
    all backends. Use this instead of defining local _positive() functions
    in individual modules.
    """
    if xp is torch:  # type: ignore[comparison-overlap]
        eps_tensor = torch.tensor(eps, dtype=value.dtype, device=value.device)
        return torch.clamp(value, min=eps_tensor)
    elif xp is jnp:  # type: ignore[comparison-overlap]
        return jnp.clip(value, eps, None)
    return np.clip(value, eps, None)


def log_factorial(value, xp):
    """Compute log(n!) = log(Gamma(n+1)) across backends.

    Parameters
    ----------
    value : array
        Non-negative values (typically integers for factorial)
    xp : module
        Array namespace (np, torch, or jnp)

    Returns
    -------
    array
        Log-factorial values
    """
    if xp is torch:  # type: ignore[comparison-overlap]
        return torch.lgamma(value + 1.0)
    elif xp is jnp:  # type: ignore[comparison-overlap]
        from jax.scipy.special import gammaln

        return gammaln(value + 1.0)
    # NumPy - use scipy for vectorized operation
    from scipy.special import gammaln

    return gammaln(np.asarray(value) + 1.0)


def log_gamma(value, xp):
    """Compute log(Gamma(x)) across backends.

    Parameters
    ----------
    value : array
        Positive values
    xp : module
        Array namespace (np, torch, or jnp)

    Returns
    -------
    array
        Log-gamma values
    """
    if xp is torch:  # type: ignore[comparison-overlap]
        return torch.lgamma(value)
    elif xp is jnp:  # type: ignore[comparison-overlap]
        from jax.scipy.special import gammaln

        return gammaln(value)
    # NumPy - use scipy for vectorized operation
    from scipy.special import gammaln

    return gammaln(np.asarray(value))


def digamma(value, xp):
    """Digamma function (derivative of log-gamma) across backends."""
    if xp is torch:  # type: ignore[comparison-overlap]
        return torch.digamma(value)
    elif xp is jnp:  # type: ignore[comparison-overlap]
        from jax.scipy.special import digamma as jax_digamma

        return jax_digamma(value)
    # NumPy
    from scipy import special

    return special.digamma(value)


__all__ = [
    "as_namespace_array",
    "clip_probability",
    "digamma",
    "ensure_positive",
    "is_jax",
    "is_torch",
    "log_factorial",
    "log_gamma",
    "namespace",
    "namespace_from_backend",
    "ones_like",
]
