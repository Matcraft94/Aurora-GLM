# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Backend implementations for distributed computing.

This module provides integration with distributed computing frameworks
like Dask and Ray.
"""

# Check availability of backends
try:
    import dask

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

try:
    import ray

    HAS_RAY = True
except ImportError:
    HAS_RAY = False


def available_backends() -> list[str]:
    """Return list of available distributed backends."""
    backends = ["local"]  # Local chunked processing always available
    if HAS_DASK:
        backends.append("dask")
    if HAS_RAY:
        backends.append("ray")
    return backends


__all__ = ["HAS_DASK", "HAS_RAY", "available_backends"]
