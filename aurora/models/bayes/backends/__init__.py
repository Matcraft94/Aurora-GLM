# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Backend implementations for Bayesian inference.

This module provides backends for different probabilistic programming
libraries (NumPyro, PyMC).
"""

from __future__ import annotations

# Check availability of backends
try:
    import numpyro

    HAS_NUMPYRO = True
except ImportError:
    HAS_NUMPYRO = False

try:
    import pymc

    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False


def available_backends() -> list[str]:
    """Return list of available Bayesian backends."""
    backends = []
    if HAS_NUMPYRO:
        backends.append("numpyro")
    if HAS_PYMC:
        backends.append("pymc")
    return backends


def get_default_backend() -> str:
    """Get the default available backend."""
    if HAS_NUMPYRO:
        return "numpyro"
    elif HAS_PYMC:
        return "pymc"
    else:
        raise ImportError(
            "No Bayesian backend available. Install numpyro or pymc:\n"
            "  pip install numpyro jax jaxlib\n"
            "  pip install pymc arviz"
        )


__all__ = ["HAS_NUMPYRO", "HAS_PYMC", "available_backends", "get_default_backend"]
