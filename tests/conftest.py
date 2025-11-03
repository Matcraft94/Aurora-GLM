"""Shared pytest fixtures for Aurora-GLM tests."""
from __future__ import annotations

import pytest

from aurora.core.backends import get_backend
from aurora.utils import BackendNotAvailableError


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
