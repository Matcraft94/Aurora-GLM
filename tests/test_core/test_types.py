"""Tests for core type definitions."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

from aurora.core import types


def test_array_alias_includes_numpy_ndarray():
    members = get_args(types.Array)
    assert np.ndarray in members


def test_scalar_alias_covers_numeric_scalars():
    members = get_args(types.Scalar)
    assert {int, float, complex}.issubset(set(members))


def test_shape_alias_is_tuple_of_ints():
    assert types.Shape == tuple[int, ...]


def test_optimization_callback_signature():
    arg_types, return_type = get_args(types.OptimizationCallback)
    assert return_type is type(None)
    assert tuple(arg_types) == (int, types.Array, float)


def test_torch_tensor_alias_matches_tensor():
    torch = pytest.importorskip("torch")
    assert types.TorchTensor is torch.Tensor


def test_exports_include_protocols():
    expected = {
        "Array",
        "ArrayLike",
        "Scalar",
        "Shape",
        "DType",
        "Distribution",
        "Link",
        "Optimizer",
        "OptimizationResult",
    }
    assert expected.issubset(set(types.__all__))
