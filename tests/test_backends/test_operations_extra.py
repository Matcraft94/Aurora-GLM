# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Extra tests for aurora.core.backends.operations covering torch/JAX dispatch
branches via mocked namespaces.

These tests exercise every ``elif xp is torch:`` / ``elif xp is jnp:`` branch
without requiring torch or JAX to be installed.  The approach is to patch the
module-level ``torch`` / ``jnp`` names inside ``operations.py`` with MagicMock
objects whose methods record calls and return predictable values.  Then we pass
the mock as ``xp`` to each function, verifying that the torch/JAX-specific API
is invoked (e.g. ``torch.cat`` instead of ``np.concatenate``).

This complements ``test_operations_full.py`` which tests the real backends
(when available) and the NumPy/fallback paths.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from numpy.testing import assert_allclose

import aurora.core.backends.operations as _ops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_xp(name: str = "mock_xp"):
    """Build a MagicMock that mimics a numerical namespace (torch / jnp)."""
    xp = MagicMock(name=name)

    # linalg sub-namespace
    linalg = MagicMock(name=f"{name}.linalg")
    xp.linalg = linalg

    # Common functions used as attributes
    xp.eye = MagicMock(name=f"{name}.eye", return_value=np.eye(2))
    xp.zeros = MagicMock(name=f"{name}.zeros", return_value=np.zeros(2))
    xp.ones = MagicMock(name=f"{name}.ones", return_value=np.ones(2))
    xp.diag = MagicMock(name=f"{name}.diag", return_value=np.array([1.0]))
    xp.trace = MagicMock(name=f"{name}.trace", return_value=5.0)
    xp.matmul = MagicMock(name=f"{name}.matmul", return_value=np.eye(2))
    xp.sum = MagicMock(name=f"{name}.sum", return_value=6.0)
    xp.mean = MagicMock(name=f"{name}.mean", return_value=2.0)
    xp.sqrt = MagicMock(name=f"{name}.sqrt", return_value=np.array([1.0]))
    xp.exp = MagicMock(name=f"{name}.exp", return_value=np.array([1.0]))
    xp.log = MagicMock(name=f"{name}.log", return_value=np.array([0.0]))
    xp.abs = MagicMock(name=f"{name}.abs", return_value=np.array([1.0]))
    xp.concatenate = MagicMock(name=f"{name}.concatenate", return_value=np.array([1.0]))
    xp.stack = MagicMock(name=f"{name}.stack", return_value=np.array([1.0]))
    xp.float64 = np.float64

    # torch-specific aliases
    xp.cat = MagicMock(name=f"{name}.cat", return_value=np.array([1.0]))
    xp.clamp = MagicMock(name=f"{name}.clamp", return_value=np.array([1.0]))
    xp.tensor = MagicMock(name=f"{name}.tensor", return_value=np.ones(2))
    xp.device = MagicMock(name=f"{name}.device", return_value="cpu")

    # max for torch returns a named tuple with .values
    _max_result = MagicMock()
    _max_result.values = np.array([5.0])
    xp.max = MagicMock(name=f"{name}.max", return_value=_max_result)

    return xp


# ---------------------------------------------------------------------------
# get_namespace -- mocked torch / jax
# ---------------------------------------------------------------------------


class TestGetNamespaceMocked:
    """Test get_namespace by mocking the module-level torch / jax imports."""

    def test_torch_backend_with_mock(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.object(_ops, "torch", mock_torch):
            xp, dev = _ops.get_namespace("torch")
            assert xp is mock_torch

    def test_torch_backend_cuda_default(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = "cuda:0"
        with patch.object(_ops, "torch", mock_torch):
            xp, dev = _ops.get_namespace("torch")
            assert xp is mock_torch
            mock_torch.device.assert_called_with("cuda")

    def test_torch_backend_cpu_fallback(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        with patch.object(_ops, "torch", mock_torch):
            xp, dev = _ops.get_namespace("torch")
            mock_torch.device.assert_called_with("cpu")

    def test_pytorch_alias_with_mock(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        with patch.object(_ops, "torch", mock_torch):
            xp, _ = _ops.get_namespace("pytorch")
            assert xp is mock_torch

    def test_jax_backend_with_mock(self):
        mock_jnp = MagicMock()
        with patch.object(_ops, "jnp", mock_jnp):
            xp, dev = _ops.get_namespace("jax")
            assert xp is mock_jnp
            assert dev is None

    def test_torch_not_installed(self):
        with patch.object(_ops, "torch", None):
            with pytest.raises(ImportError, match="PyTorch"):
                _ops.get_namespace("torch")

    def test_jax_not_installed(self):
        with patch.object(_ops, "jnp", None):
            with pytest.raises(ImportError, match="JAX"):
                _ops.get_namespace("jax")


# ---------------------------------------------------------------------------
# to_backend_array -- torch / jax branches via mock
# ---------------------------------------------------------------------------


class TestToBackendArrayMocked:
    """Exercise the torch and jnp conversion branches in to_backend_array."""

    def test_torch_output_with_mock(self):
        mock_torch = MagicMock()
        mock_torch.float64 = np.float64
        expected = np.array([1.0, 2.0])
        mock_torch.tensor.return_value = expected

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.to_backend_array([1.0, 2.0], mock_torch)
            mock_torch.tensor.assert_called_once()
            assert result is expected

    def test_torch_output_with_dtype(self):
        mock_torch = MagicMock()
        expected = np.array([1.0, 2.0])
        mock_torch.tensor.return_value = expected
        mock_torch.float32 = np.float32

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.to_backend_array([1, 2], mock_torch, dtype=np.float32)
            call_kwargs = mock_torch.tensor.call_args
            # The function sets dtype=mock_torch.float32
            assert call_kwargs is not None

    def test_jax_output_with_mock(self):
        mock_jnp = MagicMock()
        mock_jnp.float64 = np.float64
        expected = np.array([1.0, 2.0])
        mock_jnp.array.return_value = expected

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.to_backend_array([1.0, 2.0], mock_jnp)
            mock_jnp.array.assert_called_once()

    def test_jax_output_with_dtype(self):
        mock_jnp = MagicMock()
        expected = np.array([1.0, 2.0])
        mock_jnp.array.return_value = expected

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.to_backend_array([1, 2], mock_jnp, dtype=np.float32)
            mock_jnp.array.assert_called_once()

    def test_jax_input_converted_via_device_buffer(self):
        """When input has .device_buffer (jax array), it is converted to numpy."""
        class FakeJaxArray:
            device_buffer = True

        fake_arr = FakeJaxArray()

        result = _ops.to_backend_array(fake_arr, np)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# to_numpy -- torch / jax branches via mock
# ---------------------------------------------------------------------------


class TestToNumpyMocked:
    def test_jax_array_via_device_buffer(self):
        class FakeJaxArray:
            device_buffer = True

        mock_jax = MagicMock()
        with patch.object(_ops, "jax", mock_jax):
            result = _ops.to_numpy(FakeJaxArray())
            assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Linear algebra -- torch branches via mock xp
# ---------------------------------------------------------------------------


class TestLinalgTorchMocked:
    """Verify that every linalg function hits the torch branch when xp is torch."""

    @pytest.fixture()
    def mock_torch(self):
        return _make_mock_xp("mock_torch")

    def test_solve_calls_torch_linalg(self, mock_torch):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        mock_torch.linalg.solve.return_value = np.linalg.solve(A, b)

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.solve(A, b, mock_torch)
            mock_torch.linalg.solve.assert_called_once_with(A, b)

    def test_cholesky_calls_torch_linalg(self, mock_torch):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        mock_torch.linalg.cholesky.return_value = np.linalg.cholesky(A)

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.cholesky(A, mock_torch)
            mock_torch.linalg.cholesky.assert_called_once_with(A)

    def test_inv_calls_torch_linalg(self, mock_torch):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_torch.linalg.inv.return_value = np.linalg.inv(A)

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.inv(A, mock_torch)
            mock_torch.linalg.inv.assert_called_once_with(A)

    def test_det_calls_torch_linalg(self, mock_torch):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_torch.linalg.det.return_value = -2.0

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.det(A, mock_torch)
            mock_torch.linalg.det.assert_called_once_with(A)

    def test_slogdet_calls_torch_linalg(self, mock_torch):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.linalg.slogdet(A)
        mock_torch.linalg.slogdet.return_value = expected

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.slogdet(A, mock_torch)
            mock_torch.linalg.slogdet.assert_called_once_with(A)

    def test_eigh_calls_torch_linalg(self, mock_torch):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        expected = np.linalg.eigh(A)
        mock_torch.linalg.eigh.return_value = expected

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.eigh(A, mock_torch)
            mock_torch.linalg.eigh.assert_called_once_with(A)

    def test_qr_calls_torch_linalg(self, mock_torch):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.linalg.qr(A)
        mock_torch.linalg.qr.return_value = expected

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.qr(A, mock_torch)
            mock_torch.linalg.qr.assert_called_once_with(A)

    def test_lstsq_calls_torch_linalg(self, mock_torch):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([5.0, 6.0])
        # torch.linalg.lstsq returns an object with .solution
        mock_result = MagicMock()
        mock_result.solution = np.array([1.0, 2.0])
        mock_torch.linalg.lstsq.return_value = mock_result

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.lstsq(A, b, mock_torch)
            mock_torch.linalg.lstsq.assert_called_once_with(A, b)
            assert result is mock_result.solution


# ---------------------------------------------------------------------------
# Linear algebra -- jnp branches via mock xp
# ---------------------------------------------------------------------------


class TestLinalgJaxMocked:
    """Verify that every linalg function hits the jnp branch when xp is jnp."""

    @pytest.fixture()
    def mock_jnp(self):
        return _make_mock_xp("mock_jnp")

    def test_solve_calls_jnp_linalg(self, mock_jnp):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        mock_jnp.linalg.solve.return_value = np.linalg.solve(A, b)

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.solve(A, b, mock_jnp)
            mock_jnp.linalg.solve.assert_called_once_with(A, b)

    def test_cholesky_calls_jnp_linalg(self, mock_jnp):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        mock_jnp.linalg.cholesky.return_value = np.linalg.cholesky(A)

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.cholesky(A, mock_jnp)
            mock_jnp.linalg.cholesky.assert_called_once_with(A)

    def test_inv_calls_jnp_linalg(self, mock_jnp):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_jnp.linalg.inv.return_value = np.linalg.inv(A)

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.inv(A, mock_jnp)
            mock_jnp.linalg.inv.assert_called_once_with(A)

    def test_det_calls_jnp_linalg(self, mock_jnp):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_jnp.linalg.det.return_value = -2.0

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.det(A, mock_jnp)
            mock_jnp.linalg.det.assert_called_once_with(A)

    def test_slogdet_calls_jnp_linalg(self, mock_jnp):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.linalg.slogdet(A)
        mock_jnp.linalg.slogdet.return_value = expected

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.slogdet(A, mock_jnp)
            mock_jnp.linalg.slogdet.assert_called_once_with(A)

    def test_eigh_calls_jnp_linalg(self, mock_jnp):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        expected = np.linalg.eigh(A)
        mock_jnp.linalg.eigh.return_value = expected

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.eigh(A, mock_jnp)
            mock_jnp.linalg.eigh.assert_called_once_with(A)

    def test_qr_calls_jnp_linalg(self, mock_jnp):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.linalg.qr(A)
        mock_jnp.linalg.qr.return_value = expected

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.qr(A, mock_jnp)
            mock_jnp.linalg.qr.assert_called_once_with(A)

    def test_lstsq_calls_jnp_linalg(self, mock_jnp):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([5.0, 6.0])
        # jnp.linalg.lstsq returns a tuple; first element is the solution
        expected_x = np.linalg.lstsq(A, b, rcond=None)[0]
        mock_jnp.linalg.lstsq.return_value = (expected_x,)

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.lstsq(A, b, mock_jnp)
            mock_jnp.linalg.lstsq.assert_called_once_with(A, b, rcond=None)
            assert result is expected_x


# ---------------------------------------------------------------------------
# Array creation -- torch branches
# ---------------------------------------------------------------------------


class TestArrayCreationTorchMocked:
    @pytest.fixture()
    def mock_torch(self):
        return _make_mock_xp("mock_torch")

    def test_eye_default_dtype(self, mock_torch):
        mock_torch.eye.return_value = np.eye(3)

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.eye(3, mock_torch)
            # torch branch sets dtype=mock_torch.float64 if none given
            mock_torch.eye.assert_called_once_with(3, dtype=mock_torch.float64, device=None)

    def test_eye_explicit_dtype(self, mock_torch):
        mock_torch.eye.return_value = np.eye(2)

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.eye(2, mock_torch, dtype=np.float32, device="cpu")
            mock_torch.eye.assert_called_once_with(2, dtype=np.float32, device="cpu")

    def test_zeros_default_dtype(self, mock_torch):
        mock_torch.zeros.return_value = np.zeros((2, 3))

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.zeros((2, 3), mock_torch)
            mock_torch.zeros.assert_called_once_with((2, 3), dtype=mock_torch.float64, device=None)

    def test_zeros_explicit_dtype(self, mock_torch):
        mock_torch.zeros.return_value = np.zeros((2,))

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.zeros((2,), mock_torch, dtype=np.float32, device="cpu")
            mock_torch.zeros.assert_called_once_with((2,), dtype=np.float32, device="cpu")

    def test_ones_default_dtype(self, mock_torch):
        mock_torch.ones.return_value = np.ones((3, 2))

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.ones((3, 2), mock_torch)
            mock_torch.ones.assert_called_once_with((3, 2), dtype=mock_torch.float64, device=None)

    def test_ones_explicit_dtype(self, mock_torch):
        mock_torch.ones.return_value = np.ones((2,))

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.ones((2,), mock_torch, dtype=np.float32, device="cpu")
            mock_torch.ones.assert_called_once_with((2,), dtype=np.float32, device="cpu")


# ---------------------------------------------------------------------------
# Array creation -- jnp branches
# ---------------------------------------------------------------------------


class TestArrayCreationJaxMocked:
    @pytest.fixture()
    def mock_jnp(self):
        return _make_mock_xp("mock_jnp")

    def test_eye_default_dtype(self, mock_jnp):
        mock_jnp.eye.return_value = np.eye(3)

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.eye(3, mock_jnp)
            mock_jnp.eye.assert_called_once_with(3, dtype=mock_jnp.float64)

    def test_eye_explicit_dtype(self, mock_jnp):
        mock_jnp.eye.return_value = np.eye(2)

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.eye(2, mock_jnp, dtype=np.float32)
            mock_jnp.eye.assert_called_once_with(2, dtype=np.float32)

    def test_zeros_default_dtype(self, mock_jnp):
        mock_jnp.zeros.return_value = np.zeros((2, 3))

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.zeros((2, 3), mock_jnp)
            mock_jnp.zeros.assert_called_once_with((2, 3), dtype=mock_jnp.float64)

    def test_zeros_explicit_dtype(self, mock_jnp):
        mock_jnp.zeros.return_value = np.zeros((2,))

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.zeros((2,), mock_jnp, dtype=np.int32)
            mock_jnp.zeros.assert_called_once_with((2,), dtype=np.int32)

    def test_ones_default_dtype(self, mock_jnp):
        mock_jnp.ones.return_value = np.ones((3, 2))

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.ones((3, 2), mock_jnp)
            mock_jnp.ones.assert_called_once_with((3, 2), dtype=mock_jnp.float64)

    def test_ones_explicit_dtype(self, mock_jnp):
        mock_jnp.ones.return_value = np.ones((2,))

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.ones((2,), mock_jnp, dtype=np.float32)
            mock_jnp.ones.assert_called_once_with((2,), dtype=np.float32)


# ---------------------------------------------------------------------------
# concatenate / stack -- torch branches (torch.cat / torch.stack)
# ---------------------------------------------------------------------------


class TestConcatenateStackTorchMocked:
    @pytest.fixture()
    def mock_torch(self):
        return _make_mock_xp("mock_torch")

    def test_concatenate_uses_torch_cat(self, mock_torch):
        arrays = [np.array([1.0]), np.array([2.0])]
        mock_torch.cat.return_value = np.array([1.0, 2.0])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.concatenate(arrays, axis=0, xp=mock_torch)
            mock_torch.cat.assert_called_once_with(arrays, dim=0)

    def test_concatenate_axis1(self, mock_torch):
        arrays = [np.array([[1.0]]), np.array([[2.0]])]
        mock_torch.cat.return_value = np.array([[1.0, 2.0]])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.concatenate(arrays, axis=1, xp=mock_torch)
            mock_torch.cat.assert_called_once_with(arrays, dim=1)

    def test_stack_uses_torch_stack(self, mock_torch):
        arrays = [np.array([1.0]), np.array([2.0])]
        mock_torch.stack.return_value = np.array([[1.0], [2.0]])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.stack(arrays, axis=0, xp=mock_torch)
            mock_torch.stack.assert_called_once_with(arrays, dim=0)

    def test_stack_axis1(self, mock_torch):
        arrays = [np.array([1.0]), np.array([2.0])]
        mock_torch.stack.return_value = np.array([[1.0, 2.0]])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.stack(arrays, axis=1, xp=mock_torch)
            mock_torch.stack.assert_called_once_with(arrays, dim=1)


# ---------------------------------------------------------------------------
# concatenate / stack -- jnp branches
# ---------------------------------------------------------------------------


class TestConcatenateStackJaxMocked:
    @pytest.fixture()
    def mock_jnp(self):
        return _make_mock_xp("mock_jnp")

    def test_concatenate_uses_jnp_concatenate(self, mock_jnp):
        arrays = [np.array([1.0]), np.array([2.0])]
        mock_jnp.concatenate.return_value = np.array([1.0, 2.0])

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.concatenate(arrays, axis=0, xp=mock_jnp)
            mock_jnp.concatenate.assert_called_once_with(arrays, axis=0)

    def test_stack_uses_jnp_stack(self, mock_jnp):
        arrays = [np.array([1.0]), np.array([2.0])]
        mock_jnp.stack.return_value = np.array([[1.0], [2.0]])

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.stack(arrays, axis=0, xp=mock_jnp)
            mock_jnp.stack.assert_called_once_with(arrays, axis=0)


# ---------------------------------------------------------------------------
# concatenate / stack -- backend inference with mocked torch / jnp
# ---------------------------------------------------------------------------


class TestConcatenateStackInferenceMocked:
    def test_concatenate_infers_torch(self):
        mock_torch = _make_mock_xp("mock_torch")

        class FakeTensor:
            pass

        fake_t = FakeTensor()
        mock_torch.Tensor = FakeTensor
        mock_torch.cat.return_value = np.array([3.0, 4.0])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.concatenate([fake_t, fake_t])
            mock_torch.cat.assert_called_once()

    def test_concatenate_infers_jnp(self):
        mock_jnp = _make_mock_xp("mock_jnp")

        class FakeJaxArray:
            device_buffer = True

        fake_arr = FakeJaxArray()
        mock_jnp.concatenate.return_value = np.array([3.0, 4.0])

        # jnp must be not-None for the inference check
        with patch.object(_ops, "jnp", mock_jnp):
            with patch.object(_ops, "torch", None):
                result = _ops.concatenate([fake_arr, fake_arr])
                mock_jnp.concatenate.assert_called_once()

    def test_stack_infers_torch(self):
        mock_torch = _make_mock_xp("mock_torch")

        class FakeTensor:
            pass

        fake_t = FakeTensor()
        mock_torch.Tensor = FakeTensor
        mock_torch.stack.return_value = np.array([[1.0], [1.0]])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.stack([fake_t, fake_t])
            mock_torch.stack.assert_called_once()

    def test_stack_infers_jnp(self):
        mock_jnp = _make_mock_xp("mock_jnp")

        class FakeJaxArray:
            device_buffer = True

        fake_arr = FakeJaxArray()
        mock_jnp.stack.return_value = np.array([[1.0], [1.0]])

        with patch.object(_ops, "jnp", mock_jnp):
            with patch.object(_ops, "torch", None):
                result = _ops.stack([fake_arr, fake_arr])
                mock_jnp.stack.assert_called_once()


# ---------------------------------------------------------------------------
# diag, trace, matmul, transpose -- torch branches
# ---------------------------------------------------------------------------


class TestMatrixOpsTorchMocked:
    @pytest.fixture()
    def mock_torch(self):
        return _make_mock_xp("mock_torch")

    def test_diag_uses_torch_diag(self, mock_torch):
        v = np.array([1.0, 2.0])
        mock_torch.diag.return_value = np.diag(v)

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.diag(v, mock_torch)
            mock_torch.diag.assert_called_once_with(v)

    def test_trace_uses_torch_trace(self, mock_torch):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_torch.trace.return_value = 5.0

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.trace(A, mock_torch)
            mock_torch.trace.assert_called_once_with(A)

    def test_matmul_uses_torch_matmul(self, mock_torch):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        mock_torch.matmul.return_value = A @ B

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.matmul(A, B, mock_torch)
            mock_torch.matmul.assert_called_once_with(A, B)

    def test_transpose_uses_t_attribute(self, mock_torch):
        A = MagicMock()
        A.T = np.array([[1.0, 3.0], [2.0, 4.0]])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.transpose(A, mock_torch)
            assert result is A.T


# ---------------------------------------------------------------------------
# diag, trace, matmul, transpose -- jnp branches
# ---------------------------------------------------------------------------


class TestMatrixOpsJaxMocked:
    @pytest.fixture()
    def mock_jnp(self):
        return _make_mock_xp("mock_jnp")

    def test_diag_uses_jnp_diag(self, mock_jnp):
        v = np.array([1.0, 2.0])
        mock_jnp.diag.return_value = np.diag(v)

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.diag(v, mock_jnp)
            mock_jnp.diag.assert_called_once_with(v)

    def test_trace_uses_jnp_trace(self, mock_jnp):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_jnp.trace.return_value = 5.0

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.trace(A, mock_jnp)
            mock_jnp.trace.assert_called_once_with(A)

    def test_matmul_uses_jnp_matmul(self, mock_jnp):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        mock_jnp.matmul.return_value = A @ B

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.matmul(A, B, mock_jnp)
            mock_jnp.matmul.assert_called_once_with(A, B)

    def test_transpose_uses_t_attribute(self, mock_jnp):
        A = MagicMock()
        A.T = np.array([[1.0, 3.0], [2.0, 4.0]])

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.transpose(A, mock_jnp)
            assert result is A.T


# ---------------------------------------------------------------------------
# Reductions: sum, mean, max -- torch branches
# ---------------------------------------------------------------------------


class TestReductionsTorchMocked:
    @pytest.fixture()
    def mock_torch(self):
        mock = _make_mock_xp("mock_torch")
        # sum / mean should return arrays for axis case
        mock.sum = MagicMock(return_value=np.array([4.0, 6.0]))
        mock.mean = MagicMock(return_value=np.array([2.0, 6.0]))
        # max with axis returns namedtuple with .values
        _max_result = MagicMock()
        _max_result.values = np.array([3.0, 5.0])
        mock.max = MagicMock(return_value=_max_result)
        return mock

    def test_sum_no_axis(self, mock_torch):
        a = np.array([1.0, 2.0, 3.0])
        mock_torch.sum.return_value = 6.0

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.sum(a, xp=mock_torch)
            mock_torch.sum.assert_called_once_with(a)

    def test_sum_with_axis(self, mock_torch):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.sum(a, axis=0, xp=mock_torch)
            mock_torch.sum.assert_called_once_with(a, dim=0)

    def test_mean_no_axis(self, mock_torch):
        a = np.array([2.0, 4.0, 6.0])
        mock_torch.mean.return_value = 4.0

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.mean(a, xp=mock_torch)
            mock_torch.mean.assert_called_once_with(a)

    def test_mean_with_axis(self, mock_torch):
        a = np.array([[1.0, 3.0], [5.0, 7.0]])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.mean(a, axis=1, xp=mock_torch)
            mock_torch.mean.assert_called_once_with(a, dim=1)

    def test_max_no_axis(self, mock_torch):
        a = np.array([1.0, 5.0, 3.0])
        _flat_max = MagicMock()
        _flat_max.item.return_value = 5.0
        mock_torch.max.return_value = _flat_max

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.max(a, xp=mock_torch)
            mock_torch.max.assert_called_once_with(a)

    def test_max_with_axis(self, mock_torch):
        a = np.array([[1.0, 5.0], [3.0, 2.0]])
        _axis_max = MagicMock()
        _axis_max.values = np.array([3.0, 5.0])
        mock_torch.max.return_value = _axis_max

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.max(a, axis=0, xp=mock_torch)
            mock_torch.max.assert_called_once_with(a, dim=0)


# ---------------------------------------------------------------------------
# Reductions: sum, mean, max -- jnp branches
# ---------------------------------------------------------------------------


class TestReductionsJaxMocked:
    @pytest.fixture()
    def mock_jnp(self):
        mock = _make_mock_xp("mock_jnp")
        mock.sum = MagicMock(return_value=6.0)
        mock.mean = MagicMock(return_value=2.0)
        mock.max = MagicMock(return_value=5.0)
        return mock

    def test_sum_with_axis(self, mock_jnp):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_jnp.sum.return_value = np.array([4.0, 6.0])

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.sum(a, axis=0, xp=mock_jnp)
            mock_jnp.sum.assert_called_once_with(a, axis=0)

    def test_sum_no_axis(self, mock_jnp):
        a = np.array([1.0, 2.0, 3.0])
        mock_jnp.sum.return_value = 6.0

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.sum(a, xp=mock_jnp)
            mock_jnp.sum.assert_called_once_with(a, axis=None)

    def test_mean_with_axis(self, mock_jnp):
        a = np.array([[1.0, 3.0], [5.0, 7.0]])
        mock_jnp.mean.return_value = np.array([2.0, 6.0])

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.mean(a, axis=1, xp=mock_jnp)
            mock_jnp.mean.assert_called_once_with(a, axis=1)

    def test_mean_no_axis(self, mock_jnp):
        a = np.array([2.0, 4.0, 6.0])
        mock_jnp.mean.return_value = 4.0

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.mean(a, xp=mock_jnp)
            mock_jnp.mean.assert_called_once_with(a, axis=None)

    def test_max_with_axis(self, mock_jnp):
        a = np.array([[1.0, 5.0], [3.0, 2.0]])
        mock_jnp.max.return_value = np.array([3.0, 5.0])

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.max(a, axis=0, xp=mock_jnp)
            mock_jnp.max.assert_called_once_with(a, axis=0)

    def test_max_no_axis(self, mock_jnp):
        a = np.array([1.0, 5.0, 3.0])
        mock_jnp.max.return_value = 5.0

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.max(a, xp=mock_jnp)
            mock_jnp.max.assert_called_once_with(a, axis=None)


# ---------------------------------------------------------------------------
# Reductions: inference paths with mocked backends
# ---------------------------------------------------------------------------


class TestReductionInferenceMocked:
    def test_sum_infers_torch(self):
        mock_torch = _make_mock_xp("mock_torch")

        class FakeTensor:
            pass

        fake_a = FakeTensor()
        mock_torch.Tensor = FakeTensor
        mock_torch.sum.return_value = 42.0

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.sum(fake_a)
            mock_torch.sum.assert_called_once()

    def test_sum_infers_jnp(self):
        mock_jnp = _make_mock_xp("mock_jnp")

        class FakeJaxArray:
            device_buffer = True

        fake_a = FakeJaxArray()
        mock_jnp.sum.return_value = 42.0

        with patch.object(_ops, "jnp", mock_jnp):
            with patch.object(_ops, "torch", None):
                result = _ops.sum(fake_a)
                mock_jnp.sum.assert_called_once()

    def test_mean_infers_torch(self):
        mock_torch = _make_mock_xp("mock_torch")

        class FakeTensor:
            pass

        fake_a = FakeTensor()
        mock_torch.Tensor = FakeTensor
        mock_torch.mean.return_value = 21.0

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.mean(fake_a)
            mock_torch.mean.assert_called_once()

    def test_mean_infers_jnp(self):
        mock_jnp = _make_mock_xp("mock_jnp")

        class FakeJaxArray:
            device_buffer = True

        fake_a = FakeJaxArray()
        mock_jnp.mean.return_value = 21.0

        with patch.object(_ops, "jnp", mock_jnp):
            with patch.object(_ops, "torch", None):
                result = _ops.mean(fake_a)
                mock_jnp.mean.assert_called_once()

    def test_max_infers_torch(self):
        mock_torch = _make_mock_xp("mock_torch")

        class FakeTensor:
            pass

        fake_a = FakeTensor()
        mock_torch.Tensor = FakeTensor
        _flat_max = MagicMock()
        _flat_max.item.return_value = 99.0
        mock_torch.max.return_value = _flat_max

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.max(fake_a)
            mock_torch.max.assert_called_once()

    def test_max_infers_jnp(self):
        mock_jnp = _make_mock_xp("mock_jnp")

        class FakeJaxArray:
            device_buffer = True

        fake_a = FakeJaxArray()
        mock_jnp.max.return_value = 99.0

        with patch.object(_ops, "jnp", mock_jnp):
            with patch.object(_ops, "torch", None):
                result = _ops.max(fake_a)
                mock_jnp.max.assert_called_once()


# ---------------------------------------------------------------------------
# Element-wise: sqrt, exp, log, abs -- torch branches
# ---------------------------------------------------------------------------


class TestElementwiseTorchMocked:
    @pytest.fixture()
    def mock_torch(self):
        return _make_mock_xp("mock_torch")

    def test_sqrt_uses_torch_sqrt(self, mock_torch):
        a = np.array([1.0, 4.0, 9.0])
        mock_torch.sqrt.return_value = np.array([1.0, 2.0, 3.0])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.sqrt(a, mock_torch)
            mock_torch.sqrt.assert_called_once_with(a)

    def test_exp_uses_torch_exp(self, mock_torch):
        a = np.array([0.0, 1.0])
        mock_torch.exp.return_value = np.array([1.0, np.e])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.exp(a, mock_torch)
            mock_torch.exp.assert_called_once_with(a)

    def test_log_uses_torch_log(self, mock_torch):
        a = np.array([1.0, np.e])
        mock_torch.log.return_value = np.array([0.0, 1.0])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.log(a, mock_torch)
            mock_torch.log.assert_called_once_with(a)

    def test_abs_uses_torch_abs(self, mock_torch):
        a = np.array([-3.0, 0.0, 4.0])
        mock_torch.abs.return_value = np.array([3.0, 0.0, 4.0])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.abs(a, mock_torch)
            mock_torch.abs.assert_called_once_with(a)

    def test_clip_uses_torch_clamp(self, mock_torch):
        a = np.array([-5.0, 0.5, 10.0])
        mock_torch.clamp.return_value = np.array([-1.0, 0.5, 5.0])

        with patch.object(_ops, "torch", mock_torch):
            result = _ops.clip(a, -1.0, 5.0, mock_torch)
            mock_torch.clamp.assert_called_once_with(a, -1.0, 5.0)


# ---------------------------------------------------------------------------
# Element-wise: sqrt, exp, log, abs, clip -- jnp branches
# ---------------------------------------------------------------------------


class TestElementwiseJaxMocked:
    @pytest.fixture()
    def mock_jnp(self):
        return _make_mock_xp("mock_jnp")

    def test_sqrt_uses_jnp_sqrt(self, mock_jnp):
        a = np.array([1.0, 4.0, 9.0])
        mock_jnp.sqrt.return_value = np.array([1.0, 2.0, 3.0])

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.sqrt(a, mock_jnp)
            mock_jnp.sqrt.assert_called_once_with(a)

    def test_exp_uses_jnp_exp(self, mock_jnp):
        a = np.array([0.0, 1.0])
        mock_jnp.exp.return_value = np.array([1.0, np.e])

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.exp(a, mock_jnp)
            mock_jnp.exp.assert_called_once_with(a)

    def test_log_uses_jnp_log(self, mock_jnp):
        a = np.array([1.0, np.e])
        mock_jnp.log.return_value = np.array([0.0, 1.0])

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.log(a, mock_jnp)
            mock_jnp.log.assert_called_once_with(a)

    def test_abs_uses_jnp_abs(self, mock_jnp):
        a = np.array([-3.0, 0.0, 4.0])
        mock_jnp.abs.return_value = np.array([3.0, 0.0, 4.0])

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.abs(a, mock_jnp)
            mock_jnp.abs.assert_called_once_with(a)

    def test_clip_uses_jnp_clip(self, mock_jnp):
        a = np.array([-5.0, 0.5, 10.0])
        mock_jnp.clip.return_value = np.array([-1.0, 0.5, 5.0])

        with patch.object(_ops, "jnp", mock_jnp):
            result = _ops.clip(a, -1.0, 5.0, mock_jnp)
            mock_jnp.clip.assert_called_once_with(a, -1.0, 5.0)


# ---------------------------------------------------------------------------
# to_backend_array -- default dtype paths for torch and jax
# ---------------------------------------------------------------------------


class TestToBackendArrayDefaultDtype:
    """Cover lines 100-101 (torch default dtype) and 104-105 (jnp default dtype)."""

    def test_torch_default_dtype_applied(self):
        mock_torch = MagicMock()
        mock_torch.tensor.return_value = np.array([1.0, 2.0])

        with patch.object(_ops, "torch", mock_torch):
            _ops.to_backend_array([1.0, 2.0], mock_torch)
            # The function should set dtype=mock_torch.float64 when dtype is None
            call_args = mock_torch.tensor.call_args
            assert call_args[1]["dtype"] is mock_torch.float64

    def test_jax_default_dtype_applied(self):
        mock_jnp = MagicMock()
        mock_jnp.array.return_value = np.array([1.0, 2.0])

        with patch.object(_ops, "jnp", mock_jnp):
            _ops.to_backend_array([1.0, 2.0], mock_jnp)
            call_args = mock_jnp.array.call_args
            assert call_args[1]["dtype"] is mock_jnp.float64
