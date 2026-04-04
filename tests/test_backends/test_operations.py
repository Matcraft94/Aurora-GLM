# SPDX-License-Identifier: MIT
"""Tests for aurora.core.backends.operations module."""

import numpy as np
import pytest

from aurora.core.backends.operations import (
    abs,
    cholesky,
    clip,
    concatenate,
    det,
    diag,
    eigh,
    exp,
    eye,
    get_namespace,
    inv,
    log,
    lstsq,
    matmul,
    max,
    mean,
    ones,
    qr,
    solve,
    slogdet,
    stack,
    sum,
    sqrt,
    to_backend_array,
    to_numpy,
    trace,
    transpose,
    zeros,
)

np.random.seed(42)


# ---------------------------------------------------------------------------
# get_namespace
# ---------------------------------------------------------------------------


class TestGetNamespace:
    def test_numpy_returns_np_and_none_device(self):
        xp, device = get_namespace("numpy")
        assert xp is np
        assert device is None

    def test_numpy_case_insensitive(self):
        xp, device = get_namespace("NumPy")
        assert xp is np

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_namespace("tensorflow")

    def test_pytorch_alias(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        xp, device = get_namespace("pytorch")
        assert xp is torch
        assert device is not None


# ---------------------------------------------------------------------------
# to_backend_array / to_numpy
# ---------------------------------------------------------------------------


class TestToBackendArray:
    def test_list_to_numpy(self):
        result = to_backend_array([1.0, 2.0, 3.0], np)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_numpy_passthrough(self):
        arr = np.array([1.0, 2.0])
        result = to_backend_array(arr, np)
        np.testing.assert_array_equal(result, arr)

    def test_dtype_kwarg(self):
        result = to_backend_array([1, 2, 3], np, dtype=np.float32)
        assert result.dtype == np.float32

    def test_pandas_like_object(self):
        class FakeSeries:
            values = np.array([4.0, 5.0])

        result = to_backend_array(FakeSeries(), np)
        np.testing.assert_array_equal(result, [4.0, 5.0])


class TestToNumpy:
    def test_ndarray_passthrough(self):
        arr = np.array([1.0, 2.0])
        assert to_numpy(arr) is arr

    def test_list_conversion(self):
        result = to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_torch_tensor(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        t = torch.tensor([1.0, 2.0])
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------


class TestSolve:
    def test_simple_system(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        x = solve(A, b, np)
        np.testing.assert_allclose(A @ x, b, rtol=1e-5, atol=1e-8)

    def test_identity_system(self):
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        x = solve(A, b, np)
        np.testing.assert_allclose(x, b, rtol=1e-5, atol=1e-8)


class TestCholesky:
    def test_psd_matrix(self):
        np.random.seed(42)
        A = np.random.randn(4, 4)
        A = A @ A.T + np.eye(4)  # positive definite
        L = cholesky(A, np)
        np.testing.assert_allclose(L @ L.T, A, rtol=1e-5, atol=1e-8)

    def test_lower_triangular(self):
        A = np.array([[4.0, 0.0], [0.0, 9.0]])
        L = cholesky(A, np)
        # Should be lower triangular
        np.testing.assert_allclose(L[0, 1], 0.0, atol=1e-12)


class TestInv:
    def test_inverse_identity(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        A_inv = inv(A, np)
        np.testing.assert_allclose(A @ A_inv, np.eye(2), rtol=1e-5, atol=1e-8)

    def test_diagonal_inverse(self):
        A = np.diag([2.0, 4.0, 5.0])
        A_inv = inv(A, np)
        np.testing.assert_allclose(A @ A_inv, np.eye(3), rtol=1e-5, atol=1e-8)


class TestDet:
    def test_identity_determinant(self):
        assert np.isclose(det(np.eye(3), np), 1.0)

    def test_singular_matrix(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        assert np.isclose(det(A, np), 0.0, atol=1e-12)

    def test_known_determinant(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.isclose(det(A, np), -2.0)


class TestSlogdet:
    def test_identity(self):
        sign, logdet = slogdet(np.eye(3), np)
        assert np.isclose(sign, 1.0)
        assert np.isclose(logdet, 0.0, atol=1e-12)

    def test_negative_det(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        sign, logdet = slogdet(A, np)
        assert sign < 0  # det = -2


class TestEigh:
    def test_symmetric_eigenvalues(self):
        np.random.seed(42)
        A = np.random.randn(3, 3)
        A = A + A.T
        eigenvalues, eigenvectors = eigh(A, np)
        # Eigenvalues should be ascending
        assert all(eigenvalues[i] <= eigenvalues[i + 1] for i in range(len(eigenvalues) - 1))

    def test_eigendecomposition_reconstructs(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        eigenvalues, eigenvectors = eigh(A, np)
        reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        np.testing.assert_allclose(reconstructed, A, rtol=1e-5, atol=1e-8)


class TestQR:
    def test_orthogonality(self):
        np.random.seed(42)
        A = np.random.randn(4, 3)
        Q, R = qr(A, np)
        np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), rtol=1e-5, atol=1e-8)

    def test_reconstructs(self):
        np.random.seed(42)
        A = np.random.randn(4, 3)
        Q, R = qr(A, np)
        np.testing.assert_allclose(Q @ R, A, rtol=1e-5, atol=1e-8)


class TestLstsq:
    def test_overdetermined_system(self):
        np.random.seed(42)
        A = np.random.randn(10, 3)
        x_true = np.array([1.0, 2.0, 3.0])
        b = A @ x_true
        x = lstsq(A, b, np)
        np.testing.assert_allclose(x, x_true, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Array creation
# ---------------------------------------------------------------------------


class TestEye:
    def test_shape_and_values(self):
        I = eye(4, np)
        assert I.shape == (4, 4)
        np.testing.assert_array_equal(I, np.eye(4))

    def test_dtype(self):
        I = eye(3, np, dtype=np.float32)
        assert I.dtype == np.float32


class TestZeros:
    def test_shape(self):
        z = zeros((3, 4), np)
        assert z.shape == (3, 4)
        assert np.all(z == 0)

    def test_dtype(self):
        z = zeros((2,), np, dtype=np.int32)
        assert z.dtype == np.int32


class TestOnes:
    def test_shape(self):
        o = ones((2, 5), np)
        assert o.shape == (2, 5)
        assert np.all(o == 1)


# ---------------------------------------------------------------------------
# Array manipulation
# ---------------------------------------------------------------------------


class TestConcatenate:
    def test_axis0(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = concatenate([a, b], axis=0, xp=np)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])

    def test_axis1(self):
        a = np.array([[1.0], [2.0]])
        b = np.array([[3.0], [4.0]])
        result = concatenate([a, b], axis=1, xp=np)
        assert result.shape == (2, 2)

    def test_infer_backend(self):
        a = np.array([1.0])
        b = np.array([2.0])
        result = concatenate([a, b])
        np.testing.assert_array_equal(result, [1.0, 2.0])


class TestStack:
    def test_default_axis(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = stack([a, b], xp=np)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[0], [1.0, 2.0])

    def test_axis_neg1(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = stack([a, b], axis=-1, xp=np)
        assert result.shape == (2, 2)


class TestDiag:
    def test_1d_creates_matrix(self):
        v = np.array([1.0, 2.0, 3.0])
        result = diag(v, np)
        np.testing.assert_array_equal(result, np.diag([1.0, 2.0, 3.0]))

    def test_2d_extracts_diagonal(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = diag(A, np)
        np.testing.assert_array_equal(result, [1.0, 4.0])


class TestTrace:
    def test_identity(self):
        assert np.isclose(trace(np.eye(4), np), 4.0)

    def test_general(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.isclose(trace(A, np), 5.0)


class TestMatmul:
    def test_matrix_multiply(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = matmul(A, B, np)
        expected = A @ B
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


class TestTranspose:
    def test_2d(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = transpose(A, np)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, A.T)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


class TestSum:
    def test_flat(self):
        a = np.array([1.0, 2.0, 3.0])
        assert np.isclose(sum(a, xp=np), 6.0)

    def test_axis0(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = sum(a, axis=0, xp=np)
        np.testing.assert_allclose(result, [4.0, 6.0])

    def test_infer_backend(self):
        a = np.array([1.0, 2.0])
        assert np.isclose(sum(a), 3.0)


class TestMean:
    def test_flat(self):
        a = np.array([2.0, 4.0, 6.0])
        assert np.isclose(mean(a, xp=np), 4.0)

    def test_axis1(self):
        a = np.array([[1.0, 3.0], [5.0, 7.0]])
        result = mean(a, axis=1, xp=np)
        np.testing.assert_allclose(result, [2.0, 6.0])

    def test_infer_backend(self):
        a = np.array([3.0, 6.0])
        assert np.isclose(mean(a), 4.5)


# ---------------------------------------------------------------------------
# Element-wise operations
# ---------------------------------------------------------------------------


class TestSqrt:
    def test_basic(self):
        a = np.array([1.0, 4.0, 9.0])
        result = sqrt(a, np)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], rtol=1e-5, atol=1e-8)


class TestExp:
    def test_zeros(self):
        a = np.zeros(3)
        result = exp(a, np)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0], rtol=1e-5, atol=1e-8)

    def test_values(self):
        a = np.array([0.0, 1.0])
        result = exp(a, np)
        np.testing.assert_allclose(result[1], np.e, rtol=1e-5)


class TestLog:
    def test_exp_inverse(self):
        a = np.array([1.0, np.e, np.e ** 2])
        result = log(a, np)
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0], rtol=1e-5, atol=1e-8)


class TestAbs:
    def test_mixed_signs(self):
        a = np.array([-3.0, 0.0, 4.0])
        result = abs(a, np)
        np.testing.assert_allclose(result, [3.0, 0.0, 4.0])


class TestMax:
    def test_flat(self):
        a = np.array([1.0, 5.0, 3.0])
        assert np.isclose(max(a, xp=np), 5.0)

    def test_axis0(self):
        a = np.array([[1.0, 5.0], [3.0, 2.0]])
        result = max(a, axis=0, xp=np)
        np.testing.assert_allclose(result, [3.0, 5.0])

    def test_infer_backend(self):
        a = np.array([10.0, 20.0])
        assert np.isclose(max(a), 20.0)


class TestClip:
    def test_clips_low_and_high(self):
        a = np.array([-5.0, 0.5, 10.0])
        result = clip(a, -1.0, 5.0, np)
        np.testing.assert_allclose(result, [-1.0, 0.5, 5.0])

    def test_no_clip_needed(self):
        a = np.array([2.0, 3.0])
        result = clip(a, 0.0, 10.0, np)
        np.testing.assert_allclose(result, [2.0, 3.0])


# ---------------------------------------------------------------------------
# Torch backend integration (skipped if unavailable)
# ---------------------------------------------------------------------------


class TestTorchBackend:
    @pytest.fixture(autouse=True)
    def _check_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        self.torch = torch
        self.xp = torch

    def test_solve(self):
        import torch
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        b = torch.tensor([5.0, 7.0], dtype=torch.float64)
        x = solve(A, b, torch)
        np.testing.assert_allclose(x.numpy(), [1.6, 1.8], rtol=1e-5)

    def test_eye(self):
        import torch
        I = eye(3, torch)
        assert I.shape == (3, 3)
        np.testing.assert_allclose(I.numpy(), np.eye(3), atol=1e-12)

    def test_zeros_and_ones(self):
        import torch
        z = zeros((2, 3), torch)
        assert z.shape == (2, 3)
        assert torch.all(z == 0)
        o = ones((2, 3), torch)
        assert torch.all(o == 1)

    def test_elementwise(self):
        import torch
        a = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        np.testing.assert_allclose(sqrt(a, torch).numpy(), [1.0, 2.0, 3.0], rtol=1e-5)
        np.testing.assert_allclose(exp(torch.zeros(3, dtype=torch.float64), torch).numpy(), [1.0, 1.0, 1.0], rtol=1e-5)
        np.testing.assert_allclose(log(torch.tensor([1.0, np.e], dtype=torch.float64), torch).numpy(), [0.0, 1.0], rtol=1e-5)

    def test_sum_mean(self):
        import torch
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        np.testing.assert_allclose(sum(a, xp=torch).numpy(), 6.0, rtol=1e-5)
        np.testing.assert_allclose(mean(a, xp=torch).numpy(), 2.0, rtol=1e-5)

    def test_det(self):
        import torch
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        np.testing.assert_allclose(det(A, torch).numpy(), -2.0, rtol=1e-5)

    def test_clip(self):
        import torch
        a = torch.tensor([-5.0, 0.5, 10.0], dtype=torch.float64)
        result = clip(a, -1.0, 5.0, torch)
        np.testing.assert_allclose(result.numpy(), [-1.0, 0.5, 5.0])

    def test_concatenate(self):
        import torch
        a = torch.tensor([1.0, 2.0], dtype=torch.float64)
        b = torch.tensor([3.0, 4.0], dtype=torch.float64)
        result = concatenate([a, b], xp=torch)
        np.testing.assert_allclose(result.numpy(), [1.0, 2.0, 3.0, 4.0])

    def test_stack(self):
        import torch
        a = torch.tensor([1.0, 2.0], dtype=torch.float64)
        b = torch.tensor([3.0, 4.0], dtype=torch.float64)
        result = stack([a, b], xp=torch)
        assert result.shape == (2, 2)

    def test_to_numpy_from_torch(self):
        import torch
        t = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)

    def test_to_backend_array_torch(self):
        import torch
        result = to_backend_array([1.0, 2.0], torch)
        assert isinstance(result, torch.Tensor)
        np.testing.assert_allclose(result.numpy(), [1.0, 2.0], rtol=1e-5)

    def test_get_namespace_torch(self):
        import torch
        xp, device = get_namespace("torch")
        assert xp is torch

    def test_cholesky(self):
        import torch
        np.random.seed(42)
        A = np.random.randn(3, 3)
        A = A @ A.T + np.eye(3)
        A_t = torch.tensor(A, dtype=torch.float64)
        L = cholesky(A_t, torch)
        np.testing.assert_allclose(L.numpy() @ L.numpy().T, A, rtol=1e-5, atol=1e-8)

    def test_inv(self):
        import torch
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        A_inv = inv(A, torch)
        np.testing.assert_allclose(A_inv.numpy() @ A.numpy(), np.eye(2), rtol=1e-5, atol=1e-8)

    def test_eigh(self):
        import torch
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        eigenvalues, eigenvectors = eigh(A, torch)
        reconstructed = eigenvectors.numpy() @ np.diag(eigenvalues.numpy()) @ eigenvectors.numpy().T
        np.testing.assert_allclose(reconstructed, A.numpy(), rtol=1e-5, atol=1e-8)

    def test_qr(self):
        import torch
        np.random.seed(42)
        A = np.random.randn(4, 3)
        A_t = torch.tensor(A, dtype=torch.float64)
        Q, R = qr(A_t, torch)
        np.testing.assert_allclose(Q.numpy() @ R.numpy(), A, rtol=1e-5, atol=1e-8)

    def test_lstsq(self):
        import torch
        np.random.seed(42)
        A = np.random.randn(10, 3)
        x_true = np.array([1.0, 2.0, 3.0])
        b = A @ x_true
        A_t = torch.tensor(A, dtype=torch.float64)
        b_t = torch.tensor(b, dtype=torch.float64)
        x = lstsq(A_t, b_t, torch)
        np.testing.assert_allclose(x.numpy(), x_true, rtol=1e-5, atol=1e-8)

    def test_transpose(self):
        import torch
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
        result = transpose(A, torch)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result.numpy(), A.numpy().T)

    def test_matmul(self):
        import torch
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64)
        result = matmul(A, B, torch)
        expected = A.numpy() @ B.numpy()
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-8)

    def test_slogdet(self):
        import torch
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        sign, logdet = slogdet(A, torch)
        assert sign.item() < 0  # det = -2

    def test_diag(self):
        import torch
        v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = diag(v, torch)
        np.testing.assert_allclose(result.numpy(), np.diag([1.0, 2.0, 3.0]))

    def test_trace(self):
        import torch
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        assert np.isclose(trace(A, torch).item(), 5.0)

    def test_abs(self):
        import torch
        a = torch.tensor([-3.0, 0.0, 4.0], dtype=torch.float64)
        result = abs(a, torch)
        np.testing.assert_allclose(result.numpy(), [3.0, 0.0, 4.0])

    def test_max(self):
        import torch
        a = torch.tensor([1.0, 5.0, 3.0], dtype=torch.float64)
        assert np.isclose(max(a, xp=torch).item(), 5.0)
