# SPDX-License-Identifier: MIT
"""Comprehensive tests for aurora.core.backends.operations and
aurora.core.backends.pytorch_backend.

The existing test_operations.py covers many NumPy and some Torch paths but
misses JAX paths, several edge cases, and the PyTorchBackend class entirely.
This file fills those gaps to raise coverage on the ~176 missed statements.

Pattern: np.random.seed(42), np.testing.assert_allclose(..., rtol=1e-5, atol=1e-8).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

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
    max as ops_max,
    mean,
    ones,
    qr,
    slogdet,
    solve,
    sqrt,
    stack,
    sum as ops_sum,
    to_backend_array,
    to_numpy,
    trace,
    transpose,
    zeros,
)

# ---------------------------------------------------------------------------
# Optional backends
# ---------------------------------------------------------------------------
try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    HAS_JAX = False

np.random.seed(42)


# ===========================================================================
# get_namespace -- edge cases
# ===========================================================================


class TestGetNamespaceExtra:
    """Extra coverage for get_namespace beyond what exists."""

    def test_numpy_lowercase(self):
        xp, dev = get_namespace("numpy")
        assert xp is np
        assert dev is None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_namespace("cupy")

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_with_explicit_device(self):
        xp, dev = get_namespace("torch", device="cpu")
        assert xp is torch
        assert dev == torch.device("cpu")

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_alias(self):
        xp, _ = get_namespace("pytorch")
        assert xp is torch

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax(self):
        xp, dev = get_namespace("jax")
        assert xp is jnp
        assert dev is None

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_not_installed_error(self):
        """If jnp were None the function would raise ImportError."""
        # jnp is not None since we have JAX; just confirm the happy path
        xp, _ = get_namespace("jax")
        assert xp is jnp

    @pytest.mark.skipif(HAS_TORCH, reason="Test only runs without torch")
    def test_torch_import_error(self):
        with pytest.raises(ImportError, match="PyTorch"):
            get_namespace("torch")

    @pytest.mark.skipif(HAS_JAX, reason="Test only runs without JAX")
    def test_jax_import_error(self):
        with pytest.raises(ImportError, match="JAX"):
            get_namespace("jax")


# ===========================================================================
# to_backend_array -- torch / jax paths
# ===========================================================================


class TestToBackendArrayExtra:
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_output(self):
        result = to_backend_array([1.0, 2.0], torch)
        assert isinstance(result, torch.Tensor)
        assert_allclose(result.numpy(), [1.0, 2.0], rtol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_tensor_input_roundtrip(self):
        t = torch.tensor([3.0, 4.0], dtype=torch.float64)
        result = to_backend_array(t, np)
        assert isinstance(result, np.ndarray)
        assert_allclose(result, [3.0, 4.0], rtol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_dtype_kwarg(self):
        result = to_backend_array([1, 2], torch, dtype=torch.float32)
        assert result.dtype == torch.float32

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_device_kwarg(self):
        result = to_backend_array([1.0, 2.0], torch, device=torch.device("cpu"))
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_output(self):
        result = to_backend_array([1.0, 2.0], jnp)
        assert_allclose(np.asarray(result), [1.0, 2.0], rtol=1e-5)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_dtype_kwarg(self):
        result = to_backend_array([1, 2], jnp, dtype=jnp.float32)
        assert result.dtype == jnp.float32

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_array_input_to_numpy(self):
        """JAX array has .device_buffer so it should be converted to numpy."""
        a = jnp.array([5.0, 6.0])
        result = to_backend_array(a, np)
        assert isinstance(result, np.ndarray)
        assert_allclose(result, [5.0, 6.0], rtol=1e-5)

    def test_fallback_unknown_xp(self):
        """If xp is something unexpected, it should still use numpy."""

        class FakeXp:
            pass

        result = to_backend_array([1.0, 2.0], FakeXp())
        assert isinstance(result, np.ndarray)

    def test_pandas_like_values(self):
        class FakeSeries:
            values = np.array([10.0, 20.0])

        result = to_backend_array(FakeSeries(), np)
        assert_allclose(result, [10.0, 20.0])


# ===========================================================================
# to_numpy -- extra paths
# ===========================================================================


class TestToNumpyExtra:
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_passthrough(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
        assert_allclose(result, [1.0, 2.0])

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_conversion(self):
        a = jnp.array([1.0, 2.0])
        result = to_numpy(a)
        assert isinstance(result, np.ndarray)
        assert_allclose(result, [1.0, 2.0])

    def test_generic_input(self):
        result = to_numpy([3, 4])
        assert isinstance(result, np.ndarray)
        assert_allclose(result, [3, 4])


# ===========================================================================
# Linear algebra -- JAX backend coverage
# ===========================================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestLinalgJAX:
    """Cover the jnp branches of every linalg function."""

    def test_solve(self):
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        b = jnp.array([5.0, 7.0])
        x = solve(A, b, jnp)
        assert_allclose(np.asarray(x), np.linalg.solve(np.asarray(A), np.asarray(b)), rtol=1e-5)

    def test_cholesky(self):
        np.random.seed(42)
        A_np = np.random.randn(3, 3)
        A_np = A_np @ A_np.T + np.eye(3)
        A = jnp.array(A_np)
        L = cholesky(A, jnp)
        assert_allclose(np.asarray(L @ L.T), A_np, rtol=1e-4)

    def test_inv(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        A_inv = inv(A, jnp)
        np.testing.assert_allclose(
            np.asarray(A_inv) @ np.asarray(A), np.eye(2), rtol=1e-4
        )

    def test_det(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        d = det(A, jnp)
        assert_allclose(float(d), -2.0, rtol=1e-5)

    def test_slogdet(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        sign, logdet = slogdet(A, jnp)
        assert float(sign) < 0

    def test_eigh(self):
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        eigenvalues, eigenvectors = eigh(A, jnp)
        reconstructed = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T
        assert_allclose(np.asarray(reconstructed), np.asarray(A), rtol=1e-4)

    def test_qr(self):
        np.random.seed(42)
        A_np = np.random.randn(4, 3)
        A = jnp.array(A_np)
        Q, R = qr(A, jnp)
        assert_allclose(np.asarray(Q @ R), A_np, rtol=1e-4)

    def test_lstsq(self):
        np.random.seed(42)
        A_np = np.random.randn(10, 3)
        x_true = np.array([1.0, 2.0, 3.0])
        b_np = A_np @ x_true
        x = lstsq(jnp.array(A_np), jnp.array(b_np), jnp)
        assert_allclose(np.asarray(x), x_true, rtol=1e-4)


# ===========================================================================
# Array creation -- JAX / Torch extra paths
# ===========================================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestArrayCreationJAX:
    def test_eye(self):
        I = eye(3, jnp)
        assert_allclose(np.asarray(I), np.eye(3), rtol=1e-10)

    def test_eye_dtype(self):
        I = eye(2, jnp, dtype=jnp.float32)
        assert I.dtype == jnp.float32

    def test_zeros(self):
        z = zeros((2, 3), jnp)
        assert_allclose(np.asarray(z), np.zeros((2, 3)), rtol=1e-10)

    def test_zeros_dtype(self):
        z = zeros((2,), jnp, dtype=jnp.int32)
        assert z.dtype == jnp.int32

    def test_ones(self):
        o = ones((3, 2), jnp)
        assert_allclose(np.asarray(o), np.ones((3, 2)), rtol=1e-10)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestArrayCreationTorchExtra:
    def test_eye_with_device(self):
        I = eye(3, torch, device=torch.device("cpu"))
        assert I.shape == (3, 3)
        assert_allclose(I.numpy(), np.eye(3), atol=1e-12)

    def test_eye_dtype(self):
        I = eye(2, torch, dtype=torch.float32)
        assert I.dtype == torch.float32

    def test_zeros_with_device(self):
        z = zeros((2, 3), torch, device=torch.device("cpu"))
        assert z.shape == (2, 3)

    def test_ones_with_device(self):
        o = ones((2, 3), torch, device=torch.device("cpu"))
        assert o.shape == (2, 3)


# ===========================================================================
# Array manipulation -- JAX paths + infer-backend paths
# ===========================================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestConcatenateStackJAX:
    def test_concatenate(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([3.0, 4.0])
        result = concatenate([a, b], axis=0, xp=jnp)
        assert_allclose(np.asarray(result), [1.0, 2.0, 3.0, 4.0])

    def test_concatenate_infer(self):
        a = jnp.array([1.0])
        b = jnp.array([2.0])
        result = concatenate([a, b])
        assert_allclose(np.asarray(result), [1.0, 2.0])

    def test_stack(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([3.0, 4.0])
        result = stack([a, b], axis=0, xp=jnp)
        assert result.shape == (2, 2)

    def test_stack_infer(self):
        a = jnp.array([1.0])
        b = jnp.array([2.0])
        result = stack([a, b])
        assert result.shape == (2, 1)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestConcatenateStackTorchInfer:
    def test_concatenate_infer(self):
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        result = concatenate([a, b])
        assert_allclose(result.numpy(), [1.0, 2.0])

    def test_stack_infer(self):
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        result = stack([a, b])
        assert result.shape == (2, 1)


# ===========================================================================
# diag, trace, matmul, transpose -- JAX paths
# ===========================================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestMatrixOpsJAX:
    def test_diag_create(self):
        v = jnp.array([1.0, 2.0, 3.0])
        result = diag(v, jnp)
        assert_allclose(np.asarray(result), np.diag([1.0, 2.0, 3.0]), rtol=1e-10)

    def test_diag_extract(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        d = diag(A, jnp)
        assert_allclose(np.asarray(d), [1.0, 4.0])

    def test_trace(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        t = trace(A, jnp)
        assert_allclose(float(t), 5.0)

    def test_matmul(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        C = matmul(A, B, jnp)
        assert_allclose(np.asarray(C), np.asarray(A) @ np.asarray(B), rtol=1e-5)

    def test_transpose(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        T = transpose(A, jnp)
        assert T.shape == (2, 3)
        assert_allclose(np.asarray(T), np.asarray(A).T)


# ===========================================================================
# Reductions -- sum, mean, max -- JAX paths + infer paths
# ===========================================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestReductionsJAX:
    def test_sum_flat(self):
        a = jnp.array([1.0, 2.0, 3.0])
        s = ops_sum(a, xp=jnp)
        assert_allclose(float(s), 6.0)

    def test_sum_axis(self):
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        s = ops_sum(a, axis=0, xp=jnp)
        assert_allclose(np.asarray(s), [4.0, 6.0])

    def test_sum_infer(self):
        a = jnp.array([1.0, 2.0])
        s = ops_sum(a)
        assert_allclose(float(s), 3.0)

    def test_mean_flat(self):
        a = jnp.array([2.0, 4.0, 6.0])
        m = mean(a, xp=jnp)
        assert_allclose(float(m), 4.0)

    def test_mean_axis(self):
        a = jnp.array([[1.0, 3.0], [5.0, 7.0]])
        m = mean(a, axis=1, xp=jnp)
        assert_allclose(np.asarray(m), [2.0, 6.0])

    def test_mean_infer(self):
        a = jnp.array([3.0, 6.0])
        m = mean(a)
        assert_allclose(float(m), 4.5)

    def test_max_flat(self):
        a = jnp.array([1.0, 5.0, 3.0])
        m = ops_max(a, xp=jnp)
        assert_allclose(float(m), 5.0)

    def test_max_axis(self):
        a = jnp.array([[1.0, 5.0], [3.0, 2.0]])
        m = ops_max(a, axis=0, xp=jnp)
        assert_allclose(np.asarray(m), [3.0, 5.0])

    def test_max_infer(self):
        a = jnp.array([10.0, 20.0])
        m = ops_max(a)
        assert_allclose(float(m), 20.0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestReductionsTorchInfer:
    def test_sum_infer(self):
        a = torch.tensor([1.0, 2.0])
        s = ops_sum(a)
        assert_allclose(s.item(), 3.0)

    def test_mean_infer(self):
        a = torch.tensor([3.0, 6.0])
        m = mean(a)
        assert_allclose(m.item(), 4.5)

    def test_max_infer(self):
        a = torch.tensor([10.0, 20.0])
        m = ops_max(a)
        assert_allclose(m.item(), 20.0)

    def test_sum_axis(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        s = ops_sum(a, axis=0, xp=torch)
        assert_allclose(s.numpy(), [4.0, 6.0])

    def test_mean_axis(self):
        a = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
        m = mean(a, axis=1, xp=torch)
        assert_allclose(m.numpy(), [2.0, 6.0])

    def test_max_axis(self):
        a = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        m = ops_max(a, axis=0, xp=torch)
        assert_allclose(m.numpy(), [3.0, 5.0])


# ===========================================================================
# Element-wise -- sqrt, exp, log, abs, clip -- JAX paths
# ===========================================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestElementwiseJAX:
    def test_sqrt(self):
        a = jnp.array([1.0, 4.0, 9.0])
        r = sqrt(a, jnp)
        assert_allclose(np.asarray(r), [1.0, 2.0, 3.0], rtol=1e-5)

    def test_exp(self):
        a = jnp.array([0.0, 1.0])
        r = exp(a, jnp)
        assert_allclose(np.asarray(r), [1.0, np.e], rtol=1e-5)

    def test_log(self):
        a = jnp.array([1.0, np.e, np.e ** 2])
        r = log(a, jnp)
        assert_allclose(np.asarray(r), [0.0, 1.0, 2.0], rtol=1e-5)

    def test_abs(self):
        a = jnp.array([-3.0, 0.0, 4.0])
        r = abs(a, jnp)
        assert_allclose(np.asarray(r), [3.0, 0.0, 4.0])

    def test_clip(self):
        a = jnp.array([-5.0, 0.5, 10.0])
        r = clip(a, -1.0, 5.0, jnp)
        assert_allclose(np.asarray(r), [-1.0, 0.5, 5.0])


# ===========================================================================
# Fallback paths -- unknown xp uses numpy
# ===========================================================================


class TestFallbackPaths:
    """When xp is not np/torch/jnp the else branch should fall back to numpy."""

    class FakeXp:
        pass

    def test_solve_fallback(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        x = solve(A, b, self.FakeXp())
        assert_allclose(A @ x, b, rtol=1e-5)

    def test_cholesky_fallback(self):
        A = np.eye(2) * 4.0
        L = cholesky(A, self.FakeXp())
        assert_allclose(L @ L.T, A, rtol=1e-5)

    def test_inv_fallback(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        A_inv = inv(A, self.FakeXp())
        assert_allclose(A @ A_inv, np.eye(2), rtol=1e-5, atol=1e-14)

    def test_det_fallback(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_allclose(det(A, self.FakeXp()), -2.0, rtol=1e-5)

    def test_slogdet_fallback(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        sign, _ = slogdet(A, self.FakeXp())
        assert sign < 0

    def test_eigh_fallback(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        eigenvalues, _ = eigh(A, self.FakeXp())
        assert_allclose(eigenvalues, np.linalg.eigvalsh(A), rtol=1e-5)

    def test_qr_fallback(self):
        np.random.seed(42)
        A = np.random.randn(4, 3)
        Q, R = qr(A, self.FakeXp())
        assert_allclose(Q @ R, A, rtol=1e-5)

    def test_lstsq_fallback(self):
        np.random.seed(42)
        A = np.random.randn(10, 3)
        x_true = np.array([1.0, 2.0, 3.0])
        b = A @ x_true
        x = lstsq(A, b, self.FakeXp())
        assert_allclose(x, x_true, rtol=1e-5)

    def test_eye_fallback(self):
        I = eye(3, self.FakeXp())
        assert_allclose(I, np.eye(3))

    def test_zeros_fallback(self):
        z = zeros((2, 3), self.FakeXp())
        assert z.shape == (2, 3)
        assert np.all(z == 0)

    def test_ones_fallback(self):
        o = ones((2, 3), self.FakeXp())
        assert o.shape == (2, 3)
        assert np.all(o == 1)

    def test_concatenate_fallback(self):
        a = np.array([1.0])
        b = np.array([2.0])
        r = concatenate([a, b], xp=self.FakeXp())
        assert_allclose(r, [1.0, 2.0])

    def test_stack_fallback(self):
        a = np.array([1.0])
        b = np.array([2.0])
        r = stack([a, b], xp=self.FakeXp())
        assert r.shape == (2, 1)

    def test_diag_fallback(self):
        v = np.array([1.0, 2.0])
        r = diag(v, self.FakeXp())
        assert_allclose(r, np.diag([1.0, 2.0]))

    def test_trace_fallback(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_allclose(trace(A, self.FakeXp()), 5.0)

    def test_matmul_fallback(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        assert_allclose(matmul(A, B, self.FakeXp()), A @ B, rtol=1e-5)

    def test_transpose_fallback(self):
        A = np.array([[1, 2], [3, 4]])
        assert_allclose(transpose(A, self.FakeXp()), A.T)

    def test_sqrt_fallback(self):
        a = np.array([4.0, 9.0])
        assert_allclose(sqrt(a, self.FakeXp()), [2.0, 3.0], rtol=1e-5)

    def test_exp_fallback(self):
        a = np.array([0.0])
        assert_allclose(exp(a, self.FakeXp()), [1.0], rtol=1e-5)

    def test_log_fallback(self):
        a = np.array([1.0, np.e])
        assert_allclose(log(a, self.FakeXp()), [0.0, 1.0], rtol=1e-5)

    def test_abs_fallback(self):
        a = np.array([-3.0, 4.0])
        assert_allclose(abs(a, self.FakeXp()), [3.0, 4.0])

    def test_clip_fallback(self):
        a = np.array([-5.0, 0.5, 10.0])
        r = clip(a, -1.0, 5.0, self.FakeXp())
        assert_allclose(r, [-1.0, 0.5, 5.0])


# ===========================================================================
# PyTorch backend class -- pytorch_backend.py (0% coverage)
# ===========================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestPyTorchBackendClass:
    """Cover aurora.core.backends.pytorch_backend.PyTorchBackend."""

    @pytest.fixture()
    def backend(self):
        from aurora.core.backends.pytorch_backend import PyTorchBackend

        return PyTorchBackend()

    def test_init(self, backend):
        assert backend._device is not None
        assert backend._dtype is not None

    def test_array_from_list(self, backend):
        t = backend.array([1.0, 2.0, 3.0])
        assert isinstance(t, torch.Tensor)
        assert_allclose(t.cpu().numpy(), [1.0, 2.0, 3.0])

    def test_array_from_tensor(self, backend):
        original = torch.tensor([4.0, 5.0])
        t = backend.array(original)
        assert isinstance(t, torch.Tensor)
        assert_allclose(t.cpu().numpy(), [4.0, 5.0])

    def test_array_dtype_conversion(self, backend):
        t = backend.array([1, 2, 3], dtype=torch.float32)
        assert t.dtype == torch.float32

    def test_as_numpy_from_tensor(self, backend):
        t = torch.tensor([1.0, 2.0])
        result = backend.as_numpy(t)
        assert isinstance(result, np.ndarray)
        assert_allclose(result, [1.0, 2.0])

    def test_as_numpy_passthrough_non_tensor(self, backend):
        data = [1, 2, 3]
        assert backend.as_numpy(data) is data

    def test_grad(self, backend):
        def f(x):
            return (x ** 2).sum()

        grad_fn = backend.grad(f)
        x = torch.tensor([3.0, -2.0], dtype=torch.float64)
        g = grad_fn(x)
        assert_allclose(g.numpy(), [6.0, -4.0], rtol=1e-5)

    def test_grad_empty_args_raises(self, backend):
        grad_fn = backend.grad(lambda x: x.sum())
        with pytest.raises(ValueError, match="at least one positional"):
            grad_fn()

    def test_grad_non_scalar_raises(self, backend):
        def f(x):
            return x

        grad_fn = backend.grad(f)
        with pytest.raises(ValueError, match="scalar"):
            grad_fn(torch.tensor([1.0, 2.0], dtype=torch.float64))

    def test_jit(self, backend):
        def f(x):
            return x * 2

        compiled = backend.jit(f)
        t = torch.tensor([1.0, 2.0])
        result = compiled(t)
        assert_allclose(result.numpy(), [2.0, 4.0])

    def test_device_put_tensor(self, backend):
        t = torch.tensor([1.0, 2.0])
        result = backend.device_put(t)
        assert isinstance(result, torch.Tensor)
        assert result.device.type in {"cpu", "cuda"}

    def test_device_put_list(self, backend):
        result = backend.device_put([5.0, 6.0])
        assert isinstance(result, torch.Tensor)
        assert_allclose(result.cpu().numpy(), [5.0, 6.0])

    def test_vmap(self, backend):
        def square(x):
            return x * x

        vmap_fn = backend.vmap(square)
        data = backend.array([0.0, 1.0, 2.0, 3.0])
        result = vmap_fn(data)
        assert_allclose(result.cpu().numpy(), [0.0, 1.0, 4.0, 9.0], rtol=1e-5)

    def test_vmap_fallback(self, backend, monkeypatch):
        """Cover the fallback path when torch.vmap is not available."""
        monkeypatch.delattr(torch, "vmap", raising=False)

        def increment(x):
            return x + 1

        vmap_fn = backend.vmap(increment)
        data = backend.array([1.0, 2.0, 3.0])
        result = vmap_fn(data)
        assert_allclose(result.cpu().numpy(), [2.0, 3.0, 4.0])

    def test_vmap_fallback_non_standard_axes_raises(self, backend, monkeypatch):
        """Fallback vmap should reject non-zero in_axes/out_axes."""
        monkeypatch.delattr(torch, "vmap", raising=False)

        with pytest.raises(Exception, match="torch.vmap not available"):
            backend.vmap(lambda x: x, in_axes=1)

    def test_partial(self, backend):
        def add(a, b):
            return a + b

        add5 = backend.partial(add, 5)
        result = add5(3)
        assert result == 8

    def test_create_backend_factory(self):
        from aurora.core.backends.pytorch_backend import PyTorchBackend, create_backend

        b = create_backend()
        assert isinstance(b, PyTorchBackend)


# ===========================================================================
# Torch element-wise extra (clip, sqrt, exp, log, abs, max with torch xp)
# ===========================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestTorchElementwiseExtra:
    def test_sqrt(self):
        a = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        r = sqrt(a, torch)
        assert_allclose(r.numpy(), [1.0, 2.0, 3.0], rtol=1e-5)

    def test_exp(self):
        a = torch.tensor([0.0, 1.0], dtype=torch.float64)
        r = exp(a, torch)
        assert_allclose(r.numpy(), [1.0, np.e], rtol=1e-5)

    def test_log(self):
        a = torch.tensor([1.0, np.e], dtype=torch.float64)
        r = log(a, torch)
        assert_allclose(r.numpy(), [0.0, 1.0], rtol=1e-5)

    def test_abs(self):
        a = torch.tensor([-3.0, 0.0, 4.0], dtype=torch.float64)
        r = abs(a, torch)
        assert_allclose(r.numpy(), [3.0, 0.0, 4.0])

    def test_clip(self):
        a = torch.tensor([-5.0, 0.5, 10.0], dtype=torch.float64)
        r = clip(a, -1.0, 5.0, torch)
        assert_allclose(r.numpy(), [-1.0, 0.5, 5.0])

    def test_max_flat(self):
        a = torch.tensor([1.0, 5.0, 3.0], dtype=torch.float64)
        assert_allclose(ops_max(a, xp=torch).item(), 5.0)

    def test_max_axis(self):
        a = torch.tensor([[1.0, 5.0], [3.0, 2.0]], dtype=torch.float64)
        r = ops_max(a, axis=0, xp=torch)
        assert_allclose(r.numpy(), [3.0, 5.0])


# ===========================================================================
# NumPy-path tests (xp=np) — direct coverage of the primary code paths
# ===========================================================================


class TestToNumpyNumpyPaths:
    """Cover the NumPy-specific branches in to_numpy."""

    def test_ndarray_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert result is arr  # same object

    def test_list_input(self):
        result = to_numpy([4, 5, 6])
        assert isinstance(result, np.ndarray)
        assert_allclose(result, [4, 5, 6])

    def test_scalar_input(self):
        result = to_numpy(7.0)
        assert isinstance(result, np.ndarray)

    def test_nested_list(self):
        result = to_numpy([[1, 2], [3, 4]])
        assert result.shape == (2, 2)
        assert_allclose(result, [[1, 2], [3, 4]])


class TestToBackendArrayNumpyPaths:
    """Cover the NumPy branch (xp is np) in to_backend_array."""

    def test_list_to_numpy(self):
        result = to_backend_array([1.0, 2.0, 3.0], np)
        assert isinstance(result, np.ndarray)
        assert_allclose(result, [1.0, 2.0, 3.0])

    def test_scalar_to_numpy(self):
        result = to_backend_array(5.0, np)
        assert isinstance(result, np.ndarray)

    def test_ndarray_to_numpy(self):
        arr = np.array([10.0, 20.0])
        result = to_backend_array(arr, np)
        assert isinstance(result, np.ndarray)
        assert_allclose(result, [10.0, 20.0])

    def test_dtype_kwarg(self):
        result = to_backend_array([1, 2, 3], np, dtype=np.float32)
        assert result.dtype == np.float32

    def test_pandas_like_input(self):
        class FakeDF:
            values = np.array([100.0, 200.0])

        result = to_backend_array(FakeDF(), np)
        assert_allclose(result, [100.0, 200.0])


class TestLinalgNumPyPaths:
    """Cover the xp is np branches for all linalg functions."""

    def test_solve(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        x = solve(A, b, np)
        assert_allclose(A @ x, b, rtol=1e-10)

    def test_solve_1x1(self):
        x = solve(np.array([[4.0]]), np.array([8.0]), np)
        assert_allclose(x, [2.0], rtol=1e-10)

    def test_cholesky(self):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        L = cholesky(A, np)
        assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_inv(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        A_inv = inv(A, np)
        assert_allclose(A @ A_inv, np.eye(2), atol=1e-12)

    def test_inv_1x1(self):
        A_inv = inv(np.array([[5.0]]), np)
        assert_allclose(A_inv, [[0.2]], rtol=1e-10)

    def test_det(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_allclose(det(A, np), -2.0, rtol=1e-10)

    def test_det_identity(self):
        assert_allclose(det(np.eye(3), np), 1.0, rtol=1e-10)

    def test_slogdet(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        sign, logabsdet = slogdet(A, np)
        assert sign < 0
        assert_allclose(logabsdet, np.log(2.0), rtol=1e-10)

    def test_slogdet_positive(self):
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        sign, logabsdet = slogdet(A, np)
        assert sign > 0
        assert_allclose(logabsdet, np.log(6.0), rtol=1e-10)

    def test_eigh(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        eigenvalues, eigenvectors = eigh(A, np)
        assert_allclose(eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T, A, rtol=1e-10)

    def test_qr(self):
        np.random.seed(42)
        A = np.random.randn(4, 3)
        Q, R = qr(A, np)
        assert_allclose(Q @ R, A, rtol=1e-10)
        # Reduced mode: Q is 4x3, check Q^T Q = I(3)
        assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)

    def test_qr_square(self):
        np.random.seed(7)
        A = np.random.randn(3, 3)
        Q, R = qr(A, np)
        assert_allclose(Q @ R, A, rtol=1e-10)

    def test_lstsq(self):
        np.random.seed(42)
        A = np.random.randn(10, 3)
        x_true = np.array([1.0, 2.0, 3.0])
        b = A @ x_true
        x = lstsq(A, b, np)
        assert_allclose(x, x_true, rtol=1e-10)

    def test_lstsq_non_square(self):
        """Overdetermined system (more rows than cols)."""
        np.random.seed(42)
        A = np.random.randn(20, 3)
        b = np.random.randn(20)
        x = lstsq(A, b, np)
        # Verify it's a least-squares solution: A^T A x = A^T b
        assert_allclose(A.T @ A @ x, A.T @ b, rtol=1e-8)

    def test_lstsq_1x1(self):
        x = lstsq(np.array([[3.0]]), np.array([9.0]), np)
        assert_allclose(x, [3.0], rtol=1e-10)


class TestArrayCreationNumPyPaths:
    """Cover the xp is np branches for eye, zeros, ones with dtype."""

    def test_eye_default(self):
        I = eye(4, np)
        assert_allclose(I, np.eye(4))

    def test_eye_dtype(self):
        I = eye(3, np, dtype=np.float32)
        assert I.dtype == np.float32

    def test_eye_int(self):
        I = eye(2, np, dtype=np.int32)
        assert I.dtype == np.int32

    def test_zeros_default(self):
        z = zeros((3, 4), np)
        assert z.shape == (3, 4)
        assert np.all(z == 0)
        assert z.dtype == np.float64

    def test_zeros_dtype(self):
        z = zeros((2,), np, dtype=np.int32)
        assert z.dtype == np.int32

    def test_ones_default(self):
        o = ones((2, 5), np)
        assert o.shape == (2, 5)
        assert np.all(o == 1)

    def test_ones_dtype(self):
        o = ones((3,), np, dtype=np.float32)
        assert o.dtype == np.float32


class TestConcatenateStackNumPyPaths:
    """Cover the xp is np and xp-inference branches for concatenate and stack."""

    def test_concatenate_explicit_np(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = concatenate([a, b], axis=0, xp=np)
        assert_allclose(result, [1.0, 2.0, 3.0, 4.0])

    def test_concatenate_axis1(self):
        a = np.array([[1.0], [2.0]])
        b = np.array([[3.0], [4.0]])
        result = concatenate([a, b], axis=1, xp=np)
        assert result.shape == (2, 2)
        assert_allclose(result, [[1.0, 3.0], [2.0, 4.0]])

    def test_concatenate_infer_np(self):
        """xp=None should infer np from numpy arrays."""
        a = np.array([1.0])
        b = np.array([2.0])
        result = concatenate([a, b])
        assert_allclose(result, [1.0, 2.0])

    def test_stack_explicit_np(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = stack([a, b], axis=0, xp=np)
        assert result.shape == (2, 2)
        assert_allclose(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_stack_axis1(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = stack([a, b], axis=1, xp=np)
        assert result.shape == (2, 2)
        assert_allclose(result, [[1.0, 3.0], [2.0, 4.0]])

    def test_stack_infer_np(self):
        a = np.array([5.0])
        b = np.array([6.0])
        result = stack([a, b])
        assert result.shape == (2, 1)


class TestMatrixOpsNumPyPaths:
    """Cover the xp is np branches for diag, trace, matmul, transpose."""

    def test_diag_create(self):
        v = np.array([1.0, 2.0, 3.0])
        result = diag(v, np)
        assert_allclose(result, np.diag([1.0, 2.0, 3.0]))

    def test_diag_extract(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        d = diag(A, np)
        assert_allclose(d, [1.0, 4.0])

    def test_trace(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = trace(A, np)
        assert_allclose(t, 5.0)

    def test_trace_3x3(self):
        A = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        assert_allclose(trace(A, np), 6.0)

    def test_matmul(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        C = matmul(A, B, np)
        assert_allclose(C, A @ B, rtol=1e-10)

    def test_matmul_rectangular(self):
        A = np.ones((3, 2))
        B = np.ones((2, 4))
        C = matmul(A, B, np)
        assert C.shape == (3, 4)
        assert_allclose(C, 2.0 * np.ones((3, 4)))

    def test_transpose(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        T = transpose(A, np)
        assert T.shape == (2, 3)
        assert_allclose(T, A.T)


class TestReductionsNumPyPaths:
    """Cover the xp is np and xp-inference branches for sum, mean, max."""

    def test_sum_flat(self):
        a = np.array([1.0, 2.0, 3.0])
        assert_allclose(ops_sum(a, xp=np), 6.0)

    def test_sum_axis0(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        s = ops_sum(a, axis=0, xp=np)
        assert_allclose(s, [4.0, 6.0])

    def test_sum_axis1(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        s = ops_sum(a, axis=1, xp=np)
        assert_allclose(s, [3.0, 7.0])

    def test_sum_infer_np(self):
        a = np.array([5.0, 10.0])
        assert_allclose(ops_sum(a), 15.0)

    def test_mean_flat(self):
        a = np.array([2.0, 4.0, 6.0])
        assert_allclose(mean(a, xp=np), 4.0)

    def test_mean_axis(self):
        a = np.array([[1.0, 3.0], [5.0, 7.0]])
        m = mean(a, axis=1, xp=np)
        assert_allclose(m, [2.0, 6.0])

    def test_mean_infer_np(self):
        a = np.array([3.0, 6.0])
        assert_allclose(mean(a), 4.5)

    def test_max_flat(self):
        a = np.array([1.0, 5.0, 3.0])
        assert_allclose(ops_max(a, xp=np), 5.0)

    def test_max_axis0(self):
        a = np.array([[1.0, 5.0], [3.0, 2.0]])
        m = ops_max(a, axis=0, xp=np)
        assert_allclose(m, [3.0, 5.0])

    def test_max_infer_np(self):
        a = np.array([10.0, 20.0])
        assert_allclose(ops_max(a), 20.0)


class TestElementwiseNumPyPaths:
    """Cover the xp is np branches for sqrt, exp, log, abs, clip."""

    def test_sqrt(self):
        a = np.array([1.0, 4.0, 9.0])
        assert_allclose(sqrt(a, np), [1.0, 2.0, 3.0], rtol=1e-10)

    def test_exp(self):
        a = np.array([0.0, 1.0])
        assert_allclose(exp(a, np), [1.0, np.e], rtol=1e-10)

    def test_log(self):
        a = np.array([1.0, np.e, np.e ** 2])
        assert_allclose(log(a, np), [0.0, 1.0, 2.0], rtol=1e-5)

    def test_abs(self):
        a = np.array([-3.0, 0.0, 4.0])
        assert_allclose(abs(a, np), [3.0, 0.0, 4.0])

    def test_clip(self):
        a = np.array([-5.0, 0.5, 10.0])
        r = clip(a, -1.0, 5.0, np)
        assert_allclose(r, [-1.0, 0.5, 5.0])

    def test_clip_no_change(self):
        a = np.array([2.0, 3.0, 4.0])
        r = clip(a, 0.0, 10.0, np)
        assert_allclose(r, [2.0, 3.0, 4.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
