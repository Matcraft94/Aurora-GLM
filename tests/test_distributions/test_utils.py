"""Tests for aurora.distributions._utils multi-backend utilities.

Covers: namespace(), namespace_from_backend(), as_namespace_array(),
ones_like(), clip_probability(), ensure_positive(), log_factorial(),
log_gamma(), digamma(), is_torch(), is_jax().
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aurora.distributions._utils import (
    as_namespace_array,
    clip_probability,
    digamma,
    ensure_positive,
    is_jax,
    is_torch,
    log_factorial,
    log_gamma,
    namespace,
    namespace_from_backend,
    ones_like,
)

# Try importing optional backends
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# =============================================================================
# is_torch / is_jax
# =============================================================================


class TestIsTorch:
    """Test is_torch() detector."""

    def test_numpy_array(self):
        assert is_torch(np.array([1.0])) is False

    def test_python_float(self):
        assert is_torch(3.14) is False

    def test_list(self):
        assert is_torch([1, 2, 3]) is False

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_tensor(self):
        assert is_torch(torch.tensor([1.0])) is True


class TestIsJax:
    """Test is_jax() detector."""

    def test_numpy_array(self):
        assert is_jax(np.array([1.0])) is False

    def test_python_float(self):
        assert is_jax(3.14) is False

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_jax_array(self):
        assert is_jax(jnp.array([1.0])) is True


# =============================================================================
# namespace()
# =============================================================================


class TestNamespace:
    """Test namespace() backend detection."""

    def test_numpy_array(self):
        xp = namespace(np.array([1.0, 2.0]))
        assert xp is np

    def test_python_float(self):
        xp = namespace(1.0, 2.0)
        assert xp is np

    def test_no_args(self):
        xp = namespace()
        assert xp is np

    def test_list_input(self):
        xp = namespace([1.0, 2.0])
        assert xp is np

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_tensor(self):
        xp = namespace(torch.tensor([1.0, 2.0]))
        assert xp is torch

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_mixed_torch_numpy(self):
        """Torch tensor should win over numpy."""
        xp = namespace(np.array([1.0]), torch.tensor([2.0]))
        assert xp is torch

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_jax_array(self):
        xp = namespace(jnp.array([1.0, 2.0]))
        assert xp is jnp


# =============================================================================
# namespace_from_backend()
# =============================================================================


class TestNamespaceFromBackend:
    """Test namespace_from_backend() string-based dispatch."""

    def test_numpy(self):
        xp, device = namespace_from_backend("numpy")
        assert xp is np
        assert device is None

    def test_numpy_case_insensitive(self):
        xp, _ = namespace_from_backend("NumPy")
        assert xp is np

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch(self):
        xp, device = namespace_from_backend("torch")
        assert xp is torch
        assert device is not None

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_pytorch_alias(self):
        xp, device = namespace_from_backend("pytorch")
        assert xp is torch

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_custom_device(self):
        xp, device = namespace_from_backend("torch", device="cpu")
        assert str(device) == "cpu"

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_jax(self):
        xp, device = namespace_from_backend("jax")
        assert xp is jnp
        assert device is None

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            namespace_from_backend("tensorflow")


# =============================================================================
# as_namespace_array()
# =============================================================================


class TestAsNamespaceArray:
    """Test as_namespace_array() conversion."""

    def test_numpy_from_scalar(self):
        result = as_namespace_array(3.0, np)
        assert isinstance(result, np.ndarray)
        assert result == 3.0

    def test_numpy_from_list(self):
        result = as_namespace_array([1.0, 2.0, 3.0], np)
        assert isinstance(result, np.ndarray)
        assert_allclose(result, [1.0, 2.0, 3.0])

    def test_numpy_from_ndarray(self):
        arr = np.array([1.0, 2.0])
        result = as_namespace_array(arr, np)
        assert isinstance(result, np.ndarray)

    def test_numpy_dtype_matching(self):
        like = np.array([1.0], dtype=np.float32)
        result = as_namespace_array(5.0, np, like=like)
        assert result.dtype == np.float32

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_from_scalar(self):
        result = as_namespace_array(3.0, torch)
        assert isinstance(result, torch.Tensor)
        assert result.item() == 3.0

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_from_list(self):
        result = as_namespace_array([1.0, 2.0], torch)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2,)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_dtype_matching(self):
        like = torch.tensor([1.0], dtype=torch.float64)
        result = as_namespace_array(5.0, torch, like=like)
        assert result.dtype == torch.float64

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_device_matching(self):
        like = torch.tensor([1.0], device="cpu")
        result = as_namespace_array(5.0, torch, like=like)
        assert result.device == like.device


# =============================================================================
# ones_like()
# =============================================================================


class TestOnesLike:
    """Test ones_like() across backends."""

    def test_numpy_ones_like(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = ones_like(arr)
        assert isinstance(result, np.ndarray)
        assert_allclose(result, [1.0, 1.0, 1.0])
        assert result.shape == arr.shape

    def test_numpy_ones_like_2d(self):
        arr = np.zeros((3, 4))
        result = ones_like(arr)
        assert result.shape == (3, 4)
        assert_allclose(result, np.ones((3, 4)))

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_ones_like(self):
        arr = torch.tensor([1.0, 2.0, 3.0])
        result = ones_like(arr)
        assert isinstance(result, torch.Tensor)
        assert_allclose(result.numpy(), [1.0, 1.0, 1.0])


# =============================================================================
# clip_probability()
# =============================================================================


class TestClipProbability:
    """Test clip_probability() across backends."""

    def test_numpy_clips_below(self):
        prob = np.array([0.0, 0.5, 1.0])
        result = clip_probability(prob, np, eps=1e-9)
        assert result[0] >= 1e-9
        assert result[1] == pytest.approx(0.5)
        assert result[2] <= 1.0 - 1e-9

    def test_numpy_clips_above(self):
        prob = np.array([1.0, 0.5, 0.0])
        result = clip_probability(prob, np, eps=1e-9)
        assert result[0] <= 1.0 - 1e-9
        assert result[2] >= 1e-9

    def test_numpy_valid_unchanged(self):
        prob = np.array([0.1, 0.5, 0.9])
        result = clip_probability(prob, np, eps=1e-9)
        assert_allclose(result, prob)

    def test_custom_eps(self):
        prob = np.array([0.0, 1.0])
        result = clip_probability(prob, np, eps=0.01)
        assert result[0] == pytest.approx(0.01)
        assert result[1] == pytest.approx(0.99)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_clips(self):
        prob = torch.tensor([0.0, 0.5, 1.0])
        result = clip_probability(prob, torch, eps=1e-9)
        assert result[0].item() >= 1e-9
        assert result[2].item() <= 1.0 - 1e-9


# =============================================================================
# ensure_positive()
# =============================================================================


class TestEnsurePositive:
    """Test ensure_positive() across backends."""

    def test_numpy_positive_unchanged(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = ensure_positive(arr, np)
        assert_allclose(result, arr)

    def test_numpy_clips_negative(self):
        arr = np.array([-1.0, 0.0, 0.5])
        result = ensure_positive(arr, np, eps=1e-12)
        assert np.all(result >= 1e-12)
        assert result[2] == pytest.approx(0.5)

    def test_numpy_zero_clipped(self):
        arr = np.array([0.0, -0.0])
        result = ensure_positive(arr, np, eps=1e-10)
        assert np.all(result >= 1e-10)

    def test_custom_eps(self):
        arr = np.array([-5.0, 0.0, 0.001])
        result = ensure_positive(arr, np, eps=0.01)
        assert result[0] == pytest.approx(0.01)
        assert result[1] == pytest.approx(0.01)
        assert result[2] == pytest.approx(0.01)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_clips(self):
        arr = torch.tensor([-1.0, 0.0, 2.0])
        result = ensure_positive(arr, torch, eps=1e-12)
        assert result[0].item() >= 1e-12
        assert result[2].item() == pytest.approx(2.0)


# =============================================================================
# log_factorial()
# =============================================================================


class TestLogFactorial:
    """Test log_factorial() across backends."""

    def test_numpy_0(self):
        result = log_factorial(np.array(0.0), np)
        assert_allclose(result, 0.0, atol=1e-10)  # log(0!) = 0

    def test_numpy_1(self):
        result = log_factorial(np.array(1.0), np)
        assert_allclose(result, 0.0, atol=1e-10)  # log(1!) = 0

    def test_numpy_5(self):
        result = log_factorial(np.array(5.0), np)
        assert_allclose(result, np.log(120.0), rtol=1e-10)

    def test_numpy_10(self):
        result = log_factorial(np.array(10.0), np)
        assert_allclose(result, np.log(3628800.0), rtol=1e-10)

    def test_numpy_array(self):
        values = np.array([0.0, 1.0, 3.0, 5.0])
        result = log_factorial(values, np)
        expected = np.array([0.0, 0.0, np.log(6.0), np.log(120.0)])
        assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_scalar(self):
        result = log_factorial(torch.tensor(5.0), torch)
        assert_allclose(result.item(), np.log(120.0), rtol=1e-10)


# =============================================================================
# log_gamma()
# =============================================================================


class TestLogGamma:
    """Test log_gamma() across backends."""

    def test_numpy_1(self):
        result = log_gamma(np.array(1.0), np)
        assert_allclose(result, 0.0, atol=1e-10)  # Gamma(1) = 1

    def test_numpy_5(self):
        result = log_gamma(np.array(5.0), np)
        assert_allclose(result, np.log(24.0), rtol=1e-10)  # Gamma(5) = 4! = 24

    def test_numpy_half(self):
        result = log_gamma(np.array(0.5), np)
        from scipy.special import gammaln

        assert_allclose(result, gammaln(0.5), rtol=1e-10)

    def test_numpy_array(self):
        values = np.array([1.0, 2.0, 5.0])
        result = log_gamma(values, np)
        expected = np.array([0.0, 0.0, np.log(24.0)])
        assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_scalar(self):
        result = log_gamma(torch.tensor(5.0), torch)
        assert_allclose(result.item(), np.log(24.0), rtol=1e-10)


# =============================================================================
# digamma()
# =============================================================================


class TestDigamma:
    """Test digamma() across backends."""

    def test_numpy_1(self):
        """digamma(1) = -gamma (Euler-Mascheroni constant)."""
        from scipy import special

        result = digamma(np.array(1.0), np)
        assert_allclose(result, special.digamma(1.0), rtol=1e-10)

    def test_numpy_array(self):
        from scipy import special

        values = np.array([0.5, 1.0, 2.0, 5.0])
        result = digamma(values, np)
        expected = special.digamma(values)
        assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_scalar(self):
        from scipy import special

        result = digamma(torch.tensor(1.0), torch)
        assert_allclose(result.item(), special.digamma(1.0), rtol=1e-5)
