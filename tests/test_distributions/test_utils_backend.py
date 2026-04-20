"""Tests for torch and JAX backend branches in aurora.distributions._utils.

Uses unittest.mock.MagicMock to simulate torch and jax so that the backend-
specific code paths are exercised even when those libraries are not installed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose


# ---------------------------------------------------------------------------
# Helpers for building mock torch / jax modules
# ---------------------------------------------------------------------------


def _make_mock_torch():
    """Build a mock ``torch`` module with the attributes _utils expects."""
    mock = MagicMock(name="mock_torch")

    # torch.Tensor — a sentinel class so isinstance checks work
    mock.Tensor = type("MockTensor", (), {})

    # torch.cuda
    mock.cuda = MagicMock(name="mock_torch.cuda")
    mock.cuda.is_available.return_value = False

    # torch.device
    mock.device = MagicMock(name="mock_torch.device", return_value="cpu")

    # torch.get_default_dtype
    mock.get_default_dtype.return_value = np.float64

    # torch.as_tensor — just return a numpy array wrapped in a thin wrapper
    def _as_tensor(value, dtype=None, device=None):
        arr = np.asarray(value, dtype=dtype)
        sentinel = MagicMock(name="MockTensorInstance")
        sentinel.__class__ = mock.Tensor
        # Make it behave like an array for numpy operations
        sentinel.numpy.return_value = arr
        sentinel.detach.return_value.cpu.return_value.numpy.return_value = arr
        sentinel.item.side_effect = lambda: arr.item()
        sentinel.shape = arr.shape
        sentinel.dtype = dtype or np.float64
        sentinel.device = device or "cpu"
        return sentinel

    mock.as_tensor = MagicMock(side_effect=_as_tensor)

    # torch.ones_like
    def _ones_like(value):
        arr = np.ones_like(np.asarray(value, dtype=np.float64))
        sentinel = MagicMock(name="MockTensorInstance")
        sentinel.__class__ = mock.Tensor
        sentinel.numpy.return_value = arr
        return sentinel

    mock.ones_like = MagicMock(side_effect=_ones_like)

    # torch.tensor — for creating eps / one tensors
    def _tensor_fn(value, dtype=None, device=None):
        arr = np.asarray(value, dtype=dtype)
        sentinel = MagicMock(name="MockTensorInstance")
        sentinel.__class__ = mock.Tensor
        sentinel.numpy.return_value = arr
        sentinel.item.side_effect = lambda: arr.item()
        sentinel.dtype = dtype or np.float64
        sentinel.device = device or "cpu"
        return sentinel

    mock.tensor = MagicMock(side_effect=_tensor_fn)

    # torch.clamp — supports both positional and keyword min/max:
    #   torch.clamp(value, min_val, max_val)  (positional)
    #   torch.clamp(value, min=..., max=...)  (keyword)
    def _clamp(value, *args, min=None, max=None):
        arr = np.asarray(value, dtype=np.float64)
        min_val = min if min is not None else (args[0] if len(args) > 0 else None)
        max_val = max if max is not None else (args[1] if len(args) > 1 else None)
        if min_val is not None:
            arr = np.maximum(arr, np.asarray(min_val))
        if max_val is not None:
            arr = np.minimum(arr, np.asarray(max_val))
        sentinel = MagicMock(name="MockTensorInstance")
        sentinel.__class__ = mock.Tensor
        sentinel.numpy.return_value = arr
        return sentinel

    mock.clamp = MagicMock(side_effect=_clamp)

    # torch.lgamma
    from scipy.special import gammaln

    def _lgamma(value):
        arr = np.asarray(value, dtype=np.float64)
        result = gammaln(arr)
        sentinel = MagicMock(name="MockTensorInstance")
        sentinel.__class__ = mock.Tensor
        sentinel.item.side_effect = lambda: float(result)
        return sentinel

    mock.lgamma = MagicMock(side_effect=_lgamma)

    # torch.digamma
    from scipy import special

    def _digamma(value):
        arr = np.asarray(value, dtype=np.float64)
        result = special.digamma(arr)
        sentinel = MagicMock(name="MockTensorInstance")
        sentinel.__class__ = mock.Tensor
        sentinel.item.side_effect = lambda: float(result)
        return sentinel

    mock.digamma = MagicMock(side_effect=_digamma)

    return mock


def _make_mock_jax():
    """Build a mock ``jax`` module and ``jax.numpy`` submodule."""
    mock_jax = MagicMock(name="mock_jax")
    mock_jnp = MagicMock(name="mock_jax.numpy")

    # jax.Array — sentinel class
    mock_jax.Array = type("MockJaxArray", (), {})

    # jnp.clip — real computation
    def _clip(value, a_min=None, a_max=None):
        arr = np.asarray(value, dtype=np.float64)
        return np.clip(arr, a_min, a_max)

    mock_jnp.clip = MagicMock(side_effect=_clip)

    # jnp.array — real computation
    def _array(value, dtype=None):
        return np.asarray(value, dtype=dtype)

    mock_jnp.array = MagicMock(side_effect=_array)

    # jnp.float64
    mock_jnp.float64 = np.float64

    return mock_jax, mock_jnp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_torch():
    """Install a mock torch module and yield it; restore on teardown."""
    mt = _make_mock_torch()
    patches = [
        patch.dict(sys.modules, {"torch": mt}),
    ]
    # Also patch the module-level references inside _utils
    with patches[0]:
        import aurora.distributions._utils as _utils_mod

        original_torch = _utils_mod.torch
        _utils_mod.torch = mt

        yield mt

        _utils_mod.torch = original_torch


@pytest.fixture()
def mock_jax():
    """Install mock jax/jax.numpy modules and yield them; restore on teardown."""
    mj, mjnp = _make_mock_jax()
    patches = [
        patch.dict(sys.modules, {"jax": mj, "jax.numpy": mjnp}),
    ]
    with patches[0]:
        import aurora.distributions._utils as _utils_mod

        original_jax = _utils_mod.jax
        original_jnp = _utils_mod.jnp
        _utils_mod.jax = mj
        _utils_mod.jnp = mjnp

        yield mj, mjnp

        _utils_mod.jax = original_jax
        _utils_mod.jnp = original_jnp


# ---------------------------------------------------------------------------
# Test: namespace() — torch branch
# ---------------------------------------------------------------------------


class TestNamespaceTorch:
    """Test namespace() with a mocked torch tensor."""

    def test_torch_detection(self, mock_torch):
        import aurora.distributions._utils as u

        fake_tensor = MagicMock(name="fake_torch_tensor")
        fake_tensor.__class__ = mock_torch.Tensor
        xp = u.namespace(fake_tensor)
        assert xp is mock_torch

    def test_torch_beats_numpy(self, mock_torch):
        import aurora.distributions._utils as u

        fake_tensor = MagicMock(name="fake_torch_tensor")
        fake_tensor.__class__ = mock_torch.Tensor
        xp = u.namespace(np.array([1.0]), fake_tensor)
        assert xp is mock_torch


# ---------------------------------------------------------------------------
# Test: namespace() — jax branch
# ---------------------------------------------------------------------------


class TestNamespaceJax:
    """Test namespace() with a mocked JAX array."""

    def test_jax_detection_via_device_buffer(self, mock_jax):
        mj, mjnp = mock_jax
        import aurora.distributions._utils as u

        fake_jax = MagicMock(name="fake_jax_array")
        fake_jax.device_buffer = True  # has device_buffer attribute
        xp = u.namespace(fake_jax)
        assert xp is mjnp

    def test_jax_detection_via_Array_instance(self, mock_jax):
        mj, mjnp = mock_jax
        import aurora.distributions._utils as u

        fake_jax = MagicMock(name="fake_jax_array")
        fake_jax.__class__ = mj.Array
        xp = u.namespace(fake_jax)
        assert xp is mjnp


# ---------------------------------------------------------------------------
# Test: namespace_from_backend() — torch branch
# ---------------------------------------------------------------------------


class TestNamespaceFromBackendTorch:
    def test_torch_default_device(self, mock_torch):
        import aurora.distributions._utils as u

        xp, device = u.namespace_from_backend("torch")
        assert xp is mock_torch
        mock_torch.cuda.is_available.assert_called()

    def test_torch_custom_device(self, mock_torch):
        import aurora.distributions._utils as u

        xp, device = u.namespace_from_backend("torch", device="cpu")
        assert xp is mock_torch
        mock_torch.device.assert_called_with("cpu")


# ---------------------------------------------------------------------------
# Test: namespace_from_backend() — jax branch
# ---------------------------------------------------------------------------


class TestNamespaceFromBackendJax:
    def test_jax(self, mock_jax):
        mj, mjnp = mock_jax
        import aurora.distributions._utils as u

        xp, device = u.namespace_from_backend("jax")
        assert xp is mjnp
        assert device is None

    def test_jax_not_installed_raises(self):
        import aurora.distributions._utils as u

        # When jnp is None (the default state if jax is not installed)
        # this should raise ImportError
        if u.jnp is None:
            with pytest.raises(ImportError, match="JAX is not installed"):
                u.namespace_from_backend("jax")


# ---------------------------------------------------------------------------
# Test: as_namespace_array() — torch branch
# ---------------------------------------------------------------------------


class TestAsNamespaceArrayTorch:
    def test_scalar_to_torch(self, mock_torch):
        import aurora.distributions._utils as u

        result = u.as_namespace_array(3.0, mock_torch)
        mock_torch.as_tensor.assert_called_once()

    def test_with_like_dtype(self, mock_torch):
        import aurora.distributions._utils as u

        like = MagicMock(name="like_tensor")
        like.dtype = np.float32
        like.device = "cpu"
        u.as_namespace_array(5.0, mock_torch, like=like)
        call_args = mock_torch.as_tensor.call_args
        assert call_args[1]["dtype"] == np.float32

    def test_explicit_device(self, mock_torch):
        import aurora.distributions._utils as u

        u.as_namespace_array(1.0, mock_torch, device="cpu")
        call_args = mock_torch.as_tensor.call_args
        assert call_args[1]["device"] == "cpu"


# ---------------------------------------------------------------------------
# Test: as_namespace_array() — jax branch
# ---------------------------------------------------------------------------


class TestAsNamespaceArrayJax:
    def test_scalar_to_jax(self, mock_jax):
        mj, mjnp = mock_jax
        import aurora.distributions._utils as u

        u.as_namespace_array(3.0, mjnp)
        mjnp.array.assert_called_once()

    def test_with_like_dtype(self, mock_jax):
        mj, mjnp = mock_jax
        import aurora.distributions._utils as u

        like = MagicMock(name="like_jax_array")
        like.dtype = np.float32
        u.as_namespace_array(5.0, mjnp, like=like)
        call_args = mjnp.array.call_args
        assert call_args[1]["dtype"] == np.float32


# ---------------------------------------------------------------------------
# Test: ones_like() — torch branch
# ---------------------------------------------------------------------------


class TestOnesLikeTorch:
    def test_torch_ones_like(self, mock_torch):
        import aurora.distributions._utils as u

        fake_tensor = MagicMock(name="fake_torch_tensor")
        fake_tensor.__class__ = mock_torch.Tensor
        result = u.ones_like(fake_tensor)
        mock_torch.ones_like.assert_called_once_with(fake_tensor)


# ---------------------------------------------------------------------------
# Test: clip_probability() — torch branch
# ---------------------------------------------------------------------------


class TestClipProbabilityTorch:
class TestClipProbabilityJax:
    def test_jax_clips(self, mock_jax):
        mj, mjnp = mock_jax
        import aurora.distributions._utils as u

        prob = np.array([0.0, 0.5, 1.0])
        result = u.clip_probability(prob, mjnp, eps=1e-9)
        mjnp.clip.assert_called_once()
        assert_allclose(result, [1e-9, 0.5, 1.0 - 1e-9])


# ---------------------------------------------------------------------------
# Test: ensure_positive() — torch branch
# ---------------------------------------------------------------------------


class TestEnsurePositiveTorch:
class TestEnsurePositiveJax:
    def test_jax_ensure_positive(self, mock_jax):
        mj, mjnp = mock_jax
        import aurora.distributions._utils as u

        value = np.array([-1.0, 0.0, 2.0])
        result = u.ensure_positive(value, mjnp, eps=1e-12)
        mjnp.clip.assert_called_once_with(value, 1e-12, None)
        assert_allclose(result, [1e-12, 1e-12, 2.0])


# ---------------------------------------------------------------------------
# Test: log_factorial() — torch branch
# ---------------------------------------------------------------------------


class TestLogFactorialTorch:
    def test_torch_log_factorial(self, mock_torch):
        import aurora.distributions._utils as u

        value = np.float64(5.0)
        result = u.log_factorial(value, mock_torch)
        mock_torch.lgamma.assert_called_once()
        # lgamma receives (value + 1.0) => 6.0
        call_arg = mock_torch.lgamma.call_args[0][0]
        assert_allclose(np.asarray(call_arg), 6.0)


# ---------------------------------------------------------------------------
# Test: log_factorial() — jax branch
# ---------------------------------------------------------------------------


class TestLogFactorialJax:
    def test_jax_log_factorial(self, mock_jax):
        mj, mjnp = mock_jax
        import aurora.distributions._utils as u

        value = np.float64(5.0)
        # The jax branch imports gammaln from jax.scipy.special, so we must
        # patch that import path.
        fake_gammaln = MagicMock(side_effect=lambda x: np.array(6.0))
        with patch.dict(sys.modules, {"jax.scipy.special": MagicMock(gammaln=fake_gammaln)}):
            result = u.log_factorial(value, mjnp)
        fake_gammaln.assert_called_once()


# ---------------------------------------------------------------------------
# Test: log_gamma() — torch branch
# ---------------------------------------------------------------------------


class TestLogGammaTorch:
    def test_torch_log_gamma(self, mock_torch):
        import aurora.distributions._utils as u

        value = np.float64(5.0)
        result = u.log_gamma(value, mock_torch)
        mock_torch.lgamma.assert_called_once()
        call_arg = mock_torch.lgamma.call_args[0][0]
        assert_allclose(np.asarray(call_arg), 5.0)


# ---------------------------------------------------------------------------
# Test: log_gamma() — jax branch
# ---------------------------------------------------------------------------


class TestLogGammaJax:
    def test_jax_log_gamma(self, mock_jax):
        mj, mjnp = mock_jax
        import aurora.distributions._utils as u

        value = np.float64(5.0)
        fake_gammaln = MagicMock(side_effect=lambda x: np.array(3.178))
        with patch.dict(sys.modules, {"jax.scipy.special": MagicMock(gammaln=fake_gammaln)}):
            result = u.log_gamma(value, mjnp)
        fake_gammaln.assert_called_once()


# ---------------------------------------------------------------------------
# Test: digamma() — torch branch
# ---------------------------------------------------------------------------


class TestDigammaTorch:
    def test_torch_digamma(self, mock_torch):
        import aurora.distributions._utils as u

        value = np.float64(1.0)
        result = u.digamma(value, mock_torch)
        mock_torch.digamma.assert_called_once()


# ---------------------------------------------------------------------------
# Test: digamma() — jax branch
# ---------------------------------------------------------------------------


class TestDigammaJax:
    def test_jax_digamma(self, mock_jax):
        mj, mjnp = mock_jax
        import aurora.distributions._utils as u

        value = np.float64(1.0)
        fake_digamma = MagicMock(side_effect=lambda x: np.array(-0.5772))
        with patch.dict(sys.modules, {"jax.scipy.special": MagicMock(digamma=fake_digamma)}):
            result = u.digamma(value, mjnp)
        fake_digamma.assert_called_once()
