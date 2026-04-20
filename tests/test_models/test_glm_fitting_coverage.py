# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Targeted tests to improve coverage of aurora/models/glm/fitting.py.

Focuses on uncovered branches:
- fit_glm with explicit backend="numpy" (conversion path, lines ~271-321)
- _weighted_least_squares non-numpy branches (PyTorch/JAX tensor ops, lines ~512-570)
- Backend-specific helpers: _concat_columns, _clamp_positive, _sqrt (lines ~663-693)
- _matvec non-numpy path, _coerce_family / _coerce_link error branches
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from aurora.models.glm.fitting import (
    _clamp_positive,
    _concat_columns,
    _coerce_family,
    _coerce_link,
    _matvec,
    _ones_column,
    _reciprocal,
    _solve_normal_equation_numpy,
    _sqrt,
    _to_python_float,
    _weighted_least_squares,
    _zeros_vector,
    fit_glm,
)
from aurora.distributions.families import GaussianFamily, PoissonFamily
from aurora.distributions.links import IdentityLink, LogLink


# ---------------------------------------------------------------------------
# fit_glm with explicit backend="numpy"
# ---------------------------------------------------------------------------


class TestFitGlmExplicitBackendNumpy:
    """Cover the backend-is-not-None branch (lines ~271-321)."""

    @staticmethod
    def _make_data(n=80, seed=1):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2))
        true_coef = np.array([1.5, -0.8])
        y = 0.5 + X @ true_coef + rng.normal(scale=0.1, size=n)
        return X, y, true_coef, 0.5

    def test_explicit_numpy_backend_converges(self):
        X, y, true_coef, intercept = self._make_data()
        result = fit_glm(X, y, family="gaussian", backend="numpy", max_iter=50)
        assert result.converged_
        np.testing.assert_allclose(result.coef_, true_coef, atol=0.15)
        assert result.intercept_ == pytest.approx(intercept, abs=0.15)

    def test_explicit_numpy_with_weights_and_offset(self):
        """Cover lines 275-278, 309-315 (weights/offset with backend param)."""
        X, y, _, _ = self._make_data()
        weights = np.ones(X.shape[0])
        offset = np.zeros(X.shape[0])
        result = fit_glm(
            X, y, family="gaussian", backend="numpy",
            weights=weights, offset=offset, max_iter=50,
        )
        assert result.converged_

    def test_explicit_numpy_1d_X(self):
        """Cover lines 293-299: 1D X input reshaping with backend param."""
        rng = np.random.default_rng(3)
        x = rng.normal(size=60)
        y = 2.0 * x + 1.0 + rng.normal(scale=0.2, size=60)
        result = fit_glm(x, y, family="gaussian", backend="numpy", max_iter=50)
        assert result.converged_
        assert result.coef_.shape[0] == 1

    def test_explicit_numpy_weights_2d_reshaped(self):
        """Cover line 312-313: 2D weights reshaped to 1D with backend param."""
        X, y, _, _ = self._make_data()
        weights = np.ones((X.shape[0], 1))  # 2D column
        result = fit_glm(X, y, family="gaussian", backend="numpy", weights=weights, max_iter=50)
        assert result.converged_

    def test_explicit_numpy_offset_2d_reshaped(self):
        """Cover line 314-315: 2D offset reshaped to 1D with backend param."""
        X, y, _, _ = self._make_data()
        offset = np.zeros((X.shape[0], 1))  # 2D column
        result = fit_glm(X, y, family="gaussian", backend="numpy", offset=offset, max_iter=50)
        assert result.converged_

    def test_explicit_numpy_y_2d_reshaped(self):
        """Cover line 302-303: 2D y reshaped to 1D."""
        X, y, _, _ = self._make_data()
        y_2d = y.reshape(-1, 1)
        result = fit_glm(X, y_2d, family="gaussian", backend="numpy", max_iter=50)
        assert result.converged_

    def test_explicit_numpy_no_intercept(self):
        """Cover lines 346-348: fit_intercept=False path with backend param."""
        X, y, true_coef, _ = self._make_data()
        result = fit_glm(X, y, family="gaussian", backend="numpy", fit_intercept=False, max_iter=50)
        assert result.intercept_ is None
        assert result.converged_

    def test_explicit_numpy_poisson(self):
        """Non-Gaussian family through backend="numpy" path."""
        rng = np.random.default_rng(5)
        X = rng.normal(size=(100, 2))
        eta = 0.3 + X @ np.array([0.5, -0.3])
        mu = np.exp(eta)
        y = rng.poisson(mu)
        result = fit_glm(X, y, family="poisson", link="log", backend="numpy", max_iter=80)
        assert result.converged_
        assert result.deviance_ < result.null_deviance_

    def test_auto_backend_1d_x(self):
        """Cover lines 293-299 via auto-detection (backend=None) with 1D X."""
        rng = np.random.default_rng(7)
        x = rng.normal(size=50)
        y = 3.0 * x + rng.normal(scale=0.1, size=50)
        result = fit_glm(x, y, family="gaussian", max_iter=50)
        assert result.converged_

    def test_auto_backend_weights_offset_2d(self):
        """Cover lines 318-321: auto-detect backend with 2D weights/offset."""
        X, y, _, _ = self._make_data()
        weights = np.ones((X.shape[0], 1))
        offset = np.zeros((X.shape[0], 1))
        result = fit_glm(X, y, family="gaussian", weights=weights, offset=offset, max_iter=50)
        assert result.converged_


# ---------------------------------------------------------------------------
# _weighted_least_squares with mocked non-numpy backends
# ---------------------------------------------------------------------------


class TestWeightedLeastSquaresNonNumpy:
    """Cover the PyTorch/JAX branches in _weighted_least_squares (lines ~512-570)."""

    @staticmethod
    def _make_mock_torch_xp():
        """Create a mock xp that simulates PyTorch namespace."""
        xp = MagicMock()
        # .eye(device=...) and .tensor(...) for ridge scalar creation
        xp.eye = MagicMock(return_value=np.eye(2))
        xp.tensor = MagicMock(return_value=1e-8)
        # linalg.solve
        xp.linalg = MagicMock()
        xp.linalg.solve = MagicMock(return_value=np.array([[1.0], [2.0]]))
        return xp

    @staticmethod
    def _make_mock_jax_xp():
        """Create a mock xp that simulates JAX namespace (no .tensor, no .cat)."""
        xp = MagicMock()
        # JAX has .eye but no .tensor
        xp.eye = MagicMock(return_value=np.eye(2))
        del xp.tensor  # JAX does not have .tensor
        xp.linalg = MagicMock()
        xp.linalg.solve = MagicMock(return_value=np.array([[1.0], [2.0]]))
        return xp

    def _make_torch_like_arrays(self):
        """Create arrays that mimic torch tensors (have .contiguous(), .transpose())."""
        X = MagicMock()
        z = np.array([1.0, 2.0, 3.0])

        # Make X behave like a torch tensor with matmul
        X_np = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        X.contiguous = MagicMock(return_value=X_np)

        # transpose returns a callable that raises TypeError (so it falls to .T)
        def transpose_side_effect(*args, **kwargs):
            raise TypeError("transpose takes no positional args")

        X.transpose = MagicMock(side_effect=transpose_side_effect)
        X.T = X_np.T
        X.shape = (3, 2)
        X.__matmul__ = MagicMock(return_value=np.zeros(3))
        X.__rmatmul__ = MagicMock(return_value=np.zeros(2))

        return X, z

    def test_jax_path_transpose_raises_typeerror(self):
        """Cover lines 519-523: .transpose callable but raises TypeError -> falls to .T."""
        xp = self._make_mock_jax_xp()

        X_mock = MagicMock()
        X_mock.transpose = MagicMock(side_effect=TypeError("no args"))
        X_mock.T = np.array([[1.0, 0.5, 1.0], [0.5, 1.0, 0.0]])
        X_mock.shape = (3, 2)
        X_mock.dtype = np.float64

        z = np.array([1.0, 2.0, 3.0])
        result = _weighted_least_squares(xp, X_mock, z)
        assert result.shape == (2,)

    def test_pytorch_ridge_scalar_creation(self):
        """Cover lines 546-553: PyTorch ridge scalar with dtype and device kwargs."""
        xp = self._make_mock_torch_xp()
        X_data = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

        X_mock = MagicMock()
        X_mock.transpose = MagicMock(return_value=X_data.T)
        X_mock.shape = (3, 2)
        X_mock.dtype = np.float64
        X_mock.device = "cpu"

        z = np.array([1.0, 2.0, 3.0])
        result = _weighted_least_squares(xp, X_mock, z)

        # Verify tensor was called with dtype and device kwargs
        xp.tensor.assert_called_once()
        call_kwargs = xp.tensor.call_args
        assert "dtype" in call_kwargs[1]
        assert "device" in call_kwargs[1]

    def test_gram_has_no_dtype(self):
        """Cover line 530: dtype is None when gram has no .dtype."""
        xp = MagicMock()
        xp.eye = MagicMock(return_value=np.eye(2))
        del xp.tensor
        xp.linalg.solve = MagicMock(return_value=np.array([[0.5], [1.5]]))

        X_data = np.array([[1.0, 0.0], [0.0, 1.0]])
        X_mock = MagicMock()
        X_mock.transpose = MagicMock(return_value=X_data.T)
        X_mock.shape = (2, 2)
        # Explicitly no dtype
        del X_mock.dtype
        del X_mock.device

        z = np.array([1.0, 2.0])
        result = _weighted_least_squares(xp, X_mock, z)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# Backend-specific helper functions
# ---------------------------------------------------------------------------


class TestConcatColumns:
    """Cover _concat_columns for all backend paths (lines 663-672)."""

    def test_numpy_path(self):
        left = np.ones((4, 1))
        right = np.zeros((4, 2))
        result = _concat_columns(np, left, right)
        assert result.shape == (4, 3)
        np.testing.assert_array_equal(result[:, :1], left)
        np.testing.assert_array_equal(result[:, 1:], right)

    def test_pytorch_style_cat(self):
        """Cover line 668-669: xp.cat path (PyTorch)."""
        left = np.ones((4, 1))
        right = np.zeros((4, 2))
        xp = MagicMock()
        xp.cat = MagicMock(return_value=np.concatenate([left, right], axis=1))
        result = _concat_columns(xp, left, right)
        xp.cat.assert_called_once()
        assert result.shape == (4, 3)

    def test_jax_style_concatenate(self):
        """Cover line 671-672: xp.concatenate path (JAX, no .cat)."""
        left = np.ones((4, 1))
        right = np.zeros((4, 2))
        xp = MagicMock()
        del xp.cat  # No .cat attribute
        xp.concatenate = MagicMock(return_value=np.concatenate([left, right], axis=1))
        result = _concat_columns(xp, left, right)
        xp.concatenate.assert_called_once()
        assert result.shape == (4, 3)


class TestClampPositive:
    """Cover _clamp_positive for all backend paths (lines 675-693)."""

    def test_numpy_path(self):
        val = np.array([-1.0, 0.0, 0.5, 2.0])
        result = _clamp_positive(val, np)
        np.testing.assert_array_equal(result, np.array([1e-12, 1e-12, 0.5, 2.0]))

    def test_pytorch_style_clamp(self):
        """Cover lines 679-690: PyTorch .clamp path with tensor creation."""
        val = np.array([0.5, 2.0])
        xp = MagicMock()
        # PyTorch has .clamp
        xp.clamp = MagicMock(return_value=np.clip(val, 1e-12, None))
        xp.tensor = MagicMock(return_value=1e-12)

        # Create a mock value with dtype and device attrs
        val_mock = MagicMock()
        val_mock.dtype = np.float64
        val_mock.device = "cpu"

        result = _clamp_positive(val_mock, xp)
        xp.clamp.assert_called_once()
        xp.tensor.assert_called_once()

    def test_pytorch_clamp_no_dtype_device(self):
        """Cover _clamp_positive PyTorch path when value has no dtype/device."""
        xp = MagicMock()
        xp.clamp = MagicMock(return_value=np.array([1e-12]))
        xp.tensor = MagicMock(return_value=1e-12)

        val_mock = MagicMock(spec=[])  # No attributes at all
        result = _clamp_positive(val_mock, xp)
        xp.tensor.assert_called_once_with(1e-12)

    def test_jax_style_clip(self):
        """Cover lines 691-693: JAX .clip path (no .clamp)."""
        val = np.array([-1.0, 0.5])
        xp = MagicMock()
        del xp.clamp  # JAX does not have .clamp
        xp.clip = MagicMock(return_value=np.clip(val, 1e-12, None))
        result = _clamp_positive(val, xp)
        xp.clip.assert_called_once()


class TestSqrt:
    """Cover _sqrt for all backend paths (lines 700-704)."""

    def test_numpy_path(self):
        val = np.array([1.0, 4.0, 9.0])
        result = _sqrt(val, np)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_non_numpy_path(self):
        """Cover lines 703-704: xp.sqrt for non-numpy backends."""
        val = np.array([1.0, 4.0, 9.0])
        xp = MagicMock()
        xp.sqrt = MagicMock(return_value=np.sqrt(val))
        result = _sqrt(val, xp)
        xp.sqrt.assert_called_once_with(val)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


class TestReciprocal:
    """Cover _reciprocal (line 697)."""

    def test_reciprocal(self):
        val = np.array([2.0, 4.0, 0.5])
        result = _reciprocal(val, np)
        np.testing.assert_allclose(result, [0.5, 0.25, 2.0])


# ---------------------------------------------------------------------------
# _matvec non-numpy path
# ---------------------------------------------------------------------------


class TestMatvec:
    """Cover _matvec non-numpy branches (lines 633-636)."""

    def test_numpy_path(self):
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        vec = np.array([0.5, 1.5])
        result = _matvec(np, mat, vec)
        np.testing.assert_allclose(result, [3.5, 7.5])

    def test_callable_matmul(self):
        """Cover line 634-635: matrix has callable .matmul method."""
        expected = np.array([3.5, 7.5])
        mat = MagicMock()
        mat.matmul = MagicMock(return_value=expected)
        vec = np.array([0.5, 1.5])
        xp = MagicMock()
        result = _matvec(xp, mat, vec)
        mat.matmul.assert_called_once_with(vec)
        np.testing.assert_array_equal(result, expected)

    def test_at_operator_fallback(self):
        """Cover line 636: no .matmul, falls back to @ operator."""
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        vec = np.array([0.5, 1.5])
        # Use an xp that is not np
        xp = MagicMock()
        result = _matvec(xp, mat, vec)
        np.testing.assert_allclose(result, [3.5, 7.5])


# ---------------------------------------------------------------------------
# _to_python_float
# ---------------------------------------------------------------------------


class TestToPythonFloat:
    """Cover _to_python_float (lines 707-710)."""

    def test_has_item(self):
        """Cover line 709: value has .item() method (torch scalar)."""
        val = MagicMock()
        val.item = MagicMock(return_value=3.14)
        result = _to_python_float(val)
        assert result == 3.14
        assert isinstance(result, float)

    def test_no_item(self):
        """Cover line 710: plain float value."""
        result = _to_python_float(2.718)
        assert result == 2.718
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _coerce_family and _coerce_link error branches
# ---------------------------------------------------------------------------


class TestCoerceFamilyErrors:

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="family must be"):
            _coerce_family(42)  # type: ignore[arg-type]


class TestCoerceLinkErrors:

    def test_invalid_type(self):
        fam = GaussianFamily()
        with pytest.raises(TypeError, match="link must be"):
            _coerce_link(42, fam)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _zeros_vector and _ones_column with device
# ---------------------------------------------------------------------------


class TestZerosOnesWithDevice:
    """Cover _zeros_vector and _ones_column with dtype and device attrs."""

    def test_zeros_with_device(self):
        """Cover lines 644-647: device kwarg propagated."""
        xp = MagicMock()
        xp.zeros = MagicMock(return_value=np.zeros(5))
        like = MagicMock()
        like.dtype = np.float32
        like.device = "cuda:0"
        result = _zeros_vector(xp, 5, like=like)
        xp.zeros.assert_called_once_with((5,), dtype=np.float32, device="cuda:0")

    def test_ones_with_device(self):
        """Cover lines 656-658: device kwarg propagated."""
        xp = MagicMock()
        xp.ones = MagicMock(return_value=np.ones((5, 1)))
        like = MagicMock()
        like.dtype = np.float32
        like.device = "cuda:0"
        result = _ones_column(xp, 5, like=like)
        xp.ones.assert_called_once_with((5, 1), dtype=np.float32, device="cuda:0")

    def test_zeros_no_dtype_no_device(self):
        """Cover case where like has no dtype or device."""
        xp = MagicMock()
        xp.zeros = MagicMock(return_value=np.zeros(3))
        like = MagicMock(spec=[])  # No attributes
        result = _zeros_vector(xp, 3, like=like)
        xp.zeros.assert_called_once_with((3,))

    def test_ones_no_dtype_no_device(self):
        """Cover case where like has no dtype or device."""
        xp = MagicMock()
        xp.ones = MagicMock(return_value=np.ones((3, 1)))
        like = MagicMock(spec=[])  # No attributes
        result = _ones_column(xp, 3, like=like)
        xp.ones.assert_called_once_with((3, 1))


# ---------------------------------------------------------------------------
# _solve_normal_equation_numpy
# ---------------------------------------------------------------------------


class TestSolveNormalEquationNumpy:
    """Cover _solve_normal_equation_numpy (lines 573-582)."""

    def test_simple_solve(self):
        gram = np.array([[2.0, 0.5], [0.5, 1.0]])
        rhs = np.array([1.0, 0.5])
        result = _solve_normal_equation_numpy(gram, rhs)
        # Verify A*x = b
        np.testing.assert_allclose(gram @ result, rhs, atol=1e-6)

    def test_singular_matrix_with_jitter(self):
        """Cover lines 577-582: retry with increasing jitter on singular matrix."""
        # Nearly singular matrix
        gram = np.array([[1e-15, 0.0], [0.0, 1e-15]])
        rhs = np.array([1.0, 1.0])
        result = _solve_normal_equation_numpy(gram, rhs)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# fit_glm edge cases
# ---------------------------------------------------------------------------


class TestFitGlmEdgeCases:

    def test_mismatched_samples_raises(self):
        """Cover line 306: ValueError for mismatched X/y samples."""
        X = np.ones((10, 2))
        y = np.ones(5)
        with pytest.raises(ValueError, match="same number of samples"):
            fit_glm(X, y, family="gaussian")

    def test_string_family_and_link(self):
        """Cover _coerce_family and _coerce_link string resolution."""
        rng = np.random.default_rng(9)
        X = rng.normal(size=(50, 2))
        y = rng.normal(size=50)
        result = fit_glm(X, y, family="gaussian", link="identity")
        assert result.converged_

    def test_poisson_with_offset_converges(self):
        """Cover offset processing through the IRLS loop."""
        rng = np.random.default_rng(11)
        X = rng.normal(size=(100, 2))
        eta = 0.2 + X @ np.array([0.4, -0.2])
        offset = np.ones(100) * 0.5
        mu = np.exp(eta + offset)
        y = rng.poisson(mu)
        result = fit_glm(X, y, family="poisson", link="log", offset=offset, max_iter=80)
        assert result.converged_

    def test_fit_intercept_false_null_deviance_equals_deviance(self):
        """Cover line 369: null_deviance == deviance when fit_intercept=False."""
        rng = np.random.default_rng(13)
        X = rng.normal(size=(50, 2))
        y = rng.normal(size=50)
        result = fit_glm(X, y, family="gaussian", fit_intercept=False, max_iter=50)
        assert result.intercept_ is None
        assert result.null_deviance_ == result.deviance_

    def test_explicit_family_and_link_objects(self):
        """Cover _coerce_family and _coerce_link with pre-built objects."""
        rng = np.random.default_rng(15)
        X = rng.normal(size=(60, 2))
        eta = X @ np.array([0.3, -0.1])
        mu = np.exp(eta)
        y = rng.poisson(mu)
        fam = PoissonFamily()
        link = LogLink()
        result = fit_glm(X, y, family=fam, link=link, max_iter=80)
        assert result.converged_
