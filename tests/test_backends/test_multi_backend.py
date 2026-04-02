"""Tests for multi-backend support in Aurora-GLM.

This module tests that fit_glm and fit_gamm work correctly with different
computational backends (NumPy, PyTorch, JAX).
"""

import numpy as np
import pytest

from aurora.models.gamm import fit_gamm
from aurora.models.glm import fit_glm


# Helper functions - must be defined before use in decorators
def _torch_available():
    """Check if PyTorch is available."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _torch_cuda_available():
    """Check if PyTorch CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _jax_available():
    """Check if JAX is available."""
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


# Test data generation
def generate_glm_data(n=100, p=3, seed=42):
    """Generate simple GLM test data."""
    np.random.seed(seed)
    X = np.random.randn(n, p)
    beta_true = np.array([1.0, -0.5, 0.3])
    y = X @ beta_true + np.random.randn(n) * 0.5
    return X, y, beta_true


def generate_gamm_data(n_groups=10, n_per_group=20, seed=42):
    """Generate simple GAMM test data with random intercepts."""
    np.random.seed(seed)
    n = n_groups * n_per_group

    # Fixed effect
    x = np.random.randn(n)
    beta = 1.5

    # Random intercepts
    group_effects = np.random.randn(n_groups) * 0.5
    groups = np.repeat(np.arange(n_groups), n_per_group)

    # Response
    y = beta * x + group_effects[groups] + np.random.randn(n) * 0.3

    return x, y, groups, beta


class TestGLMBackends:
    """Test fit_glm with different backends."""

    def test_numpy_backend(self):
        """Test fit_glm with NumPy backend (default)."""
        X, y, _ = generate_glm_data()

        result = fit_glm(X, y, family="gaussian", backend="numpy")

        assert result.converged_
        assert len(result.coef_) == 3
        assert result.intercept_ is not None

    def test_numpy_backend_explicit(self):
        """Test fit_glm with explicit NumPy backend."""
        X, y, _ = generate_glm_data()

        result = fit_glm(X, y, family="gaussian", backend="numpy")

        assert result.converged_

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
    def test_torch_cpu_backend(self):
        """Test fit_glm with PyTorch CPU backend."""
        X, y, _ = generate_glm_data()

        result = fit_glm(X, y, family="gaussian", backend="torch", device="cpu")

        assert result.converged_
        assert len(result.coef_) == 3

    @pytest.mark.skipif(not _torch_cuda_available(), reason="PyTorch CUDA not available")
    def test_torch_gpu_backend(self):
        """Test fit_glm with PyTorch GPU backend."""
        X, y, _ = generate_glm_data()

        result = fit_glm(X, y, family="gaussian", backend="torch", device="cuda")

        assert result.converged_
        assert len(result.coef_) == 3

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_backend(self):
        """Test fit_glm with JAX backend."""
        X, y, _ = generate_glm_data()

        result = fit_glm(X, y, family="gaussian", backend="jax")

        assert result.converged_
        assert len(result.coef_) == 3

    def test_backends_produce_similar_results(self):
        """Test that different backends produce similar results."""
        X, y, _ = generate_glm_data()

        # NumPy result
        result_numpy = fit_glm(X, y, family="gaussian", backend="numpy")

        # PyTorch result (if available)
        if _torch_available():
            result_torch = fit_glm(X, y, family="gaussian", backend="torch", device="cpu")

            # Coefficients should be very close
            np.testing.assert_allclose(
                result_numpy.coef_,
                result_torch.coef_,
                rtol=1e-5,
                err_msg="NumPy and PyTorch results differ",
            )

        # JAX result (if available)
        if _jax_available():
            result_jax = fit_glm(X, y, family="gaussian", backend="jax")

            # JAX uses float32 by default, so we need looser tolerance
            np.testing.assert_allclose(
                result_numpy.coef_,
                result_jax.coef_,
                rtol=1e-3,  # Looser tolerance for float32 vs float64
                err_msg="NumPy and JAX results differ",
            )


class TestGAMMBackends:
    """Test fit_gamm with different backends."""

    def test_numpy_backend(self):
        """Test fit_gamm with NumPy backend."""
        import pandas as pd

        x, y, groups, _ = generate_gamm_data()
        data = pd.DataFrame({"y": y, "x": x, "group": groups})

        result = fit_gamm(
            formula="y ~ x + (1 | group)", data=data, covariance="identity", backend="numpy"
        )

        assert result.converged
        assert len(result.beta_parametric) == 2  # intercept + x

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
    def test_torch_cpu_backend(self):
        """Test fit_gamm with PyTorch CPU backend."""
        import pandas as pd

        x, y, groups, _ = generate_gamm_data()
        data = pd.DataFrame({"y": y, "x": x, "group": groups})

        result = fit_gamm(
            formula="y ~ x + (1 | group)",
            data=data,
            covariance="identity",
            backend="torch",
            device="cpu",
        )

        assert result.converged
        assert len(result.beta_parametric) == 2

    @pytest.mark.skipif(not _torch_cuda_available(), reason="PyTorch CUDA not available")
    def test_torch_gpu_backend(self):
        """Test fit_gamm with PyTorch GPU backend."""
        import pandas as pd

        x, y, groups, _ = generate_gamm_data()
        data = pd.DataFrame({"y": y, "x": x, "group": groups})

        result = fit_gamm(
            formula="y ~ x + (1 | group)",
            data=data,
            covariance="identity",
            backend="torch",
            device="cuda",
        )

        assert result.converged

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_backend(self):
        """Test fit_gamm with JAX backend."""
        import pandas as pd

        x, y, groups, _ = generate_gamm_data()
        data = pd.DataFrame({"y": y, "x": x, "group": groups})

        result = fit_gamm(
            formula="y ~ x + (1 | group)", data=data, covariance="identity", backend="jax"
        )

        assert result.converged


class TestBackendOperations:
    """Test backend-agnostic operations module."""

    def test_get_namespace_numpy(self):
        """Test get_namespace returns NumPy for numpy backend."""
        from aurora.core.backends.operations import get_namespace

        xp, device = get_namespace("numpy")
        assert xp is np
        assert device is None

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
    def test_get_namespace_torch(self):
        """Test get_namespace returns torch for torch backend."""
        import torch

        from aurora.core.backends.operations import get_namespace

        xp, device = get_namespace("torch", "cpu")
        assert xp is torch
        assert device == torch.device("cpu")

    def test_to_backend_array_numpy(self):
        """Test to_backend_array with NumPy."""
        from aurora.core.backends.operations import to_backend_array

        data = [1, 2, 3]
        result = to_backend_array(data, np)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
    def test_to_backend_array_torch(self):
        """Test to_backend_array with PyTorch."""
        import torch

        from aurora.core.backends.operations import to_backend_array

        data = [1.0, 2.0, 3.0]
        device = torch.device("cpu")
        result = to_backend_array(data, torch, device)

        assert isinstance(result, torch.Tensor)
        assert result.device == device

    def test_linear_algebra_operations(self):
        """Test basic linear algebra operations."""
        from aurora.core.backends.operations import eye, inv, solve

        # Create test matrix
        A = np.array([[4, 1], [1, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)

        # Test solve
        x = solve(A, b, np)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, expected)

        # Test inv
        A_inv = inv(A, np)
        np.testing.assert_allclose(A @ A_inv, np.eye(2), atol=1e-10)

        # Test eye
        identity = eye(3, np)
        np.testing.assert_array_equal(identity, np.eye(3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
