"""Tests for SGD optimizers."""

import pytest
import numpy as np

from aurora.models.distributed import (
    SGDOptimizer,
    AdamOptimizer,
    AdaGradOptimizer,
)


class TestSGDOptimizer:
    """Tests for vanilla SGD optimizer."""

    def test_basic_update(self):
        """Test basic parameter update."""
        opt = SGDOptimizer(learning_rate=0.1)
        params = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])

        new_params = opt.step(params, grad)

        expected = params - 0.1 * grad
        np.testing.assert_allclose(new_params, expected)

    def test_momentum(self):
        """Test momentum accumulation."""
        opt = SGDOptimizer(learning_rate=0.1, momentum=0.9)
        params = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])

        # First step
        params = opt.step(params, grad)

        # Second step with same gradient
        params_2 = opt.step(params, grad)

        # Momentum should accumulate, making second update larger
        # velocity = 0.9 * grad + grad = 1.9 * grad
        assert np.abs(params_2[0] - params[0]) > np.abs(opt.learning_rate * grad[0])

    def test_weight_decay(self):
        """Test L2 regularization."""
        opt = SGDOptimizer(learning_rate=0.1, weight_decay=0.01)
        params = np.array([10.0, 20.0])
        grad = np.zeros(2)

        # With zero gradient, weight decay alone drives params toward zero
        new_params = opt.step(params, grad)

        assert np.all(new_params < params)

    def test_reset(self):
        """Test optimizer reset."""
        opt = SGDOptimizer(learning_rate=0.1, momentum=0.9)
        params = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])

        # Accumulate momentum
        opt.step(params, grad)
        assert opt._velocity is not None

        # Reset
        opt.reset()
        assert opt._velocity is None


class TestAdamOptimizer:
    """Tests for Adam optimizer."""

    def test_basic_update(self):
        """Test basic parameter update."""
        opt = AdamOptimizer(learning_rate=0.1)
        params = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])

        new_params = opt.step(params, grad)

        # Should have moved in negative gradient direction
        assert np.all(new_params < params)

    def test_convergence(self):
        """Test Adam converges on simple quadratic."""
        opt = AdamOptimizer(learning_rate=0.5)

        # Minimize f(x) = x^2, gradient = 2x
        x = np.array([10.0])

        for _ in range(200):
            grad = 2 * x
            x = opt.step(x, grad)

        # Should be close to minimum at 0
        assert np.abs(x[0]) < 1.0

    def test_bias_correction(self):
        """Test that bias correction works."""
        opt = AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999)
        params = np.array([1.0])
        grad = np.array([1.0])

        # First few steps should have bias correction
        for i in range(10):
            params = opt.step(params, grad)

        # Steps should be reasonable size
        assert np.abs(params[0]) < 10.0

    def test_reset(self):
        """Test optimizer reset."""
        opt = AdamOptimizer()
        params = np.array([1.0])
        grad = np.array([0.1])

        opt.step(params, grad)
        assert opt._m is not None
        assert opt._v is not None
        assert opt._t > 0

        opt.reset()
        assert opt._m is None
        assert opt._v is None
        assert opt._t == 0


class TestAdaGradOptimizer:
    """Tests for AdaGrad optimizer."""

    def test_basic_update(self):
        """Test basic parameter update."""
        opt = AdaGradOptimizer(learning_rate=0.1)
        params = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])

        new_params = opt.step(params, grad)

        assert np.all(new_params < params)

    def test_adaptive_learning_rate(self):
        """Test that learning rate decreases over time."""
        opt = AdaGradOptimizer(learning_rate=1.0)
        params = np.array([10.0])
        grad = np.array([1.0])

        # Record step sizes
        step_sizes = []
        for _ in range(10):
            old_params = params.copy()
            params = opt.step(params, grad)
            step_sizes.append(np.abs(params[0] - old_params[0]))

        # Step sizes should decrease (due to accumulated squared gradients)
        for i in range(1, len(step_sizes)):
            assert step_sizes[i] < step_sizes[i - 1]

    def test_reset(self):
        """Test optimizer reset."""
        opt = AdaGradOptimizer()
        params = np.array([1.0])
        grad = np.array([0.1])

        opt.step(params, grad)
        assert opt._g_sum is not None

        opt.reset()
        assert opt._g_sum is None
