# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Tests for aurora.validation.metrics.regression module."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.validation.metrics.regression import (
    mean_absolute_error,
    mean_squared_error,
    r_squared,
    root_mean_squared_error,
)


# ---------------------------------------------------------------------------
# mean_squared_error
# ---------------------------------------------------------------------------


class TestMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == 0.0

    def test_known_mse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        # Only one error: (3-4)^2 = 1, mean = 1/3
        np.testing.assert_allclose(mean_squared_error(y_true, y_pred), 1.0 / 3, atol=1e-10)

    def test_rmse_mode(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        np.testing.assert_allclose(rmse, np.sqrt(1.0 / 3), atol=1e-10)

    def test_weighted_mse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        weights = np.array([0.0, 0.0, 1.0])
        # Only weight on last element: error = 1
        np.testing.assert_allclose(mean_squared_error(y_true, y_pred, sample_weight=weights), 1.0, atol=1e-10)

    def test_list_input(self):
        """Should accept Python lists."""
        mse = mean_squared_error([1.0, 2.0], [1.0, 3.0])
        assert mse > 0

    def test_zero_weights_raises(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="positive sum"):
            mean_squared_error(y_true, y_pred, sample_weight=np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# mean_absolute_error
# ---------------------------------------------------------------------------


class TestMAE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(y, y) == 0.0

    def test_known_mae(self):
        y_true = np.array([1.0, 3.0, 5.0])
        y_pred = np.array([2.0, 3.0, 5.0])
        # errors: 1, 0, 0 → MAE = 1/3
        np.testing.assert_allclose(mean_absolute_error(y_true, y_pred), 1.0 / 3, atol=1e-10)

    def test_weighted_mae(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        weights = np.array([1.0, 0.0, 0.0])
        # only first error matters: |1| = 1
        np.testing.assert_allclose(mean_absolute_error(y_true, y_pred, sample_weight=weights), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# root_mean_squared_error
# ---------------------------------------------------------------------------


class TestRMSE:
    def test_equals_mse_squared_false(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 5.0])
        rmse = root_mean_squared_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred, squared=False)
        np.testing.assert_allclose(rmse, mse, atol=1e-10)

    def test_weighted_rmse(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 3.0])
        weights = np.array([1.0, 3.0])
        rmse = root_mean_squared_error(y_true, y_pred, sample_weight=weights)
        assert rmse > 0


# ---------------------------------------------------------------------------
# r_squared
# ---------------------------------------------------------------------------


class TestRSquared:
    def test_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(r_squared(y, y), 1.0, atol=1e-10)

    def test_no_fit(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.full(3, np.mean(y_true))  # Predict mean
        # R² should be 0 for predicting the mean
        np.testing.assert_allclose(r_squared(y_true, y_pred), 0.0, atol=1e-10)

    def test_negative_r2(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])  # Very bad predictions
        assert r_squared(y_true, y_pred) < 0

    def test_weighted_r2(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        weights = np.array([1.0, 1.0, 10.0])
        r2 = r_squared(y_true, y_pred, sample_weight=weights)
        assert 0 < r2 <= 1

    def test_constant_y_true(self):
        """When y_true is constant, R² returns 0."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 6.0, 5.0])
        np.testing.assert_allclose(r_squared(y_true, y_pred), 0.0, atol=1e-10)

    def test_zero_weights_raises(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="positive sum"):
            r_squared(y_true, y_pred, sample_weight=np.array([-1.0, 0.0]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            mean_squared_error(np.array([1.0, 2.0]), np.array([1.0]))

    def test_list_conversion(self):
        """Lists should be auto-converted to arrays."""
        mse = mean_squared_error([1, 2, 3], [1, 2, 4])
        assert isinstance(mse, float)

    def test_torch_tensor_input(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        y_t = torch.tensor([1.0, 2.0, 3.0])
        y_p = torch.tensor([1.0, 2.0, 4.0])
        mse = mean_squared_error(y_t, y_p)
        np.testing.assert_allclose(mse, 1.0 / 3, atol=1e-5)
