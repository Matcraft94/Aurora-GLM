# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Tests for aurora.validation.metrics.classification module."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.validation.metrics.classification import (
    accuracy_score,
    brier_score_loss,
    concordance_index,
    confusion_matrix,
    f1_score,
    log_loss,
    precision,
    recall,
    roc_auc,
)


class TestAccuracyScore:
    def test_perfect_binary(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        np.testing.assert_allclose(accuracy_score(y_true, y_pred), 1.0)

    def test_partial_binary(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        np.testing.assert_allclose(accuracy_score(y_true, y_pred), 0.8)

    def test_normalize_false(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        np.testing.assert_allclose(accuracy_score(y_true, y_pred, normalize=False), 4.0)

    def test_weighted(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        weights = np.array([1.0, 1.0, 2.0, 2.0])
        expected = np.average(y_true == y_pred, weights=weights)
        np.testing.assert_allclose(accuracy_score(y_true, y_pred, sample_weight=weights), expected)

    def test_multiclass(self):
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        np.testing.assert_allclose(accuracy_score(y_true, y_pred), 0.8)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="identical shapes"):
            accuracy_score(np.array([0, 1]), np.array([0, 1, 1]))

    def test_zero_weight_sum_raises(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        with pytest.raises(ValueError, match="positive sum"):
            accuracy_score(y_true, y_pred, sample_weight=np.array([0.0, 0.0]))

    def test_list_inputs(self):
        assert accuracy_score([0, 1, 1], [0, 1, 0]) == pytest.approx(2.0 / 3)


class TestLogLoss:
    def test_binary_manual(self):
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8])
        eps = 1e-15
        p = np.clip(y_prob, eps, 1.0 - eps)
        expected = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        np.testing.assert_allclose(log_loss(y_true, y_prob), expected, rtol=1e-5)

    def test_binary_weighted(self):
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8])
        weights = np.array([1.0, 2.0, 3.0])
        eps = 1e-15
        p = np.clip(y_prob, eps, 1.0 - eps)
        losses = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        expected = np.average(losses, weights=weights)
        np.testing.assert_allclose(log_loss(y_true, y_prob, sample_weight=weights), expected, rtol=1e-5)

    def test_multiclass(self):
        y_true = np.array([0, 2, 1])
        y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3]])
        expected = -np.mean(np.log([0.7, 0.6, 0.5]))
        np.testing.assert_allclose(log_loss(y_true, y_prob), expected, rtol=1e-5)

    def test_multiclass_weighted(self):
        y_true = np.array([0, 2, 1])
        y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3]])
        weights = np.array([1.0, 2.0, 3.0])
        expected = np.average([-np.log(0.7), -np.log(0.6), -np.log(0.5)], weights=weights)
        np.testing.assert_allclose(log_loss(y_true, y_prob, sample_weight=weights), expected, rtol=1e-5)

    def test_non_binary_labels_with_1d_prob_raises(self):
        with pytest.raises(ValueError, match="not binary"):
            log_loss(np.array([0, 1, 2]), np.array([0.2, 0.8, 0.4]))

    def test_multiclass_sample_mismatch_raises(self):
        y_true = np.array([0, 1])
        y_prob = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        with pytest.raises(ValueError, match="same number of samples"):
            log_loss(y_true, y_prob)

    def test_multiclass_rows_not_summing_to_one_raises(self):
        y_true = np.array([0, 1])
        y_prob = np.array([[0.5, 0.6], [0.3, 0.8]])
        with pytest.raises(ValueError, match="sum to 1.0"):
            log_loss(y_true, y_prob)

    def test_multiclass_labels_out_of_range_raises(self):
        y_true = np.array([0, 3])
        y_prob = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])
        with pytest.raises(ValueError, match="out of range"):
            log_loss(y_true, y_prob)

    def test_2d_y_true_raises(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            log_loss(np.array([[0, 1], [1, 0]]), np.array([0.5, 0.5]))


class TestBrierScoreLoss:
    def test_manual(self):
        y_true = np.array([1, 0, 1, 1])
        y_prob = np.array([0.9, 0.2, 0.8, 0.7])
        expected = np.mean((y_true - y_prob) ** 2)
        np.testing.assert_allclose(brier_score_loss(y_true, y_prob), expected, rtol=1e-5)

    def test_weighted(self):
        y_true = np.array([1, 0, 1, 1])
        y_prob = np.array([0.9, 0.2, 0.8, 0.7])
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.average((y_true - y_prob) ** 2, weights=weights)
        np.testing.assert_allclose(brier_score_loss(y_true, y_prob, sample_weight=weights), expected, rtol=1e-5)

    def test_non_binary_labels_raises(self):
        with pytest.raises(ValueError, match="binary labels"):
            brier_score_loss(np.array([0, 1, 2]), np.array([0.1, 0.9, 0.5]))

    def test_2d_prob_raises(self):
        y_true = np.array([0, 1])
        y_prob = np.array([[0.1, 0.9], [0.8, 0.2]])
        with pytest.raises(ValueError, match="1D binary"):
            brier_score_loss(y_true, y_prob)


class TestConfusionMatrix:
    def test_perfect_binary(self):
        cm = confusion_matrix(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0]))
        np.testing.assert_array_equal(cm, np.array([[2, 0], [0, 2]]))

    def test_all_wrong(self):
        cm = confusion_matrix(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]))
        np.testing.assert_array_equal(cm, np.array([[0, 2], [2, 0]]))

    def test_mixed(self):
        cm = confusion_matrix(np.array([0, 0, 1, 1, 1, 0]), np.array([0, 1, 1, 0, 1, 0]))
        np.testing.assert_array_equal(cm, np.array([[2, 1], [1, 2]]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="identical shapes"):
            confusion_matrix(np.array([0, 1]), np.array([0]))

    def test_non_binary_raises(self):
        with pytest.raises(ValueError, match="binary labels"):
            confusion_matrix(np.array([0, 1, 2]), np.array([0, 1, 2]))


class TestPrecision:
    def test_perfect(self):
        np.testing.assert_allclose(precision(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0])), 1.0)

    def test_no_positive_predictions(self):
        assert precision(np.array([0, 1, 1, 0]), np.array([0, 0, 0, 0])) == 0.0

    def test_mixed(self):
        np.testing.assert_allclose(precision(np.array([0, 0, 1, 1, 1, 0]), np.array([0, 1, 1, 0, 1, 0])), 2.0 / 3, rtol=1e-5)

    def test_weighted(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        expected = 3.0 / (3.0 + 2.0)
        np.testing.assert_allclose(precision(y_true, y_pred, sample_weight=weights), expected, rtol=1e-5)


class TestRecall:
    def test_perfect(self):
        np.testing.assert_allclose(recall(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0])), 1.0)

    def test_no_positive_true(self):
        assert recall(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1])) == 0.0

    def test_mixed(self):
        np.testing.assert_allclose(recall(np.array([0, 0, 1, 1, 1, 0]), np.array([0, 1, 1, 0, 1, 0])), 2.0 / 3, rtol=1e-5)

    def test_weighted(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        expected = 3.0 / (3.0 + 4.0)
        np.testing.assert_allclose(recall(y_true, y_pred, sample_weight=weights), expected, rtol=1e-5)


class TestF1Score:
    def test_perfect(self):
        np.testing.assert_allclose(f1_score(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0])), 1.0)

    def test_zero_prec_and_recall(self):
        assert f1_score(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])) == 0.0

    def test_mixed(self):
        p, r = 2.0 / 3, 2.0 / 3
        expected = 2 * p * r / (p + r)
        np.testing.assert_allclose(f1_score(np.array([0, 0, 1, 1, 1, 0]), np.array([0, 1, 1, 0, 1, 0])), expected, rtol=1e-5)

    def test_weighted(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        prec = precision(y_true, y_pred, sample_weight=weights)
        rec = recall(y_true, y_pred, sample_weight=weights)
        expected = 2 * prec * rec / (prec + rec)
        np.testing.assert_allclose(f1_score(y_true, y_pred, sample_weight=weights), expected, rtol=1e-5)


class TestConcordanceIndex:
    def test_perfect(self):
        np.testing.assert_allclose(concordance_index(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2])), 1.0)

    def test_random_scores(self):
        np.testing.assert_allclose(concordance_index(np.array([0, 1, 1, 0]), np.full(4, 0.5)), 0.5)

    def test_partial(self):
        np.testing.assert_allclose(concordance_index(np.array([0, 1, 0, 1]), np.array([0.1, 0.8, 0.6, 0.4])), 0.75)

    def test_weighted_uniform_same_as_unweighted(self):
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_score = np.array([0.1, 0.9, 0.7, 0.3, 0.8, 0.2])
        c_uw = concordance_index(y_true, y_score)
        c_w = concordance_index(y_true, y_score, sample_weight=np.ones(6))
        np.testing.assert_allclose(c_w, c_uw, rtol=1e-5)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            concordance_index(np.array([]), np.array([]))

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            concordance_index(np.array([0, 1]), np.array([0.1]))

    def test_non_finite_score_raises(self):
        with pytest.raises(ValueError, match="finite"):
            concordance_index(np.array([0, 1]), np.array([0.5, np.inf]))

    def test_all_one_class_raises(self):
        with pytest.raises(ValueError):
            concordance_index(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))

    def test_wrong_weight_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            concordance_index(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2]), sample_weight=np.array([1.0, 2.0]))

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            concordance_index(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2]), sample_weight=np.array([1.0, -1.0, 1.0, 1.0]))

    def test_non_finite_weight_raises(self):
        with pytest.raises(ValueError, match="finite"):
            concordance_index(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2]), sample_weight=np.array([1.0, np.nan, 1.0, 1.0]))

    def test_alternative_binary_encoding(self):
        np.testing.assert_allclose(concordance_index(np.array([-1, 1, 1, -1]), np.array([0.1, 0.9, 0.8, 0.2])), 1.0)


class TestRocAuc:
    def test_perfect(self):
        np.testing.assert_allclose(roc_auc(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2])), 1.0)

    def test_equals_concordance_index(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_score = np.array([0.2, 0.7, 0.4, 0.6, 0.1, 0.9])
        np.testing.assert_allclose(roc_auc(y_true, y_score), concordance_index(y_true, y_score), rtol=1e-10)

    def test_weighted(self):
        np.testing.assert_allclose(roc_auc(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2]), sample_weight=np.ones(4)), 1.0)
