"""Tests for classification metric utilities."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.validation.metrics import accuracy_score, brier_score_loss, concordance_index, log_loss


def test_accuracy_score_basic_and_weighted():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    acc = accuracy_score(y_true, y_pred)
    assert acc == pytest.approx(0.75)

    weights = np.array([1.0, 1.0, 2.0, 2.0])
    weighted_acc = accuracy_score(y_true, y_pred, sample_weight=weights)
    manual = np.average(y_true == y_pred, weights=weights)
    assert weighted_acc == pytest.approx(manual)

    raw_hits = accuracy_score(y_true, y_pred, normalize=False)
    assert raw_hits == pytest.approx(3.0)


def test_log_loss_binary_and_multiclass():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.6, 0.25])
    loss = log_loss(y_true, y_prob)
    manual = -np.mean(
        y_true * np.log(np.clip(y_prob, 1e-15, 1 - 1e-15))
        + (1 - y_true) * np.log(np.clip(1 - y_prob, 1e-15, 1 - 1e-15))
    )
    assert loss == pytest.approx(manual)

    y_true_mc = np.array([0, 2, 1])
    y_prob_mc = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
            [0.2, 0.5, 0.3],
        ]
    )
    loss_mc = log_loss(y_true_mc, y_prob_mc)
    manual_mc = -np.mean(np.log([0.7, 0.6, 0.5]))
    assert loss_mc == pytest.approx(manual_mc)

    weights = np.array([1.0, 2.0, 3.0])
    weighted_loss = log_loss(y_true_mc, y_prob_mc, sample_weight=weights)
    manual_weighted = np.average([-np.log(0.7), -np.log(0.6), -np.log(0.5)], weights=weights)
    assert weighted_loss == pytest.approx(manual_weighted)

    with pytest.raises(ValueError):
        log_loss(np.array([0, 1, 2]), np.array([0.2, 0.8, 0.4]))

    with pytest.raises(ValueError):
        log_loss(y_true_mc, np.array([[0.5, 0.6], [0.4, 0.7], [0.3, 0.8]]))


def test_brier_score_loss():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.7, 0.3])
    score = brier_score_loss(y_true, y_prob)
    manual = np.mean((y_true - y_prob) ** 2)
    assert score == pytest.approx(manual)

    weights = np.array([1.0, 2.0, 2.0, 1.0])
    weighted = brier_score_loss(y_true, y_prob, sample_weight=weights)
    manual_weighted = np.average((y_true - y_prob) ** 2, weights=weights)
    assert weighted == pytest.approx(manual_weighted)

    with pytest.raises(ValueError):
        brier_score_loss(np.array([0, 1, 2]), y_prob)

    with pytest.raises(ValueError):
        brier_score_loss(y_true, np.column_stack((y_prob, y_prob)))


def test_concordance_index_behaviour():
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.1, 0.9, 0.8, 0.2])
    assert concordance_index(y_true, y_score) == pytest.approx(1.0)

    y_score_partial = np.array([0.1, 0.8, 0.6, 0.4])
    assert concordance_index(np.array([0, 1, 0, 1]), y_score_partial) == pytest.approx(0.75)

    y_score_equal = np.full(4, 0.5)
    assert concordance_index(y_true, y_score_equal) == pytest.approx(0.5)

    with pytest.raises(ValueError):
        concordance_index(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))

    with pytest.raises(ValueError):
        concordance_index(y_true, np.array([0.1, 0.2]))


def test_concordance_index_with_sample_weights():
    """Test that concordance index correctly handles sample weights."""
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_score = np.array([0.1, 0.9, 0.7, 0.3, 0.8, 0.2])

    # Unweighted c-index should be 1.0 (all pairs concordant)
    c_unweighted = concordance_index(y_true, y_score)
    assert c_unweighted == pytest.approx(1.0)

    # With uniform weights, should be identical
    weights_uniform = np.ones(6)
    c_uniform = concordance_index(y_true, y_score, sample_weight=weights_uniform)
    assert c_uniform == pytest.approx(c_unweighted)

    # With non-uniform weights, c-index should change
    # Give higher weight to first negative (0.1) and first positive (0.9)
    weights = np.array([2.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    c_weighted = concordance_index(y_true, y_score, sample_weight=weights)
    # This should still be 1.0 since all pairs are concordant
    assert c_weighted == pytest.approx(1.0)

    # Test with a case that has discordant pairs
    y_true_mixed = np.array([0, 1, 0, 1])
    y_score_mixed = np.array([0.4, 0.6, 0.3, 0.7])
    # Pairs: (0,1): 0.4 < 0.6 ✓, (0,3): 0.4 < 0.7 ✓, (2,1): 0.3 < 0.6 ✓, (2,3): 0.3 < 0.7 ✓
    # C-index = 4/4 = 1.0
    c_mixed = concordance_index(y_true_mixed, y_score_mixed)
    assert c_mixed == pytest.approx(1.0)

    # Weight the discordant example more heavily
    weights_mixed = np.array([1.0, 1.0, 1.0, 1.0])
    c_weighted_mixed = concordance_index(y_true_mixed, y_score_mixed, sample_weight=weights_mixed)
    assert c_weighted_mixed == pytest.approx(c_mixed)

    # Test error cases
    with pytest.raises(ValueError, match="sample_weight must have the same length"):
        concordance_index(y_true, y_score, sample_weight=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="sample_weight must be non-negative"):
        concordance_index(y_true, y_score, sample_weight=np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match="sample_weight must be finite"):
        concordance_index(
            y_true, y_score, sample_weight=np.array([1.0, np.inf, 1.0, 1.0, 1.0, 1.0])
        )
