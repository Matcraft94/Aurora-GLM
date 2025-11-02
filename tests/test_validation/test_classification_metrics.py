"""Tests for classification metric utilities."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.validation.metrics import accuracy_score, brier_score_loss, log_loss


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
