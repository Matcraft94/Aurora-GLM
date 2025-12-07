# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Classification-oriented evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def accuracy_score(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Any | None = None,
    normalize: bool = True,
) -> float:
    """Compute classification accuracy, optionally weighted."""

    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    _validate_shape(true, pred)

    hits = (true == pred).astype(np.float64)
    total = _weighted_sum(hits, sample_weight)
    if normalize:
        weight_sum = _weighted_sum(np.ones_like(hits), sample_weight)
        if weight_sum <= 0.0:
            raise ValueError("sample weights must have positive sum")
        return float(total / weight_sum)
    return float(total)


def log_loss(
    y_true: Any,
    y_prob: Any,
    *,
    eps: float = 1e-15,
    sample_weight: Any | None = None,
) -> float:
    """Compute the negative log-likelihood for probabilistic predictions."""

    true = _to_numpy(y_true)
    prob = _to_numpy(y_prob)

    if true.ndim != 1:
        raise ValueError("y_true must be a one-dimensional array of labels")
    if prob.ndim == 1:
        if not _is_binary_labels(true):
            raise ValueError("Binary probabilities provided but labels are not binary")
        p = np.clip(prob, eps, 1.0 - eps)
        losses = -(true * np.log(p) + (1.0 - true) * np.log(1.0 - p))
    elif prob.ndim == 2:
        n_samples, n_classes = prob.shape
        if true.shape[0] != n_samples:
            raise ValueError("y_true and y_prob must share the same number of samples")
        prob = np.clip(prob, eps, 1.0 - eps)
        row_sums = prob.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Each probability row must sum to 1.0")
        labels = true.astype(int)
        if np.any(labels < 0) or np.any(labels >= n_classes):
            raise ValueError("Labels out of range for provided probability matrix")
        losses = -np.log(prob[np.arange(n_samples), labels])
    else:  # pragma: no cover - defensive branch
        raise ValueError("y_prob must be one- or two-dimensional")

    return float(_weighted_mean(losses, sample_weight))


def brier_score_loss(
    y_true: Any,
    y_prob: Any,
    *,
    sample_weight: Any | None = None,
) -> float:
    """Compute the Brier score for probabilistic binary predictions."""

    true = _to_numpy(y_true)
    prob = _to_numpy(y_prob)

    if prob.ndim != 1:
        raise ValueError("Brier score currently supports 1D binary probabilities only")
    if not _is_binary_labels(true):
        raise ValueError("Brier score requires binary labels")

    diff = true - prob
    losses = diff * diff
    return float(_weighted_mean(losses, sample_weight))


def concordance_index(
    y_true: Any, y_score: Any, *, sample_weight: Any | None = None
) -> float:
    """Compute the concordance index (c-statistic) for binary outcomes.

    The concordance index measures the probability that for a randomly selected
    pair of positive and negative samples, the positive sample has a higher
    predicted score than the negative sample.

    Parameters
    ----------
    y_true : array-like
        Binary labels (must contain exactly two unique values).
    y_score : array-like
        Predicted scores or probabilities.
    sample_weight : array-like, optional
        Sample weights. If provided, the concordance computation is weighted.

    Returns
    -------
    float
        The concordance index, ranging from 0.0 (perfect inverse discrimination)
        to 1.0 (perfect discrimination). A value of 0.5 indicates random predictions.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths, are empty, contain non-finite scores,
        or lack both positive and negative examples.
    """

    true = _to_numpy(y_true).reshape(-1)
    score = _to_numpy(y_score).reshape(-1)

    if true.shape[0] != score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if true.size == 0:
        raise ValueError("Inputs must be non-empty")
    if not np.all(np.isfinite(score)):
        raise ValueError("Predicted scores must be finite")

    binary = _to_binary_labels(true)

    # Handle sample weights
    if sample_weight is None:
        weights = np.ones_like(binary)
    else:
        weights = _to_numpy(sample_weight).reshape(-1)
        if weights.shape[0] != binary.shape[0]:
            raise ValueError("sample_weight must have the same length as y_true")
        if np.any(weights < 0):
            raise ValueError("sample_weight must be non-negative")
        if not np.all(np.isfinite(weights)):
            raise ValueError("sample_weight must be finite")

    n_pos = float(np.sum(binary * weights))
    n_neg = float(np.sum((1.0 - binary) * weights))
    if n_pos <= 0.0 or n_neg <= 0.0:
        raise ValueError(
            "Concordance index requires both positive and negative examples"
        )

    order = np.argsort(-score, kind="mergesort")
    y_sorted = binary[order]
    score_sorted = score[order]
    weights_sorted = weights[order]

    pos_weight_seen = 0.0
    neg_weight_seen = 0.0
    concordant = 0.0
    ties = 0.0

    n = score_sorted.size
    i = 0
    while i < n:
        s_val = score_sorted[i]
        j = i + 1
        while j < n and np.isclose(score_sorted[j], s_val, rtol=1e-12, atol=1e-12):
            j += 1

        group_y = y_sorted[i:j]
        group_w = weights_sorted[i:j]
        pos_weight_group = float(np.sum(group_y * group_w))
        neg_weight_group = float(np.sum((1.0 - group_y) * group_w))

        concordant += neg_weight_group * pos_weight_seen
        ties += pos_weight_group * neg_weight_group
        # discordant count included implicitly via pos_weight_group * neg_weight_seen, but not required for index

        pos_weight_seen += pos_weight_group
        neg_weight_seen += neg_weight_group
        i = j

    total_pairs = n_pos * n_neg
    if total_pairs <= 0.0:
        raise ValueError("No comparable pairs available for concordance computation")

    return (concordant + 0.5 * ties) / total_pairs


def _is_binary_labels(labels: np.ndarray) -> bool:
    unique = np.unique(labels)
    return (
        np.array_equal(unique, [0])
        or np.array_equal(unique, [1])
        or np.array_equal(unique, [0, 1])
    )


def _to_binary_labels(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    if unique.size != 2:
        raise ValueError("Binary labels required for concordance index")
    positive = unique.max()
    return (labels == positive).astype(np.float64)


def _weighted_sum(values: np.ndarray, sample_weight: Any | None) -> float:
    if sample_weight is None:
        return float(np.sum(values))
    weights = _to_numpy(sample_weight)
    if weights.shape != values.shape:
        weights = np.broadcast_to(weights, values.shape)
    return float(np.sum(weights * values))


def _weighted_mean(values: np.ndarray, sample_weight: Any | None) -> float:
    total = _weighted_sum(values, sample_weight)
    if sample_weight is None:
        return total / values.size
    weights = _to_numpy(sample_weight)
    if weights.shape != values.shape:
        weights = np.broadcast_to(weights, values.shape)
    weight_sum = np.sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("sample weights must have positive sum")
    return total / weight_sum


def _validate_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError("Inputs must have identical shapes")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def confusion_matrix(y_true: Any, y_pred: Any) -> np.ndarray:
    """Compute confusion matrix for binary classification.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_pred : array-like
        Predicted binary labels (0 or 1).

    Returns
    -------
    cm : ndarray, shape (2, 2)
        Confusion matrix where cm[i, j] is the count of samples
        with true label i and predicted label j.
        [[TN, FP],
         [FN, TP]]
    """
    true = _to_numpy(y_true).astype(int)
    pred = _to_numpy(y_pred).astype(int)
    _validate_shape(true, pred)

    if not _is_binary_labels(true) or not _is_binary_labels(pred):
        raise ValueError("confusion_matrix requires binary labels (0 and 1)")

    # Compute confusion matrix
    tn = np.sum((true == 0) & (pred == 0))
    fp = np.sum((true == 0) & (pred == 1))
    fn = np.sum((true == 1) & (pred == 0))
    tp = np.sum((true == 1) & (pred == 1))

    return np.array([[tn, fp], [fn, tp]], dtype=int)


def precision(y_true: Any, y_pred: Any, *, sample_weight: Any | None = None) -> float:
    """Compute precision (positive predictive value).

    Precision = TP / (TP + FP)

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    sample_weight : array-like, optional
        Sample weights.

    Returns
    -------
    precision : float
        Precision score. Returns 0.0 if no positive predictions.
    """
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    _validate_shape(true, pred)

    if sample_weight is None:
        tp = np.sum((true == 1) & (pred == 1))
        fp = np.sum((true == 0) & (pred == 1))
    else:
        weights = _to_numpy(sample_weight)
        tp = np.sum(weights * (true == 1) * (pred == 1))
        fp = np.sum(weights * (true == 0) * (pred == 1))

    denominator = tp + fp
    if denominator == 0:
        return 0.0
    return float(tp / denominator)


def recall(y_true: Any, y_pred: Any, *, sample_weight: Any | None = None) -> float:
    """Compute recall (sensitivity, true positive rate).

    Recall = TP / (TP + FN)

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    sample_weight : array-like, optional
        Sample weights.

    Returns
    -------
    recall : float
        Recall score. Returns 0.0 if no positive samples.
    """
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    _validate_shape(true, pred)

    if sample_weight is None:
        tp = np.sum((true == 1) & (pred == 1))
        fn = np.sum((true == 1) & (pred == 0))
    else:
        weights = _to_numpy(sample_weight)
        tp = np.sum(weights * (true == 1) * (pred == 1))
        fn = np.sum(weights * (true == 1) * (pred == 0))

    denominator = tp + fn
    if denominator == 0:
        return 0.0
    return float(tp / denominator)


def f1_score(y_true: Any, y_pred: Any, *, sample_weight: Any | None = None) -> float:
    """Compute F1 score (harmonic mean of precision and recall).

    F1 = 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    sample_weight : array-like, optional
        Sample weights.

    Returns
    -------
    f1 : float
        F1 score. Returns 0.0 if both precision and recall are 0.
    """
    prec = precision(y_true, y_pred, sample_weight=sample_weight)
    rec = recall(y_true, y_pred, sample_weight=sample_weight)

    denominator = prec + rec
    if denominator == 0:
        return 0.0
    return float(2 * prec * rec / denominator)


def roc_auc(y_true: Any, y_score: Any, *, sample_weight: Any | None = None) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

    This is equivalent to the concordance index for binary classification.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted scores or probabilities.
    sample_weight : array-like, optional
        Sample weights.

    Returns
    -------
    auc : float
        ROC-AUC score, ranging from 0.0 to 1.0.
        A value of 0.5 indicates random predictions.
    """
    return concordance_index(y_true, y_score, sample_weight=sample_weight)


__all__ = [
    "accuracy_score",
    "log_loss",
    "brier_score_loss",
    "concordance_index",
    "confusion_matrix",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
]
