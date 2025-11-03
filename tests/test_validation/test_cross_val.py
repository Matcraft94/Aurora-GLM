"""Tests for cross-validation utilities."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models.glm import fit_glm
from aurora.validation.metrics import mean_squared_error
from aurora.validation.cross_val import CrossValResult, KFold, StratifiedKFold, cross_val_score


def _dataset(seed: int = 2024) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 3))
    coef = np.array([0.8, -0.4, 0.25])
    intercept = 1.1
    noise = rng.normal(scale=0.5, size=X.shape[0])
    y = intercept + X @ coef + noise
    return X, y


def _classification_dataset(seed: int = 1234) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(150, 4))
    logits = X @ np.array([0.6, -0.9, 0.3, 0.2]) + 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)
    return X, y


def test_kfold_split_produces_non_overlapping_partitions():
    X, _ = _dataset()
    splitter = KFold(n_splits=6)
    folds = list(splitter.split(X))

    assert len(folds) == 6
    n_samples = X.shape[0]
    seen = np.zeros(n_samples, dtype=int)

    for train_idx, test_idx in folds:
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        seen[test_idx] += 1

    assert np.all(seen == 1)

    with pytest.raises(ValueError):
        list(KFold(n_splits=1).split(X))

    with pytest.raises(ValueError):
        list(KFold(n_splits=500).split(X))


def _scoring_function(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    predictions = model.predict(X_val)
    return -mean_squared_error(y_val, predictions)


def _classification_score(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    probs = model.predict(X_val)
    preds = (probs >= 0.5).astype(int)
    return float(np.mean(preds == y_val.astype(int)))


def test_cross_val_score_returns_expected_length_and_variability():
    X, y = _dataset()
    scores = cross_val_score(
        fit_glm,
        _scoring_function,
        X,
        y,
        n_splits=4,
        shuffle=True,
        random_state=42,
        fit_kwargs={"family": "gaussian", "link": None, "max_iter": 60, "tol": 1e-9},
    )

    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))
    assert not np.allclose(scores, scores[0])


def test_cross_val_score_can_return_summary_result():
    X, y = _dataset()
    result = cross_val_score(
        fit_glm,
        _scoring_function,
        X,
        y,
        n_splits=5,
        shuffle=True,
        random_state=101,
        fit_kwargs={"family": "gaussian", "link": None},
        return_result=True,
    )

    assert isinstance(result, CrossValResult)
    assert result.scores.shape == (5,)
    expected_mean = float(np.mean(result.scores))
    expected_std = float(np.std(result.scores, ddof=1))
    assert result.mean == expected_mean
    assert pytest.approx(expected_std, rel=1e-9, abs=1e-9) == result.std


def test_cross_val_score_with_deterministic_shuffling():
    X, y = _dataset()
    kwargs = {
        "fit_kwargs": {"family": "gaussian", "link": None},
        "n_splits": 3,
        "shuffle": True,
        "random_state": 123,
    }

    first = cross_val_score(fit_glm, _scoring_function, X, y, **kwargs)
    second = cross_val_score(fit_glm, _scoring_function, X, y, **kwargs)
    assert np.allclose(first, second)

    different = cross_val_score(
        fit_glm,
        _scoring_function,
        X,
        y,
        n_splits=3,
        shuffle=True,
        random_state=456,
        fit_kwargs={"family": "gaussian", "link": None},
    )
    assert not np.allclose(first, different)


def test_cross_val_score_accepts_custom_splitter_object():
    class TwoFoldSplitter:
        def __init__(self) -> None:
            self.calls = 0

        def split(self, X, y=None):  # noqa: ANN001 - match cross_val usage
            self.calls += 1
            n_samples = X.shape[0]
            midpoint = n_samples // 2
            idx = np.arange(n_samples)
            yield idx[:midpoint], idx[midpoint:]
            yield idx[midpoint:], idx[:midpoint]

    X, y = _dataset()
    splitter = TwoFoldSplitter()
    scores = cross_val_score(
        fit_glm,
        _scoring_function,
        X,
        y,
        splitter=splitter,
        fit_kwargs={"family": "gaussian", "link": None},
    )

    assert splitter.calls == 0
    assert scores.shape == (2,)


def test_cross_val_score_with_stratified_identifier():
    X, y = _classification_dataset()
    scores = cross_val_score(
        fit_glm,
        _classification_score,
        X,
        y,
        splitter="stratified",
        n_splits=4,
        shuffle=True,
        random_state=11,
        fit_kwargs={"family": "binomial", "link": None, "max_iter": 60, "tol": 1e-9},
    )

    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))


def test_cross_val_score_stratified_identifier_requires_class_support():
    X = np.linspace(0.0, 1.0, num=6).reshape(3, 2)
    y = np.array([0, 0, 1])

    with pytest.raises(ValueError):
        cross_val_score(
            fit_glm,
            _classification_score,
            X,
            y,
            splitter="stratified",
            n_splits=3,
            fit_kwargs={"family": "binomial", "link": None},
        )


def test_cross_val_score_reuses_splitter_instance_safely():
    X, y = _classification_dataset(seed=7)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=21)

    scores_first = cross_val_score(
        fit_glm,
        _classification_score,
        X,
        y,
        splitter=splitter,
        fit_kwargs={"family": "binomial", "link": None, "max_iter": 60, "tol": 1e-9},
    )

    scores_second = cross_val_score(
        fit_glm,
        _classification_score,
        X,
        y,
        splitter=splitter,
        fit_kwargs={"family": "binomial", "link": None, "max_iter": 60, "tol": 1e-9},
    )

    assert np.allclose(scores_first, scores_second)


def test_stratified_kfold_preserves_class_counts():
    X, y = _classification_dataset()
    splitter = StratifiedKFold(n_splits=5)

    total_counts = np.bincount(y)
    folds = list(splitter.split(X, y))
    assert len(folds) == 5

    for train_idx, test_idx in folds:
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        fold_counts = np.bincount(y[test_idx], minlength=total_counts.size)
        expected = total_counts / splitter.n_splits
        # Allow small variance due to integer division
        assert np.all(np.abs(fold_counts - expected) <= 1.0)


def test_stratified_kfold_shuffle_is_reproducible():
    X, y = _classification_dataset(seed=42)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
    first = list(splitter.split(X, y))
    second = list(splitter.split(X, y))
    assert all(np.array_equal(f1[0], f2[0]) and np.array_equal(f1[1], f2[1]) for f1, f2 in zip(first, second))

    different = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=9).split(X, y))
    assert any(
        not np.array_equal(f1[1], f2[1])
        for f1, f2 in zip(first, different)
    )
