"""Tests for cross-validation utilities."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models.glm import fit_glm
from aurora.validation.metrics import mean_squared_error
from aurora.validation.cross_val import KFold, cross_val_score


def _dataset(seed: int = 2024) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 3))
    coef = np.array([0.8, -0.4, 0.25])
    intercept = 1.1
    noise = rng.normal(scale=0.5, size=X.shape[0])
    y = intercept + X @ coef + noise
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
