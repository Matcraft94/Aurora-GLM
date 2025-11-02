"""Cross-validation scoring utilities."""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .split import KFold

FitFunc = Callable[..., Any]
ScoreFunc = Callable[[Any, Sequence[Any], Sequence[Any]], float]


def cross_val_score(
    fit_func: FitFunc,
    score_func: ScoreFunc,
    X: Sequence[Any],
    y: Sequence[Any],
    *,
    n_splits: int = 5,
    shuffle: bool = False,
    random_state: int | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    score_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate a model using K-fold cross-validation.

    ``fit_func`` must accept the training design matrix and response as its first
    two positional arguments and return a fitted model object. ``score_func`` must
    accept the fitted model, validation design matrix, and response, returning a
    scalar score where larger values indicate better performance.
    """

    if fit_kwargs is None:
        fit_kwargs = {}
    if score_kwargs is None:
        score_kwargs = {}

    X_np = _to_numpy(X)
    y_np = _to_numpy(y)

    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores: list[float] = []

    for train_idx, test_idx in splitter.split(X_np, y_np):
        X_train = X_np[train_idx]
        y_train = y_np[train_idx]
        X_test = X_np[test_idx]
        y_test = y_np[test_idx]

        model = fit_func(X_train, y_train, **fit_kwargs)
        score = score_func(model, X_test, y_test, **score_kwargs)
        scores.append(float(score))

    return np.asarray(scores, dtype=np.float64)


def _to_numpy(value: Sequence[Any] | Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


__all__ = ["cross_val_score"]
