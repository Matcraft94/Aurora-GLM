# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Cross-validation scoring utilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np

from .split import KFold, StratifiedKFold

FitFunc = Callable[..., Any]
ScoreFunc = Callable[[Any, Sequence[Any], Sequence[Any]], float]


@dataclass(frozen=True)
class CrossValResult:
    """Aggregate statistics for a cross-validation run."""

    scores: np.ndarray
    mean: float
    std: float


def cross_val_score(
    fit_func: FitFunc,
    score_func: ScoreFunc,
    X: Sequence[Any],
    y: Sequence[Any],
    *,
    n_splits: int = 5,
    shuffle: bool = False,
    random_state: int | None = None,
    splitter: str | Any | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    score_kwargs: dict[str, Any] | None = None,
    return_result: bool = False,
) -> np.ndarray | CrossValResult:
    """Evaluate a model using K-fold cross-validation.

    ``fit_func`` must accept the training design matrix and response as its first
    two positional arguments and return a fitted model object. ``score_func`` must
    accept the fitted model, validation design matrix, and response, returning a
    scalar score where larger values indicate better performance.

    Set ``return_result`` to ``True`` to obtain a ``CrossValResult`` with summary
    statistics in addition to the fold scores.
    """

    if fit_kwargs is None:
        fit_kwargs = {}
    if score_kwargs is None:
        score_kwargs = {}

    X_np = _to_numpy(X)
    y_np = _to_numpy(y)

    splitter_obj = _resolve_splitter(
        splitter,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    scores: list[float] = []

    for train_idx, test_idx in splitter_obj.split(X_np, y_np):
        X_train = X_np[train_idx]
        y_train = y_np[train_idx]
        X_test = X_np[test_idx]
        y_test = y_np[test_idx]

        model = fit_func(X_train, y_train, **fit_kwargs)
        score = score_func(model, X_test, y_test, **score_kwargs)
        scores.append(float(score))

    scores_arr = np.asarray(scores, dtype=np.float64)
    if not return_result:
        return scores_arr

    mean = float(np.mean(scores_arr)) if scores_arr.size else float("nan")
    std = float(np.std(scores_arr, ddof=1)) if scores_arr.size > 1 else 0.0
    return CrossValResult(scores=scores_arr, mean=mean, std=std)


def _to_numpy(value: Sequence[Any] | Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _resolve_splitter(
    splitter: str | Any | None,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int | None,
) -> Any:
    if splitter is None:
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    if isinstance(splitter, str):
        key = splitter.lower()
        if key in {"kfold", "k-fold"}:
            return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        if key in {"stratified", "stratifiedkfold", "stratified-kfold"}:
            return StratifiedKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
        raise ValueError(f"Unknown splitter identifier: {splitter!r}")

    if hasattr(splitter, "split") and callable(splitter.split):
        return _clone_splitter(splitter)

    raise TypeError(
        "splitter must be None, a string identifier, or an object with a split() method"
    )


def _clone_splitter(splitter: Any) -> Any:
    if is_dataclass(splitter):
        return replace(splitter)
    try:
        return copy.deepcopy(splitter)
    except Exception:  # pragma: no cover - fallback branch
        return splitter


__all__ = ["CrossValResult", "cross_val_score"]
