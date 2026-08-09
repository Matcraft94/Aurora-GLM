# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Robustness diagnostics of the production GLM IRLS loop.

Covers step-halving, separation detection, non-convergence warnings,
condition-number monitoring, and rank-deficiency/aliasing reporting.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from aurora.models.glm import fit_glm


class TestSeparationDetection:
    def test_complete_separation_warns_without_overflow(self):
        """Complete separation must produce R-style warnings, not raw overflow."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([0, 0, 0, 1, 1, 1])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit_glm(X, y, family="binomial", max_iter=100)

        messages = [str(w.message) for w in caught]
        assert any("fitted probabilities numerically 0 or 1" in m for m in messages)
        # No raw numerical overflow should leak to the user
        assert not any("overflow" in m.lower() for m in messages)
        assert np.all(np.isfinite(np.asarray(result.mu_)))

    def test_well_behaved_binomial_no_separation_warning(self):
        rng = np.random.default_rng(5)
        n = 300
        X = rng.standard_normal((n, 1))
        p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * X[:, 0])))
        y = rng.binomial(1, p)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_glm(X, y, family="binomial")

        messages = [str(w.message) for w in caught]
        assert not any("fitted probabilities" in m for m in messages)


class TestNonConvergenceWarning:
    def test_zero_max_iter_warns(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 2))
        y = rng.normal(size=100)
        with pytest.warns(RuntimeWarning, match="failed to converge"):
            result = fit_glm(X, y, family="gaussian", max_iter=0)
        assert not result.converged_
        assert result.n_iter_ == 0

    def test_converged_fit_does_not_warn(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 2))
        y = 1.0 + X @ np.array([0.5, -0.3]) + rng.normal(scale=0.5, size=200)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit_glm(X, y, family="gaussian")
        assert result.converged_
        assert not any("failed to converge" in str(w.message) for w in caught)


class TestConditionNumberMonitoring:
    def test_ill_conditioned_design_warns(self):
        """x2 = x1 + 1e-9·ε must trigger the κ > 1e8 warning."""
        rng = np.random.default_rng(9)
        n = 200
        x1 = rng.standard_normal(n)
        X = np.column_stack([x1, x1 + 1e-9 * rng.standard_normal(n)])
        y = rng.standard_normal(n)

        with pytest.warns(RuntimeWarning, match="Ill-conditioned"):
            fit_glm(X, y, family="gaussian")

    def test_condition_number_recorded(self):
        rng = np.random.default_rng(9)
        n = 200
        x1 = rng.standard_normal(n)
        X = np.column_stack([x1, x1 + 1e-9 * rng.standard_normal(n)])
        y = rng.standard_normal(n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_glm(X, y, family="gaussian")
        assert result.condition_number_ is not None
        assert result.condition_number_ > 1e8


class TestRankDeficiencyReporting:
    def test_duplicate_column_reports_rank(self):
        rng = np.random.default_rng(11)
        X_base = rng.standard_normal((100, 2))
        X = np.column_stack([X_base, X_base[:, 0]])
        y = rng.standard_normal(100)

        with pytest.warns(RuntimeWarning, match="rank deficient"):
            result = fit_glm(X, y, family="gaussian")
        assert result.rank_ == 3  # intercept + 2 independent columns
        assert np.all(np.isfinite(result.coef_))

    def test_full_rank_design_reports_full_rank(self):
        rng = np.random.default_rng(12)
        X = rng.standard_normal((100, 2))
        y = rng.standard_normal(100)
        result = fit_glm(X, y, family="gaussian")
        assert result.rank_ == 3
