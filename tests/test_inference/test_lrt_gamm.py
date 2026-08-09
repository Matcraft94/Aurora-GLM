# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Integration tests: likelihood_ratio_test with real GAMMResult objects.

Covers the attribute harmonization between the anova module and
``GAMMResult`` (``variance_components``, ``residual_variance``,
``log_likelihood`` — no trailing underscores), the Self & Liang (1987)
boundary correction, and the documented PQL/REML caveats.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import stats as sst

from aurora.inference.anova import (
    _detect_boundary_conditions,
    likelihood_ratio_test,
)
from aurora.models.gamm import RandomEffect, fit_gamm


@pytest.fixture
def gaussian_gamm_pair():
    """Two nested Gaussian GAMMs with random intercepts."""
    np.random.seed(42)
    n_groups, n_per = 12, 15
    groups = np.repeat(np.arange(n_groups), n_per)
    n = len(groups)
    x = np.random.randn(n)
    b = np.random.randn(n_groups) * 0.5
    y = 1 + 0.8 * x + b[groups] + np.random.randn(n) * 0.3

    re = RandomEffect(grouping="g")
    full = fit_gamm(
        y=y,
        X=np.column_stack([np.ones(n), x]),
        random_effects=[re],
        groups_data={"g": groups},
        family="gaussian",
    )
    reduced = fit_gamm(
        y=y,
        X=np.ones((n, 1)),
        random_effects=[re],
        groups_data={"g": groups},
        family="gaussian",
    )
    return reduced, full


class TestGAMMBoundaryDetection:
    """_detect_boundary_conditions understands GAMMResult attribute names."""

    def test_no_boundary_on_real_fit(self, gaussian_gamm_pair):
        _, full = gaussian_gamm_pair
        # Fitted with genuine random-effect variance: nothing at boundary
        assert full.variance_components[0][0, 0] > 1e-6
        assert _detect_boundary_conditions(full) == []

    def test_detects_boundary_variance_component(self, gaussian_gamm_pair):
        _, full = gaussian_gamm_pair
        at_boundary = dataclasses.replace(full, variance_components=[np.array([[1e-15]])])
        assert "variance_component_0" in _detect_boundary_conditions(at_boundary)

    def test_detects_boundary_residual_variance(self, gaussian_gamm_pair):
        _, full = gaussian_gamm_pair
        at_boundary = dataclasses.replace(full, residual_variance=1e-15)
        assert "residual_variance" in _detect_boundary_conditions(at_boundary)


class TestGAMMLRT:
    """LRT runs on GAMMResult (log_likelihood without underscore)."""

    def test_lrt_runs_and_warns_about_reml(self, gaussian_gamm_pair):
        reduced, full = gaussian_gamm_pair
        with pytest.warns(RuntimeWarning, match="REML"):
            result = likelihood_ratio_test(reduced, full)

        assert result.df == len(full.beta_parametric) - len(reduced.beta_parametric)
        assert result.statistic >= 0
        assert 0 <= result.p_value <= 1
        assert np.isclose(result.statistic, 2 * (full.log_likelihood - reduced.log_likelihood))

    def test_lrt_applies_boundary_correction(self, gaussian_gamm_pair):
        reduced, full = gaussian_gamm_pair
        at_boundary = dataclasses.replace(full, variance_components=[np.array([[0.0]])])
        with pytest.warns(RuntimeWarning, match="REML"):
            result = likelihood_ratio_test(reduced, at_boundary)

        assert result.boundary_correction_applied is True
        assert result.boundary_conditions is not None
        # df = 1: mixture 0.5·χ²₁ + 0.5·χ²₀ with χ²₀ a point mass at 0,
        # so p = 0.5·P(χ²₁ > t) for t > 0 (Self & Liang 1987, case 5).
        assert result.df == 1
        expected_p = 0.5 * sst.chi2.sf(result.statistic, 1)
        assert np.isclose(result.p_value, expected_p, rtol=1e-10)

    def test_lrt_without_boundary_uses_standard_chi2(self, gaussian_gamm_pair):
        reduced, full = gaussian_gamm_pair
        with pytest.warns(RuntimeWarning, match="REML"):
            result = likelihood_ratio_test(reduced, full)

        assert result.boundary_correction_applied is False
        assert result.boundary_conditions is None
        assert np.isclose(result.p_value, sst.chi2.sf(result.statistic, result.df))


class TestPQLWarning:
    """Non-Gaussian (PQL) GAMMs trigger the conditional-likelihood warning."""

    def test_lrt_on_pql_result_warns(self):
        np.random.seed(123)
        n_groups, n_per = 8, 12
        groups = np.repeat(np.arange(n_groups), n_per)
        n = len(groups)
        x = np.random.randn(n)
        b = np.random.randn(n_groups) * 0.3
        mu = np.exp(0.8 + 0.4 * x + b[groups])
        y = np.random.poisson(mu)

        re = RandomEffect(grouping="g")
        common = {"random_effects": [re], "groups_data": {"g": groups}, "family": "poisson"}
        full = fit_gamm(y=y, X=np.column_stack([np.ones(n), x]), **common)
        reduced = fit_gamm(y=y, X=np.ones((n, 1)), **common)

        with pytest.warns(RuntimeWarning, match="PQL"):
            result = likelihood_ratio_test(reduced, full)

        assert result.statistic >= 0
        assert 0 <= result.p_value <= 1
