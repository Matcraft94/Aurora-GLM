# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Offset handling in the core IRLS optimizer (dense and sparse routes).

Regression tests for the dense-route offset bug: the working response z was
built from the linear predictor *with* offset while the WLS step solves
against X alone, so β absorbed the offset and it doubled every iteration.
Both routes must match statsmodels (Poisson + offset).
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import sparse

from aurora.core.backends import get_backend
from aurora.core.optimization.irls import irls
from aurora.distributions.families import PoissonFamily
from aurora.distributions.links import LogLink


@pytest.fixture
def poisson_offset_case():
    rng = np.random.default_rng(7)
    n = 300
    X = rng.standard_normal((n, 2))
    design = np.column_stack([np.ones(n), X])
    offset = rng.uniform(0.2, 1.5, n)
    eta = design @ np.array([0.3, 0.5, -0.4]) + offset
    y = rng.poisson(np.exp(eta)).astype(float)
    return design, y, offset


def _poisson_loss(design, y, offset):
    def loss(beta):
        mu = np.exp(design @ beta + offset)
        return float(-np.sum(y * np.log(mu) - mu))

    return loss


def _statsmodels_fit(design, y, offset):
    return sm.GLM(y, design, family=sm.families.Poisson(), offset=offset).fit()


class TestDenseOffset:
    def test_dense_offset_matches_statsmodels(self, poisson_offset_case):
        design, y, offset = poisson_offset_case
        family = PoissonFamily()
        result = irls(
            _poisson_loss(design, y, offset),
            np.zeros(design.shape[1]),
            backend=get_backend("numpy"),
            design_matrix=design,
            response=y,
            link=LogLink(),
            variance_fn=family.variance,
            offset=offset,
            max_iter=100,
            tol=1e-10,
        )
        sm_res = _statsmodels_fit(design, y, offset)
        assert result.success
        np.testing.assert_allclose(result.x, sm_res.params, atol=1e-8)

    def test_dense_offset_converges_quickly(self, poisson_offset_case):
        """With the offset handled correctly, canonical-link IRLS needs few
        iterations (the buggy version failed to converge in 100)."""
        design, y, offset = poisson_offset_case
        family = PoissonFamily()
        result = irls(
            _poisson_loss(design, y, offset),
            np.zeros(design.shape[1]),
            backend=get_backend("numpy"),
            design_matrix=design,
            response=y,
            link=LogLink(),
            variance_fn=family.variance,
            offset=offset,
            max_iter=25,
            tol=1e-8,
        )
        assert result.success
        assert result.nit <= 25


class TestSparseOffset:
    def test_sparse_offset_matches_statsmodels(self, poisson_offset_case):
        design, y, offset = poisson_offset_case
        family = PoissonFamily()
        result = irls(
            _poisson_loss(design, y, offset),
            np.zeros(design.shape[1]),
            design_matrix=sparse.csr_matrix(design),
            response=y,
            link=LogLink(),
            variance_fn=family.variance,
            offset=offset,
            max_iter=100,
            tol=1e-10,
        )
        sm_res = _statsmodels_fit(design, y, offset)
        assert result.success
        np.testing.assert_allclose(result.x, sm_res.params, atol=1e-8)

    def test_dense_and_sparse_agree(self, poisson_offset_case):
        design, y, offset = poisson_offset_case
        family = PoissonFamily()
        kwargs = {
            "response": y,
            "link": LogLink(),
            "variance_fn": family.variance,
            "offset": offset,
            "max_iter": 100,
            "tol": 1e-10,
        }
        dense = irls(
            _poisson_loss(design, y, offset),
            np.zeros(design.shape[1]),
            backend=get_backend("numpy"),
            design_matrix=design,
            **kwargs,
        )
        sparse_res = irls(
            _poisson_loss(design, y, offset),
            np.zeros(design.shape[1]),
            design_matrix=sparse.csr_matrix(design),
            **kwargs,
        )
        np.testing.assert_allclose(dense.x, sparse_res.x, atol=1e-8)
