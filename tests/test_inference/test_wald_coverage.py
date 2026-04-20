# SPDX-License-Identifier: MIT
"""Additional coverage tests for aurora.inference.hypothesis.wald module.

Covers edge cases: singular covariance matrices, _combine_parameters,
_full_covariance branches, _regularized_gamma_p continued-fraction path,
_chi_square_sf boundary values, and _symmetrise_matrix.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aurora.inference.hypothesis.wald import (
    _chi_square_sf,
    _combine_parameters,
    _full_covariance,
    _quadratic_form,
    _regularized_gamma_p,
    _symmetrise_matrix,
    wald_test,
)
from aurora.models.glm import fit_glm


def _make_mock_result(coef, intercept=None, cov=None):
    """Create a mock result object for Wald tests."""

    class R:
        pass

    r = R()
    r.coef_ = np.asarray(coef, dtype=float)
    r.intercept_ = intercept
    p = len(coef) + (1 if intercept is not None else 0)
    r.coef_cov_ = cov if cov is not None else np.eye(p) * 0.1
    return r


class TestCombineParameters:
    """Cover _combine_parameters branches."""

    def test_with_intercept(self):
        r = _make_mock_result([1.0, 2.0], intercept=0.5)
        result = _combine_parameters(r, include_intercept=True)
        assert_allclose(result, [0.5, 1.0, 2.0])

    def test_without_intercept(self):
        r = _make_mock_result([1.0, 2.0], intercept=0.5)
        result = _combine_parameters(r, include_intercept=False)
        assert_allclose(result, [1.0, 2.0])

    def test_none_intercept(self):
        r = _make_mock_result([1.0, 2.0], intercept=None)
        result = _combine_parameters(r, include_intercept=True)
        assert_allclose(result, [1.0, 2.0])


class TestFullCovariance:
    """Cover _full_covariance branches."""

    def test_with_intercept_returns_full(self):
        cov = np.eye(3) * 0.5
        r = _make_mock_result([1.0, 2.0], intercept=0.5, cov=cov)
        result = _full_covariance(r, include_intercept=True)
        assert result is cov

    def test_intercept_none_with_include_raises(self):
        cov = np.eye(2) * 0.5
        r = _make_mock_result([1.0, 2.0], intercept=None, cov=cov)
        with pytest.raises(ValueError, match="intercept"):
            _full_covariance(r, include_intercept=True)

    def test_excludes_intercept_rows(self):
        cov = np.array([[0.1, 0.01, 0.0], [0.01, 0.2, 0.0], [0.0, 0.0, 0.3]])
        r = _make_mock_result([1.0, 2.0], intercept=0.5, cov=cov)
        result = _full_covariance(r, include_intercept=False)
        assert result.shape == (2, 2)
        assert_allclose(result, cov[1:, 1:])

    def test_no_intercept_include_false(self):
        cov = np.eye(2) * 0.5
        r = _make_mock_result([1.0, 2.0], intercept=None, cov=cov)
        result = _full_covariance(r, include_intercept=False)
        assert result is cov


class TestSymmetriseMatrix:
    """Cover _symmetrise_matrix."""

    def test_already_symmetric(self):
        M = np.array([[1.0, 0.5], [0.5, 2.0]])
        result = _symmetrise_matrix(M)
        assert_allclose(result, M)

    def test_asymmetric_input(self):
        M = np.array([[1.0, 3.0], [0.0, 2.0]])
        result = _symmetrise_matrix(M)
        assert_allclose(result, 0.5 * (M + M.T))
        assert_allclose(result[0, 1], result[1, 0])


class TestQuadraticForm:
    """Cover _quadratic_form including LinAlg fallback."""

    def test_identity_covariance(self):
        diff = np.array([1.0, 2.0])
        cov = np.eye(2)
        qf = _quadratic_form(diff, cov)
        assert_allclose(qf, 5.0)

    def test_singular_covariance_uses_pinv(self):
        """Singular matrix should fall back to pseudoinverse."""
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
        diff = np.array([1.0, 1.0])
        qf = _quadratic_form(diff, cov)
        assert np.isfinite(qf)

    def test_general_covariance(self):
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        diff = np.array([1.0, 1.0])
        qf = _quadratic_form(diff, cov)
        # diff^T cov^{-1} diff
        expected = diff @ np.linalg.solve(cov, diff)
        assert_allclose(qf, expected, rtol=1e-10)


class TestChiSquareSf:
    """Cover _chi_square_sf boundary conditions."""

    def test_zero_value_returns_one(self):
        assert _chi_square_sf(0.0, 2.0) == 1.0

    def test_negative_value_returns_one(self):
        assert _chi_square_sf(-5.0, 3.0) == 1.0

    def test_large_value_returns_near_zero(self):
        result = _chi_square_sf(100.0, 2.0)
        assert result < 1e-10

    def test_result_bounded(self):
        """Result should be in [0, 1]."""
        for val in [0.5, 5.0, 50.0]:
            for df in [1.0, 3.0, 10.0]:
                result = _chi_square_sf(val, df)
                assert 0.0 <= result <= 1.0


class TestRegularizedGammaP:
    """Cover _regularized_gamma_p series and continued-fraction paths."""

    def test_series_path_x_small(self):
        """x < a + 1 uses series expansion."""
        result = _regularized_gamma_p(2.0, 1.0)
        assert 0.0 < result < 1.0

    def test_cf_path_x_large(self):
        """x >= a + 1 uses continued fraction."""
        result = _regularized_gamma_p(2.0, 10.0)
        assert 0.9 < result < 1.0

    def test_x_zero_returns_zero(self):
        assert _regularized_gamma_p(2.0, 0.0) == 0.0

    def test_negative_a_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _regularized_gamma_p(-1.0, 1.0)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _regularized_gamma_p(2.0, -1.0)


class TestWaldTestMultiConstraint:
    """Cover multi-constraint Wald test with real fit."""

    def test_multi_constraint_with_values(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(300, 3))
        y = 0.5 + X @ [1.0, -0.5, 0.3] + rng.normal(scale=0.3, size=300)
        result = fit_glm(X, y, family="gaussian", max_iter=50)

        contrast = np.array(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
        test = wald_test(result, contrast, value=[1.0, -0.5])
        assert test["df"] == pytest.approx(2.0)
        assert test["p_value"] >= 0.0

    def test_single_constraint_value_scalar(self):
        """Scalar value broadcast to n_constraints."""
        r = _make_mock_result([1.0, 2.0], intercept=0.5, cov=np.eye(3) * 0.1)
        test = wald_test(r, [1.0, 0.0, 0.0], value=5.0)
        assert test["df"] == pytest.approx(1.0)

    def test_value_array_mismatch_raises(self):
        r = _make_mock_result([1.0, 2.0], intercept=0.5, cov=np.eye(3) * 0.1)
        with pytest.raises(ValueError, match="value dimension"):
            wald_test(r, [1.0, 0.0, 0.0], value=[1.0, 2.0])

    def test_3d_contrast_raises(self):
        r = _make_mock_result([1.0, 2.0], intercept=0.5, cov=np.eye(3) * 0.1)
        with pytest.raises(ValueError, match="vector or matrix"):
            wald_test(r, np.ones((2, 3, 4)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
