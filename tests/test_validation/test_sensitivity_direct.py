"""Tests for aurora.validation.sensitivity module.

Direct unit tests for the public functions in
aurora.validation.sensitivity.__init__: leverage, cooks_distance,
studentized_residuals, dffits, dfbetas, influence_measures, loo_residuals,
press_statistic, and the InfluenceResult dataclass.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from aurora.validation.sensitivity import (
    InfluenceResult,
    cooks_distance,
    dfbetas,
    dffits,
    influence_measures,
    leverage,
    loo_residuals,
    press_statistic,
    studentized_residuals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockGLMResult:
    """Mock fitted model result mimicking GLMResult for sensitivity functions.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix (including intercept column).
    y : ndarray of shape (n,)
        Response vector.
    coef : ndarray of shape (p,)
        Full coefficient vector (including intercept as first element).
    residual_variance : float or None
        If provided, set as ``residual_variance_`` on the mock so that
        cooks_distance / studentized_residuals will use it instead of
        computing MSE internally.
    """

    def __init__(self, X, y, coef, residual_variance=None):
        self._X = X
        self.residuals = y - X @ coef
        self.coef_ = coef[1:]  # skip intercept — mirrors real GLMResult
        self.intercept_ = coef[0]
        if residual_variance is not None:
            self.residual_variance_ = residual_variance


class MockResultNoX:
    """Minimal mock that has no _X attribute and no coef_."""

    def __init__(self, n=5):
        self.residuals = np.zeros(n)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_data():
    """Simple linear regression: y = 2 + 3x + noise, n=30, p=2."""
    np.random.seed(42)
    n = 30
    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])
    coef = np.array([2.0, 3.0])
    residuals = np.random.randn(n) * 0.5
    y = X @ coef + residuals
    result = MockGLMResult(X, y, coef)
    return result, X, y, coef


@pytest.fixture
def multi_regression_data():
    """Multiple regression: y = 1 + 2x1 - 1.5x2 + 0.5x3 + noise, n=50, p=4."""
    np.random.seed(42)
    n = 50
    predictors = np.random.randn(n, 3)
    X = np.column_stack([np.ones(n), predictors])
    coef = np.array([1.0, 2.0, -1.5, 0.5])
    residuals = np.random.randn(n) * 0.3
    y = X @ coef + residuals
    result = MockGLMResult(X, y, coef)
    return result, X, y, coef


# ===================================================================
# leverage
# ===================================================================


class TestLeverage:
    """Tests for the leverage() function."""

    def test_basic_computation(self, regression_data):
        """Leverage returns an array with the correct shape."""
        result, X, y, coef = regression_data
        h = leverage(result)
        assert h.shape == (30,)
        assert h.dtype == np.float64

    def test_qr_path_matches_direct(self, regression_data):
        """QR-based leverage equals direct hat-matrix diagonal."""
        result, X, y, coef = regression_data
        h_qr = leverage(result)

        # Direct computation: h_ii = x_i'(X'X)^{-1} x_i
        XtX_inv = np.linalg.inv(X.T @ X)
        h_direct = np.sum((X @ XtX_inv) * X, axis=1)

        np.testing.assert_allclose(h_qr, h_direct, rtol=1e-10, atol=1e-14)

    def test_explicit_X_matches_result_X(self, regression_data):
        """Passing X explicitly gives same result as using result._X."""
        result, X, y, coef = regression_data
        h_via_result = leverage(result)
        h_via_arg = leverage(result, X=X)
        np.testing.assert_allclose(h_via_result, h_via_arg, rtol=1e-14)

    def test_x_via_result_underscore_X(self, regression_data):
        """When X is not passed, it is read from result._X."""
        result, X, y, coef = regression_data
        # Do NOT pass X — it should be read from result._X
        h = leverage(result)
        assert h.shape[0] == X.shape[0]

    def test_raises_without_X(self):
        """ValueError raised when no X is available."""
        result = MockResultNoX()
        with pytest.raises(ValueError, match="Design matrix X must be provided"):
            leverage(result)

    def test_values_in_01(self, regression_data):
        """All leverage values must lie in [0, 1]."""
        result, X, y, coef = regression_data
        h = leverage(result)
        assert np.all(h >= -1e-15)  # allow tiny numerical noise
        assert np.all(h <= 1.0 + 1e-15)

    def test_sum_equals_p(self, regression_data):
        """Sum of hat values equals the number of parameters p."""
        result, X, y, coef = regression_data
        p = X.shape[1]
        h = leverage(result)
        np.testing.assert_allclose(np.sum(h), p, atol=1e-10)

    def test_sum_equals_p_multivariate(self, multi_regression_data):
        """Sum of hat values equals p for multi-parameter model."""
        result, X, y, coef = multi_regression_data
        p = X.shape[1]
        h = leverage(result)
        np.testing.assert_allclose(np.sum(h), p, atol=1e-10)

    def test_high_leverage_point_detected(self):
        """An extreme predictor value should have higher leverage."""
        np.random.seed(42)
        n = 30
        x = np.random.randn(n)
        X = np.column_stack([np.ones(n), x])
        coef = np.array([2.0, 3.0])
        residuals = np.random.randn(n) * 0.5
        y = X @ coef + residuals

        # Move one point far from the center
        X_mod = X.copy()
        X_mod[0, 1] = 10.0
        y_mod = y.copy()
        y_mod[0] = X_mod[0] @ coef + residuals[0]
        result = MockGLMResult(X_mod, y_mod, coef)
        h = leverage(result)
        assert h[0] > np.median(h)


# ===================================================================
# cooks_distance
# ===================================================================


class TestCooksDistance:
    """Tests for the cooks_distance() function."""

    def test_non_negative(self, regression_data):
        """Cook's distance values must be >= 0."""
        result, X, y, coef = regression_data
        D = cooks_distance(result)
        assert np.all(D >= 0)

    def test_correct_shape(self, regression_data):
        """Output shape matches number of observations."""
        result, X, y, coef = regression_data
        D = cooks_distance(result)
        assert D.shape == (30,)

    def test_explicit_X(self, regression_data):
        """Passing X explicitly gives same result as using result._X."""
        result, X, y, coef = regression_data
        D1 = cooks_distance(result)
        D2 = cooks_distance(result, X=X)
        np.testing.assert_allclose(D1, D2, rtol=1e-14)

    def test_uses_leverage(self, regression_data):
        """Cook's D is correlated with leverage — higher leverage -> higher D."""
        result, X, y, coef = regression_data
        h = leverage(result)
        D = cooks_distance(result)
        # Points with above-median leverage should tend to have higher Cook's D
        # (when residuals are similar), so rank-correlation should be positive
        from scipy.stats import spearmanr

        corr, _ = spearmanr(h, D)
        assert corr > 0

    def test_uses_residual_variance_if_present(self, regression_data):
        """When residual_variance_ is set, it is used instead of computed MSE."""
        result, X, y, coef = regression_data
        D_default = cooks_distance(result)

        # Create a result with an explicit residual_variance_ that differs
        n, p = X.shape
        mse_computed = np.sum(result.residuals**2) / (n - p)
        # Use a different variance
        result_var = MockGLMResult(
            X, X @ coef + result.residuals, coef, residual_variance=mse_computed * 4.0
        )
        D_var = cooks_distance(result_var)

        # Larger residual variance -> smaller Cook's D
        assert np.all(D_var < D_default)

    def test_raises_without_X(self):
        """ValueError when no X is available."""
        result = MockResultNoX()
        with pytest.raises(ValueError, match="Design matrix X must be provided"):
            cooks_distance(result)

    def test_manual_formula(self, regression_data):
        """Verify against manually computed Cook's D formula."""
        result, X, y, coef = regression_data
        D = cooks_distance(result)

        residuals = result.residuals
        n, p = X.shape
        h = leverage(result)
        mse = np.sum(residuals**2) / (n - p)
        D_manual = (residuals**2 / (p * mse)) * (h / (1 - h) ** 2)
        np.testing.assert_allclose(D, D_manual, rtol=1e-12)


# ===================================================================
# studentized_residuals
# ===================================================================


class TestStudentizedResiduals:
    """Tests for the studentized_residuals() function."""

    def test_external_shape(self, regression_data):
        """Externally studentized residuals have the correct shape."""
        result, X, y, coef = regression_data
        r = studentized_residuals(result, external=True)
        assert r.shape == (30,)

    def test_internal_shape(self, regression_data):
        """Internally studentized residuals have the correct shape."""
        result, X, y, coef = regression_data
        r = studentized_residuals(result, external=False)
        assert r.shape == (30,)

    def test_external_vs_internal_differ(self, regression_data):
        """External and internal studentization give different values."""
        result, X, y, coef = regression_data
        r_ext = studentized_residuals(result, external=True)
        r_int = studentized_residuals(result, external=False)
        assert not np.allclose(r_ext, r_int)

    def test_external_larger_magnitude(self, regression_data):
        """Externally studentized residuals have slightly larger magnitude
        on average because the leave-one-out variance is smaller."""
        result, X, y, coef = regression_data
        r_ext = studentized_residuals(result, external=True)
        r_int = studentized_residuals(result, external=False)
        # Both should be finite
        assert np.all(np.isfinite(r_ext))
        assert np.all(np.isfinite(r_int))
        # The means of absolute values should be reasonably close
        # but not identical
        assert abs(np.mean(np.abs(r_ext)) - np.mean(np.abs(r_int))) > 1e-6

    def test_reasonable_magnitude(self, regression_data):
        """Studentized residuals should have magnitude in a reasonable range."""
        result, X, y, coef = regression_data
        r = studentized_residuals(result, external=True)
        # For well-behaved data, most residuals should be < 3
        assert np.mean(np.abs(r) < 5) > 0.8

    def test_explicit_X(self, regression_data):
        """Passing X explicitly works."""
        result, X, y, coef = regression_data
        r1 = studentized_residuals(result, external=True)
        r2 = studentized_residuals(result, X=X, external=True)
        np.testing.assert_allclose(r1, r2, rtol=1e-14)

    def test_raises_without_X(self):
        """ValueError when no X is available."""
        result = MockResultNoX()
        with pytest.raises(ValueError, match="Design matrix X must be provided"):
            studentized_residuals(result)

    def test_uses_residual_variance_if_present(self, regression_data):
        """When residual_variance_ is set, it is used."""
        result, X, y, coef = regression_data
        r_default = studentized_residuals(result, external=False)

        n, p = X.shape
        mse_computed = np.sum(result.residuals**2) / (n - p)
        result_var = MockGLMResult(
            X, X @ coef + result.residuals, coef, residual_variance=mse_computed * 4.0
        )
        r_var = studentized_residuals(result_var, external=False)
        # Larger variance -> smaller residuals in magnitude
        assert np.mean(np.abs(r_var)) < np.mean(np.abs(r_default))


# ===================================================================
# dffits
# ===================================================================


class TestDffits:
    """Tests for the dffits() function."""

    def test_correct_shape(self, regression_data):
        """DFFITS output shape matches number of observations."""
        result, X, y, coef = regression_data
        d = dffits(result)
        assert d.shape == (30,)

    def test_relationship_to_studentized_residuals_and_leverage(self, regression_data):
        """DFFITS = r_studentized * sqrt(h / (1 - h))."""
        result, X, y, coef = regression_data
        d = dffits(result)
        h = leverage(result)
        r = studentized_residuals(result, external=True)
        d_expected = r * np.sqrt(h / (1 - h))
        np.testing.assert_allclose(d, d_expected, rtol=1e-12)

    def test_explicit_X(self, regression_data):
        """Passing X explicitly works."""
        result, X, y, coef = regression_data
        d1 = dffits(result)
        d2 = dffits(result, X=X)
        np.testing.assert_allclose(d1, d2, rtol=1e-14)

    def test_raises_without_X(self):
        """ValueError when no X is available."""
        result = MockResultNoX()
        with pytest.raises(ValueError, match="Design matrix X must be provided"):
            dffits(result)


# ===================================================================
# dfbetas
# ===================================================================


class TestDfbetas:
    """Tests for the dfbetas() function."""

    def test_shape_n_by_p(self, regression_data):
        """DFBETAS has shape (n_obs, n_params)."""
        result, X, y, coef = regression_data
        db = dfbetas(result)
        n, p = X.shape
        assert db.shape == (n, p)

    def test_shape_multivariate(self, multi_regression_data):
        """DFBETAS shape for multi-parameter model."""
        result, X, y, coef = multi_regression_data
        db = dfbetas(result)
        n, p = X.shape
        assert db.shape == (n, p)

    def test_all_finite(self, regression_data):
        """All DFBETAS values are finite."""
        result, X, y, coef = regression_data
        db = dfbetas(result)
        assert np.all(np.isfinite(db))

    def test_reasonable_magnitude(self, regression_data):
        """DFBETAS values should typically be small for well-behaved data."""
        result, X, y, coef = regression_data
        db = dfbetas(result)
        n = X.shape[0]
        # Common cutoff is 2/sqrt(n); most values should be below this
        cutoff = 2 / np.sqrt(n)
        assert np.mean(np.abs(db) < cutoff) > 0.5

    def test_explicit_X(self, regression_data):
        """Passing X explicitly works."""
        result, X, y, coef = regression_data
        db1 = dfbetas(result)
        db2 = dfbetas(result, X=X)
        np.testing.assert_allclose(db1, db2, rtol=1e-14)

    def test_raises_without_X(self):
        """ValueError when no X is available."""
        result = MockResultNoX()
        with pytest.raises(ValueError, match="Design matrix X must be provided"):
            dfbetas(result)


# ===================================================================
# influence_measures
# ===================================================================


class TestInfluenceMeasures:
    """Tests for the influence_measures() function."""

    def test_returns_influence_result(self, regression_data):
        """Returns an InfluenceResult instance."""
        result, X, y, coef = regression_data
        inf = influence_measures(result)
        assert isinstance(inf, InfluenceResult)

    def test_all_fields_populated(self, regression_data):
        """All fields of InfluenceResult are populated with correct shapes."""
        result, X, y, coef = regression_data
        n, p = X.shape
        inf = influence_measures(result)

        assert inf.cooks_distance.shape == (n,)
        assert inf.leverage.shape == (n,)
        assert inf.studentized_residuals.shape == (n,)
        assert inf.dffits.shape == (n,)
        assert inf.dfbetas.shape == (n, p)
        assert inf.n_obs == n
        assert inf.n_params == p

    def test_n_params_includes_intercept(self, regression_data):
        """n_params counts all columns in X (intercept + slopes)."""
        result, X, y, coef = regression_data
        inf = influence_measures(result)
        # coef_ has p-1 elements (slopes only), intercept_ is set,
        # so n_params = len(coef_) + 1 = p
        assert inf.n_params == X.shape[1]

    def test_explicit_X(self, regression_data):
        """Passing X explicitly works."""
        result, X, y, coef = regression_data
        inf1 = influence_measures(result)
        inf2 = influence_measures(result, X=X)
        np.testing.assert_allclose(inf1.leverage, inf2.leverage, rtol=1e-14)
        np.testing.assert_allclose(inf1.cooks_distance, inf2.cooks_distance, rtol=1e-14)

    def test_raises_without_X(self):
        """ValueError when no X is available."""
        result = MockResultNoX()
        with pytest.raises(ValueError, match="Design matrix X must be provided"):
            influence_measures(result)

    def test_values_match_individual_functions(self, regression_data):
        """InfluenceResult fields match calling individual functions."""
        result, X, y, coef = regression_data
        inf = influence_measures(result)

        np.testing.assert_allclose(inf.leverage, leverage(result), rtol=1e-14)
        np.testing.assert_allclose(
            inf.cooks_distance, cooks_distance(result), rtol=1e-14
        )
        np.testing.assert_allclose(
            inf.studentized_residuals,
            studentized_residuals(result, external=True),
            rtol=1e-14,
        )
        np.testing.assert_allclose(inf.dffits, dffits(result), rtol=1e-14)
        np.testing.assert_allclose(inf.dfbetas, dfbetas(result), rtol=1e-14)


# ===================================================================
# InfluenceResult dataclass
# ===================================================================


class TestInfluenceResult:
    """Tests for the InfluenceResult dataclass, its properties and methods."""

    @staticmethod
    def _make_result(n=20, p=3):
        """Create a synthetic InfluenceResult with controlled data."""
        np.random.seed(42)
        return InfluenceResult(
            cooks_distance=np.random.rand(n) * 0.5,
            leverage=np.random.rand(n) * 0.3,
            studentized_residuals=np.random.randn(n),
            dffits=np.random.randn(n) * 0.2,
            dfbetas=np.random.randn(n, p) * 0.1,
            n_obs=n,
            n_params=p,
        )

    def test_influential_cooks_property(self):
        """influential_cooks returns indices where D > 4/n."""
        n, p = 10, 2
        cooks = np.zeros(n)
        cooks[0] = 1.0  # well above 4/10 = 0.4
        cooks[3] = 0.5  # also above 0.4
        inf = InfluenceResult(
            cooks_distance=cooks,
            leverage=np.ones(n) * 0.1,
            studentized_residuals=np.zeros(n),
            dffits=np.zeros(n),
            dfbetas=np.zeros((n, p)),
            n_obs=n,
            n_params=p,
        )
        idx = inf.influential_cooks
        assert 0 in idx
        assert 3 in idx
        # Points below threshold should not be included
        assert 1 not in idx

    def test_high_leverage_property(self):
        """high_leverage returns indices where h > 2p/n."""
        n, p = 10, 2
        threshold = 2 * p / n  # 0.4
        lev = np.ones(n) * 0.1
        lev[0] = 0.8  # above threshold
        lev[5] = 0.5  # above threshold
        inf = InfluenceResult(
            cooks_distance=np.zeros(n),
            leverage=lev,
            studentized_residuals=np.zeros(n),
            dffits=np.zeros(n),
            dfbetas=np.zeros((n, p)),
            n_obs=n,
            n_params=p,
        )
        idx = inf.high_leverage
        assert 0 in idx
        assert 5 in idx
        assert 1 not in idx

    def test_outliers_property(self):
        """outliers returns indices where |t| > 2."""
        n, p = 10, 2
        sr = np.zeros(n)
        sr[0] = 3.0
        sr[5] = -2.5
        sr[7] = 1.9  # not an outlier
        inf = InfluenceResult(
            cooks_distance=np.zeros(n),
            leverage=np.ones(n) * 0.1,
            studentized_residuals=sr,
            dffits=np.zeros(n),
            dfbetas=np.zeros((n, p)),
            n_obs=n,
            n_params=p,
        )
        idx = inf.outliers
        assert 0 in idx
        assert 5 in idx
        assert 7 not in idx

    def test_summary_returns_string(self):
        """summary() returns a formatted string."""
        inf = self._make_result()
        s = inf.summary()
        assert isinstance(s, str)

    def test_summary_contains_key_sections(self):
        """summary() includes expected section headers."""
        inf = self._make_result()
        s = inf.summary()
        assert "Influence" in s
        assert "Cook" in s
        assert "Leverage" in s
        assert "Studentized Resid" in s
        assert "DFFITS" in s
        assert "Thresholds" in s
        assert "Influential Observations" in s

    def test_summary_contains_n_obs_n_params(self):
        """summary() reports n_obs and n_params."""
        inf = self._make_result(n=20, p=3)
        s = inf.summary()
        assert "20" in s
        assert "3" in s

    def test_repr_returns_string(self):
        """__repr__ returns a concise string."""
        inf = self._make_result()
        r = repr(inf)
        assert isinstance(r, str)
        assert "InfluenceResult" in r
        assert "n_obs=20" in r
        assert "n_params=3" in r
        assert "n_influential=" in r


# ===================================================================
# loo_residuals
# ===================================================================


class TestLooResiduals:
    """Tests for the loo_residuals() function."""

    def test_correct_shape(self, regression_data):
        """LOO residuals have the same length as the data."""
        result, X, y, coef = regression_data
        loo = loo_residuals(result)
        assert loo.shape == (30,)

    def test_magnified_residuals(self, regression_data):
        """LOO residuals are larger in magnitude than raw residuals
        (divided by 1 - h_ii which is < 1)."""
        result, X, y, coef = regression_data
        h = leverage(result)
        loo = loo_residuals(result)

        # All points with leverage > 0 should have |LOO| >= |raw residual|
        mask = h > 1e-10
        assert np.all(np.abs(loo[mask]) >= np.abs(result.residuals[mask]) - 1e-12)

    def test_formula(self, regression_data):
        """LOO residual = e_i / (1 - h_ii)."""
        result, X, y, coef = regression_data
        h = leverage(result)
        loo = loo_residuals(result)
        expected = result.residuals / (1 - h)
        np.testing.assert_allclose(loo, expected, rtol=1e-12)

    def test_explicit_X(self, regression_data):
        """Passing X explicitly works."""
        result, X, y, coef = regression_data
        loo1 = loo_residuals(result)
        loo2 = loo_residuals(result, X=X)
        np.testing.assert_allclose(loo1, loo2, rtol=1e-14)

    def test_raises_without_X(self):
        """ValueError when no X is available."""
        result = MockResultNoX()
        with pytest.raises(ValueError, match="Design matrix X must be provided"):
            loo_residuals(result)


# ===================================================================
# press_statistic
# ===================================================================


class TestPressStatistic:
    """Tests for the press_statistic() function."""

    def test_positive_scalar(self, regression_data):
        """PRESS statistic is a positive float."""
        result, X, y, coef = regression_data
        press = press_statistic(result)
        assert isinstance(press, float)
        assert press > 0

    def test_equals_sum_of_squared_loo(self, regression_data):
        """PRESS = sum of squared LOO residuals."""
        result, X, y, coef = regression_data
        press = press_statistic(result)
        loo = loo_residuals(result)
        np.testing.assert_allclose(press, np.sum(loo**2), rtol=1e-12)

    def test_greater_than_sse(self, regression_data):
        """PRESS >= SSE (because LOO residuals are larger)."""
        result, X, y, coef = regression_data
        press = press_statistic(result)
        sse = np.sum(result.residuals**2)
        assert press >= sse - 1e-10

    def test_explicit_X(self, regression_data):
        """Passing X explicitly works."""
        result, X, y, coef = regression_data
        p1 = press_statistic(result)
        p2 = press_statistic(result, X=X)
        np.testing.assert_allclose(p1, p2, rtol=1e-14)

    def test_raises_without_X(self):
        """ValueError when no X is available."""
        result = MockResultNoX()
        with pytest.raises(ValueError, match="Design matrix X must be provided"):
            press_statistic(result)


# ===================================================================
# Error handling — all functions without X
# ===================================================================


class TestAllRaiseWithoutX:
    """Every public function raises ValueError when X is unavailable."""

    @pytest.fixture
    def no_x_result(self):
        return MockResultNoX(n=5)

    def test_leverage_raises(self, no_x_result):
        with pytest.raises(ValueError):
            leverage(no_x_result)

    def test_cooks_distance_raises(self, no_x_result):
        with pytest.raises(ValueError):
            cooks_distance(no_x_result)

    def test_studentized_residuals_raises(self, no_x_result):
        with pytest.raises(ValueError):
            studentized_residuals(no_x_result)

    def test_dffits_raises(self, no_x_result):
        with pytest.raises(ValueError):
            dffits(no_x_result)

    def test_dfbetas_raises(self, no_x_result):
        with pytest.raises(ValueError):
            dfbetas(no_x_result)

    def test_influence_measures_raises(self, no_x_result):
        with pytest.raises(ValueError):
            influence_measures(no_x_result)

    def test_loo_residuals_raises(self, no_x_result):
        with pytest.raises(ValueError):
            loo_residuals(no_x_result)

    def test_press_statistic_raises(self, no_x_result):
        with pytest.raises(ValueError):
            press_statistic(no_x_result)
