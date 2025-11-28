"""Tests for influence diagnostics and sensitivity analysis.

This tests both the low-level aurora.validation.sensitivity module and
the integrated diagnostics available through GLMResult.diagnostics_.
"""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models import fit_glm
from aurora.inference.diagnostics import glm_diagnostics


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n = 50
    X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    beta = np.array([1.0, 2.0, -1.0])
    y = X @ beta + np.random.randn(n) * 0.5
    return X, y


@pytest.fixture
def sample_data_with_outlier():
    """Generate sample data with an outlier."""
    np.random.seed(42)
    n = 50
    X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    beta = np.array([1.0, 2.0, -1.0])
    y = X @ beta + np.random.randn(n) * 0.5
    
    # Add an outlier
    y[0] = 100.0  # Extreme value
    
    return X, y


@pytest.fixture
def sample_data_with_leverage_point():
    """Generate sample data with a high leverage point."""
    np.random.seed(42)
    n = 50
    X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    beta = np.array([1.0, 2.0, -1.0])
    y = X @ beta + np.random.randn(n) * 0.5
    
    # Add a high leverage point
    X[0, 1] = 10.0  # Far from center
    X[0, 2] = 10.0
    
    return X, y


@pytest.fixture
def fitted_model(sample_data):
    """Fit a GLM model."""
    X, y = sample_data
    return fit_glm(X, y, family="gaussian")


class TestGLMDiagnostics:
    """Tests for GLM diagnostics via result.diagnostics_."""

    def test_diagnostics_basic(self, fitted_model):
        """Test basic diagnostics computation."""
        diag = fitted_model.diagnostics_
        assert diag is not None
        
    def test_diagnostics_has_leverage(self, fitted_model):
        """Test diagnostics has leverage."""
        diag = fitted_model.diagnostics_
        assert hasattr(diag, 'leverage')
        assert len(diag.leverage) > 0

    def test_diagnostics_has_cooks_distance(self, fitted_model):
        """Test diagnostics has Cook's distance."""
        diag = fitted_model.diagnostics_
        assert hasattr(diag, 'cooks_distance')
        assert len(diag.cooks_distance) > 0

    def test_diagnostics_has_studentized_residuals(self, fitted_model):
        """Test diagnostics has studentized residuals."""
        diag = fitted_model.diagnostics_
        assert hasattr(diag, 'studentized_residuals')
        assert len(diag.studentized_residuals) > 0

    def test_diagnostics_has_dfbetas(self, fitted_model):
        """Test diagnostics has DFBETAS."""
        diag = fitted_model.diagnostics_
        assert hasattr(diag, 'dfbetas')
        assert diag.dfbetas is not None


class TestLeverage:
    """Tests for leverage (hat values)."""

    def test_leverage_bounds(self, fitted_model):
        """Test leverage values are in valid range."""
        diag = fitted_model.diagnostics_
        lev = diag.leverage
        # Leverage should be in (0, 1) for regular OLS
        assert np.all(lev >= 0)
        assert np.all(lev <= 1)

    def test_leverage_sum(self, sample_data, fitted_model):
        """Test leverage values sum to approximately number of parameters."""
        X, y = sample_data
        diag = fitted_model.diagnostics_
        lev = diag.leverage
        # For Gaussian GLM with identity link, sum of leverage = p
        # Allow for intercept term
        p = X.shape[1]  # Number of columns in design matrix
        # Note: With intercept added, p becomes p+1, but X already has intercept column
        np.testing.assert_allclose(np.sum(lev), p, rtol=0.1)

    def test_leverage_detects_extreme(self, sample_data_with_leverage_point):
        """Test leverage detects high leverage points."""
        X, y = sample_data_with_leverage_point
        model = fit_glm(X, y, family="gaussian")
        diag = model.diagnostics_
        lev = diag.leverage
        
        # First point should have high leverage
        assert lev[0] > np.median(lev)


class TestCooksDistance:
    """Tests for Cook's distance."""

    def test_cooks_distance_non_negative(self, fitted_model):
        """Test Cook's distance is non-negative."""
        diag = fitted_model.diagnostics_
        cd = diag.cooks_distance
        assert np.all(cd >= 0)

    def test_cooks_distance_detects_outlier(self, sample_data_with_outlier):
        """Test Cook's distance detects outliers."""
        X, y = sample_data_with_outlier
        model = fit_glm(X, y, family="gaussian")
        diag = model.diagnostics_
        cd = diag.cooks_distance
        
        # First point (outlier) should have high Cook's D
        assert cd[0] > np.median(cd)

    def test_cooks_distance_cutoff(self, sample_data, fitted_model):
        """Test Cook's distance against common cutoffs."""
        X, y = sample_data
        diag = fitted_model.diagnostics_
        cd = diag.cooks_distance
        n = len(y)
        
        # Common cutoff is 4/n
        cutoff = 4 / n
        # Most points should be below cutoff in well-behaved data
        assert np.mean(cd < cutoff) > 0.5


class TestStudentizedResiduals:
    """Tests for studentized residuals."""

    def test_studentized_approximately_standard(self, fitted_model):
        """Test studentized residuals are approximately standard normal."""
        diag = fitted_model.diagnostics_
        sr = diag.studentized_residuals
        
        # Mean should be near 0
        assert abs(np.mean(sr)) < 0.5
        
        # SD should be near 1
        assert 0.5 < np.std(sr) < 2.0

    def test_studentized_detects_outlier(self, sample_data_with_outlier):
        """Test studentized residuals detect outliers."""
        X, y = sample_data_with_outlier
        model = fit_glm(X, y, family="gaussian")
        diag = model.diagnostics_
        sr = diag.studentized_residuals
        
        # First point should have large absolute studentized residual
        assert abs(sr[0]) > 2.0


class TestDFBETAS:
    """Tests for DFBETAS."""

    def test_dfbetas_shape(self, sample_data, fitted_model):
        """Test DFBETAS has correct shape."""
        X, y = sample_data
        diag = fitted_model.diagnostics_
        db = diag.dfbetas
        
        assert db.shape[0] == len(y)
        # Number of columns should match parameters
        assert db.shape[1] > 0

    def test_dfbetas_cutoff(self, sample_data, fitted_model):
        """Test DFBETAS against common cutoff."""
        X, y = sample_data
        diag = fitted_model.diagnostics_
        db = diag.dfbetas
        n = len(y)
        
        # Common cutoff is 2/sqrt(n)
        cutoff = 2 / np.sqrt(n)
        
        # Most entries should be below cutoff
        assert np.mean(np.abs(db) < cutoff) > 0.5


class TestDiagnosticsSummary:
    """Tests for diagnostics summary functionality."""

    def test_summary_exists(self, fitted_model):
        """Test summary array exists."""
        diag = fitted_model.diagnostics_
        assert hasattr(diag, 'summary')
        assert diag.summary is not None

    def test_summary_columns_exist(self, fitted_model):
        """Test summary columns are defined."""
        diag = fitted_model.diagnostics_
        assert hasattr(diag, 'summary_columns')
        assert len(diag.summary_columns) > 0

    def test_summary_shape(self, sample_data, fitted_model):
        """Test summary has correct shape."""
        X, y = sample_data
        diag = fitted_model.diagnostics_
        
        assert diag.summary.shape[0] == len(y)
        assert diag.summary.shape[1] == len(diag.summary_columns)


class TestResidualTypes:
    """Tests for different residual types."""

    def test_response_residuals(self, fitted_model):
        """Test response residuals exist."""
        diag = fitted_model.diagnostics_
        assert hasattr(diag, 'response_residuals')
        assert len(diag.response_residuals) > 0

    def test_pearson_residuals(self, fitted_model):
        """Test Pearson residuals exist."""
        diag = fitted_model.diagnostics_
        assert hasattr(diag, 'pearson_residuals')
        assert len(diag.pearson_residuals) > 0

    def test_deviance_residuals(self, fitted_model):
        """Test deviance residuals exist."""
        diag = fitted_model.diagnostics_
        assert hasattr(diag, 'deviance_residuals')
        assert len(diag.deviance_residuals) > 0

    def test_working_residuals(self, fitted_model):
        """Test working residuals exist."""
        diag = fitted_model.diagnostics_
        assert hasattr(diag, 'working_residuals')
        assert len(diag.working_residuals) > 0


class TestInfluenceIntegration:
    """Integration tests for influence diagnostics."""

    def test_all_measures_computed(self, fitted_model):
        """Test all influence measures are computed."""
        diag = fitted_model.diagnostics_
        
        # All should be non-None
        assert diag.leverage is not None
        assert diag.cooks_distance is not None
        assert diag.studentized_residuals is not None
        assert diag.dfbetas is not None

    def test_influence_poisson(self):
        """Test influence measures with Poisson family."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        mu = np.exp(X @ np.array([1.0, 0.3]))
        y = np.random.poisson(mu)
        
        model = fit_glm(X, y, family="poisson")
        diag = model.diagnostics_
        
        assert len(diag.leverage) == n
        assert len(diag.cooks_distance) == n

    def test_influence_binomial(self):
        """Test influence measures with binomial family."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        p = 1 / (1 + np.exp(-X @ np.array([0.0, 1.0])))
        y = (np.random.rand(n) < p).astype(float)
        
        model = fit_glm(X, y, family="binomial")
        diag = model.diagnostics_
        
        assert len(diag.leverage) == n
        assert len(diag.cooks_distance) == n

    def test_high_leverage_high_influence(self, sample_data_with_leverage_point):
        """Test that high leverage points tend to have influence potential."""
        X, y = sample_data_with_leverage_point
        model = fit_glm(X, y, family="gaussian")
        diag = model.diagnostics_
        
        # High leverage point should have above-average leverage
        assert diag.leverage[0] > np.mean(diag.leverage)


class TestGlmDiagnosticsFunction:
    """Tests for glm_diagnostics function directly."""

    def test_glm_diagnostics_direct_call(self, fitted_model):
        """Test calling glm_diagnostics directly."""
        diag = glm_diagnostics(fitted_model)
        
        assert diag.leverage is not None
        assert diag.cooks_distance is not None
        assert diag.studentized_residuals is not None

    def test_glm_diagnostics_matches_property(self, fitted_model):
        """Test glm_diagnostics matches diagnostics_ property."""
        diag1 = fitted_model.diagnostics_
        diag2 = glm_diagnostics(fitted_model)
        
        # Both should return GLMDiagnosticResult with same values
        np.testing.assert_array_equal(diag1.leverage, diag2.leverage)
        np.testing.assert_array_equal(diag1.cooks_distance, diag2.cooks_distance)


class TestEdgeCases:
    """Test edge cases for diagnostics."""

    def test_small_sample(self):
        """Test diagnostics with small sample."""
        np.random.seed(42)
        n = 10
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.5
        
        model = fit_glm(X, y, family="gaussian")
        diag = model.diagnostics_
        
        assert len(diag.leverage) == n
        assert len(diag.cooks_distance) == n

    def test_many_predictors(self):
        """Test diagnostics with many predictors."""
        np.random.seed(42)
        n = 100
        p = 10
        X = np.column_stack([np.ones(n), np.random.randn(n, p-1)])
        beta = np.random.randn(p)
        y = X @ beta + np.random.randn(n) * 0.5
        
        model = fit_glm(X, y, family="gaussian")
        diag = model.diagnostics_
        
        assert len(diag.leverage) == n
        # DFBETAS includes intercept added by fit_glm, so shape is (n, p+1)
        assert diag.dfbetas.shape[0] == n
        assert diag.dfbetas.shape[1] >= p
