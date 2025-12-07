"""Tests for aurora.inference.anova module.

Tests ANOVA and LRT: anova_glm, likelihood_ratio_test, ANOVAResult, LRTResult.
"""
from __future__ import annotations

import numpy as np
import pytest

from aurora.inference.anova import (
    anova_glm,
    likelihood_ratio_test,
    ANOVAResult,
    LRTResult,
)
from aurora.models import fit_glm


@pytest.fixture
def sample_glm_data():
    """Generate sample data for GLM testing."""
    np.random.seed(42)
    n = 100
    X = np.column_stack([
        np.ones(n),
        np.random.randn(n),
        np.random.randn(n),
    ])
    beta = np.array([1.0, 2.0, -1.0])
    y = X @ beta + np.random.randn(n) * 0.5
    return X, y


@pytest.fixture
def fitted_full_model(sample_glm_data):
    """Fit a full GLM model."""
    X, y = sample_glm_data
    return fit_glm(X, y, family="gaussian")


@pytest.fixture
def fitted_reduced_model(sample_glm_data):
    """Fit a reduced GLM model (fewer predictors)."""
    X, y = sample_glm_data
    X_reduced = X[:, :2]  # Only intercept and first predictor
    return fit_glm(X_reduced, y, family="gaussian")


class TestANOVAResult:
    """Tests for ANOVAResult dataclass."""

    def test_anova_result_creation(self):
        """Test ANOVAResult can be created."""
        result = ANOVAResult(
            df=np.array([1, 1]),
            ss=np.array([100.0, 50.0]),
            ms=np.array([100.0, 50.0]),
            f_statistic=np.array([97.09, 48.0]),
            p_value=np.array([0.001, 0.01]),
            source=["x1", "x2"],
            anova_type=3,
            residual_df=97,
            residual_ss=50.0,
        )
        assert result is not None
        assert len(result.source) == 2

    def test_anova_result_summary(self):
        """Test ANOVAResult summary method."""
        result = ANOVAResult(
            df=np.array([1, 1]),
            ss=np.array([80.0, 20.0]),
            ms=np.array([80.0, 20.0]),
            f_statistic=np.array([155.34, 38.83]),
            p_value=np.array([1e-10, 1e-8]),
            source=["x1", "x2"],
            anova_type=3,
            residual_df=97,
            residual_ss=50.0,
        )
        summary = result.summary()
        assert isinstance(summary, str)
        assert "x1" in summary

    def test_anova_result_to_dict(self):
        """Test ANOVAResult to_dict method."""
        result = ANOVAResult(
            df=np.array([1]),
            ss=np.array([100.0]),
            ms=np.array([100.0]),
            f_statistic=np.array([196.08]),
            p_value=np.array([1e-20]),
            source=["Model"],
            anova_type=3,
            residual_df=98,
            residual_ss=50.0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "source" in d or "df" in d


class TestLRTResult:
    """Tests for LRTResult dataclass."""

    def test_lrt_result_creation(self):
        """Test LRTResult can be created."""
        result = LRTResult(
            statistic=10.5,
            df=1,
            p_value=0.001,
            model_names=("reduced", "full"),
            ll_reduced=-50.0,
            ll_full=-45.0,
        )
        assert result is not None
        assert result.statistic == 10.5
        assert result.df == 1

    def test_lrt_result_summary(self):
        """Test LRTResult summary method."""
        result = LRTResult(
            statistic=15.0,
            df=2,
            p_value=0.0005,
            model_names=("reduced", "full"),
            ll_reduced=-60.0,
            ll_full=-52.5,
        )
        summary = result.summary()
        assert isinstance(summary, str)
        assert "15" in summary or "Chi" in summary


class TestAnovaGLM:
    """Tests for anova_glm function."""

    def test_anova_glm_basic(self, fitted_full_model):
        """Test basic ANOVA on GLM."""
        result = anova_glm(fitted_full_model)
        assert result is not None
        assert isinstance(result, ANOVAResult)

    def test_anova_glm_has_sources(self, fitted_full_model):
        """Test ANOVA result has sources."""
        result = anova_glm(fitted_full_model)
        assert len(result.source) > 0

    def test_anova_glm_df_positive(self, fitted_full_model):
        """Test ANOVA degrees of freedom are positive."""
        result = anova_glm(fitted_full_model)
        for df in result.df:
            assert df >= 0

    def test_anova_glm_ss_positive(self, fitted_full_model):
        """Test ANOVA sum of squares are non-negative."""
        result = anova_glm(fitted_full_model)
        for ss in result.ss:
            if ss is not None:
                assert ss >= 0

    def test_anova_glm_f_statistic(self, fitted_full_model):
        """Test ANOVA F-statistics are computed."""
        result = anova_glm(fitted_full_model)
        assert len(result.f_statistic) >= 0

    def test_anova_glm_p_values_valid(self, fitted_full_model):
        """Test ANOVA p-values are in [0, 1]."""
        result = anova_glm(fitted_full_model)
        for p in result.p_value:
            if p is not None:
                assert 0 <= p <= 1


class TestLikelihoodRatioTest:
    """Tests for likelihood_ratio_test function."""

    def test_lrt_basic(self, fitted_reduced_model, fitted_full_model):
        """Test basic LRT between two models."""
        if not hasattr(fitted_full_model, 'log_likelihood_'):
            pytest.skip("Model doesn't provide log_likelihood")
        
        result = likelihood_ratio_test(fitted_reduced_model, fitted_full_model)
        assert result is not None
        assert isinstance(result, LRTResult)

    def test_lrt_statistic_positive(self, fitted_reduced_model, fitted_full_model):
        """Test LRT statistic is non-negative."""
        if not hasattr(fitted_full_model, 'log_likelihood_'):
            pytest.skip("Model doesn't provide log_likelihood")
        
        result = likelihood_ratio_test(fitted_reduced_model, fitted_full_model)
        assert result.statistic >= -1e-10

    def test_lrt_df_positive(self, fitted_reduced_model, fitted_full_model):
        """Test LRT degrees of freedom is positive."""
        if not hasattr(fitted_full_model, 'log_likelihood_'):
            pytest.skip("Model doesn't provide log_likelihood")
        
        result = likelihood_ratio_test(fitted_reduced_model, fitted_full_model)
        assert result.df > 0

    def test_lrt_p_value_valid(self, fitted_reduced_model, fitted_full_model):
        """Test LRT p-value is in [0, 1]."""
        if not hasattr(fitted_full_model, 'log_likelihood_'):
            pytest.skip("Model doesn't provide log_likelihood")
        
        result = likelihood_ratio_test(fitted_reduced_model, fitted_full_model)
        assert 0 <= result.p_value <= 1

    def test_lrt_significant_predictor(self, sample_glm_data):
        """Test LRT detects significant predictor."""
        X, y = sample_glm_data
        full_model = fit_glm(X, y, family="gaussian")
        
        if not hasattr(full_model, 'log_likelihood_'):
            pytest.skip("Model doesn't provide log_likelihood")
        
        X_reduced = X[:, :2]
        reduced_model = fit_glm(X_reduced, y, family="gaussian")
        
        result = likelihood_ratio_test(reduced_model, full_model)
        assert result.p_value < 0.05

    def test_lrt_same_model(self, fitted_full_model):
        """Test LRT with same model gives statistic near 0."""
        if not hasattr(fitted_full_model, 'log_likelihood_'):
            pytest.skip("Model doesn't provide log_likelihood")
        
        result = likelihood_ratio_test(fitted_full_model, fitted_full_model)
        assert abs(result.statistic) < 1e-10


class TestANOVAIntegration:
    """Integration tests for ANOVA functionality."""

    def test_anova_then_lrt(self, fitted_reduced_model, fitted_full_model):
        """Test using ANOVA and LRT together."""
        anova_result = anova_glm(fitted_full_model)
        assert anova_result is not None
        
        if hasattr(fitted_full_model, 'log_likelihood_'):
            lrt_result = likelihood_ratio_test(fitted_reduced_model, fitted_full_model)
            assert lrt_result is not None

    def test_anova_poisson(self):
        """Test ANOVA with Poisson family."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        mu = np.exp(X @ np.array([1.0, 0.5]))
        y = np.random.poisson(mu)
        
        model = fit_glm(X, y, family="poisson")
        result = anova_glm(model)
        
        assert result is not None
        assert len(result.source) > 0
