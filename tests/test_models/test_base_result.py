"""Tests for aurora.models.base.base_result module.

Tests the unified result hierarchy: BaseResult, LinearModelResult, MixedModelResultBase.
"""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models.base.base_result import (
    BaseResult,
    LinearModelResult,
    MixedModelResultBase,
)


class TestLinearModelResult:
    """Tests for LinearModelResult class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample LinearModelResult for testing."""
        np.random.seed(42)
        n, p = 100, 2  # 2 features + intercept
        X = np.column_stack([np.ones(n), np.random.randn(n, p)])
        beta_true = np.array([1.0, 2.0, -1.0])  # intercept + 2 coefs
        y = X @ beta_true + np.random.randn(n) * 0.5
        
        # Fit simple OLS
        XtX = X.T @ X
        Xty = X.T @ y
        coefficients = np.linalg.solve(XtX, Xty)
        fitted = X @ coefficients
        residuals = y - fitted
        residual_var = np.sum(residuals**2) / (n - 3)
        
        return LinearModelResult(
            coef=coefficients[1:],  # coefficients without intercept
            intercept=coefficients[0],
            fitted_values=fitted,
            residuals=residuals,
            residual_variance=residual_var,
            converged=True,
            n_iter=1,
            n_obs=n,
            X=X,
            y=y,
        )

    def test_init_basic(self, sample_result):
        """Test basic initialization."""
        assert sample_result.coef_ is not None
        assert len(sample_result.coef_) == 2
        assert sample_result.n_obs_ == 100

    def test_coefficients_property(self, sample_result):
        """Test coefficients property includes intercept."""
        coef = sample_result.coefficients
        assert isinstance(coef, np.ndarray)
        assert coef.shape == (3,)  # intercept + 2 coefs

    def test_fitted_values_property(self, sample_result):
        """Test fitted_values property."""
        fitted = sample_result.fitted_values
        assert isinstance(fitted, np.ndarray)
        assert fitted.shape == (100,)

    def test_residuals_property(self, sample_result):
        """Test residuals property."""
        resid = sample_result.residuals
        assert isinstance(resid, np.ndarray)
        assert resid.shape == (100,)
        # Residuals should sum to approximately zero for OLS with intercept
        assert abs(np.sum(resid)) < 1e-10

    def test_r_squared(self, sample_result):
        """Test R-squared calculation."""
        r2 = sample_result.r_squared
        assert 0 <= r2 <= 1
        # With true linear model and low noise, R² should be high
        assert r2 > 0.9

    def test_adj_r_squared(self, sample_result):
        """Test adjusted R-squared calculation."""
        adj_r2 = sample_result.adj_r_squared
        assert not np.isnan(adj_r2)
        assert adj_r2 <= sample_result.r_squared  # Adjusted is always <= R²

    def test_df_residual(self, sample_result):
        """Test degrees of freedom calculation."""
        df = sample_result.df_residual
        assert df == 100 - 3  # n - p

    def test_df_model(self, sample_result):
        """Test model degrees of freedom."""
        df_model = sample_result.df_model
        assert df_model == 2  # p - 1 (excluding intercept from count)

    def test_n_features(self, sample_result):
        """Test n_features property."""
        assert sample_result.n_features == 2

    def test_predict_with_design_matrix(self, sample_result):
        """Test prediction with new design matrix."""
        # coef has 2 elements, so X needs 2 columns (intercept handled separately)
        X_new = np.random.randn(10, 2)
        predictions = sample_result.predict(X_new)
        assert predictions.shape == (10,)

    def test_summary(self, sample_result):
        """Test summary method returns string."""
        summary = sample_result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_to_dict(self, sample_result):
        """Test to_dict method."""
        d = sample_result.to_dict()
        assert isinstance(d, dict)
        assert "coefficients" in d
        assert "converged" in d

    def test_converged_flag(self, sample_result):
        """Test converged flag."""
        assert sample_result.converged is True
        assert sample_result.converged_ is True

    def test_repr(self, sample_result):
        """Test __repr__ method."""
        repr_str = repr(sample_result)
        assert "LinearModelResult" in repr_str
        assert "n_obs=100" in repr_str


class TestMixedModelResultBase:
    """Tests for MixedModelResultBase class."""

    @pytest.fixture
    def sample_mixed_result(self):
        """Create a sample MixedModelResultBase for testing."""
        np.random.seed(42)
        n, p, q = 100, 3, 10
        
        # Fixed effects (including intercept)
        fixed_effects = np.array([1.0, 2.0, -1.0])
        
        # Random effects (10 groups)
        random_effects = np.random.randn(q) * 0.5
        
        # Variance components
        variance_components = {
            "residual": 1.0,
            "group": 0.25,
        }
        
        # Design matrices
        X = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
        group_idx = np.repeat(np.arange(q), n // q)
        Z = np.zeros((n, q))
        Z[np.arange(n), group_idx] = 1
        
        # Compute fitted and linear predictor
        linear_predictor = X @ fixed_effects + Z @ random_effects
        fitted = linear_predictor  # For Gaussian, identity link
        y = fitted + np.random.randn(n)
        
        return MixedModelResultBase(
            fixed_effects=fixed_effects,
            random_effects=random_effects,
            variance_components=variance_components,
            log_likelihood=-50.0,
            fitted_values=fitted,
            linear_predictor=linear_predictor,
            converged=True,
            n_iter=10,
            n_obs=n,
            X=X,
            Z=Z,
            y=y,
        )

    def test_init_mixed(self, sample_mixed_result):
        """Test initialization of mixed model result."""
        assert sample_mixed_result.random_effects_ is not None
        assert len(sample_mixed_result.random_effects_) == 10

    def test_random_effects_property(self, sample_mixed_result):
        """Test random effects property (b alias)."""
        re = sample_mixed_result.b
        assert isinstance(re, np.ndarray)
        assert re.shape == (10,)

    def test_fixed_effects_property(self, sample_mixed_result):
        """Test fixed effects property (beta alias)."""
        fe = sample_mixed_result.beta
        assert isinstance(fe, np.ndarray)
        assert fe.shape == (3,)

    def test_variance_components_property(self, sample_mixed_result):
        """Test variance_components property."""
        vc = sample_mixed_result.variance_components_
        assert isinstance(vc, dict)
        assert "residual" in vc
        assert "group" in vc

    def test_n_fixed_effects(self, sample_mixed_result):
        """Test n_fixed_effects property."""
        assert sample_mixed_result.n_fixed_effects == 3

    def test_n_random_effects(self, sample_mixed_result):
        """Test n_random_effects property."""
        assert sample_mixed_result.n_random_effects == 10

    def test_coefficients_combined(self, sample_mixed_result):
        """Test coefficients combines fixed and random."""
        coef = sample_mixed_result.coefficients
        assert len(coef) == 13  # 3 fixed + 10 random

    def test_linear_predictor(self, sample_mixed_result):
        """Test linear_predictor property."""
        eta = sample_mixed_result.linear_predictor
        assert len(eta) == 100

    def test_residuals(self, sample_mixed_result):
        """Test residuals computation."""
        resid = sample_mixed_result.residuals
        assert len(resid) == 100

    def test_predict_fixed_only(self, sample_mixed_result):
        """Test prediction with fixed effects only."""
        X_new = np.column_stack([np.ones(5), np.random.randn(5, 2)])
        pred = sample_mixed_result.predict(X_new, include_random=False)
        assert pred.shape == (5,)

    def test_predict_with_random(self, sample_mixed_result):
        """Test prediction with random effects."""
        X_new = np.column_stack([np.ones(5), np.random.randn(5, 2)])
        Z_new = np.zeros((5, 10))
        Z_new[:, 0] = 1  # All new obs in group 0
        
        pred = sample_mixed_result.predict(X_new, Z=Z_new, include_random=True)
        assert pred.shape == (5,)

    def test_summary(self, sample_mixed_result):
        """Test summary method."""
        summary = sample_mixed_result.summary()
        assert isinstance(summary, str)
        assert "Mixed Model" in summary
        assert "Fixed Effects" in summary

    def test_to_dict_includes_random(self, sample_mixed_result):
        """Test to_dict includes random effects information."""
        d = sample_mixed_result.to_dict()
        assert "random_effects" in d
        assert "variance_components" in d
        assert "fixed_effects" in d


class TestBaseResultAbstract:
    """Tests for BaseResult abstract class."""

    def test_cannot_instantiate_base(self):
        """Test that BaseResult cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseResult(converged=True, n_iter=1, n_obs=10)

    def test_inheritance(self):
        """Test inheritance relationships."""
        assert issubclass(LinearModelResult, BaseResult)
        assert issubclass(MixedModelResultBase, BaseResult)


class TestResultEdgeCases:
    """Test edge cases and error handling."""

    def test_minimal_model(self):
        """Test with minimal model (intercept only)."""
        result = LinearModelResult(
            coef=np.array([]),  # No coefficients, intercept only
            intercept=1.0,
            fitted_values=np.array([1.0, 1.0, 1.0]),
            residuals=np.array([0.0, 0.1, -0.1]),
            residual_variance=0.02,
            converged=True,
            n_iter=1,
            n_obs=3,
            X=np.ones((3, 1)),
            y=np.array([1.0, 1.1, 0.9]),
        )
        assert result.n_features == 0
        assert len(result.coefficients) == 1  # Just intercept
        assert result.df_residual == 2

    def test_no_intercept_model(self):
        """Test model without intercept."""
        result = LinearModelResult(
            coef=np.array([2.0]),
            intercept=None,  # No intercept
            fitted_values=np.array([2.0, 4.0, 6.0]),
            residuals=np.array([0.1, -0.1, 0.0]),
            residual_variance=0.01,
            converged=True,
            n_iter=1,
            n_obs=3,
            X=np.array([[1], [2], [3]]),
            y=np.array([2.1, 3.9, 6.0]),
        )
        assert result.intercept_ is None
        assert len(result.coefficients) == 1  # Just the single coef
        assert result.n_features == 1
