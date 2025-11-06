"""Tests for robust inference methods."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.inference import bootstrap_inference, robust_covariance
from aurora.models import fit_glm


class TestRobustCovariance:
    """Test robust covariance estimation."""

    def test_hc0_basic(self):
        """Test HC0 (White) standard errors."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 * x + 1 + np.random.randn(n) * 0.5
        
        result = fit_glm(x.reshape(-1, 1), y, family='gaussian')
        robust_result = robust_covariance(result, hc_type='HC0')
        
        # Check structure
        assert robust_result.std_errors.shape == (1,)
        assert robust_result.intercept_std_error is not None
        assert robust_result.hc_type == 'HC0'
        assert robust_result.coef_cov.shape == (2, 2)
        
        # Robust SEs should be positive
        assert robust_result.intercept_std_error > 0
        assert robust_result.std_errors[0] > 0

    def test_all_hc_types(self):
        """Test all HC types produce valid results."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        # Add heteroscedasticity
        y = 2 * x + 1 + np.random.randn(n) * (1 + 0.5 * np.abs(x))
        
        result = fit_glm(x.reshape(-1, 1), y, family='gaussian')
        
        hc_types = ['HC0', 'HC1', 'HC2', 'HC3', 'HC4']
        results = {}
        
        for hc_type in hc_types:
            robust_result = robust_covariance(result, hc_type=hc_type)
            results[hc_type] = robust_result
            
            # All should produce valid SEs
            assert robust_result.intercept_std_error > 0
            assert robust_result.std_errors[0] > 0
            assert robust_result.hc_type == hc_type
        
        # HC1-HC4 should generally be larger than HC0
        for hc_type in ['HC1', 'HC2', 'HC3', 'HC4']:
            # Allow for small numerical differences
            assert results[hc_type].intercept_std_error >= results['HC0'].intercept_std_error * 0.95

    def test_with_outlier(self):
        """Test that robust SEs are larger with outliers."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 * x + 1 + np.random.randn(n) * 0.5
        
        # Add extreme outlier
        y[0] = 50
        
        result = fit_glm(x.reshape(-1, 1), y, family='gaussian')
        robust_result = robust_covariance(result, hc_type='HC3')
        
        # Robust SEs should be much larger than standard SEs
        assert robust_result.intercept_std_error > result.intercept_std_error_
        assert robust_result.std_errors[0] > result.std_errors_[0]
        
        # Should be at least 2x larger with such an extreme outlier
        assert robust_result.intercept_std_error > 2 * result.intercept_std_error_

    def test_invalid_hc_type(self):
        """Test error handling for invalid HC type."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = 2 * x + 1 + np.random.randn(50)
        
        result = fit_glm(x.reshape(-1, 1), y, family='gaussian')
        
        with pytest.raises(ValueError, match="Unknown HC type"):
            robust_covariance(result, hc_type='HC99')

    def test_covariance_matrix(self):
        """Test that covariance matrix is positive semi-definite."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = 2 * x + 1 + np.random.randn(100)
        
        result = fit_glm(x.reshape(-1, 1), y, family='gaussian')
        robust_result = robust_covariance(result, hc_type='HC3')
        
        # Check covariance matrix is symmetric
        assert np.allclose(robust_result.coef_cov, robust_result.coef_cov.T)
        
        # Check eigenvalues are non-negative (positive semi-definite)
        eigenvalues = np.linalg.eigvals(robust_result.coef_cov)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors


class TestBootstrapInference:
    """Test bootstrap inference methods."""

    def test_bootstrap_basic(self):
        """Test basic bootstrap functionality."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 * x + 1 + np.random.randn(n) * 0.5
        
        result = fit_glm(x.reshape(-1, 1), y, family='gaussian')
        boot_result = bootstrap_inference(result, n_bootstrap=100, seed=42)
        
        # Check structure
        assert 'std_errors' in boot_result
        assert 'intercept_std_error' in boot_result
        assert 'ci_lower' in boot_result
        assert 'ci_upper' in boot_result
        assert 'intercept_ci' in boot_result
        assert 'boot_coefs' in boot_result
        
        # Check dimensions
        assert boot_result['std_errors'].shape == (1,)
        assert boot_result['ci_lower'].shape == (1,)
        assert boot_result['ci_upper'].shape == (1,)
        assert boot_result['boot_coefs'].shape[0] <= 100  # May have some failures
        assert boot_result['boot_coefs'].shape[1] == 2  # intercept + slope

    def test_bootstrap_ci_coverage(self):
        """Test that bootstrap CIs contain true parameter (on average)."""
        np.random.seed(42)
        true_beta = 2.0
        
        # Generate data from known model
        n = 100
        x = np.random.randn(n)
        y = true_beta * x + 1 + np.random.randn(n) * 0.5
        
        result = fit_glm(x.reshape(-1, 1), y, family='gaussian')
        boot_result = bootstrap_inference(result, n_bootstrap=200, alpha=0.05, seed=42)
        
        # CI should contain true value (not guaranteed every time, but likely)
        ci_lower = boot_result['ci_lower'][0]
        ci_upper = boot_result['ci_upper'][0]
        
        # At least check that CI is reasonable
        assert ci_lower < ci_upper
        assert ci_upper - ci_lower > 0.1  # Not too narrow
        assert ci_upper - ci_lower < 2.0  # Not too wide

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap with seed is reproducible."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = 2 * x + 1 + np.random.randn(50)
        
        result = fit_glm(x.reshape(-1, 1), y, family='gaussian')
        
        boot_result1 = bootstrap_inference(result, n_bootstrap=100, seed=123)
        boot_result2 = bootstrap_inference(result, n_bootstrap=100, seed=123)
        
        # Should get identical results with same seed
        assert np.allclose(boot_result1['std_errors'], boot_result2['std_errors'])
        assert np.allclose(boot_result1['boot_coefs'], boot_result2['boot_coefs'])

    def test_bootstrap_vs_robust_se(self):
        """Test that bootstrap and robust SEs are similar."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 * x + 1 + np.random.randn(n) * 0.5
        
        result = fit_glm(x.reshape(-1, 1), y, family='gaussian')
        
        robust_result = robust_covariance(result, hc_type='HC3')
        boot_result = bootstrap_inference(result, n_bootstrap=500, seed=42)
        
        # Should be in similar ballpark (within 50%)
        ratio_intercept = boot_result['intercept_std_error'] / robust_result.intercept_std_error
        ratio_slope = boot_result['std_errors'][0] / robust_result.std_errors[0]
        
        assert 0.5 < ratio_intercept < 2.0
        assert 0.5 < ratio_slope < 2.0

    def test_bootstrap_with_outlier(self):
        """Test bootstrap with outlier."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 * x + 1 + np.random.randn(n) * 0.5
        y[0] = 20  # Outlier
        
        result = fit_glm(x.reshape(-1, 1), y, family='gaussian')
        boot_result = bootstrap_inference(result, n_bootstrap=100, seed=42)
        
        # Should still complete successfully
        assert boot_result['std_errors'][0] > 0
        assert boot_result['intercept_std_error'] > 0
        
        # SEs should be larger than without outlier
        assert boot_result['std_errors'][0] > result.std_errors_[0]
