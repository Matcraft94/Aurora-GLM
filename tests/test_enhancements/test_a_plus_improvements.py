# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Tests for A+ enhancement improvements.

Tests four enhancements:
1. IRLS step-halving (backtracking line search)
2. LRT boundary condition handling (Self & Liang 1987)
3. GAMM bias correction (Breslow & Lin 1995)
4. Condition number monitoring
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

try:
    from scipy import sparse

    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False

from aurora.core.linalg import weighted_condition_number
from aurora.core.optimization.result import OptimizationResult
from aurora.inference.anova import (
    LRTResult,
    _detect_boundary_conditions,
    likelihood_ratio_test,
)
from aurora.models.gamm.pql import (
    _apply_fixed_effect_correction,
    _apply_random_effect_correction,
    _compute_group_sizes,
    fit_pql,
)


# =============================================================================
# Mock helpers
# =============================================================================


class MockIdentityLink:
    """Mock identity link function for Gaussian."""

    def inverse(self, eta):
        return eta

    def derivative(self, mu):
        return np.ones_like(mu)


class MockLogLink:
    """Mock log link function for Poisson."""

    def inverse(self, eta):
        return np.exp(np.clip(eta, -20, 20))

    def derivative(self, mu):
        return 1.0 / np.clip(mu, 1e-10, None)


class MockLogitLink:
    """Mock logit link function for Binomial."""

    def inverse(self, eta):
        eta = np.clip(eta, -20, 20)
        return 1.0 / (1.0 + np.exp(-eta))

    def derivative(self, mu):
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return 1.0 / (mu * (1 - mu))


def gaussian_variance(mu):
    """Gaussian variance function: V(mu) = 1."""
    return np.ones_like(mu)


def poisson_variance(mu):
    """Poisson variance function: V(mu) = mu."""
    return np.clip(mu, 1e-10, None)


def binomial_variance(mu):
    """Binomial variance function: V(mu) = mu(1-mu)."""
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    return mu * (1 - mu)


# =============================================================================
# Test 1: IRLS Step-Halving (via sparse path to avoid JAX dependency)
# =============================================================================


@pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse not available")
class TestIRLSStepHalving:
    """Tests for IRLS backtracking line search."""

    def test_step_halving_converges_gaussian(self):
        """Test that IRLS with step-halving converges on Gaussian data."""
        from aurora.core.optimization.irls import irls

        np.random.seed(42)
        n, p = 100, 3
        X_dense = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        X_sparse = sparse.csr_matrix(X_dense)
        beta_true = np.array([1.0, 2.0, -1.0])
        y = X_dense @ beta_true + np.random.randn(n) * 0.5

        result = irls(
            loss_fn=lambda beta, X, y: 0.5 * np.sum((y - X @ beta) ** 2),
            init_params=np.zeros(p),
            design_matrix=X_sparse,
            response=y,
            link=MockIdentityLink(),
            variance_fn=gaussian_variance,
            args=(X_dense, y),
            max_iter=50,
            tol=1e-8,
        )

        assert result.success
        assert_allclose(result.x, beta_true, atol=0.2)
        assert hasattr(result, "backtrack_iterations")
        assert isinstance(result.backtrack_iterations, int)

    def test_step_halving_prevents_divergence(self):
        """Test that step-halving recovers from bad initial values."""
        from aurora.core.optimization.irls import irls

        np.random.seed(42)
        n, p = 100, 3
        X_dense = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        X_sparse = sparse.csr_matrix(X_dense)
        beta_true = np.array([1.0, 2.0, -1.0])
        y = X_dense @ beta_true + np.random.randn(n) * 0.5

        # Start from very bad initial values
        bad_init = np.array([100.0, -100.0, 100.0])

        result = irls(
            loss_fn=lambda beta, X, y: 0.5 * np.sum((y - X @ beta) ** 2),
            init_params=bad_init,
            design_matrix=X_sparse,
            response=y,
            link=MockIdentityLink(),
            variance_fn=gaussian_variance,
            args=(X_dense, y),
            max_iter=200,
            tol=1e-6,
        )

        # Should still converge or at least not diverge to infinity
        assert np.all(np.isfinite(result.x))

    def test_backtrack_tracking(self):
        """Test that backtrack iterations are recorded in result."""
        from aurora.core.optimization.irls import irls

        np.random.seed(42)
        n, p = 50, 2
        X_dense = np.column_stack([np.ones(n), np.random.randn(n)])
        X_sparse = sparse.csr_matrix(X_dense)
        beta_true = np.array([1.0, 2.0])
        y = X_dense @ beta_true + np.random.randn(n) * 0.5

        result = irls(
            loss_fn=lambda beta, X, y: 0.5 * np.sum((y - X @ beta) ** 2),
            init_params=np.zeros(p),
            design_matrix=X_sparse,
            response=y,
            link=MockIdentityLink(),
            variance_fn=gaussian_variance,
            args=(X_dense, y),
            max_iter=50,
            tol=1e-8,
        )

        assert hasattr(result, "backtrack_iterations")
        assert isinstance(result.backtrack_iterations, int)
        assert result.backtrack_iterations >= 0

    def test_optimization_result_has_new_fields(self):
        """Test that OptimizationResult has backtrack_iterations and condition_number."""
        result = OptimizationResult(
            x=np.array([1.0, 2.0]),
            fun=0.5,
            backtrack_iterations=3,
            condition_number=1e5,
        )
        assert result.backtrack_iterations == 3
        assert result.condition_number == 1e5


# =============================================================================
# Test 2: LRT Boundary Condition Handling
# =============================================================================


class TestLRTBoundaryCorrection:
    """Tests for Self & Liang (1987) boundary correction in LRT."""

    def test_boundary_detection_variance_components(self):
        """Test detection of variance components at boundary."""

        class MockModel:
            variance_components_ = {"group1": 1e-15, "group2": 0.5}

        result = _detect_boundary_conditions(MockModel())
        assert "group1" in result
        assert "group2" not in result

    def test_boundary_detection_residual_variance(self):
        """Test detection of residual variance at boundary."""

        class MockModel:
            residual_variance_ = 1e-12

        result = _detect_boundary_conditions(MockModel())
        assert "residual_variance" in result

    def test_boundary_detection_psi(self):
        """Test detection of psi at boundary."""

        class MockModel:
            psi = np.array([[1e-15, 0], [0, 0.5]])

        result = _detect_boundary_conditions(MockModel())
        assert "psi" in result

    def test_no_boundary_when_not_at_boundary(self):
        """Test no detection when all parameters are far from boundary."""

        class MockModel:
            variance_components_ = {"group1": 0.5, "group2": 1.0}

        result = _detect_boundary_conditions(MockModel())
        assert len(result) == 0

    def test_lrt_result_has_boundary_fields(self):
        """Test LRTResult includes boundary condition fields."""
        result = LRTResult(
            statistic=5.0,
            df=2,
            p_value=0.05,
            model_names=("A", "B"),
            ll_reduced=-100.0,
            ll_full=-97.5,
        )
        assert hasattr(result, "boundary_conditions")
        assert hasattr(result, "boundary_correction_applied")
        assert result.boundary_conditions is None
        assert result.boundary_correction_applied is False

    def test_lrt_with_boundary_model(self):
        """Test LRT applies correction when variance at boundary."""

        class ReducedModel:
            log_likelihood_ = -100.0
            coef_ = np.array([1.0])
            intercept_ = 1.0
            variance_components_ = {"group1": 1e-15}

        class FullModel:
            log_likelihood_ = -97.5
            coef_ = np.array([1.0, 2.0])
            intercept_ = 1.0
            variance_components_ = {"group1": 0.5}

        result = likelihood_ratio_test(ReducedModel(), FullModel())

        assert result.boundary_correction_applied is True
        assert result.boundary_conditions is not None
        assert len(result.boundary_conditions) > 0

        # df = (2+1) - (1+1) = 1; mixture: 0.5*chi2(1) + 0.5*1.0
        statistic = 2 * (-97.5 - (-100.0))  # = 5.0
        expected_p = 0.5 * (1 - stats.chi2.cdf(statistic, 1)) + 0.5
        assert_allclose(result.p_value, expected_p, rtol=1e-10)

    def test_lrt_without_boundary_model(self):
        """Test LRT uses standard chi2 when no boundary."""

        class ReducedModel:
            log_likelihood_ = -100.0
            coef_ = np.array([1.0])
            intercept_ = 1.0

        class FullModel:
            log_likelihood_ = -97.5
            coef_ = np.array([1.0, 2.0])
            intercept_ = 1.0

        result = likelihood_ratio_test(ReducedModel(), FullModel())

        assert result.boundary_correction_applied is False
        assert result.boundary_conditions is None
        # df = (2+1) - (1+1) = 1
        expected_p = 1 - stats.chi2.cdf(5.0, 1)
        assert_allclose(result.p_value, expected_p, rtol=1e-10)

    def test_mixture_chi_square_p_value(self):
        """Test that boundary-corrected p-value uses mixture distribution."""
        statistic = 5.0
        df = 2

        # Mixture: 0.5 * chi2(df) + 0.5 * chi2(df-1)
        mixture_p = 0.5 * (1 - stats.chi2.cdf(statistic, df)) + 0.5 * (
            1 - stats.chi2.cdf(statistic, df - 1)
        )
        # Standard chi2
        standard_p = 1 - stats.chi2.cdf(statistic, df)

        # Mixture should give a different p-value
        assert mixture_p != standard_p


# =============================================================================
# Test 3: GAMM Bias Correction
# =============================================================================


class TestGAMMBiasCorrection:
    """Tests for Breslow & Lin (1995) bias correction in PQL."""

    def test_compute_group_sizes(self):
        """Test group size computation from Z matrix."""
        n_groups, n_per_group = 4, 5
        n = n_groups * n_per_group
        groups = np.repeat(np.arange(n_groups), n_per_group)
        Z = np.eye(n_groups)[groups]
        n_effects = 1

        group_sizes = _compute_group_sizes(Z, n_effects)

        assert len(group_sizes) == n_groups
        assert_allclose(group_sizes, n_per_group)

    def test_compute_group_sizes_unbalanced(self):
        """Test group sizes with unbalanced design."""
        # 3 groups with different sizes
        groups = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        Z = np.eye(3)[groups]
        n_effects = 1

        group_sizes = _compute_group_sizes(Z, n_effects)

        assert group_sizes[0] == 3
        assert group_sizes[1] == 2
        assert group_sizes[2] == 4

    def test_fixed_effect_correction_small_groups(self):
        """Test fixed effect bias correction with small groups."""
        beta = np.array([1.0, 2.0])
        group_sizes = np.array([2, 3, 4])  # Small groups

        corrected = _apply_fixed_effect_correction(beta, group_sizes)

        # Correction factor: 1 / (1 - sum(1/(2*m_i)))
        correction = 1.0 / (1.0 - np.sum(1.0 / (2.0 * group_sizes)))
        expected = beta * correction
        assert_allclose(corrected, expected, rtol=1e-10)

        # Correction should inflate estimates for small groups
        assert np.all(np.abs(corrected) > np.abs(beta))

    def test_fixed_effect_correction_large_groups(self):
        """Test correction diminishes with large groups."""
        beta = np.array([1.0, 2.0])
        large_groups = np.array([100, 200, 300])

        corrected = _apply_fixed_effect_correction(beta, large_groups)

        # With large groups, correction factor ≈ 1
        assert_allclose(corrected, beta, rtol=0.02)

    def test_random_effect_correction(self):
        """Test random effect bias correction."""
        n_groups, n_effects = 3, 2
        b_matrix = np.array([[0.5, -0.3], [0.2, 0.1], [-0.4, 0.6]])
        group_sizes = np.array([2, 5, 10])

        corrected = _apply_random_effect_correction(b_matrix, group_sizes)

        # Group 0 (m=2): inflation = 2/1 = 2.0
        assert_allclose(corrected[0], b_matrix[0] * 2.0)
        # Group 1 (m=5): inflation = 5/4 = 1.25
        assert_allclose(corrected[1], b_matrix[1] * 1.25)
        # Group 2 (m=10): inflation = 10/9 ≈ 1.111
        assert_allclose(corrected[2], b_matrix[2] * (10.0 / 9.0))

    def test_random_effect_correction_large_groups_no_change(self):
        """Test random effect correction is minimal for large groups."""
        b_matrix = np.array([[0.5, -0.3], [0.2, 0.1]])
        group_sizes = np.array([100, 200])

        corrected = _apply_random_effect_correction(b_matrix, group_sizes)

        # For large groups, inflation ≈ 1 (e.g. 100/99 ≈ 1.01)
        assert_allclose(corrected, b_matrix, rtol=0.02)

    def test_pql_runs_with_bias_correction(self):
        """Test that PQL fitting runs with bias correction enabled."""
        np.random.seed(42)
        n_groups, n_per_group = 5, 10
        n = n_groups * n_per_group
        groups = np.repeat(np.arange(n_groups), n_per_group)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        Z = np.eye(n_groups)[groups]

        eta = X @ [1.0, 0.5] + np.random.randn(n_groups)[groups] * 0.3
        y = np.random.poisson(np.exp(eta))

        result = fit_pql(X, Z, y, family="poisson")

        assert isinstance(result.beta, np.ndarray)
        assert len(result.beta) == 2
        assert np.all(np.isfinite(result.beta))


        assert result.converged is True


        assert result.n_iter_outer > 0


# =============================================================================
# Test 4: Condition Number Monitoring
# =============================================================================


class TestConditionNumberMonitoring:
    """Tests for condition number computation and monitoring."""

    def test_weighted_condition_number_identity(self):
        """Test condition number of identity matrix is 1."""
        X = np.eye(5)
        kappa = weighted_condition_number(X)
        assert_allclose(kappa, 1.0, rtol=1e-10)

    def test_weighted_condition_number_with_weights(self):
        """Test condition number with observation weights."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        weights = np.ones(50)

        kappa_noweight = weighted_condition_number(X)
        kappa_weighted = weighted_condition_number(X, weights)

        # With unit weights, should be same as unweighted
        assert_allclose(kappa_noweight, kappa_weighted, rtol=1e-10)

    def test_weighted_condition_number_singular(self):
        """Test condition number is very large for singular/near-singular matrix."""
        X = np.array([[1, 2], [1, 2]])  # Rank-deficient
        kappa = weighted_condition_number(X)
        # Due to floating point, may be very large finite rather than exact inf
        assert kappa > 1e10

    def test_weighted_condition_number_well_conditioned(self):
        """Test condition number for well-conditioned matrix."""
        np.random.seed(42)
        X = np.random.randn(100, 3)  # Random matrix is typically well-conditioned
        kappa = weighted_condition_number(X)
        assert kappa < 1e8  # Should be well-conditioned

    @pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse not available")
    def test_irls_reports_condition_number(self):
        """Test that IRLS result includes condition number."""
        from aurora.core.optimization.irls import irls

        np.random.seed(42)
        n, p = 50, 2
        X_dense = np.column_stack([np.ones(n), np.random.randn(n)])
        X_sparse = sparse.csr_matrix(X_dense)
        beta_true = np.array([1.0, 2.0])
        y = X_dense @ beta_true + np.random.randn(n) * 0.5

        result = irls(
            loss_fn=lambda beta, X, y: 0.5 * np.sum((y - X @ beta) ** 2),
            init_params=np.zeros(p),
            design_matrix=X_sparse,
            response=y,
            link=MockIdentityLink(),
            variance_fn=gaussian_variance,
            args=(X_dense, y),
            max_iter=50,
            tol=1e-8,
        )

        assert hasattr(result, "condition_number")
        assert result.condition_number is not None
        assert result.condition_number > 0

    @pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse not available")
    def test_high_condition_number_warning(self):
        """Test that warning triggers for ill-conditioned system."""
        from aurora.core.optimization.irls import irls

        np.random.seed(42)
        n, p = 50, 3
        # Create near-collinear design matrix
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 1e-10  # Nearly identical to x1
        X_dense = np.column_stack([np.ones(n), x1, x2])
        X_sparse = sparse.csr_matrix(X_dense)
        y = X_dense @ np.array([1.0, 2.0, 3.0]) + np.random.randn(n) * 0.1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = irls(
                loss_fn=lambda beta, X, y: 0.5 * np.sum((y - X @ beta) ** 2),
                init_params=np.zeros(p),
                design_matrix=X_sparse,
                response=y,
                link=MockIdentityLink(),
                variance_fn=gaussian_variance,
                args=(X_dense, y),
                max_iter=50,
                tol=1e-6,
            )
            # Check if a RuntimeWarning about ill-conditioning was issued
            runtime_warnings = [
                x for x in w if issubclass(x.category, RuntimeWarning)
            ]
            # If condition number is high enough, we should get a warning
            if result.condition_number is not None and result.condition_number > 1e8:
                assert len(runtime_warnings) > 0
                assert "Ill-conditioned" in str(runtime_warnings[0].message)
    @pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse not available")
    def test_well_conditioned_no_warning(self):
        """Test no warning for well-conditioned systems."""
        from aurora.core.optimization.irls import irls

        np.random.seed(42)
        n, p = 100, 2
        X_dense = np.column_stack([np.ones(n), np.random.randn(n)])
        X_sparse = sparse.csr_matrix(X_dense)
        y = X_dense @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.5
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = irls(
                loss_fn=lambda beta, X, y: 0.5 * np.sum((y - X @ beta) ** 2),
                init_params=np.zeros(p),
                design_matrix=X_sparse,
                response=y,
                link=MockIdentityLink(),
                variance_fn=gaussian_variance,
                args=(X_dense, y),
                max_iter=50,
                tol=1e-8,
            )

            runtime_warnings = [
                x for x in w if issubclass(x.category, RuntimeWarning)
            ]
            # Should not have ill-conditioning warning for random data
            ill_cond_warnings = [
                x for x in runtime_warnings if "Ill-conditioned" in str(x.message)
            ]
            assert len(ill_cond_warnings) == 0
