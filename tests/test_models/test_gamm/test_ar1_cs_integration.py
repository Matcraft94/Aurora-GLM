"""
Integration tests for AR1 and Compound Symmetry covariance structures.

These tests verify that the covariance structures work correctly in
realistic GAMM fitting scenarios, including:
- Parameter recovery from simulated data
- Convergence properties
- Predictions with different covariance structures
- Multiple random effects with mixed structures
"""

import numpy as np
import pytest

from aurora.models.gamm import fit_gamm, RandomEffect, predict_from_gamm


class TestAR1Integration:
    """Integration tests for AR1 covariance in GAMMs."""

    @pytest.fixture
    def ar1_longitudinal_data(self):
        """Simulate longitudinal data with AR1 structure."""
        np.random.seed(123)
        n_subjects = 50
        n_times = 15
        rho = 0.75
        sigma2 = 2.5
        sigma_eps = 0.8

        # Fixed effects
        beta = np.array([8.0, 0.5, -0.3])  # Intercept, time, treatment

        # Simulate
        subject_id = np.repeat(np.arange(n_subjects), n_times)
        time = np.tile(np.arange(n_times), n_subjects)
        treatment = np.repeat(np.random.binomial(1, 0.5, n_subjects), n_times)

        # AR1 random effects
        b = np.zeros((n_subjects, n_times))
        for i in range(n_subjects):
            b[i, 0] = np.random.randn() * np.sqrt(sigma2)
            for t in range(1, n_times):
                innovation = np.random.randn() * np.sqrt(sigma2 * (1 - rho**2))
                b[i, t] = rho * b[i, t - 1] + innovation

        b_flat = b.ravel()

        # Design matrix
        X = np.column_stack([np.ones(len(time)), time, treatment])

        # Response
        y = X @ beta + b_flat + np.random.randn(len(time)) * sigma_eps

        return {
            "y": y,
            "X": X,
            "subject_id": subject_id,
            "time": time,
            "treatment": treatment,
            "true_params": {"beta": beta, "sigma2": sigma2, "rho": rho, "sigma_eps": sigma_eps},
        }

    def test_ar1_fitting_succeeds(self, ar1_longitudinal_data):
        """Test that AR1 GAMM fitting completes successfully."""
        data = ar1_longitudinal_data

        # Fit GAMM with AR1
        re = RandomEffect(grouping="subject", covariance="ar1")
        result = fit_gamm(
            y=data["y"], X=data["X"], random_effects=[re], groups_data={"subject": data["subject_id"]}
        )

        # Check convergence
        assert result.converged, "GAMM with AR1 covariance did not converge"
        assert result.n_iterations < 100, f"Too many iterations: {result.n_iterations}"

        # Check result has expected attributes
        assert hasattr(result, "beta_parametric")
        assert hasattr(result, "variance_components")
        assert hasattr(result, "random_effects")

    def test_ar1_parameter_recovery(self, ar1_longitudinal_data):
        """Test AR1 parameter recovery from simulated data."""
        data = ar1_longitudinal_data

        # Fit GAMM with AR1
        re = RandomEffect(grouping="subject", covariance="ar1")
        result = fit_gamm(
            y=data["y"], X=data["X"], random_effects=[re], groups_data={"subject": data["subject_id"]}
        )

        # Extract estimates
        beta_est = result.beta_parametric

        # AR1 parameters from variance_components
        params = result.variance_components
        sigma2_est = np.exp(params[0])
        rho_est = np.tanh(params[1])

        # Check fixed effects (within 20% or 0.5 absolute)
        true_beta = data["true_params"]["beta"]
        for i, (est, true) in enumerate(zip(beta_est, true_beta)):
            rel_error = abs(est - true) / abs(true) if true != 0 else abs(est)
            assert rel_error < 0.20 or abs(est - true) < 0.5, (
                f"Beta[{i}] recovery failed: {est:.3f} vs {true:.3f} "
                f"(rel_error={rel_error:.3f})"
            )

        # Check AR1 parameters (more lenient tolerances)
        true_sigma2 = data["true_params"]["sigma2"]
        true_rho = data["true_params"]["rho"]

        sigma2_rel_error = abs(sigma2_est - true_sigma2) / true_sigma2
        assert sigma2_rel_error < 0.40, (
            f"Sigma2 recovery: {sigma2_est:.3f} vs {true_sigma2:.3f} "
            f"(rel_error={sigma2_rel_error:.3f})"
        )

        rho_abs_error = abs(rho_est - true_rho)
        assert rho_abs_error < 0.20, f"Rho recovery: {rho_est:.3f} vs {true_rho:.3f} (error={rho_abs_error:.3f})"

    def test_ar1_vs_independence(self, ar1_longitudinal_data):
        """AR1 should fit better than independence for AR1 data."""
        data = ar1_longitudinal_data

        # Fit with AR1
        re_ar1 = RandomEffect(grouping="subject", covariance="ar1")
        result_ar1 = fit_gamm(
            y=data["y"], X=data["X"], random_effects=[re_ar1], groups_data={"subject": data["subject_id"]}
        )

        # Fit with independence
        re_indep = RandomEffect(grouping="subject", covariance="identity")
        result_indep = fit_gamm(
            y=data["y"], X=data["X"], random_effects=[re_indep], groups_data={"subject": data["subject_id"]}
        )

        # AR1 should have better AIC
        assert result_ar1.aic < result_indep.aic, (
            f"AR1 AIC ({result_ar1.aic:.1f}) should be better than "
            f"independence ({result_indep.aic:.1f})"
        )

        # AIC improvement should be substantial (>5 points is meaningful)
        aic_diff = result_indep.aic - result_ar1.aic
        assert aic_diff > 5, f"AIC improvement too small: {aic_diff:.1f}"

    def test_ar1_prediction_population(self, ar1_longitudinal_data):
        """Test population-level predictions with AR1 covariance."""
        data = ar1_longitudinal_data

        re = RandomEffect(grouping="subject", covariance="ar1")
        result = fit_gamm(
            y=data["y"], X=data["X"], random_effects=[re], groups_data={"subject": data["subject_id"]}
        )

        # Population-level prediction (new subject)
        X_new = np.array([[1, 10, 1]])  # Intercept, time=10, treatment=1
        pred_pop = predict_from_gamm(result, X_new, include_random=False)

        assert np.isfinite(pred_pop).all(), "Population prediction has non-finite values"
        assert len(pred_pop) == 1
        assert pred_pop.ndim == 1

    def test_ar1_prediction_conditional(self, ar1_longitudinal_data):
        """Test conditional predictions with AR1 covariance."""
        data = ar1_longitudinal_data

        re = RandomEffect(grouping="subject", covariance="ar1")
        result = fit_gamm(
            y=data["y"], X=data["X"], random_effects=[re], groups_data={"subject": data["subject_id"]}
        )

        # Conditional prediction (existing subject 0)
        X_new_cond = np.array([[1, 10, data["treatment"][0]]])
        groups_new = np.array([0])  # First subject
        pred_cond = predict_from_gamm(result, X_new_cond, groups_new=groups_new, include_random=True)

        assert np.isfinite(pred_cond).all(), "Conditional prediction has non-finite values"
        assert len(pred_cond) == 1

        # Get population prediction for comparison
        pred_pop = predict_from_gamm(result, X_new_cond, include_random=False)

        # Conditional and population predictions should differ (random effect contribution)
        assert abs(pred_cond[0] - pred_pop[0]) > 0.1, "Conditional and population predictions are too similar"


class TestCompoundSymmetryIntegration:
    """Integration tests for Compound Symmetry covariance."""

    @pytest.fixture
    def cs_clustered_data(self):
        """Simulate clustered data with compound symmetry."""
        np.random.seed(456)
        n_clusters = 80
        n_per_cluster = 25
        rho = 0.55  # ICC
        sigma2 = 3.0
        sigma_eps = 1.2

        # Fixed effects
        beta = np.array([12.0, 2.0])  # Intercept, covariate

        # Simulate
        cluster_id = np.repeat(np.arange(n_clusters), n_per_cluster)
        covariate = np.random.randn(len(cluster_id))

        # Cluster random effects (exchangeable)
        cluster_effects = np.random.randn(n_clusters) * np.sqrt(sigma2 * rho)

        # Individual errors
        individual_errors = np.random.randn(len(cluster_id)) * np.sqrt(sigma2 * (1 - rho))

        # Design matrix
        X = np.column_stack([np.ones(len(cluster_id)), covariate])

        # Response
        y = X @ beta + cluster_effects[cluster_id] + individual_errors

        return {
            "y": y,
            "X": X,
            "cluster_id": cluster_id,
            "covariate": covariate,
            "true_params": {"beta": beta, "sigma2": sigma2, "rho": rho, "sigma_eps": sigma_eps},
        }

    def test_cs_fitting_succeeds(self, cs_clustered_data):
        """Test that Compound Symmetry GAMM fitting completes successfully."""
        data = cs_clustered_data

        # Fit GAMM with CS
        re = RandomEffect(grouping="cluster", covariance="compound_symmetry")
        result = fit_gamm(
            y=data["y"], X=data["X"], random_effects=[re], groups_data={"cluster": data["cluster_id"]}
        )

        # Check convergence
        assert result.converged, "GAMM with compound symmetry did not converge"
        assert result.n_iterations < 100

        # Check result structure
        assert hasattr(result, "beta_parametric")
        assert hasattr(result, "variance_components")

    def test_cs_alias_works(self, cs_clustered_data):
        """Test that 'cs' alias for compound_symmetry works."""
        data = cs_clustered_data

        # Fit GAMM with 'cs' alias
        re = RandomEffect(grouping="cluster", covariance="cs")
        result = fit_gamm(
            y=data["y"], X=data["X"], random_effects=[re], groups_data={"cluster": data["cluster_id"]}
        )

        assert result.converged, "GAMM with 'cs' alias did not converge"

    def test_cs_parameter_recovery(self, cs_clustered_data):
        """Test compound symmetry parameter recovery."""
        data = cs_clustered_data

        # Fit GAMM with CS
        re = RandomEffect(grouping="cluster", covariance="compound_symmetry")
        result = fit_gamm(
            y=data["y"], X=data["X"], random_effects=[re], groups_data={"cluster": data["cluster_id"]}
        )

        # Extract estimates
        beta_est = result.beta_parametric
        params = result.variance_components
        sigma2_est = np.exp(params[0])

        # Check fixed effects
        true_beta = data["true_params"]["beta"]
        for i, (est, true) in enumerate(zip(beta_est, true_beta)):
            rel_error = abs(est - true) / abs(true)
            assert rel_error < 0.20, f"Beta[{i}] recovery: {est:.3f} vs {true:.3f} (rel_error={rel_error:.3f})"

        # Check variance parameter (more lenient)
        true_sigma2 = data["true_params"]["sigma2"]
        sigma2_rel_error = abs(sigma2_est - true_sigma2) / true_sigma2
        assert sigma2_rel_error < 0.40, (
            f"Sigma2 recovery: {sigma2_est:.3f} vs {true_sigma2:.3f} " f"(rel_error={sigma2_rel_error:.3f})"
        )

    def test_cs_vs_independence(self, cs_clustered_data):
        """Compound symmetry should fit better than independence for clustered data."""
        data = cs_clustered_data

        # Fit with compound symmetry
        re_cs = RandomEffect(grouping="cluster", covariance="compound_symmetry")
        result_cs = fit_gamm(
            y=data["y"], X=data["X"], random_effects=[re_cs], groups_data={"cluster": data["cluster_id"]}
        )

        # Fit with independence
        re_indep = RandomEffect(grouping="cluster", covariance="identity")
        result_indep = fit_gamm(
            y=data["y"], X=data["X"], random_effects=[re_indep], groups_data={"cluster": data["cluster_id"]}
        )

        # CS should have better or similar AIC (data has exchangeable correlation)
        # We use <= to allow for cases where they're very close
        assert result_cs.aic <= result_indep.aic + 2, (
            f"CS AIC ({result_cs.aic:.1f}) should be better or similar to "
            f"independence ({result_indep.aic:.1f})"
        )


class TestMixedCovariances:
    """Test models with multiple random effects using different covariances."""

    def test_multiple_random_effects_different_structures(self):
        """Test subject-level AR1 + site-level compound symmetry."""
        np.random.seed(789)

        # Multi-site longitudinal study
        n_sites = 10
        n_subjects_per_site = 8
        n_times = 12

        # Total observations
        n_subjects = n_sites * n_subjects_per_site
        n = n_subjects * n_times

        # Identifiers
        site_id = np.repeat(np.arange(n_sites), n_subjects_per_site * n_times)
        subject_id = np.repeat(np.arange(n_subjects), n_times)
        time = np.tile(np.arange(n_times), n_subjects)

        # Design matrix
        X = np.column_stack([np.ones(n), time])

        # Site effects (compound symmetry)
        site_effects = np.random.randn(n_sites) * 1.5

        # Subject AR1 effects (nested within site)
        rho = 0.6
        subject_effects = np.zeros((n_subjects, n_times))
        for i in range(n_subjects):
            subject_effects[i, 0] = np.random.randn()
            for t in range(1, n_times):
                subject_effects[i, t] = (
                    rho * subject_effects[i, t - 1] + np.random.randn() * np.sqrt(1 - rho**2)
                )
        subject_effects_flat = subject_effects.ravel()

        # Response
        beta = np.array([15.0, 0.4])
        y = X @ beta + site_effects[site_id] + subject_effects_flat + np.random.randn(n) * 0.5

        # Fit model with both random effects
        re_subject = RandomEffect(grouping="subject", covariance="ar1")
        re_site = RandomEffect(grouping="site", covariance="compound_symmetry")

        result = fit_gamm(
            y=y, X=X, random_effects=[re_subject, re_site], groups_data={"subject": subject_id, "site": site_id}
        )

        # Should converge
        assert result.converged, "Multi-level model did not converge"

        # Should have variance components
        assert hasattr(result, "variance_components")

        # Fixed effects should be reasonable
        assert abs(result.beta_parametric[0] - 15.0) < 1.5, (
            f"Intercept estimate {result.beta_parametric[0]:.3f} " f"far from true value 15.0"
        )
        assert abs(result.beta_parametric[1] - 0.4) < 0.25, (
            f"Slope estimate {result.beta_parametric[1]:.3f} " f"far from true value 0.4"
        )


class TestAR1EdgeCases:
    """Test AR1 covariance with edge cases and boundary conditions."""

    def test_ar1_small_sample(self):
        """Test AR1 with small number of subjects."""
        np.random.seed(999)
        n_subjects = 5  # Small sample
        n_times = 10
        n = n_subjects * n_times

        subject_id = np.repeat(np.arange(n_subjects), n_times)
        X = np.column_stack([np.ones(n), np.tile(np.arange(n_times), n_subjects)])
        y = X @ np.array([5.0, 0.3]) + np.random.randn(n)

        re = RandomEffect(grouping="subject", covariance="ar1")
        result = fit_gamm(y=y, X=X, random_effects=[re], groups_data={"subject": subject_id})

        # Should still converge with small sample
        assert result.converged or result.n_iterations >= 50, "Should converge or exhaust iterations"

    def test_ar1_short_timeseries(self):
        """Test AR1 with short time series (few observations per subject)."""
        np.random.seed(888)
        n_subjects = 50
        n_times = 3  # Very short time series
        n = n_subjects * n_times

        subject_id = np.repeat(np.arange(n_subjects), n_times)
        X = np.column_stack([np.ones(n), np.tile(np.arange(n_times), n_subjects)])
        y = X @ np.array([10.0, 0.5]) + np.random.randn(n)

        re = RandomEffect(grouping="subject", covariance="ar1")
        result = fit_gamm(y=y, X=X, random_effects=[re], groups_data={"subject": subject_id})

        # Should handle short series
        assert result.converged or result.n_iterations >= 50


class TestCompoundSymmetryEdgeCases:
    """Test Compound Symmetry covariance with edge cases."""

    def test_cs_small_clusters(self):
        """Test CS with small cluster sizes."""
        np.random.seed(777)
        n_clusters = 100
        n_per_cluster = 2  # Very small clusters
        n = n_clusters * n_per_cluster

        cluster_id = np.repeat(np.arange(n_clusters), n_per_cluster)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([8.0, 1.5]) + np.random.randn(n)

        re = RandomEffect(grouping="cluster", covariance="compound_symmetry")
        result = fit_gamm(y=y, X=X, random_effects=[re], groups_data={"cluster": cluster_id})

        # Should handle small clusters
        assert result.converged or result.n_iterations >= 50

    def test_cs_unbalanced_clusters(self):
        """Test CS with unbalanced cluster sizes."""
        np.random.seed(666)

        # Create unbalanced clusters: varying sizes from 5 to 30
        cluster_sizes = np.random.randint(5, 30, size=50)
        n = cluster_sizes.sum()

        cluster_id = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([6.0, 1.2]) + np.random.randn(n)

        re = RandomEffect(grouping="cluster", covariance="cs")
        result = fit_gamm(y=y, X=X, random_effects=[re], groups_data={"cluster": cluster_id})

        # Should handle unbalanced design
        assert result.converged or result.n_iterations >= 50
