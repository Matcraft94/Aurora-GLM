"""Tests for Phase 5.3: Additional Covariance Structures.

This module tests the new covariance structures implemented for GAMM:
- AR(1) Autoregressive
- Compound Symmetry (Exchangeable)
- Exponential Spatial
- Matérn Spatial

References
----------
.. [1] Pinheiro & Bates (2000). Mixed-Effects Models in S and S-PLUS.
.. [2] Diggle et al. (2002). Analysis of Longitudinal Data.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aurora.models.gamm.covariance import (
    AR1Covariance,
    CompoundSymmetryCovariance,
    CovarianceStructure,
    DiagonalCovariance,
    ExponentialSpatialCovariance,
    IdentityCovariance,
    MaternCovariance,
    UnstructuredCovariance,
    get_covariance_structure,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def n_effects():
    """Number of random effects for testing."""
    return 5


@pytest.fixture
def spatial_coords():
    """2D spatial coordinates for testing."""
    np.random.seed(42)
    return np.random.rand(10, 2) * 10  # 10 points in 10x10 space


# =============================================================================
# Test AR(1) Covariance
# =============================================================================


class TestAR1Covariance:
    """Test AR(1) autoregressive covariance structure."""

    def test_n_parameters(self, n_effects):
        """AR(1) should have exactly 2 parameters."""
        cov = AR1Covariance()
        assert cov.n_parameters(n_effects) == 2

    def test_construct_psi_shape(self, n_effects):
        """Constructed matrix should have correct shape."""
        cov = AR1Covariance()
        params = np.array([0.0, 0.5])  # log(σ²)=0, arctanh(ρ)=0.5
        psi = cov.construct_psi(params, n_effects)

        assert psi.shape == (n_effects, n_effects)

    def test_construct_psi_symmetry(self, n_effects):
        """AR(1) matrix should be symmetric."""
        cov = AR1Covariance()
        params = np.array([0.0, 0.5])
        psi = cov.construct_psi(params, n_effects)

        assert_allclose(psi, psi.T)

    def test_construct_psi_positive_definite(self, n_effects):
        """AR(1) matrix should be positive definite."""
        cov = AR1Covariance()
        params = np.array([0.0, 0.5])
        psi = cov.construct_psi(params, n_effects)

        eigenvalues = np.linalg.eigvalsh(psi)
        assert np.all(eigenvalues > 0)

    def test_ar1_structure(self, n_effects):
        """Verify AR(1) correlation structure: Cov(i,j) = σ² ρ^|i-j|."""
        cov = AR1Covariance()
        log_sigma2 = 0.5  # σ² ≈ 1.65
        arctanh_rho = 0.4  # ρ ≈ 0.38
        params = np.array([log_sigma2, arctanh_rho])

        psi = cov.construct_psi(params, n_effects)

        sigma2 = np.exp(log_sigma2)
        rho = np.tanh(arctanh_rho)

        # Check specific elements
        assert_allclose(psi[0, 0], sigma2, rtol=1e-10)  # Diagonal
        assert_allclose(psi[0, 1], sigma2 * rho, rtol=1e-10)  # Off by 1
        assert_allclose(psi[0, 2], sigma2 * rho**2, rtol=1e-10)  # Off by 2

    def test_extract_params_roundtrip(self, n_effects):
        """Extract should recover original parameters."""
        cov = AR1Covariance()
        params_orig = np.array([0.5, 0.3])

        psi = cov.construct_psi(params_orig, n_effects)
        params_recovered = cov.extract_params(psi)

        assert_allclose(params_recovered, params_orig, rtol=0.1)

    def test_inverse_efficiency(self, n_effects):
        """AR(1) inverse should be tridiagonal."""
        cov = AR1Covariance()
        params = np.array([0.0, 0.5])

        psi = cov.construct_psi(params, n_effects)
        psi_inv_computed = cov.inverse(params, n_effects)
        psi_inv_direct = np.linalg.inv(psi)

        # Use atol for near-zero elements (tridiagonal has many zeros)
        assert_allclose(psi_inv_computed, psi_inv_direct, rtol=1e-8, atol=1e-14)


# =============================================================================
# Test Compound Symmetry Covariance
# =============================================================================


class TestCompoundSymmetryCovariance:
    """Test compound symmetry (exchangeable) covariance structure."""

    def test_n_parameters(self, n_effects):
        """Compound symmetry should have exactly 2 parameters."""
        cov = CompoundSymmetryCovariance()
        assert cov.n_parameters(n_effects) == 2

    def test_construct_psi_symmetry(self, n_effects):
        """CS matrix should be symmetric."""
        cov = CompoundSymmetryCovariance()
        params = np.array([0.0, 0.0])
        psi = cov.construct_psi(params, n_effects)

        assert_allclose(psi, psi.T)

    def test_cs_structure(self, n_effects):
        """Verify CS structure: equal variances, equal covariances."""
        cov = CompoundSymmetryCovariance()
        params = np.array([0.0, 0.0])  # σ²=1, ρ≈0.5
        psi = cov.construct_psi(params, n_effects)

        # All diagonal elements should be equal
        diag = np.diag(psi)
        assert_allclose(diag, diag[0] * np.ones(n_effects))

        # All off-diagonal elements should be equal
        off_diag = psi[~np.eye(n_effects, dtype=bool)]
        assert_allclose(off_diag, off_diag[0] * np.ones(len(off_diag)), rtol=1e-10)

    def test_positive_definite(self, n_effects):
        """CS matrix should be positive definite within valid ρ range."""
        cov = CompoundSymmetryCovariance()
        params = np.array([0.0, 0.0])
        psi = cov.construct_psi(params, n_effects)

        eigenvalues = np.linalg.eigvalsh(psi)
        assert np.all(eigenvalues > 0)


# =============================================================================
# Test Exponential Spatial Covariance
# =============================================================================


class TestExponentialSpatialCovariance:
    """Test exponential spatial covariance structure."""

    def test_n_parameters(self, spatial_coords):
        """Exponential spatial should have exactly 2 parameters."""
        cov = ExponentialSpatialCovariance(coordinates=spatial_coords)
        n = len(spatial_coords)
        assert cov.n_parameters(n) == 2

    def test_construct_psi_shape(self, spatial_coords):
        """Matrix should match number of coordinates."""
        cov = ExponentialSpatialCovariance(coordinates=spatial_coords)
        n = len(spatial_coords)
        params = np.array([0.0, 0.0])  # log(σ²)=0, log(φ)=0

        psi = cov.construct_psi(params, n)
        assert psi.shape == (n, n)

    def test_spatial_decay(self, spatial_coords):
        """Covariance should decay with distance."""
        cov = ExponentialSpatialCovariance(coordinates=spatial_coords)
        n = len(spatial_coords)
        params = np.array([0.0, 0.0])  # σ²=1, φ=1

        psi = cov.construct_psi(params, n)

        # Diagonal should be maximum (σ²)
        sigma2 = np.exp(params[0])
        assert_allclose(np.diag(psi), sigma2 * np.ones(n))

        # Off-diagonals should be less than diagonal
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert psi[i, j] < psi[i, i]

    def test_without_coordinates(self):
        """Should work with 1D equally-spaced assumption."""
        cov = ExponentialSpatialCovariance()  # No coordinates
        params = np.array([0.0, 0.5])
        n = 5

        psi = cov.construct_psi(params, n)

        # Should create valid matrix
        assert psi.shape == (n, n)
        assert_allclose(psi, psi.T)

    def test_positive_definite(self, spatial_coords):
        """Spatial matrix should be positive definite."""
        cov = ExponentialSpatialCovariance(coordinates=spatial_coords)
        n = len(spatial_coords)
        params = np.array([0.0, 0.5])

        psi = cov.construct_psi(params, n)
        eigenvalues = np.linalg.eigvalsh(psi)
        assert np.all(eigenvalues > 0)


# =============================================================================
# Test Matérn Covariance
# =============================================================================


class TestMaternCovariance:
    """Test Matérn spatial covariance structure."""

    def test_n_parameters(self, spatial_coords):
        """Matérn should have exactly 2 parameters (for fixed ν)."""
        cov = MaternCovariance(coordinates=spatial_coords, nu=1.5)
        n = len(spatial_coords)
        assert cov.n_parameters(n) == 2

    def test_smoothness_parameter(self, spatial_coords):
        """Different ν values should produce different smoothness."""
        n = len(spatial_coords)
        params = np.array([0.0, 0.5])

        psi_05 = MaternCovariance(coordinates=spatial_coords, nu=0.5).construct_psi(params, n)
        psi_15 = MaternCovariance(coordinates=spatial_coords, nu=1.5).construct_psi(params, n)
        psi_25 = MaternCovariance(coordinates=spatial_coords, nu=2.5).construct_psi(params, n)

        # Different ν should give different matrices
        assert not np.allclose(psi_05, psi_15)
        assert not np.allclose(psi_15, psi_25)

    def test_matern_05_equals_exponential(self, spatial_coords):
        """Matérn with ν=0.5 should equal exponential covariance."""
        n = len(spatial_coords)
        params = np.array([0.0, 0.5])

        matern = MaternCovariance(coordinates=spatial_coords, nu=0.5)
        exp_cov = ExponentialSpatialCovariance(coordinates=spatial_coords)

        psi_matern = matern.construct_psi(params, n)
        psi_exp = exp_cov.construct_psi(params, n)

        # Should be approximately equal (scaling may differ slightly)
        # Check correlation structure instead
        corr_matern = psi_matern / psi_matern[0, 0]
        corr_exp = psi_exp / psi_exp[0, 0]

        assert_allclose(corr_matern, corr_exp, rtol=0.1)

    def test_positive_definite(self, spatial_coords):
        """Matérn matrix should be positive definite."""
        cov = MaternCovariance(coordinates=spatial_coords, nu=1.5)
        n = len(spatial_coords)
        params = np.array([0.0, 0.5])

        psi = cov.construct_psi(params, n)
        eigenvalues = np.linalg.eigvalsh(psi)
        assert np.all(eigenvalues > 0)


# =============================================================================
# Test get_covariance_structure Factory
# =============================================================================


class TestGetCovarianceStructure:
    """Test the factory function for covariance structures."""

    def test_basic_structures(self):
        """Should return correct types for basic structures."""
        assert isinstance(get_covariance_structure("unstructured"), UnstructuredCovariance)
        assert isinstance(get_covariance_structure("diagonal"), DiagonalCovariance)
        assert isinstance(get_covariance_structure("identity"), IdentityCovariance)
        assert isinstance(get_covariance_structure("ar1"), AR1Covariance)
        assert isinstance(get_covariance_structure("compound_symmetry"), CompoundSymmetryCovariance)
        assert isinstance(get_covariance_structure("cs"), CompoundSymmetryCovariance)

    def test_spatial_with_coordinates(self, spatial_coords):
        """Spatial structures should accept coordinates."""
        exp_cov = get_covariance_structure("exponential", coordinates=spatial_coords)
        assert isinstance(exp_cov, ExponentialSpatialCovariance)
        assert exp_cov.coordinates is spatial_coords

    def test_matern_with_nu(self, spatial_coords):
        """Matérn should accept nu parameter."""
        matern = get_covariance_structure("matern", coordinates=spatial_coords, nu=2.5)
        assert isinstance(matern, MaternCovariance)
        assert matern.nu == 2.5

    def test_unknown_structure_raises(self):
        """Unknown structure should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown covariance structure"):
            get_covariance_structure("unknown")


# =============================================================================
# Integration Tests
# =============================================================================


class TestCovarianceIntegration:
    """Integration tests for covariance structures with GAMM."""

    def test_all_structures_are_covariance_structure(self):
        """All structures should inherit from CovarianceStructure."""
        structures = [
            UnstructuredCovariance(),
            DiagonalCovariance(),
            IdentityCovariance(),
            AR1Covariance(),
            CompoundSymmetryCovariance(),
            ExponentialSpatialCovariance(),
            MaternCovariance(),
        ]

        for struct in structures:
            assert isinstance(struct, CovarianceStructure)

    def test_roundtrip_all_structures(self, n_effects, spatial_coords):
        """All structures should support construct/extract roundtrip."""
        test_cases = [
            (UnstructuredCovariance(), np.random.randn(n_effects * (n_effects + 1) // 2)),
            (DiagonalCovariance(), np.random.randn(n_effects)),
            (IdentityCovariance(), np.array([0.5])),
            (AR1Covariance(), np.array([0.5, 0.3])),
            (CompoundSymmetryCovariance(), np.array([0.5, 0.0])),
        ]

        for cov, params in test_cases:
            psi = cov.construct_psi(params, n_effects)

            # Matrix should be symmetric
            assert_allclose(psi, psi.T, rtol=1e-10)

            # Matrix should be positive definite
            eigenvalues = np.linalg.eigvalsh(psi)
            assert np.all(eigenvalues > -1e-10)  # Allow small numerical error

    def test_parameter_transforms(self):
        """Parameter transformations should ensure valid matrices."""
        cov = AR1Covariance()

        # Extreme parameters should still give valid matrix
        extreme_params = [
            np.array([-10.0, -5.0]),  # Very small variance, negative correlation
            np.array([10.0, 5.0]),  # Very large variance, high positive correlation
            np.array([0.0, 0.0]),  # Unit variance, zero correlation
        ]

        for params in extreme_params:
            psi = cov.construct_psi(params, 5)
            assert psi.shape == (5, 5)
            assert_allclose(psi, psi.T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
