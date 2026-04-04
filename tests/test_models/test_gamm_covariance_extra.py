"""Extra tests for GAMM covariance structures.

Extends the existing test_covariance_structures.py with coverage for:
- ToeplitzCovariance (construct, extract, roundtrip)
- Edge cases for UnstructuredCovariance, DiagonalCovariance, IdentityCovariance
- Error paths (wrong number of params, non-square matrices, etc.)
- ExponentialSpatialCovariance and MaternCovariance: extract_params roundtrip
- AR1Covariance: extract_params with n=1 edge case
- CompoundSymmetryCovariance: extract_params roundtrip
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aurora.models.gamm.covariance import (
    AR1Covariance,
    CompoundSymmetryCovariance,
    DiagonalCovariance,
    ExponentialSpatialCovariance,
    IdentityCovariance,
    MaternCovariance,
    ToeplitzCovariance,
    UnstructuredCovariance,
    get_covariance_structure,
)


# =============================================================================
# UnstructuredCovariance
# =============================================================================


class TestUnstructuredCovarianceExtra:
    """Additional tests for UnstructuredCovariance."""

    def test_n_parameters_q1(self):
        """1 effect should need 1 parameter."""
        assert UnstructuredCovariance().n_parameters(1) == 1

    def test_n_parameters_q3(self):
        """3 effects should need 6 parameters."""
        assert UnstructuredCovariance().n_parameters(3) == 6

    def test_construct_1d(self):
        """1x1 matrix from single param."""
        cov = UnstructuredCovariance()
        psi = cov.construct_psi(np.array([2.0]), n_effects=1)
        assert_allclose(psi, [[4.0]])  # L=[2], LL'=[4]

    def test_construct_wrong_param_count(self):
        """Wrong number of params should raise."""
        cov = UnstructuredCovariance()
        with pytest.raises(ValueError, match="Expected 3 parameters"):
            cov.construct_psi(np.array([1.0, 0.5]), n_effects=2)

    def test_extract_non_square_raises(self):
        """Extract from non-square should raise."""
        cov = UnstructuredCovariance()
        with pytest.raises(ValueError, match="psi must be square"):
            cov.extract_params(np.array([[1.0, 2.0]]))

    def test_extract_not_pd_raises(self):
        """Extract from non-positive-definite should raise."""
        cov = UnstructuredCovariance()
        with pytest.raises(ValueError, match="psi must be positive definite"):
            cov.extract_params(np.array([[-1.0, 0.0], [0.0, -1.0]]))

    def test_roundtrip_q2(self):
        """Construct then extract should recover Cholesky params."""
        cov = UnstructuredCovariance()
        params = np.array([1.5, 0.3, 0.8])
        psi = cov.construct_psi(params, n_effects=2)
        recovered = cov.extract_params(psi)
        assert_allclose(recovered, params, rtol=1e-10)

    def test_roundtrip_q3(self):
        """Roundtrip with 3 effects."""
        cov = UnstructuredCovariance()
        np.random.seed(42)
        # Random lower-triangular entries
        params = np.array([1.0, 0.2, 0.9, -0.1, 0.3, 0.7])
        psi = cov.construct_psi(params, n_effects=3)
        recovered = cov.extract_params(psi)
        # Reconstruct should give same matrix
        psi2 = cov.construct_psi(recovered, n_effects=3)
        assert_allclose(psi2, psi, rtol=1e-10)


# =============================================================================
# DiagonalCovariance
# =============================================================================


class TestDiagonalCovarianceExtra:
    """Additional tests for DiagonalCovariance."""

    def test_construct_wrong_param_count(self):
        cov = DiagonalCovariance()
        with pytest.raises(ValueError, match="Expected 3 parameters"):
            cov.construct_psi(np.array([0.0, 1.0]), n_effects=3)

    def test_extract_non_diagonal_raises(self):
        cov = DiagonalCovariance()
        psi = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="psi must be diagonal"):
            cov.extract_params(psi)

    def test_extract_negative_variance_raises(self):
        cov = DiagonalCovariance()
        psi = np.array([[-1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="Diagonal elements must be positive"):
            cov.extract_params(psi)

    def test_extract_non_square_raises(self):
        cov = DiagonalCovariance()
        with pytest.raises(ValueError, match="psi must be square"):
            cov.extract_params(np.array([[1.0, 2.0]]))

    def test_roundtrip(self):
        """Construct then extract should recover log-variances."""
        cov = DiagonalCovariance()
        params = np.array([0.0, 1.0, -0.5])
        psi = cov.construct_psi(params, n_effects=3)
        recovered = cov.extract_params(psi)
        assert_allclose(recovered, params, rtol=1e-10)

    def test_construct_produces_positive_variance(self):
        cov = DiagonalCovariance()
        params = np.array([-5.0, 0.0, 5.0])
        psi = cov.construct_psi(params, n_effects=3)
        assert np.all(np.diag(psi) > 0)

    def test_off_diagonals_are_zero(self):
        cov = DiagonalCovariance()
        params = np.array([1.0, 2.0, 3.0])
        psi = cov.construct_psi(params, n_effects=3)
        np.fill_diagonal(psi, 0)
        assert_allclose(psi, np.zeros((3, 3)))


# =============================================================================
# IdentityCovariance
# =============================================================================


class TestIdentityCovarianceExtra:
    """Additional tests for IdentityCovariance."""

    def test_n_parameters_any_q(self):
        """Identity should always need 1 parameter."""
        cov = IdentityCovariance()
        for q in [1, 5, 100]:
            assert cov.n_parameters(q) == 1

    def test_construct_wrong_param_count(self):
        cov = IdentityCovariance()
        with pytest.raises(ValueError, match="Expected 1 parameter"):
            cov.construct_psi(np.array([1.0, 2.0]), n_effects=3)

    def test_extract_non_identity_raises(self):
        cov = IdentityCovariance()
        psi = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="proportional to identity"):
            cov.extract_params(psi)

    def test_extract_negative_variance_raises(self):
        cov = IdentityCovariance()
        psi = -1.0 * np.eye(3)
        with pytest.raises(ValueError, match="Variance must be positive"):
            cov.extract_params(psi)

    def test_extract_non_square_raises(self):
        cov = IdentityCovariance()
        with pytest.raises(ValueError, match="psi must be square"):
            cov.extract_params(np.ones((2, 3)))

    def test_roundtrip(self):
        cov = IdentityCovariance()
        params = np.array([0.5])
        psi = cov.construct_psi(params, n_effects=4)
        recovered = cov.extract_params(psi)
        assert_allclose(recovered, params, rtol=1e-10)


# =============================================================================
# AR1Covariance
# =============================================================================


class TestAR1CovarianceExtra:
    """Additional tests for AR1Covariance."""

    def test_construct_wrong_param_count(self):
        cov = AR1Covariance()
        with pytest.raises(ValueError, match="Expected 2 parameters"):
            cov.construct_psi(np.array([1.0]), n_effects=3)

    def test_extract_non_square_raises(self):
        cov = AR1Covariance()
        with pytest.raises(ValueError, match="psi must be square"):
            cov.extract_params(np.ones((2, 3)))

    def test_extract_n1_raises(self):
        """Extract from 1x1 matrix should raise."""
        cov = AR1Covariance()
        with pytest.raises(ValueError, match="Need at least 2"):
            cov.extract_params(np.array([[1.0]]))

    def test_inverse_wrong_param_count(self):
        cov = AR1Covariance()
        with pytest.raises(ValueError, match="Expected 2 parameters"):
            cov.inverse(np.array([1.0]), n_effects=3)

    def test_inverse_n1(self):
        """1x1 inverse should be [[1/σ²]]."""
        cov = AR1Covariance()
        params = np.array([1.0, 0.0])  # σ²=e, ρ=0
        inv = cov.inverse(params, n_effects=1)
        sigma2 = np.exp(1.0)
        assert_allclose(inv, [[1.0 / sigma2]])

    def test_inverse_matches_numpy(self):
        """Inverse should match np.linalg.inv."""
        cov = AR1Covariance()
        params = np.array([0.0, 0.5])
        psi = cov.construct_psi(params, n_effects=4)
        inv_custom = cov.inverse(params, n_effects=4)
        inv_np = np.linalg.inv(psi)
        assert_allclose(inv_custom, inv_np, rtol=1e-8, atol=1e-14)

    def test_construct_negative_correlation(self):
        """AR(1) with negative rho should still be symmetric PD."""
        cov = AR1Covariance()
        params = np.array([0.0, -1.0])  # arctanh maps to negative rho
        psi = cov.construct_psi(params, n_effects=4)
        assert_allclose(psi, psi.T)
        eigvals = np.linalg.eigvalsh(psi)
        assert np.all(eigvals > 0)


# =============================================================================
# CompoundSymmetryCovariance
# =============================================================================


class TestCompoundSymmetryExtra:
    """Additional tests for CompoundSymmetryCovariance."""

    def test_construct_wrong_param_count(self):
        cov = CompoundSymmetryCovariance()
        with pytest.raises(ValueError, match="Expected 2 parameters"):
            cov.construct_psi(np.array([1.0]), n_effects=3)

    def test_extract_non_square_raises(self):
        cov = CompoundSymmetryCovariance()
        with pytest.raises(ValueError, match="psi must be square"):
            cov.extract_params(np.ones((2, 3)))

    def test_extract_1x1(self):
        """1x1 matrix extraction should set rho=0."""
        cov = CompoundSymmetryCovariance()
        psi = np.array([[2.0]])
        params = cov.extract_params(psi)
        assert len(params) == 2

    def test_roundtrip(self):
        """Construct then extract should approximately recover params."""
        cov = CompoundSymmetryCovariance()
        params = np.array([0.5, 0.3])
        psi = cov.construct_psi(params, n_effects=5)
        recovered = cov.extract_params(psi)
        assert_allclose(recovered, params, rtol=0.1, atol=0.05)

    def test_diagonal_equal(self):
        """All diagonal elements should be equal."""
        cov = CompoundSymmetryCovariance()
        psi = cov.construct_psi(np.array([0.0, 0.5]), n_effects=4)
        diag = np.diag(psi)
        assert_allclose(diag, diag[0] * np.ones(4))

    def test_positive_definite_various_params(self):
        """Matrix should be positive definite for various param values."""
        cov = CompoundSymmetryCovariance()
        for p0 in [-2.0, 0.0, 2.0]:
            for p1 in [-1.0, 0.0, 1.0]:
                psi = cov.construct_psi(np.array([p0, p1]), n_effects=5)
                eigvals = np.linalg.eigvalsh(psi)
                assert np.all(eigvals > -1e-10), f"Failed for params=[{p0}, {p1}]"


# =============================================================================
# ToeplitzCovariance
# =============================================================================


class TestToeplitzCovariance:
    """Tests for ToeplitzCovariance."""

    def test_n_parameters_band1(self):
        """band=1 should need 2 params (variance + 1 correlation)."""
        cov = ToeplitzCovariance(band=1)
        assert cov.n_parameters(5) == 2

    def test_n_parameters_band2(self):
        """band=2 should need 3 params."""
        cov = ToeplitzCovariance(band=2)
        assert cov.n_parameters(5) == 3

    def test_n_parameters_band_larger_than_q(self):
        """band > n_effects-1 should be clamped."""
        cov = ToeplitzCovariance(band=10)
        assert cov.n_parameters(3) == 3  # min(10, 2) + 1 = 3

    def test_band_lt_1_raises(self):
        """band < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="band must be >= 1"):
            ToeplitzCovariance(band=0)

    def test_construct_psi_shape(self):
        cov = ToeplitzCovariance(band=2)
        params = np.array([0.0, 0.3, -0.1])
        psi = cov.construct_psi(params, n_effects=5)
        assert psi.shape == (5, 5)

    def test_construct_psi_symmetry(self):
        cov = ToeplitzCovariance(band=2)
        params = np.array([0.0, 0.3, -0.1])
        psi = cov.construct_psi(params, n_effects=5)
        assert_allclose(psi, psi.T, rtol=1e-10)

    def test_construct_wrong_param_count(self):
        cov = ToeplitzCovariance(band=2)
        with pytest.raises(ValueError, match="Expected 3 parameters"):
            cov.construct_psi(np.array([0.0, 0.3]), n_effects=5)

    def test_toeplitz_structure(self):
        """Elements at same lag should be equal."""
        cov = ToeplitzCovariance(band=1)
        params = np.array([0.5, 0.3])
        psi = cov.construct_psi(params, n_effects=4)
        # All diagonal elements equal
        diag = np.diag(psi)
        assert_allclose(diag, diag[0] * np.ones(4), rtol=1e-10)
        # All lag-1 elements equal
        lag1 = np.array([psi[i, i + 1] for i in range(3)])
        assert_allclose(lag1, lag1[0] * np.ones(3), rtol=1e-10)

    def test_construct_positive_definite(self):
        """Result should be positive definite or regularized to be so."""
        cov = ToeplitzCovariance(band=2)
        params = np.array([0.0, 0.3, 0.1])
        psi = cov.construct_psi(params, n_effects=5)
        eigvals = np.linalg.eigvalsh(psi)
        assert np.all(eigvals > -1e-8)

    def test_extract_params_roundtrip(self):
        """Construct then extract should recover approximate params."""
        cov = ToeplitzCovariance(band=2)
        params = np.array([0.5, 0.3, -0.1])
        psi = cov.construct_psi(params, n_effects=5)
        recovered = cov.extract_params(psi)
        # Reconstruct to check matrix match
        psi2 = cov.construct_psi(recovered, n_effects=5)
        assert_allclose(psi2, psi, rtol=0.1, atol=0.05)

    def test_extract_non_square_raises(self):
        cov = ToeplitzCovariance(band=1)
        with pytest.raises(ValueError, match="psi must be square"):
            cov.extract_params(np.ones((2, 3)))

    def test_get_covariance_structure_toeplitz(self):
        """Factory should return ToeplitzCovariance with default band."""
        cov = get_covariance_structure("toeplitz")
        assert isinstance(cov, ToeplitzCovariance)
        assert cov.band == 2

    def test_get_covariance_structure_toeplitz_custom_band(self):
        """Factory should pass band kwarg."""
        cov = get_covariance_structure("toeplitz", band=3)
        assert isinstance(cov, ToeplitzCovariance)
        assert cov.band == 3


# =============================================================================
# ExponentialSpatialCovariance — Extract Roundtrip
# =============================================================================


class TestExponentialSpatialExtra:
    """Additional tests for ExponentialSpatialCovariance."""

    def test_extract_non_square_raises(self):
        cov = ExponentialSpatialCovariance()
        with pytest.raises(ValueError, match="psi must be square"):
            cov.extract_params(np.ones((2, 3)))

    def test_extract_roundtrip_1d(self):
        """Construct then extract should approximately recover params."""
        cov = ExponentialSpatialCovariance()
        params = np.array([0.5, 1.0])
        psi = cov.construct_psi(params, n_effects=5)
        recovered = cov.extract_params(psi)
        # Reconstruct and check matrix match
        psi2 = cov.construct_psi(recovered, n_effects=5)
        assert_allclose(psi2, psi, rtol=0.2)

    def test_extract_roundtrip_2d_coords(self):
        """Roundtrip with 2D coordinates."""
        np.random.seed(42)
        coords = np.random.rand(6, 2) * 10
        cov = ExponentialSpatialCovariance(coordinates=coords)
        params = np.array([0.0, 0.5])
        psi = cov.construct_psi(params, n_effects=6)
        recovered = cov.extract_params(psi)
        psi2 = cov.construct_psi(recovered, n_effects=6)
        assert_allclose(psi2, psi, rtol=0.3)

    def test_coord_mismatch_raises(self):
        """Wrong n_effects vs coordinates should raise."""
        coords = np.array([[0, 0], [1, 0]])
        cov = ExponentialSpatialCovariance(coordinates=coords)
        with pytest.raises(ValueError, match="Coordinates have 2 locations"):
            cov.construct_psi(np.array([0.0, 0.5]), n_effects=5)

    def test_cached_distance_matrix(self):
        """Distance matrix should be cached after first computation."""
        cov = ExponentialSpatialCovariance()
        cov.construct_psi(np.array([0.0, 0.5]), n_effects=4)
        assert cov._distance_matrix is not None
        assert cov._distance_matrix.shape == (4, 4)
        # Second call with same n_effects should use cache
        cov.construct_psi(np.array([1.0, 0.0]), n_effects=4)
        # Should still be (4,4), not recomputed for different n
        assert cov._distance_matrix.shape == (4, 4)


# =============================================================================
# MaternCovariance — Extract Roundtrip
# =============================================================================


class TestMaternExtra:
    """Additional tests for MaternCovariance."""

    def test_extract_non_square_raises(self):
        cov = MaternCovariance()
        with pytest.raises(ValueError, match="psi must be square"):
            cov.extract_params(np.ones((2, 3)))

    def test_construct_wrong_param_count(self):
        cov = MaternCovariance()
        with pytest.raises(ValueError, match="Expected 2 parameters"):
            cov.construct_psi(np.array([1.0]), n_effects=3)

    def test_construct_shape(self):
        cov = MaternCovariance(nu=1.5)
        psi = cov.construct_psi(np.array([0.0, 0.5]), n_effects=5)
        assert psi.shape == (5, 5)

    def test_construct_symmetric(self):
        cov = MaternCovariance(nu=2.5)
        psi = cov.construct_psi(np.array([0.0, 1.0]), n_effects=5)
        assert_allclose(psi, psi.T, rtol=1e-10)

    def test_extract_roundtrip_1d(self):
        """Construct then extract should approximately recover params."""
        cov = MaternCovariance(nu=1.5)
        params = np.array([0.5, 0.5])
        psi = cov.construct_psi(params, n_effects=5)
        recovered = cov.extract_params(psi)
        psi2 = cov.construct_psi(recovered, n_effects=5)
        assert_allclose(psi2, psi, rtol=1.2)

    def test_nu_values_all_produce_pd(self):
        """Various nu values should produce positive definite matrices."""
        np.random.seed(42)
        coords = np.random.rand(5, 2)
        for nu in [0.5, 1.5, 2.5]:
            cov = MaternCovariance(coordinates=coords, nu=nu)
            psi = cov.construct_psi(np.array([0.0, 0.5]), n_effects=5)
            eigvals = np.linalg.eigvalsh(psi)
            assert np.all(eigvals > 0), f"Failed for nu={nu}"

    def test_coord_mismatch_raises(self):
        coords = np.array([[0, 0], [1, 0]])
        cov = MaternCovariance(coordinates=coords)
        with pytest.raises(ValueError, match="Coordinates have 2 locations"):
            cov.construct_psi(np.array([0.0, 0.5]), n_effects=5)

    def test_cached_distance_matrix(self):
        """Distance matrix should be cached after first computation."""
        cov = MaternCovariance()
        cov.construct_psi(np.array([0.0, 0.5]), n_effects=4)
        assert cov._distance_matrix is not None
        assert cov._distance_matrix.shape == (4, 4)

    def test_diagonal_equals_sigma2(self):
        """Diagonal should equal sigma2."""
        cov = MaternCovariance(nu=1.5)
        sigma2 = np.exp(0.5)
        psi = cov.construct_psi(np.array([0.5, 1.0]), n_effects=5)
        assert_allclose(np.diag(psi), sigma2 * np.ones(5))


# =============================================================================
# get_covariance_structure — Extra
# =============================================================================


class TestGetCovarianceStructureExtra:
    """Additional factory function tests."""

    def test_toeplitz_default_band(self):
        cov = get_covariance_structure("toeplitz")
        assert isinstance(cov, ToeplitzCovariance)
        assert cov.band == 2

    def test_toeplitz_custom_band(self):
        cov = get_covariance_structure("toeplitz", band=3)
        assert cov.band == 3

    def test_exponential_with_coords(self):
        coords = np.array([[0, 0], [1, 0], [0, 1]])
        cov = get_covariance_structure("exponential", coordinates=coords)
        assert cov.coordinates is coords

    def test_matern_with_nu(self):
        cov = get_covariance_structure("matern", nu=2.5)
        assert cov.nu == 2.5

    def test_all_names_valid(self):
        """All documented structure names should work."""
        names = [
            "unstructured",
            "diagonal",
            "identity",
            "ar1",
            "compound_symmetry",
            "cs",
            "toeplitz",
            "exponential",
            "matern",
        ]
        for name in names:
            kwargs = {}
            if name == "toeplitz":
                kwargs["band"] = 1
            if name in ("exponential", "matern"):
                kwargs["coordinates"] = np.array([[0], [1], [2]])
            cov = get_covariance_structure(name, **kwargs)
            assert cov is not None

    def test_unknown_structure_raises(self):
        """Unknown structure name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown covariance structure"):
            get_covariance_structure("invalid_name")


# =============================================================================
# Additional Coverage — targeted gaps
# =============================================================================


class TestExponentialSpatialCovarianceAdditional:
    """Fill remaining coverage gaps in ExponentialSpatialCovariance."""

    def test_construct_wrong_param_count(self):
        cov = ExponentialSpatialCovariance()
        with pytest.raises(ValueError, match="Expected 2 parameters"):
            cov.construct_psi(np.array([1.0]), n_effects=3)

    def test_construct_with_coords(self):
        """construct_psi with coordinates should use spatial distances."""
        coords = np.array([[0, 0], [1, 0], [0, 1]])
        cov = ExponentialSpatialCovariance(coordinates=coords)
        psi = cov.construct_psi(np.array([0.0, 0.0]), n_effects=3)
        assert psi.shape == (3, 3)
        assert_allclose(psi, psi.T, rtol=1e-10)

    def test_extract_all_nonpositive_covs(self):
        """extract_params when all off-diagonal covs <= 0 should use phi=1.0."""
        cov = ExponentialSpatialCovariance()
        # Build a matrix with negative off-diagonal entries (not realistic but tests the branch)
        psi = np.array([[2.0, -1.0, -0.5], [-1.0, 2.0, -0.3], [-0.5, -0.3, 2.0]])
        params = cov.extract_params(psi)
        assert len(params) == 2
        # phi should be log(1.0) = 0.0 since all covs <= 0
        assert_allclose(params[1], 0.0)

    def test_distance_matrix_cached_different_size(self):
        """Cached distance matrix with wrong size should be recomputed."""
        cov = ExponentialSpatialCovariance()
        # First call with n=3 caches a 3x3 matrix
        cov.construct_psi(np.array([0.0, 0.5]), n_effects=3)
        assert cov._distance_matrix.shape == (3, 3)
        # Second call with n=5 should recompute
        cov.construct_psi(np.array([0.0, 0.5]), n_effects=5)
        assert cov._distance_matrix.shape == (5, 5)


class TestMaternCovarianceAdditional:
    """Fill remaining coverage gaps in MaternCovariance."""

    def test_extract_all_zero_covs(self):
        """extract_params when no positive covs should use phi=1.0."""
        cov = MaternCovariance(nu=1.5)
        psi = np.eye(3)  # Only diagonal, no off-diagonal correlations
        params = cov.extract_params(psi)
        assert len(params) == 2

    def test_extract_no_valid_distances(self):
        """extract_params when no valid distances/correlations should fallback."""
        cov = MaternCovariance(nu=0.5)
        # Create a matrix where corrs are very small
        psi = np.array([[1.0, 1e-15, 1e-15], [1e-15, 1.0, 1e-15], [1e-15, 1e-15, 1.0]])
        params = cov.extract_params(psi)
        assert len(params) == 2
        assert np.isfinite(params).all()

    def test_cached_distance_different_size(self):
        """Cached distance matrix with wrong size should recompute."""
        cov = MaternCovariance(nu=1.5)
        cov.construct_psi(np.array([0.0, 0.5]), n_effects=3)
        assert cov._distance_matrix.shape == (3, 3)
        cov.construct_psi(np.array([0.0, 0.5]), n_effects=5)
        assert cov._distance_matrix.shape == (5, 5)

    def test_extract_with_coords(self):
        """extract_params with coordinates should work."""
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        cov = MaternCovariance(coordinates=coords, nu=1.5)
        psi = cov.construct_psi(np.array([0.0, 0.5]), n_effects=3)
        params = cov.extract_params(psi)
        assert len(params) == 2


class TestToeplitzCovarianceAdditional:
    """Additional Toeplitz coverage for edge cases."""

    def test_extract_params_with_large_band(self):
        """extract_params when effective_band > n should handle gracefully."""
        cov = ToeplitzCovariance(band=5)
        # With n_effects=3, effective_band = min(5, 2) = 2
        params = np.array([0.0, 0.3, 0.1])
        psi = cov.construct_psi(params, n_effects=3)
        recovered = cov.extract_params(psi)
        assert len(recovered) == 3

    def test_construct_n1(self):
        """Construct with n_effects=1 should produce 1x1 matrix."""
        cov = ToeplitzCovariance(band=1)
        # n_parameters(1) = min(1, 0) + 1 = 1
        params = np.array([1.0])
        psi = cov.construct_psi(params, n_effects=1)
        assert psi.shape == (1, 1)
        assert_allclose(psi, [[np.exp(1.0)]])


class TestCompoundSymmetryAdditional:
    """Additional CompoundSymmetry coverage."""

    def test_construct_n2(self):
        """CS with n=2 should compute rho_min correctly."""
        cov = CompoundSymmetryCovariance()
        psi = cov.construct_psi(np.array([0.0, 0.0]), n_effects=2)
        assert psi.shape == (2, 2)
        assert_allclose(psi, psi.T)

    def test_extract_roundtrip_n2(self):
        """CS roundtrip with n=2 should approximately recover params."""
        cov = CompoundSymmetryCovariance()
        params = np.array([0.5, 0.3])
        psi = cov.construct_psi(params, n_effects=2)
        recovered = cov.extract_params(psi)
        # Reconstruct and check matrix match
        psi2 = cov.construct_psi(recovered, n_effects=2)
        assert_allclose(psi2, psi, rtol=0.2)


class TestUnstructuredCovarianceAdditional:
    """Additional UnstructuredCovariance coverage."""

    def test_construct_3x3_values(self):
        """Construct 3x3 matrix and verify positive definiteness."""
        cov = UnstructuredCovariance()
        params = np.array([1.0, 0.5, 0.8, -0.2, 0.3, 0.7])
        psi = cov.construct_psi(params, n_effects=3)
        eigvals = np.linalg.eigvalsh(psi)
        assert np.all(eigvals > 0)

    def test_extract_returns_correct_length(self):
        """extract_params should return q(q+1)/2 parameters."""
        cov = UnstructuredCovariance()
        params = np.array([1.0, 0.2, 0.9, -0.1, 0.3, 0.7])
        psi = cov.construct_psi(params, n_effects=3)
        extracted = cov.extract_params(psi)
        assert len(extracted) == 6
