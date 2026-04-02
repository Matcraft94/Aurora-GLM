"""Tests for covariance structures."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.models.gamm import (
    DiagonalCovariance,
    IdentityCovariance,
    UnstructuredCovariance,
    get_covariance_structure,
)

# Unstructured Covariance Tests


def test_unstructured_n_parameters():
    """UnstructuredCovariance should compute correct parameter count."""
    cov = UnstructuredCovariance()

    assert cov.n_parameters(1) == 1
    assert cov.n_parameters(2) == 3
    assert cov.n_parameters(3) == 6
    assert cov.n_parameters(4) == 10


def test_unstructured_construct_psi_1d():
    """UnstructuredCovariance should construct 1D covariance."""
    cov = UnstructuredCovariance()
    params = np.array([2.0])  # Single variance

    psi = cov.construct_psi(params, n_effects=1)

    assert psi.shape == (1, 1)
    assert psi[0, 0] == 4.0  # 2^2


def test_unstructured_construct_psi_2d():
    """UnstructuredCovariance should construct 2D covariance."""
    cov = UnstructuredCovariance()
    # Cholesky parameters: [L11, L21, L22]
    params = np.array([1.0, 0.5, 0.8])

    psi = cov.construct_psi(params, n_effects=2)

    # Expected: L = [[1.0, 0], [0.5, 0.8]]
    # Ψ = LL' = [[1.0, 0.5], [0.5, 1.89]]
    expected = np.array([[1.0, 0.5], [0.5, 0.89]])

    assert psi.shape == (2, 2)
    np.testing.assert_allclose(psi, expected, atol=1e-10)


def test_unstructured_extract_params_2d():
    """UnstructuredCovariance should extract parameters correctly."""
    cov = UnstructuredCovariance()

    # Create a valid covariance matrix
    psi = np.array([[2.0, 0.5], [0.5, 1.0]])

    params = cov.extract_params(psi)

    # Reconstruct and check
    psi_reconstructed = cov.construct_psi(params, n_effects=2)
    np.testing.assert_allclose(psi_reconstructed, psi, atol=1e-10)


def test_unstructured_roundtrip():
    """UnstructuredCovariance should roundtrip params->psi->params."""
    cov = UnstructuredCovariance()
    original_params = np.array([1.5, 0.3, 1.2])

    psi = cov.construct_psi(original_params, n_effects=2)
    extracted_params = cov.extract_params(psi)

    np.testing.assert_allclose(extracted_params, original_params, atol=1e-10)


def test_unstructured_wrong_param_count_raises():
    """UnstructuredCovariance should raise with wrong parameter count."""
    cov = UnstructuredCovariance()
    params = np.array([1.0, 0.5])  # Need 3 for n_effects=2

    with pytest.raises(ValueError, match="Expected 3 parameters"):
        cov.construct_psi(params, n_effects=2)


def test_unstructured_non_positive_definite_raises():
    """UnstructuredCovariance should raise for non-PD matrix."""
    cov = UnstructuredCovariance()

    # Non-positive definite matrix
    psi = np.array([[1.0, 2.0], [2.0, 1.0]])

    with pytest.raises(ValueError, match="positive definite"):
        cov.extract_params(psi)


# Diagonal Covariance Tests


def test_diagonal_n_parameters():
    """DiagonalCovariance should have q parameters."""
    cov = DiagonalCovariance()

    assert cov.n_parameters(1) == 1
    assert cov.n_parameters(3) == 3
    assert cov.n_parameters(5) == 5


def test_diagonal_construct_psi():
    """DiagonalCovariance should construct diagonal matrix."""
    cov = DiagonalCovariance()
    # Log-variances
    params = np.array([0.0, 0.5, 1.0])

    psi = cov.construct_psi(params, n_effects=3)

    # Expected variances: exp([0.0, 0.5, 1.0])
    expected = np.diag(np.exp([0.0, 0.5, 1.0]))

    assert psi.shape == (3, 3)
    np.testing.assert_allclose(psi, expected, atol=1e-10)

    # Check it's diagonal
    off_diag = psi - np.diag(np.diag(psi))
    np.testing.assert_allclose(off_diag, 0, atol=1e-15)


def test_diagonal_extract_params():
    """DiagonalCovariance should extract log-variances."""
    cov = DiagonalCovariance()

    psi = np.diag([1.0, 2.0, 3.0])

    params = cov.extract_params(psi)

    expected = np.log([1.0, 2.0, 3.0])
    np.testing.assert_allclose(params, expected, atol=1e-10)


def test_diagonal_roundtrip():
    """DiagonalCovariance should roundtrip correctly."""
    cov = DiagonalCovariance()
    original_params = np.array([0.5, 1.0, -0.5])

    psi = cov.construct_psi(original_params, n_effects=3)
    extracted_params = cov.extract_params(psi)

    np.testing.assert_allclose(extracted_params, original_params, atol=1e-10)


def test_diagonal_non_diagonal_raises():
    """DiagonalCovariance should raise for non-diagonal matrix."""
    cov = DiagonalCovariance()

    psi = np.array([[1.0, 0.1], [0.1, 1.0]])

    with pytest.raises(ValueError, match="must be diagonal"):
        cov.extract_params(psi)


def test_diagonal_negative_variance_raises():
    """DiagonalCovariance should raise for negative variances."""
    cov = DiagonalCovariance()

    psi = np.diag([1.0, -0.5, 2.0])

    with pytest.raises(ValueError, match="must be positive"):
        cov.extract_params(psi)


# Identity Covariance Tests


def test_identity_n_parameters():
    """IdentityCovariance should have 1 parameter."""
    cov = IdentityCovariance()

    assert cov.n_parameters(1) == 1
    assert cov.n_parameters(5) == 1
    assert cov.n_parameters(10) == 1


def test_identity_construct_psi():
    """IdentityCovariance should construct σ²I."""
    cov = IdentityCovariance()
    # Log-variance
    params = np.array([0.5])

    psi = cov.construct_psi(params, n_effects=3)

    # Expected: exp(0.5) * I_3
    variance = np.exp(0.5)
    expected = variance * np.eye(3)

    assert psi.shape == (3, 3)
    np.testing.assert_allclose(psi, expected, atol=1e-10)


def test_identity_extract_params():
    """IdentityCovariance should extract log(σ²)."""
    cov = IdentityCovariance()

    psi = 2.5 * np.eye(4)

    params = cov.extract_params(psi)

    assert params.shape == (1,)
    assert np.isclose(params[0], np.log(2.5))


def test_identity_roundtrip():
    """IdentityCovariance should roundtrip correctly."""
    cov = IdentityCovariance()
    original_params = np.array([1.2])

    psi = cov.construct_psi(original_params, n_effects=4)
    extracted_params = cov.extract_params(psi)

    np.testing.assert_allclose(extracted_params, original_params, atol=1e-10)


def test_identity_wrong_param_count_raises():
    """IdentityCovariance should raise with wrong parameter count."""
    cov = IdentityCovariance()
    params = np.array([0.5, 1.0])  # Need exactly 1

    with pytest.raises(ValueError, match="Expected 1 parameter"):
        cov.construct_psi(params, n_effects=3)


def test_identity_non_identity_raises():
    """IdentityCovariance should raise for non-identity proportional matrix."""
    cov = IdentityCovariance()

    # Different diagonal elements
    psi = np.diag([1.0, 2.0, 1.0])

    with pytest.raises(ValueError, match="proportional to identity"):
        cov.extract_params(psi)


def test_identity_with_off_diagonal_raises():
    """IdentityCovariance should raise for non-zero off-diagonals."""
    cov = IdentityCovariance()

    psi = np.array([[2.0, 0.1], [0.1, 2.0]])

    with pytest.raises(ValueError, match="proportional to identity"):
        cov.extract_params(psi)


# Factory function tests


def test_get_covariance_structure_unstructured():
    """get_covariance_structure should return UnstructuredCovariance."""
    cov = get_covariance_structure("unstructured")

    assert isinstance(cov, UnstructuredCovariance)


def test_get_covariance_structure_diagonal():
    """get_covariance_structure should return DiagonalCovariance."""
    cov = get_covariance_structure("diagonal")

    assert isinstance(cov, DiagonalCovariance)


def test_get_covariance_structure_identity():
    """get_covariance_structure should return IdentityCovariance."""
    cov = get_covariance_structure("identity")

    assert isinstance(cov, IdentityCovariance)


def test_get_covariance_structure_invalid_raises():
    """get_covariance_structure should raise for invalid structure."""
    with pytest.raises(ValueError, match="Unknown covariance structure"):
        get_covariance_structure("invalid")
