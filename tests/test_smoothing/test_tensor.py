"""Tests for tensor product smooths."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.smoothing.splines.bspline import BSplineBasis
from aurora.smoothing.splines.cubic import CubicSplineBasis
from aurora.smoothing.tensor import (
    fit_tensor_product,
    tensor_product_basis,
    tensor_product_penalty,
)


def test_tensor_product_basis_shape():
    """Tensor product basis should have correct shape."""
    n = 100
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)

    # Create bases
    knots1 = BSplineBasis.create_knots(x1, n_basis=8, degree=3)
    knots2 = BSplineBasis.create_knots(x2, n_basis=10, degree=3)
    basis1 = BSplineBasis(knots1, degree=3)
    basis2 = BSplineBasis(knots2, degree=3)

    # Compute tensor product
    B = tensor_product_basis(x1, x2, basis1, basis2)

    # Should be (n, p1 * p2)
    assert B.shape == (n, 8 * 10)
    assert B.shape == (n, 80)


def test_tensor_product_basis_values():
    """Tensor product basis should compute correct values."""
    n = 50
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)

    # Create simple bases
    knots1 = BSplineBasis.create_knots(x1, n_basis=5, degree=3)
    knots2 = BSplineBasis.create_knots(x2, n_basis=5, degree=3)
    basis1 = BSplineBasis(knots1, degree=3)
    basis2 = BSplineBasis(knots2, degree=3)

    # Get marginal basis matrices
    B1 = basis1.basis_matrix(x1)
    B2 = basis2.basis_matrix(x2)

    # Compute tensor product
    B_tensor = tensor_product_basis(x1, x2, basis1, basis2)

    # Check first row manually
    # B_tensor[0, :] should equal vec(B1[0, :] ⊗ B2[0, :])
    expected_row0 = np.outer(B1[0, :], B2[0, :]).ravel()
    np.testing.assert_allclose(B_tensor[0, :], expected_row0)

    # Check last row
    expected_row_last = np.outer(B1[-1, :], B2[-1, :]).ravel()
    np.testing.assert_allclose(B_tensor[-1, :], expected_row_last)


def test_tensor_product_basis_different_sizes():
    """Tensor product should work with different basis sizes."""
    n = 75
    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)

    # Different sizes
    knots1 = BSplineBasis.create_knots(x1, n_basis=6, degree=3)
    knots2 = BSplineBasis.create_knots(x2, n_basis=12, degree=3)
    basis1 = BSplineBasis(knots1, degree=3)
    basis2 = BSplineBasis(knots2, degree=3)

    B = tensor_product_basis(x1, x2, basis1, basis2)

    assert B.shape == (n, 6 * 12)


def test_tensor_product_basis_cubic_splines():
    """Tensor product should work with cubic splines."""
    n = 80
    x1 = np.linspace(-1, 1, n)
    x2 = np.linspace(-1, 1, n)

    # Create cubic spline bases
    knots1 = CubicSplineBasis.create_knots(x1, n_knots=8)
    knots2 = CubicSplineBasis.create_knots(x2, n_knots=8)
    basis1 = CubicSplineBasis(knots1)
    basis2 = CubicSplineBasis(knots2)

    B = tensor_product_basis(x1, x2, basis1, basis2)

    # Cubic splines have n_knots + 2 basis functions
    p1 = 8 + 2
    p2 = 8 + 2
    assert B.shape == (n, p1 * p2)


def test_tensor_product_penalty_shape():
    """Tensor product penalties should have correct shape."""
    p1, p2 = 8, 10
    S1 = np.eye(p1)
    S2 = np.eye(p2)

    S_x1, S_x2 = tensor_product_penalty(S1, S2, p1, p2)

    # Both should be (p1*p2, p1*p2)
    assert S_x1.shape == (p1 * p2, p1 * p2)
    assert S_x2.shape == (p1 * p2, p1 * p2)
    assert S_x1.shape == (80, 80)


def test_tensor_product_penalty_structure():
    """Tensor product penalties should have Kronecker product structure."""
    p1, p2 = 3, 4
    S1 = np.arange(9).reshape(3, 3)
    S2 = np.arange(16).reshape(4, 4)

    S_x1, S_x2 = tensor_product_penalty(S1, S2, p1, p2)

    # S_x1 = S1 ⊗ I2
    I2 = np.eye(p2)
    expected_S_x1 = np.kron(S1, I2)
    np.testing.assert_allclose(S_x1, expected_S_x1)

    # S_x2 = I1 ⊗ S2
    I1 = np.eye(p1)
    expected_S_x2 = np.kron(I1, S2)
    np.testing.assert_allclose(S_x2, expected_S_x2)


def test_tensor_product_penalty_symmetry():
    """Tensor product penalties should be symmetric."""
    p1, p2 = 5, 6
    S1 = np.random.randn(p1, p1)
    S1 = S1 @ S1.T  # Make symmetric
    S2 = np.random.randn(p2, p2)
    S2 = S2 @ S2.T  # Make symmetric

    S_x1, S_x2 = tensor_product_penalty(S1, S2, p1, p2)

    # Both should be symmetric
    np.testing.assert_allclose(S_x1, S_x1.T)
    np.testing.assert_allclose(S_x2, S_x2.T)


def test_fit_tensor_product_basic():
    """fit_tensor_product should fit interaction surface."""
    np.random.seed(42)
    n = 200

    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)

    # True interaction surface
    y_true = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)
    y = y_true + 0.1 * np.random.randn(n)

    # Create bases
    knots1 = BSplineBasis.create_knots(x1, n_basis=10, degree=3)
    knots2 = BSplineBasis.create_knots(x2, n_basis=10, degree=3)
    basis1 = BSplineBasis(knots1, degree=3)
    basis2 = BSplineBasis(knots2, degree=3)

    S1 = basis1.penalty_matrix(order=2)
    S2 = basis2.penalty_matrix(order=2)

    # Fit tensor product
    result = fit_tensor_product(
        x1, x2, y, basis1, basis2, S1, S2,
        lambda1=0.001, lambda2=0.001
    )

    # Check result structure
    assert 'coefficients' in result
    assert 'fitted_values' in result
    assert 'basis_matrix' in result
    assert 'edf' in result

    # Check shapes
    assert result['fitted_values'].shape == (n,)
    assert result['coefficients'].shape[0] == 10 * 10

    # Check fit quality
    r_squared = 1 - np.sum((y - result['fitted_values'])**2) / np.sum((y - np.mean(y))**2)
    assert r_squared > 0.7  # Should capture interaction well


def test_fit_tensor_product_with_weights():
    """fit_tensor_product should handle observation weights."""
    np.random.seed(42)
    n = 150

    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)
    y = x1 * x2 + 0.1 * np.random.randn(n)
    weights = np.random.uniform(0.5, 1.5, n)

    # Create bases
    knots1 = BSplineBasis.create_knots(x1, n_basis=8, degree=3)
    knots2 = BSplineBasis.create_knots(x2, n_basis=8, degree=3)
    basis1 = BSplineBasis(knots1, degree=3)
    basis2 = BSplineBasis(knots2, degree=3)

    S1 = basis1.penalty_matrix(order=2)
    S2 = basis2.penalty_matrix(order=2)

    result = fit_tensor_product(
        x1, x2, y, basis1, basis2, S1, S2,
        lambda1=0.01, lambda2=0.01,
        weights=weights
    )

    assert result['fitted_values'].shape == (n,)


def test_fit_tensor_product_different_smoothing():
    """fit_tensor_product should handle different smoothing parameters."""
    np.random.seed(42)
    n = 150

    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)

    # Smooth in x1, wiggly in x2
    y = np.sin(np.pi * x1) * np.sin(10 * np.pi * x2) + 0.1 * np.random.randn(n)

    # Create bases
    knots1 = BSplineBasis.create_knots(x1, n_basis=10, degree=3)
    knots2 = BSplineBasis.create_knots(x2, n_basis=15, degree=3)
    basis1 = BSplineBasis(knots1, degree=3)
    basis2 = BSplineBasis(knots2, degree=3)

    S1 = basis1.penalty_matrix(order=2)
    S2 = basis2.penalty_matrix(order=2)

    # More smoothing in x1, less in x2
    result = fit_tensor_product(
        x1, x2, y, basis1, basis2, S1, S2,
        lambda1=1.0,    # Smoother in x1
        lambda2=0.001   # More flexible in x2
    )

    # Should still provide reasonable fit
    r_squared = 1 - np.sum((y - result['fitted_values'])**2) / np.sum((y - np.mean(y))**2)
    assert r_squared > 0.5


def test_fit_tensor_product_edf():
    """EDF should be computed correctly."""
    np.random.seed(42)
    n = 100

    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)
    y = x1 + x2 + 0.1 * np.random.randn(n)

    # Small bases
    knots1 = BSplineBasis.create_knots(x1, n_basis=6, degree=3)
    knots2 = BSplineBasis.create_knots(x2, n_basis=6, degree=3)
    basis1 = BSplineBasis(knots1, degree=3)
    basis2 = BSplineBasis(knots2, degree=3)

    S1 = basis1.penalty_matrix(order=2)
    S2 = basis2.penalty_matrix(order=2)

    result = fit_tensor_product(
        x1, x2, y, basis1, basis2, S1, S2,
        lambda1=0.1, lambda2=0.1
    )

    # EDF should be positive and less than total parameters
    assert result['edf'] > 0
    assert result['edf'] < 6 * 6
    assert np.isfinite(result['edf'])


def test_tensor_product_additive_vs_interaction():
    """Tensor product should capture interactions better than additive model."""
    np.random.seed(42)
    n = 200

    x1 = np.random.uniform(-1, 1, n)
    x2 = np.random.uniform(-1, 1, n)

    # Pure interaction (no main effects)
    y_true = x1 * x2
    y = y_true + 0.1 * np.random.randn(n)

    # Create bases
    knots1 = BSplineBasis.create_knots(x1, n_basis=10, degree=3)
    knots2 = BSplineBasis.create_knots(x2, n_basis=10, degree=3)
    basis1 = BSplineBasis(knots1, degree=3)
    basis2 = BSplineBasis(knots2, degree=3)

    S1 = basis1.penalty_matrix(order=2)
    S2 = basis2.penalty_matrix(order=2)

    # Fit tensor product (captures interaction)
    result_tensor = fit_tensor_product(
        x1, x2, y, basis1, basis2, S1, S2,
        lambda1=0.01, lambda2=0.01
    )

    # Fit additive model (no interaction)
    B1 = basis1.basis_matrix(x1)
    B2 = basis2.basis_matrix(x2)
    B_additive = np.column_stack([B1, B2])

    from scipy.linalg import block_diag
    S_additive = block_diag(S1, S2)

    # Simple penalized fit
    XtX = B_additive.T @ B_additive
    Xty = B_additive.T @ y
    A = XtX + 0.01 * S_additive
    coef_additive = np.linalg.solve(A, Xty)
    fitted_additive = B_additive @ coef_additive

    # Compute R² for both
    r2_tensor = 1 - np.sum((y - result_tensor['fitted_values'])**2) / np.sum((y - np.mean(y))**2)
    r2_additive = 1 - np.sum((y - fitted_additive)**2) / np.sum((y - np.mean(y))**2)

    # Tensor product should be much better at capturing pure interaction
    assert r2_tensor > r2_additive + 0.2  # At least 20% better


def test_tensor_product_basis_reproducibility():
    """Tensor product should give same results with same inputs."""
    n = 100
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)

    knots1 = BSplineBasis.create_knots(x1, n_basis=8, degree=3)
    knots2 = BSplineBasis.create_knots(x2, n_basis=8, degree=3)
    basis1 = BSplineBasis(knots1, degree=3)
    basis2 = BSplineBasis(knots2, degree=3)

    # Compute twice
    B1 = tensor_product_basis(x1, x2, basis1, basis2)
    B2 = tensor_product_basis(x1, x2, basis1, basis2)

    np.testing.assert_array_equal(B1, B2)
