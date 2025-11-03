"""Tests for B-spline basis functions."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.smoothing.splines.bspline import BSplineBasis


def test_bspline_basis_initialization():
    """BSplineBasis should initialize with valid parameters."""
    # Create uniform knot vector for cubic B-splines
    knots = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1])  # degree 3 (cubic)
    basis = BSplineBasis(knots, degree=3)

    # For n knots and degree p: n_basis = n - p - 1
    assert basis.n_basis_ == 5  # 9 - 3 - 1 = 5
    assert basis.degree_ == 3
    np.testing.assert_array_equal(basis.knots_, knots)


def test_bspline_with_different_degrees():
    """BSplineBasis should work with different degrees."""
    knots = np.array([0, 0, 0, 0.5, 1, 1, 1])

    # Linear B-splines (degree 1)
    basis_linear = BSplineBasis(knots, degree=1)
    assert basis_linear.n_basis_ == 5  # 7 - 1 - 1

    # Quadratic B-splines (degree 2)
    basis_quad = BSplineBasis(knots, degree=2)
    assert basis_quad.n_basis_ == 4  # 7 - 2 - 1

    # Cubic B-splines (degree 3)
    basis_cubic = BSplineBasis(knots, degree=3)
    assert basis_cubic.n_basis_ == 3  # 7 - 3 - 1


def test_bspline_rejects_invalid_parameters():
    """BSplineBasis should validate input parameters."""
    # Negative degree
    with pytest.raises(ValueError, match="non-negative"):
        BSplineBasis(np.array([0, 0, 1, 1]), degree=-1)

    # Non-monotonic knots
    with pytest.raises(ValueError, match="non-decreasing"):
        BSplineBasis(np.array([0, 1, 0.5, 1]), degree=1)

    # Too few knots
    with pytest.raises(ValueError, match="at least"):
        BSplineBasis(np.array([0, 1]), degree=3)


def test_bspline_basis_matrix_shape():
    """basis_matrix should return correct shape."""
    knots = BSplineBasis.create_knots(np.linspace(0, 1, 50), n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)

    x = np.linspace(0, 1, 30)
    B = basis.basis_matrix(x)

    assert B.shape == (30, 10)


def test_bspline_partition_of_unity():
    """B-spline basis functions should sum to 1 (partition of unity)."""
    knots = BSplineBasis.create_knots(np.linspace(0, 1, 50), n_basis=8, degree=3)
    basis = BSplineBasis(knots, degree=3)

    x = np.linspace(0.1, 0.9, 20)  # Avoid boundaries
    B = basis.basis_matrix(x)

    # Sum of basis functions at each point should be 1
    row_sums = B.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_bspline_local_support():
    """B-splines should have local support."""
    knots = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1])
    basis = BSplineBasis(knots, degree=3)

    # Evaluate at middle of domain
    x = np.array([0.5])
    B = basis.basis_matrix(x)

    # Most basis functions should be zero (local support)
    # For cubic B-splines, at most degree+1 = 4 basis functions are non-zero
    non_zero = np.sum(np.abs(B[0]) > 1e-10)
    assert non_zero <= 4


def test_bspline_linear_function():
    """B-splines should exactly represent polynomials up to their degree."""
    knots = BSplineBasis.create_knots(np.linspace(0, 1, 50), n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)

    x = np.linspace(0, 1, 20)
    B = basis.basis_matrix(x)

    # Cubic B-splines should exactly represent linear functions
    # Find coefficients that give f(x) = 2x + 1
    # Since sum of basis = 1, constant is easy: all coef = constant
    # For linear: need to solve B @ coef = 2x + 1

    # Least squares fit
    target = 2 * x + 1
    coef, _, _, _ = np.linalg.lstsq(B, target, rcond=None)

    # Reconstruct
    f = B @ coef

    # Should match closely (within numerical precision)
    np.testing.assert_allclose(f, target, atol=1e-6)


def test_bspline_penalty_matrix_shape():
    """penalty_matrix should return correct shape."""
    knots = BSplineBasis.create_knots(np.linspace(0, 1, 50), n_basis=12, degree=3)
    basis = BSplineBasis(knots, degree=3)

    S = basis.penalty_matrix(order=2)

    assert S.shape == (12, 12)
    # Should be symmetric
    np.testing.assert_allclose(S, S.T, atol=1e-12)
    # Should be positive semi-definite
    eigenvalues = np.linalg.eigvals(S)
    assert np.all(eigenvalues >= -1e-10)


def test_bspline_penalty_for_polynomials():
    """Penalty should be zero for polynomials up to degree."""
    knots = BSplineBasis.create_knots(np.linspace(0, 1, 50), n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)

    S = basis.penalty_matrix(order=2)

    # Linear function should have zero second-order difference penalty
    x = np.linspace(0, 1, 20)
    B = basis.basis_matrix(x)

    # Fit linear function
    target = 2 * x + 1
    coef, _, _, _ = np.linalg.lstsq(B, target, rcond=None)

    # Second-order penalty should be small (linear has zero second derivative)
    penalty = coef @ S @ coef
    assert abs(penalty) < 0.1  # Small but may not be exactly zero due to approximation


def test_bspline_create_knots_uniform():
    """create_knots with uniform method should space knots evenly."""
    x = np.linspace(0, 10, 100)
    knots = BSplineBasis.create_knots(x, n_basis=8, degree=3, method="uniform")

    # Should have repeated boundaries
    assert knots[0] == knots[3]  # First 4 are same (degree+1)
    assert knots[-1] == knots[-4]  # Last 4 are same

    # Interior knots should be evenly spaced
    interior = knots[4:-4]
    if len(interior) > 1:
        spacing = np.diff(interior)
        np.testing.assert_allclose(spacing, spacing[0], atol=1e-10)


def test_bspline_create_knots_quantile():
    """create_knots with quantile method should place knots at quantiles."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=1000)

    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3, method="quantile")

    # Should have structure: [min]*4 + interior + [max]*4
    assert np.abs(knots[0] - x.min()) < 1e-6
    assert np.abs(knots[-1] - x.max()) < 1e-6


def test_bspline_create_knots_invalid_params():
    """create_knots should validate parameters."""
    x = np.linspace(0, 1, 50)

    # n_basis too small for degree
    with pytest.raises(ValueError, match="too small"):
        BSplineBasis.create_knots(x, n_basis=2, degree=3)

    # Invalid method
    with pytest.raises(ValueError, match="Unknown method"):
        BSplineBasis.create_knots(x, n_basis=10, degree=3, method="invalid")


def test_bspline_derivative_first_order():
    """derivative_basis_matrix should compute first derivatives correctly."""
    knots = BSplineBasis.create_knots(np.linspace(0, 1, 50), n_basis=8, degree=3)
    basis = BSplineBasis(knots, degree=3)

    x = np.linspace(0.1, 0.9, 10)

    # Approximate derivative using finite differences
    h = 1e-6
    B_plus = basis.basis_matrix(x + h)
    B_minus = basis.basis_matrix(x - h)
    dB_numeric = (B_plus - B_minus) / (2 * h)

    # Analytical derivative
    dB_analytic = basis.derivative_basis_matrix(x, order=1)

    # Should match
    np.testing.assert_allclose(dB_analytic, dB_numeric, atol=1e-4)


def test_bspline_derivative_higher_order():
    """derivative_basis_matrix should handle higher-order derivatives."""
    knots = BSplineBasis.create_knots(np.linspace(0, 1, 50), n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)

    x = np.array([0.5])

    # Second derivative (reduces degree by 2, so has more basis functions)
    d2B = basis.derivative_basis_matrix(x, order=2)
    assert d2B.shape[0] == 1  # One evaluation point
    assert d2B.shape[1] >= 10  # At least as many basis functions

    # Third derivative (equals degree)
    d3B = basis.derivative_basis_matrix(x, order=3)
    assert d3B.shape[0] == 1

    # Fourth derivative (exceeds degree) - should return zeros
    d4B = basis.derivative_basis_matrix(x, order=4)
    assert d4B.shape == (1, 10)  # Same size as original
    # All zeros since derivative order exceeds polynomial degree
    np.testing.assert_allclose(d4B, 0, atol=1e-10)


def test_bspline_scalar_input():
    """basis_matrix should handle scalar input."""
    knots = BSplineBasis.create_knots(np.linspace(0, 1, 50), n_basis=8, degree=3)
    basis = BSplineBasis(knots, degree=3)

    x = 0.5
    B = basis.basis_matrix(x)

    assert B.shape == (1, 8)


def test_bspline_boundary_behavior():
    """B-splines should interpolate at boundaries with repeated knots."""
    knots = BSplineBasis.create_knots(np.linspace(0, 1, 50), n_basis=8, degree=3)
    basis = BSplineBasis(knots, degree=3)

    # Evaluate at boundaries
    x_boundary = np.array([0.0, 1.0])
    B_boundary = basis.basis_matrix(x_boundary)

    # At boundaries, basis should sum to 1
    assert np.abs(B_boundary[0].sum() - 1.0) < 1e-6
    assert np.abs(B_boundary[1].sum() - 1.0) < 1e-6


def test_bspline_consistency_with_cubic_splines():
    """B-splines and cubic splines should both work for fitting."""
    from aurora.smoothing.splines.cubic import CubicSplineBasis

    x = np.linspace(0, 1, 50)

    # Create similar knot configurations
    interior_knots = np.array([0.25, 0.5, 0.75])

    # Cubic splines (natural splines, not necessarily partition of unity)
    cubic_basis = CubicSplineBasis(interior_knots, boundary_knots=(0.0, 1.0))

    # B-splines with similar structure
    bspline_knots = BSplineBasis.create_knots(x, n_basis=5, degree=3)
    bspline_basis = BSplineBasis(bspline_knots, degree=3)

    # Both should give reasonable basis matrices
    x_eval = np.linspace(0.1, 0.9, 20)
    B_cubic = cubic_basis.basis_matrix(x_eval)
    B_bspline = bspline_basis.basis_matrix(x_eval)

    # B-splines sum to 1 (partition of unity property)
    assert np.allclose(B_bspline.sum(axis=1), 1.0, atol=1e-10)

    # Cubic splines don't necessarily sum to 1 (different formulation)
    # Just check they're reasonable (non-negative, finite)
    assert np.all(np.isfinite(B_cubic))
    assert B_cubic.shape == (20, 5)
