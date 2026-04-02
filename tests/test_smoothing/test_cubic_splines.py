"""Tests for natural cubic spline basis functions."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.smoothing.splines.cubic import CubicSplineBasis


def test_cubic_basis_initialization():
    """CubicSplineBasis should initialize with valid knots."""
    knots = np.array([0.25, 0.5, 0.75])
    basis = CubicSplineBasis(knots)

    assert basis.n_basis_ == 5  # 3 knots + 2 = 5 basis functions
    np.testing.assert_array_equal(basis.knots_, knots)
    assert basis.boundary_knots_[0] < knots[0]
    assert basis.boundary_knots_[1] > knots[-1]


def test_cubic_basis_with_custom_boundaries():
    """CubicSplineBasis should accept custom boundary knots."""
    knots = np.array([0.3, 0.5, 0.7])
    boundary_knots = (0.0, 1.0)

    basis = CubicSplineBasis(knots, boundary_knots=boundary_knots)

    assert basis.boundary_knots_ == boundary_knots
    assert basis.n_basis_ == 5


def test_cubic_basis_rejects_invalid_knots():
    """CubicSplineBasis should raise errors for invalid knots."""
    # Non-increasing knots
    with pytest.raises(ValueError, match="strictly increasing"):
        CubicSplineBasis(np.array([0.5, 0.3, 0.7]))

    # Empty knots
    with pytest.raises(ValueError, match="At least one"):
        CubicSplineBasis(np.array([]))

    # Multidimensional knots
    with pytest.raises(ValueError, match="1-dimensional"):
        CubicSplineBasis(np.array([[0.3, 0.5]]))


def test_cubic_basis_rejects_invalid_boundaries():
    """CubicSplineBasis should validate boundary knots."""
    knots = np.array([0.3, 0.5, 0.7])

    # Boundaries don't enclose knots
    with pytest.raises(ValueError, match="enclose"):
        CubicSplineBasis(knots, boundary_knots=(0.4, 1.0))

    # Lower >= upper
    with pytest.raises(ValueError, match="Lower boundary"):
        CubicSplineBasis(knots, boundary_knots=(1.0, 0.0))


def test_cubic_basis_matrix_shape():
    """basis_matrix should return correct shape."""
    knots = np.array([0.25, 0.5, 0.75])
    basis = CubicSplineBasis(knots)

    x = np.linspace(0, 1, 50)
    B = basis.basis_matrix(x)

    assert B.shape == (50, 5)


def test_cubic_basis_matrix_at_knots():
    """Basis functions should be continuous at knots."""
    knots = np.array([0.3, 0.5, 0.7])
    basis = CubicSplineBasis(knots, boundary_knots=(0.0, 1.0))

    # Evaluate just before and after a knot
    eps = 1e-8
    knot = knots[1]  # Middle knot

    B_before = basis.basis_matrix(np.array([knot - eps]))
    B_after = basis.basis_matrix(np.array([knot + eps]))

    # Should be approximately equal (continuity)
    np.testing.assert_allclose(B_before, B_after, atol=1e-4)


def test_cubic_basis_linear_function():
    """Cubic splines should exactly represent linear functions."""
    knots = np.array([0.25, 0.5, 0.75])
    basis = CubicSplineBasis(knots, boundary_knots=(0.0, 1.0))

    x = np.linspace(0, 1, 20)
    B = basis.basis_matrix(x)

    # Linear function: f(x) = 2x + 3
    # Should be representable with just the first two basis functions
    coef = np.array([3.0, 2.0, 0.0, 0.0, 0.0])
    f = B @ coef

    expected = 2 * x + 3
    np.testing.assert_allclose(f, expected, atol=1e-10)


def test_cubic_basis_penalty_matrix_shape():
    """penalty_matrix should return correct shape."""
    knots = np.array([0.25, 0.5, 0.75])
    basis = CubicSplineBasis(knots)

    S = basis.penalty_matrix()

    assert S.shape == (5, 5)
    # Should be symmetric
    np.testing.assert_allclose(S, S.T, atol=1e-12)


def test_cubic_basis_penalty_matrix_properties():
    """Penalty matrix should have correct properties."""
    knots = np.array([0.3, 0.5, 0.7])
    basis = CubicSplineBasis(knots, boundary_knots=(0.0, 1.0))

    S = basis.penalty_matrix()

    # Penalty for linear functions should be zero
    # (first two basis functions are constant and linear)
    linear_coef = np.array([1.0, 2.0, 0.0, 0.0, 0.0])
    penalty = linear_coef @ S @ linear_coef
    assert abs(penalty) < 1e-10

    # Penalty for cubic terms should be positive
    cubic_coef = np.array([0.0, 0.0, 1.0, 0.5, 0.3])
    penalty_cubic = cubic_coef @ S @ cubic_coef
    assert penalty_cubic > 0


def test_cubic_basis_penalty_increases_with_roughness():
    """More oscillatory functions should have higher penalty."""
    knots = np.array([0.2, 0.4, 0.6, 0.8])
    basis = CubicSplineBasis(knots, boundary_knots=(0.0, 1.0))

    S = basis.penalty_matrix()

    # Smooth cubic
    smooth_coef = np.array([0.0, 0.0, 0.1, 0.1, 0.1, 0.1])
    penalty_smooth = smooth_coef @ S @ smooth_coef

    # Oscillatory cubic (alternating signs)
    oscillatory_coef = np.array([0.0, 0.0, 1.0, -1.0, 1.0, -1.0])
    penalty_oscillatory = oscillatory_coef @ S @ oscillatory_coef

    # Oscillatory should have higher penalty
    assert penalty_oscillatory > penalty_smooth


def test_create_knots_quantile():
    """create_knots with quantile method should place knots at quantiles."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=1000)

    knots = CubicSplineBasis.create_knots(x, n_knots=5, method="quantile")

    assert len(knots) == 5
    # Knots should be sorted
    assert np.all(np.diff(knots) > 0)

    # Check approximate quantile positions
    expected_quantiles = np.quantile(x, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
    np.testing.assert_allclose(knots, expected_quantiles, atol=0.1)


def test_create_knots_uniform():
    """create_knots with uniform method should space knots evenly."""
    x = np.linspace(0, 10, 100)

    knots = CubicSplineBasis.create_knots(x, n_knots=4, method="uniform")

    assert len(knots) == 4
    # Knots should be approximately evenly spaced
    spacing = np.diff(knots)
    np.testing.assert_allclose(spacing, spacing[0], atol=1e-10)


def test_create_knots_invalid_method():
    """create_knots should raise error for unknown method."""
    x = np.linspace(0, 1, 50)

    with pytest.raises(ValueError, match="Unknown method"):
        CubicSplineBasis.create_knots(x, n_knots=5, method="invalid")


def test_cubic_basis_with_pytorch_backend():
    """CubicSplineBasis should work with PyTorch tensors."""
    pytest.skip("PyTorch backend support deferred to later iteration")
    pytest.importorskip("torch")
    import torch

    knots = np.array([0.25, 0.5, 0.75])
    basis = CubicSplineBasis(knots, boundary_knots=(0.0, 1.0))

    x_torch = torch.linspace(0, 1, 20)
    B_torch = basis.basis_matrix(x_torch)

    # Should return PyTorch tensor
    assert isinstance(B_torch, torch.Tensor)
    assert B_torch.shape == (20, 5)

    # Should match NumPy result
    x_numpy = x_torch.numpy()
    B_numpy = basis.basis_matrix(x_numpy)

    np.testing.assert_allclose(B_torch.numpy(), B_numpy, atol=1e-6)


def test_cubic_basis_scalar_input():
    """basis_matrix should handle scalar input."""
    knots = np.array([0.3, 0.5, 0.7])
    basis = CubicSplineBasis(knots, boundary_knots=(0.0, 1.0))

    # Scalar input
    x = 0.5
    B = basis.basis_matrix(x)

    assert B.shape == (1, 5)


def test_cubic_basis_extrapolation():
    """Basis functions should handle extrapolation gracefully."""
    knots = np.array([0.25, 0.5, 0.75])
    basis = CubicSplineBasis(knots, boundary_knots=(0.0, 1.0))

    # Extrapolate beyond boundaries
    x = np.array([-0.5, 1.5])
    B = basis.basis_matrix(x)

    # Should not raise, and should return reasonable values
    assert B.shape == (2, 5)
    assert np.all(np.isfinite(B))
