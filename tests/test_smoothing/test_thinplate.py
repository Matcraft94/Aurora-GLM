"""Tests for thin plate splines."""

from __future__ import annotations

import numpy as np
import pytest

from aurora.smoothing.thinplate import fit_tps, select_knots, tps_basis, tps_penalty


def test_tps_basis_shape_2d():
    """TPS basis should have correct shape for 2D data."""
    n = 100
    k = 20
    d = 2

    X = np.random.randn(n, d)
    knots = X[:k]

    B = tps_basis(X, knots, d=d)

    # Should be (n, k + d + 1)
    assert B.shape == (n, k + d + 1)
    assert B.shape == (n, 23)


def test_tps_basis_shape_3d():
    """TPS basis should work for 3D data."""
    n = 80
    k = 15
    d = 3

    X = np.random.randn(n, d)
    knots = X[:k]

    B = tps_basis(X, knots, d=d)

    # Should be (n, k + d + 1)
    assert B.shape == (n, k + d + 1)
    assert B.shape == (n, 19)


def test_tps_basis_polynomial_part():
    """TPS basis should include polynomial terms correctly."""
    n = 50
    k = 10
    d = 2

    X = np.random.randn(n, d)
    knots = X[:k]

    B = tps_basis(X, knots, d=d)

    # Last d+1 columns should be [1, x1, x2]
    np.testing.assert_array_equal(B[:, k], np.ones(n))  # Intercept
    np.testing.assert_array_equal(B[:, k + 1], X[:, 0])  # x1
    np.testing.assert_array_equal(B[:, k + 2], X[:, 1])  # x2


def test_tps_basis_radial_symmetry():
    """TPS radial basis should be symmetric in distance."""
    k = 10
    d = 2

    X = np.random.randn(50, d)
    knots = X[:k]

    B = tps_basis(X, knots, d=d)

    # Check that radial part depends only on distance
    # Points equidistant from a knot should have same radial value
    knot_idx = 0
    knot = knots[knot_idx]

    # Find points at similar distance
    distances = np.linalg.norm(X - knot, axis=1)
    target_dist = distances[10]
    similar = np.abs(distances - target_dist) < 1e-10

    if np.sum(similar) > 1:
        # Radial values should be very similar
        radial_vals = B[similar, knot_idx]
        assert np.std(radial_vals) < 1e-10


def test_tps_basis_zero_distance():
    """TPS basis should handle zero distance correctly."""
    n = 20
    d = 2

    X = np.random.randn(n, d)
    knots = X.copy()  # Use same points as knots

    B = tps_basis(X, knots, d=d)

    # Diagonal of radial part should be zero (r²log(r) at r=0)
    for i in range(n):
        assert B[i, i] == 0.0  # Distance to itself is 0


def test_tps_penalty_shape():
    """TPS penalty should have correct shape."""
    k = 15
    d = 2

    knots = np.random.randn(k, d)
    S = tps_penalty(knots, d=d)

    # Should be (k + d + 1, k + d + 1)
    assert S.shape == (k + d + 1, k + d + 1)
    assert S.shape == (18, 18)


def test_tps_penalty_structure():
    """TPS penalty should only penalize radial part."""
    k = 10
    d = 2

    knots = np.random.randn(k, d)
    S = tps_penalty(knots, d=d)

    # Top-left k×k block should be non-zero
    assert np.any(S[:k, :k] != 0)

    # Bottom and right parts should be zero (polynomial not penalized)
    np.testing.assert_array_equal(S[k:, :], 0)
    np.testing.assert_array_equal(S[:, k:], 0)


def test_tps_penalty_symmetry():
    """TPS penalty should be symmetric."""
    k = 12
    d = 2

    knots = np.random.randn(k, d)
    S = tps_penalty(knots, d=d)

    np.testing.assert_allclose(S, S.T, atol=1e-14)


def test_fit_tps_basic():
    """fit_tps should fit 2D surface."""
    np.random.seed(42)
    n = 150
    d = 2

    X = np.random.uniform(-1, 1, (n, d))
    # True surface: radial function
    y_true = np.exp(-np.sum(X**2, axis=1))
    y = y_true + 0.05 * np.random.randn(n)

    # Use subset of points as knots
    knots = X[::5]  # Every 5th point

    result = fit_tps(X, y, knots=knots, lambda_=0.01)

    # Check result structure
    assert "coefficients" in result
    assert "fitted_values" in result
    assert "knots" in result
    assert "edf" in result

    # Check shapes
    assert result["fitted_values"].shape == (n,)
    assert result["knots"].shape == knots.shape

    # Check fit quality
    r_squared = 1 - np.sum((y - result["fitted_values"]) ** 2) / np.sum((y - np.mean(y)) ** 2)
    assert r_squared > 0.7  # Should capture smooth surface well


def test_fit_tps_with_weights():
    """fit_tps should handle observation weights."""
    np.random.seed(42)
    n = 100
    d = 2

    X = np.random.uniform(-1, 1, (n, d))
    y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(n)
    weights = np.random.uniform(0.5, 1.5, n)

    knots = X[::4]

    result = fit_tps(X, y, knots=knots, lambda_=0.05, weights=weights)

    assert result["fitted_values"].shape == (n,)


def test_fit_tps_no_knots():
    """fit_tps should use all points as knots if not specified."""
    np.random.seed(42)
    n = 50  # Small n for speed
    d = 2

    X = np.random.randn(n, d)
    y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(n)

    result = fit_tps(X, y, knots=None, lambda_=0.1)

    # Should use all n points as knots
    assert result["knots"].shape == (n, d)
    np.testing.assert_array_equal(result["knots"], X)


def test_fit_tps_3d():
    """fit_tps should work for 3D data."""
    np.random.seed(42)
    n = 120
    d = 3

    X = np.random.uniform(-1, 1, (n, d))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(n)

    knots = X[::6]

    result = fit_tps(X, y, knots=knots, lambda_=0.1)

    assert result["fitted_values"].shape == (n,)
    r_squared = 1 - np.sum((y - result["fitted_values"]) ** 2) / np.sum((y - np.mean(y)) ** 2)
    assert r_squared > 0.5


def test_fit_tps_smoothing_parameter():
    """Different lambda should give different smoothness."""
    np.random.seed(42)
    n = 100
    d = 2

    X = np.random.uniform(-1, 1, (n, d))
    y = np.sin(3 * X[:, 0]) + np.cos(3 * X[:, 1]) + 0.2 * np.random.randn(n)

    knots = X[::5]

    # Very small lambda (more wiggly)
    result_wiggly = fit_tps(X, y, knots=knots, lambda_=0.001)

    # Large lambda (smoother)
    result_smooth = fit_tps(X, y, knots=knots, lambda_=10.0)

    # Wiggly fit should have higher EDF
    assert result_wiggly["edf"] > result_smooth["edf"]


def test_fit_tps_edf():
    """EDF should be finite and in a valid range."""
    np.random.seed(42)
    n = 80
    d = 2

    X = np.random.randn(n, d)
    y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(n)

    knots = X[::4]
    k = knots.shape[0]

    result = fit_tps(X, y, knots=knots, lambda_=0.1)

    # With the constraint-absorbed PSD penalty the EDF is bounded by the
    # number of knots (constrained design columns) and positive
    assert np.isfinite(result["edf"])
    assert 0 < result["edf"] <= k + 1e-8


def test_tps_penalty_positive_semidefinite():
    """The constraint-absorbed TPS penalty must be positive semi-definite.

    Regression test: the raw radial energy matrix E = eta(||x_i - x_j||) is
    only conditionally PSD (its eigenvalues ranged over [-10, 11]), so
    beta' S beta could be negative in a real fit.  After absorbing the
    constraints T' delta = 0 (Wood 2003) all eigenvalues must be >= 0.
    """
    rng = np.random.default_rng(42)
    knots = rng.normal(size=(20, 2))
    S = tps_penalty(knots, d=2)

    eigvals = np.linalg.eigvalsh(S)
    assert eigvals.min() > -1e-12

    # beta' S beta >= 0 for arbitrary coefficient vectors
    beta = rng.normal(size=S.shape[0])
    assert beta @ S @ beta >= -1e-12


def test_tps_penalty_equals_energy_on_constraint_subspace():
    """The projected penalty equals the raw bending energy when T' delta = 0."""
    from scipy.spatial.distance import cdist

    from aurora.smoothing.thinplate import (
        _null_space_of_transpose,
        _tps_polynomial,
        _tps_radial,
    )

    rng = np.random.default_rng(42)
    k, d, m = 25, 2, 2
    knots = rng.normal(size=(k, d))

    E = _tps_radial(cdist(knots, knots, metric="euclidean"), d, m)
    T = _tps_polynomial(knots, d, m)
    Z = _null_space_of_transpose(T)
    M = T.shape[1]

    S = tps_penalty(knots, d=d, m=m)

    # For delta in null(T') (i.e. delta = Z gamma), delta' S delta == delta' E delta
    gamma = rng.normal(size=k - M)
    delta = Z @ gamma
    beta = np.concatenate([delta, np.zeros(M)])
    np.testing.assert_allclose(beta @ S @ beta, delta @ E @ delta, rtol=1e-10)
    # And the raw energy on this subspace is non-negative
    assert delta @ E @ delta >= -1e-12


def test_tps_fit_penalty_nonnegative():
    """beta' S beta must be non-negative in an actual fit (was -1.30)."""
    rng = np.random.default_rng(42)
    n = 150
    X = rng.uniform(-1, 1, (n, 2))
    y = np.exp(-np.sum(X**2, axis=1)) + 0.05 * rng.normal(size=n)

    knots = X[::5]
    result = fit_tps(X, y, knots=knots, lambda_=0.01)
    S = tps_penalty(knots, d=2)

    assert result["coefficients"] @ S @ result["coefficients"] >= -1e-12


def test_tps_radial_duchon_higher_dimensions():
    """Duchon (1977) radial function for d >= 4 and the CPSD sign convention.

    The radial function is eta(r) = c r^(2m-d) log(r) for even d and
    eta(r) = c r^(2m-d) for odd d, with the minimal valid m (2m > d) and
    c = (-1)^(m - ceil(d/2) + 1) the sign making the energy matrix
    conditionally positive semi-definite.  The old code inverted the
    parity, used the exponent d-2, and ignored the sign.
    """
    from aurora.smoothing.thinplate import _tps_radial

    r = np.array([[0.5], [1.0], [2.0]])

    # d = 4, minimal m = 3: eta(r) = +r^2 log(r)  (even d -> log form)
    eta4 = _tps_radial(r, d=4, m=3)
    expected4 = (r**2) * np.log(r)
    np.testing.assert_allclose(eta4, expected4, rtol=1e-12)

    # d = 5, minimal m = 3: eta(r) = -r  (odd d -> pure power, negative sign)
    eta5 = _tps_radial(r, d=5, m=3)
    np.testing.assert_allclose(eta5, -r, rtol=1e-12)

    # d = 5, m = 4: eta(r) = +r^3 (sign alternates with m)
    eta5b = _tps_radial(r, d=5, m=4)
    np.testing.assert_allclose(eta5b, r**3, rtol=1e-12)

    # Classical cases keep their signs: +r^3 (d=1), +r^2 log r (d=2), -r (d=3)
    np.testing.assert_allclose(_tps_radial(r, d=1, m=2), r**3, rtol=1e-12)
    np.testing.assert_allclose(_tps_radial(r, d=3, m=2), -r, rtol=1e-12)


def test_tps_m_validation():
    """m must satisfy the Duchon condition 2m > d."""
    rng = np.random.default_rng(42)
    knots = rng.normal(size=(30, 5))

    with pytest.raises(ValueError, match="2m > d"):
        tps_penalty(knots, d=5, m=2)

    # Minimal valid m works and yields a PSD penalty
    S = tps_penalty(knots, d=5, m=3)
    assert np.linalg.eigvalsh(S).min() > -1e-12


def test_fit_tps_4d():
    """TPS fit should work in d = 4 with the corrected Duchon formula."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.uniform(-1, 1, (n, 4))
    y = np.sum(np.sin(2 * X), axis=1) + 0.05 * rng.normal(size=n)

    result = fit_tps(X, y, knots=X[::4], lambda_=0.01)

    assert result["m"] == 3  # Minimal valid order for d = 4
    assert result["fitted_values"].shape == (n,)
    r_squared = 1 - np.sum((y - result["fitted_values"]) ** 2) / np.sum((y - np.mean(y)) ** 2)
    assert r_squared > 0.9


def test_select_knots_uniform():
    """select_knots should select uniform subset."""
    n = 1000
    d = 2

    X = np.random.randn(n, d)
    knots = select_knots(X, n_knots=50, method="uniform")

    assert knots.shape == (50, d)
    # Should be a subset of original points
    for i in range(knots.shape[0]):
        # Check if knot exists in X
        matches = np.all(np.abs(X - knots[i]) < 1e-10, axis=1)
        assert np.any(matches)


def test_select_knots_random():
    """select_knots should select random subset."""
    np.random.seed(42)
    n = 500
    d = 2

    X = np.random.randn(n, d)
    knots = select_knots(X, n_knots=30, method="random")

    assert knots.shape == (30, d)


def test_select_knots_default():
    """select_knots should use default n_knots if not specified."""
    n = 200
    d = 2

    X = np.random.randn(n, d)
    knots = select_knots(X, n_knots=None, method="uniform")

    # Should use min(n, 100)
    assert knots.shape[0] == 100


def test_select_knots_too_many():
    """select_knots should handle n_knots > n."""
    n = 50
    d = 2

    X = np.random.randn(n, d)
    knots = select_knots(X, n_knots=100, method="uniform")

    # Should return at most n knots
    assert knots.shape[0] <= n


def test_select_knots_invalid_method():
    """select_knots should reject invalid method."""
    X = np.random.randn(100, 2)

    with pytest.raises(ValueError, match="Unknown method"):
        select_knots(X, n_knots=20, method="invalid")


def test_select_knots_kmeans_returns_centroids():
    """select_knots(method='kmeans') should return cluster centroids of X."""
    rng = np.random.default_rng(42)
    # Two well-separated clusters
    cluster_a = rng.normal(loc=[0, 0], scale=0.2, size=(60, 2))
    cluster_b = rng.normal(loc=[5, 5], scale=0.2, size=(60, 2))
    X = np.vstack([cluster_a, cluster_b])

    knots = select_knots(X, n_knots=2, method="kmeans")
    assert knots.shape == (2, 2)
    # Centroids should be near the true cluster centers (0,0) and (5,5)
    # Sort by first coordinate to make assertion order-independent
    sorted_knots = knots[np.argsort(knots[:, 0])]
    np.testing.assert_allclose(sorted_knots[0], [0, 0], atol=0.15)
    np.testing.assert_allclose(sorted_knots[1], [5, 5], atol=0.15)


def test_tps_interpolation_exact():
    """TPS should interpolate exactly at knots with lambda=0."""
    np.random.seed(42)
    n = 30
    d = 2

    X = np.random.randn(n, d)
    y = np.sum(X**2, axis=1)

    # Use all points as knots with no penalty
    result = fit_tps(X, y, knots=X, lambda_=0.0)

    # Should interpolate exactly (or very close due to numerical issues)
    np.testing.assert_allclose(result["fitted_values"], y, atol=1e-6)


def test_tps_1d():
    """TPS should work for 1D data (reduces to cubic spline-like)."""
    np.random.seed(42)
    n = 60
    d = 1

    X = np.random.uniform(0, 1, (n, d))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.1 * np.random.randn(n)

    knots = X[::3]

    result = fit_tps(X, y, knots=knots, lambda_=0.01)

    assert result["fitted_values"].shape == (n,)
    r_squared = 1 - np.sum((y - result["fitted_values"]) ** 2) / np.sum((y - np.mean(y)) ** 2)
    assert r_squared > 0.5


def test_tps_basis_1d_radial():
    """TPS 1D radial function should be r³."""
    k = 5

    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]])
    knots = X[:k]

    B = tps_basis(X, knots, d=1)

    # For d=1, η(r) = r³
    # Check first radial basis (distance from knot at 0.0)
    expected = np.array([0, 1, 8, 27, 64, 125, 216, 343, 512, 729], dtype=float)
    np.testing.assert_allclose(B[:, 0], expected, atol=1e-10)


def test_tps_reproducibility():
    """TPS should give same results with same inputs."""
    np.random.seed(42)
    n = 80
    d = 2

    X = np.random.randn(n, d)
    y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(n)
    knots = X[::4]

    result1 = fit_tps(X, y, knots=knots, lambda_=0.1)
    result2 = fit_tps(X, y, knots=knots, lambda_=0.1)

    np.testing.assert_array_equal(result1["fitted_values"], result2["fitted_values"])
    np.testing.assert_array_equal(result1["coefficients"], result2["coefficients"])
