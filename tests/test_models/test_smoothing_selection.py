"""Tests for smoothing parameter selection in GAMM (Phase 5.1.3)."""

import numpy as np

from aurora.distributions.families import BinomialFamily, PoissonFamily
from aurora.models.gamm.smoothing_selection import (
    select_smoothing_gcv,
    select_smoothing_performance_iter,
)
from aurora.smoothing.splines.bspline import BSplineBasis


def test_select_smoothing_gcv_single_term():
    """Test GCV selection with single smooth term."""
    np.random.seed(42)
    n = 150
    n_groups = 15

    # Generate data
    x = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)

    # True smooth function
    f_true = 2 * np.sin(2 * np.pi * x)

    # Random effects
    b_true = np.random.randn(n_groups) * 0.4
    eta_true = 1.0 + f_true + b_true[groups]
    mu_true = np.exp(eta_true)
    y = np.random.poisson(mu_true)

    # Create smooth basis
    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Random effects design
    Z = np.zeros((n, n_groups))
    Z[np.arange(n), groups] = 1.0

    # Initial Psi
    Psi = np.array([[0.5]])

    # Select smoothing parameter
    family_obj = PoissonFamily()
    link = family_obj.default_link

    lambda_opt = select_smoothing_gcv(
        X_parametric=np.ones((n, 1)),
        X_smooth_dict={"s(x)": B},
        S_smooth_dict={"s(x)": S},
        Z=Z,
        y=y,
        family_obj=family_obj,
        link=link,
        Psi=Psi,
        verbose=False,
    )

    # Check result
    assert "s(x)" in lambda_opt
    assert lambda_opt["s(x)"] > 0
    assert np.isfinite(lambda_opt["s(x)"])
    # Should select something reasonable (not too small or too large)
    assert 1e-4 < lambda_opt["s(x)"] < 1e4


def test_select_smoothing_gcv_multiple_terms():
    """Test GCV selection with multiple smooth terms."""
    np.random.seed(123)
    n = 150
    n_groups = 15

    # Generate data
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)

    f1_true = np.sin(2 * np.pi * x1)
    f2_true = 0.5 * x2**2

    b_true = np.random.randn(n_groups) * 0.3
    eta_true = f1_true + f2_true + b_true[groups]
    mu_true = np.exp(eta_true)
    y = np.random.poisson(mu_true)

    # Create smooth bases
    knots1 = BSplineBasis.create_knots(x1, n_basis=8, degree=3)
    basis1 = BSplineBasis(knots1, degree=3)
    B1 = basis1.basis_matrix(x1)
    S1 = basis1.penalty_matrix(order=2)

    knots2 = BSplineBasis.create_knots(x2, n_basis=8, degree=3)
    basis2 = BSplineBasis(knots2, degree=3)
    B2 = basis2.basis_matrix(x2)
    S2 = basis2.penalty_matrix(order=2)

    # Random effects design
    Z = np.zeros((n, n_groups))
    Z[np.arange(n), groups] = 1.0

    Psi = np.array([[0.3]])

    # Select smoothing parameters
    family_obj = PoissonFamily()
    link = family_obj.default_link

    lambda_opt = select_smoothing_gcv(
        X_parametric=np.ones((n, 1)),
        X_smooth_dict={"s(x1)": B1, "s(x2)": B2},
        S_smooth_dict={"s(x1)": S1, "s(x2)": S2},
        Z=Z,
        y=y,
        family_obj=family_obj,
        link=link,
        Psi=Psi,
        verbose=False,
    )

    # Check results
    assert "s(x1)" in lambda_opt
    assert "s(x2)" in lambda_opt
    assert lambda_opt["s(x1)"] > 0
    assert lambda_opt["s(x2)"] > 0
    assert np.isfinite(lambda_opt["s(x1)"])
    assert np.isfinite(lambda_opt["s(x2)"])


def test_select_smoothing_gcv_binomial():
    """Test GCV selection for binomial family."""
    np.random.seed(456)
    n = 100
    n_groups = 10

    x = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)

    f_true = 2 * (x - 0.5)
    b_true = np.random.randn(n_groups) * 0.5

    eta_true = f_true + b_true[groups]
    prob_true = 1 / (1 + np.exp(-eta_true))
    y = np.random.binomial(1, prob_true)

    # Create smooth basis
    knots = BSplineBasis.create_knots(x, n_basis=8, degree=3)
    basis = BSplineBasis(knots, degree=3)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    Z = np.zeros((n, n_groups))
    Z[np.arange(n), groups] = 1.0

    Psi = np.array([[0.5]])

    # Select smoothing parameter
    family_obj = BinomialFamily()
    link = family_obj.default_link

    lambda_opt = select_smoothing_gcv(
        X_parametric=np.ones((n, 1)),
        X_smooth_dict={"s(x)": B},
        S_smooth_dict={"s(x)": S},
        Z=Z,
        y=y,
        family_obj=family_obj,
        link=link,
        Psi=Psi,
        verbose=False,
    )

    assert "s(x)" in lambda_opt
    assert lambda_opt["s(x)"] > 0


def test_select_smoothing_custom_grid():
    """Test GCV selection with custom λ grid."""
    np.random.seed(789)
    n = 100
    n_groups = 10

    x = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)

    f_true = np.sin(2 * np.pi * x)
    b_true = np.random.randn(n_groups) * 0.3

    eta_true = f_true + b_true[groups]
    mu_true = np.exp(eta_true)
    y = np.random.poisson(mu_true)

    # Create smooth basis
    knots = BSplineBasis.create_knots(x, n_basis=8, degree=3)
    basis = BSplineBasis(knots, degree=3)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    Z = np.zeros((n, n_groups))
    Z[np.arange(n), groups] = 1.0

    Psi = np.array([[0.3]])

    # Custom grid (coarser)
    lambda_grid = {"s(x)": np.array([0.01, 0.1, 1.0, 10.0, 100.0])}

    family_obj = PoissonFamily()
    link = family_obj.default_link

    lambda_opt = select_smoothing_gcv(
        X_parametric=np.ones((n, 1)),
        X_smooth_dict={"s(x)": B},
        S_smooth_dict={"s(x)": S},
        Z=Z,
        y=y,
        family_obj=family_obj,
        link=link,
        Psi=Psi,
        lambda_grid=lambda_grid,
        verbose=False,
    )

    # Should be one of the grid values
    assert lambda_opt["s(x)"] in lambda_grid["s(x)"]


def test_performance_iteration():
    """Test performance iteration algorithm."""
    np.random.seed(101)
    n = 150
    n_groups = 15

    x = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)

    f_true = 2 * np.sin(2 * np.pi * x)
    b_true = np.random.randn(n_groups) * 0.4

    eta_true = 1.0 + f_true + b_true[groups]
    mu_true = np.exp(eta_true)
    y = np.random.poisson(mu_true)

    # Create smooth basis
    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    Z = np.zeros((n, n_groups))
    Z[np.arange(n), groups] = 1.0
    Z_info = [{"n_levels": n_groups, "dim": 1, "type": "intercept"}]

    # Run performance iteration
    result = select_smoothing_performance_iter(
        X_parametric=np.ones((n, 1)),
        X_smooth_dict={"s(x)": B},
        S_smooth_dict={"s(x)": S},
        Z=Z,
        Z_info=Z_info,
        y=y,
        family="poisson",
        max_iter=3,  # Just a few iterations for testing
        verbose=False,
    )

    # Check results
    assert "lambda_opt" in result
    assert "s(x)" in result["lambda_opt"]
    assert result["lambda_opt"]["s(x)"] > 0
    assert "beta_parametric" in result
    assert "beta_smooth" in result
    assert "converged" in result


def test_automatic_selection_in_fit():
    """Test that automatic λ selection works in fit_pql_with_smooth."""
    np.random.seed(202)
    n = 120
    n_groups = 12

    x = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)

    f_true = 1.5 * np.sin(2 * np.pi * x)
    b_true = np.random.randn(n_groups) * 0.3

    eta_true = 0.5 + f_true + b_true[groups]
    mu_true = np.exp(eta_true)
    y = np.random.poisson(mu_true)

    # Create smooth basis
    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    Z = np.zeros((n, n_groups))
    Z[np.arange(n), groups] = 1.0
    Z_info = [{"n_levels": n_groups, "dim": 1, "type": "intercept"}]

    # Fit with automatic λ selection (lambda_smooth=None)
    from aurora.models.gamm.pql_smooth import fit_pql_with_smooth

    result = fit_pql_with_smooth(
        X_parametric=np.ones((n, 1)),
        X_smooth_dict={"s(x)": B},
        Z=Z,
        Z_info=Z_info,
        y=y,
        family="poisson",
        S_smooth_dict={"s(x)": S},
        lambda_smooth=None,  # Auto-select
        maxiter_outer=10,  # More iterations for GCV to converge
        verbose=False,
    )

    # Check that λ was selected automatically
    assert "s(x)" in result["smoothing_parameters"]
    # λ should be positive (valid smoothing parameter)
    assert result["smoothing_parameters"]["s(x)"] > 0
    # λ should be finite and reasonable
    assert result["smoothing_parameters"]["s(x)"] < 1e10


if __name__ == "__main__":
    # Run tests
    test_select_smoothing_gcv_single_term()
    print("✓ GCV single term test passed")

    test_select_smoothing_gcv_multiple_terms()
    print("✓ GCV multiple terms test passed")

    test_select_smoothing_gcv_binomial()
    print("✓ GCV binomial test passed")

    test_select_smoothing_custom_grid()
    print("✓ Custom grid test passed")

    test_performance_iteration()
    print("✓ Performance iteration test passed")

    test_automatic_selection_in_fit()
    print("✓ Automatic selection integration test passed")

    print("\n🎉 All smoothing selection tests passed!")
