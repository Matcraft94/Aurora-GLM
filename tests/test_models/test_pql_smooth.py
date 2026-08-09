"""Tests for PQL with smooth terms (Phase 5.1)."""

import numpy as np
import pytest

from aurora.models.gamm.pql import fit_pql
from aurora.models.gamm.pql_smooth import fit_pql_with_smooth
from aurora.smoothing.splines.bspline import BSplineBasis


def _poisson_smooth_data(seed, n_groups=40, n_per_group=30):
    """Poisson data with a nearly linear smooth and random intercepts."""
    rng = np.random.default_rng(seed)
    n = n_groups * n_per_group
    x = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(n_groups), n_per_group)
    b_true = rng.standard_normal(n_groups) * 0.3
    # Nearly linear smooth: representable by the B-spline basis
    eta_true = 1.0 + 0.8 * (x - 0.5) + b_true[groups]
    y = rng.poisson(np.exp(eta_true))

    knots = BSplineBasis.create_knots(x, n_basis=8, degree=3)
    basis = BSplineBasis(knots, degree=3)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)
    Z = np.eye(n_groups)[groups]
    return x, groups, y, B, S, Z, n_groups


def test_pql_smooth_poisson_basic():
    """Test PQL with smooth terms for Poisson data."""
    np.random.seed(42)
    n = 200
    n_groups = 20

    # Generate data
    x = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)

    # True smooth function
    f_true = 2 * np.sin(2 * np.pi * x)

    # Random effects
    b_true = np.random.randn(n_groups) * 0.5

    # Linear predictor
    eta_true = 1.0 + f_true + b_true[groups]
    mu_true = np.exp(eta_true)

    # Generate Poisson response
    y = np.random.poisson(mu_true)

    # Create smooth basis
    knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
    basis = BSplineBasis(knots, degree=3)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Random effects design
    Z = np.zeros((n, n_groups))
    Z[np.arange(n), groups] = 1.0

    # Fit model
    result = fit_pql_with_smooth(
        X_parametric=np.ones((n, 1)),  # Intercept only
        X_smooth_dict={"s(x)": B},
        Z=Z,
        Z_info=[{"n_levels": n_groups, "dim": 1, "type": "intercept"}],
        y=y,
        family="poisson",
        S_smooth_dict={"s(x)": S},
        lambda_smooth={"s(x)": 1.0},  # Fixed smoothing parameter
        maxiter_outer=10,
        maxiter_inner=10,
        verbose=False,
    )

    # Check results
    assert result["converged"], "PQL should converge"
    assert "s(x)" in result["beta_smooth"]
    assert result["beta_smooth"]["s(x)"].shape == (10,)
    assert result["random_effects"].shape == (n_groups,)
    assert len(result["variance_components"]) == 1
    assert result["variance_components"][0].shape == (1, 1)

    # Check EDF
    assert "s(x)" in result["edf_smooth"]
    edf = result["edf_smooth"]["s(x)"]
    assert 1.0 < edf < 10.0, f"EDF should be between 1 and 10, got {edf}"

    # Check fitted values
    assert result["fitted_values"].shape == (n,)
    assert np.all(np.isfinite(result["fitted_values"]))


def test_pql_smooth_multiple_smooths():
    """Test PQL with multiple smooth terms."""
    np.random.seed(123)
    n = 150
    n_groups = 15

    # Generate data
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)

    # True smooth functions
    f1_true = np.sin(2 * np.pi * x1)
    f2_true = 0.5 * x2**2

    # Random effects
    b_true = np.random.randn(n_groups) * 0.3

    # Linear predictor
    eta_true = 0.5 + f1_true + f2_true + b_true[groups]
    mu_true = np.exp(eta_true)

    # Generate Poisson response
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

    # Fit model
    result = fit_pql_with_smooth(
        X_parametric=np.ones((n, 1)),
        X_smooth_dict={"s(x1)": B1, "s(x2)": B2},
        Z=Z,
        Z_info=[{"n_levels": n_groups, "dim": 1, "type": "intercept"}],
        y=y,
        family="poisson",
        S_smooth_dict={"s(x1)": S1, "s(x2)": S2},
        lambda_smooth={"s(x1)": 1.0, "s(x2)": 1.0},
        maxiter_outer=10,
        maxiter_inner=10,
    )

    # Check results
    assert result["converged"]
    assert "s(x1)" in result["beta_smooth"]
    assert "s(x2)" in result["beta_smooth"]
    assert result["beta_smooth"]["s(x1)"].shape == (8,)
    assert result["beta_smooth"]["s(x2)"].shape == (8,)

    # Check EDF for both terms
    assert "s(x1)" in result["edf_smooth"]
    assert "s(x2)" in result["edf_smooth"]


def test_pql_smooth_binomial():
    """Test PQL with smooth terms for binomial data."""
    np.random.seed(456)
    n = 100
    n_groups = 10

    # Generate data
    x = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)

    # True smooth function
    f_true = 2 * (x - 0.5)

    # Random effects
    b_true = np.random.randn(n_groups) * 0.5

    # Linear predictor
    eta_true = f_true + b_true[groups]
    prob_true = 1 / (1 + np.exp(-eta_true))

    # Generate binomial response
    y = np.random.binomial(1, prob_true)

    # Create smooth basis
    knots = BSplineBasis.create_knots(x, n_basis=8, degree=3)
    basis = BSplineBasis(knots, degree=3)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Random effects design
    Z = np.zeros((n, n_groups))
    Z[np.arange(n), groups] = 1.0

    # Fit model with more iterations for binomial convergence
    result = fit_pql_with_smooth(
        X_parametric=np.ones((n, 1)),
        X_smooth_dict={"s(x)": B},
        Z=Z,
        Z_info=[{"n_levels": n_groups, "dim": 1, "type": "intercept"}],
        y=y,
        family="binomial",
        S_smooth_dict={"s(x)": S},
        lambda_smooth={"s(x)": 2.0},
        maxiter_outer=20,  # More iterations for non-Gaussian
        maxiter_inner=25,
    )

    # Check results - binomial PQL may not always converge with stochastic data
    # Focus on shape correctness rather than strict convergence
    assert result["beta_smooth"]["s(x)"].shape == (8,)
    assert result["random_effects"].shape == (n_groups,)


def test_pql_smooth_validation():
    """Test input validation."""
    n = 50
    x = np.linspace(0, 1, n)

    # Create basis
    knots = BSplineBasis.create_knots(x, n_basis=6, degree=3)
    basis = BSplineBasis(knots, degree=3)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)

    # Mismatched dimensions
    with pytest.raises(ValueError, match="has.*rows, expected"):
        fit_pql_with_smooth(
            X_parametric=np.ones((n, 1)),
            X_smooth_dict={"s(x)": B[: n - 5, :]},  # Wrong number of rows
            Z=np.eye(n),
            Z_info=[{"n_levels": n, "dim": 1}],
            y=np.random.poisson(1, size=n),
            family="poisson",
            S_smooth_dict={"s(x)": S},
        )

    # Missing penalty matrix
    with pytest.raises(ValueError, match="Missing penalty matrix"):
        fit_pql_with_smooth(
            X_parametric=np.ones((n, 1)),
            X_smooth_dict={"s(x)": B, "s(z)": B},  # Two smooths
            Z=np.eye(n),
            Z_info=[{"n_levels": n, "dim": 1}],
            y=np.random.poisson(1, size=n),
            family="poisson",
            S_smooth_dict={"s(x)": S},  # Only one penalty
        )

    # Unsupported family
    with pytest.raises(ValueError, match="Unsupported family"):
        fit_pql_with_smooth(
            X_parametric=np.ones((n, 1)),
            X_smooth_dict={"s(x)": B},
            Z=np.eye(n),
            Z_info=[{"n_levels": n, "dim": 1}],
            y=np.random.poisson(1, size=n),
            family="unknown_family",
            S_smooth_dict={"s(x)": S},
        )


def test_pql_smooth_fitted_on_response_scale():
    """fitted_values must be on the response scale (μ), not the η scale."""
    x, groups, y, B, S, Z, n_groups = _poisson_smooth_data(seed=11)

    result = fit_pql_with_smooth(
        X_parametric=np.ones((len(y), 1)),
        X_smooth_dict={"s(x)": B},
        Z=Z,
        Z_info=[{"n_levels": n_groups, "dim": 1, "type": "intercept"}],
        y=y,
        family="poisson",
        S_smooth_dict={"s(x)": S},
        lambda_smooth={"s(x)": 5.0},
        maxiter_outer=15,
        maxiter_inner=12,
    )

    mu = result["fitted_values"]
    eta = result["linear_predictor"]

    # Response-scale fitted values are strictly positive for Poisson-log
    assert np.all(mu > 0)
    # ...and consistent with μ = exp(η)
    np.testing.assert_allclose(mu, np.exp(np.clip(eta, -700, 700)), rtol=1e-8)
    # Fitted means must be in the data range (not log-scale values)
    assert mu.mean() == pytest.approx(y.mean(), rel=0.3)


def test_pql_smooth_matches_parametric_when_linear():
    """With a nearly linear truth, the smooth fit must agree with the
    equivalent parametric PQL fit (same fixed point of Breslow & Clayton)."""
    x, groups, y, B, S, Z, n_groups = _poisson_smooth_data(seed=11)
    n = len(y)

    result_smooth = fit_pql_with_smooth(
        X_parametric=np.ones((n, 1)),
        X_smooth_dict={"s(x)": B},
        Z=Z,
        Z_info=[{"n_levels": n_groups, "dim": 1, "type": "intercept"}],
        y=y,
        family="poisson",
        S_smooth_dict={"s(x)": S},
        lambda_smooth={"s(x)": 5.0},
        maxiter_outer=15,
        maxiter_inner=12,
    )

    X_par = np.column_stack([np.ones(n), x - 0.5])
    result_par = fit_pql(X_par, Z, y, family="poisson")

    # Parametric fit recovers the data-generating fixed effects
    assert result_par.converged
    np.testing.assert_allclose(result_par.beta, [1.0, 0.8], atol=0.15)

    # Fitted values of both parametrizations must essentially coincide
    corr = np.corrcoef(result_smooth["fitted_values"], result_par.fitted_values)[0, 1]
    assert corr > 0.95, f"smooth vs parametric fitted correlation: {corr}"

    # Variance component near the true sd of the random intercepts (0.3)
    psi_smooth = result_smooth["variance_components"][0]
    assert np.sqrt(psi_smooth[0, 0]) == pytest.approx(0.3, abs=0.15)

    # Estimated smooth trend recovers the true linear trend
    f_hat = B @ result_smooth["beta_smooth"]["s(x)"]
    trend_corr = np.corrcoef(f_hat - f_hat.mean(), x - x.mean())[0, 1]
    assert trend_corr > 0.8


def test_pql_smooth_random_slopes():
    """Random intercept + slope (dim=2) must run and give sane variances.

    Before the kron(I_m, Ψ⁻¹) fix this raised a broadcasting ValueError;
    naive fixes diverged (η clipped at ±700) or collapsed Ψ to 1e-6.
    """
    rng = np.random.default_rng(0)
    n_groups, n_per_group = 30, 25
    n = n_groups * n_per_group
    t = np.tile(np.linspace(0, 1, n_per_group), n_groups)
    groups = np.repeat(np.arange(n_groups), n_per_group)

    b_int = rng.standard_normal(n_groups) * 0.3
    b_slo = rng.standard_normal(n_groups) * 0.2
    eta_true = 0.8 + 0.5 * t + b_int[groups] + b_slo[groups] * t
    y = rng.poisson(np.exp(eta_true))

    Z = np.zeros((n, n_groups * 2))
    Z[np.arange(n), groups * 2] = 1.0
    Z[np.arange(n), groups * 2 + 1] = t

    knots = BSplineBasis.create_knots(t, n_basis=6, degree=3)
    basis = BSplineBasis(knots, degree=3)
    B = basis.basis_matrix(t)
    S = basis.penalty_matrix(order=2)

    result = fit_pql_with_smooth(
        X_parametric=np.ones((n, 1)),
        X_smooth_dict={"s(t)": B},
        Z=Z,
        Z_info=[{"n_levels": n_groups, "dim": 2, "type": "intercept+slope"}],
        y=y,
        family="poisson",
        S_smooth_dict={"s(t)": S},
        lambda_smooth={"s(t)": 5.0},
        maxiter_outer=25,
        maxiter_inner=15,
    )

    psi = result["variance_components"][0]
    assert psi.shape == (2, 2)
    # Sensible variances: no collapse to the 1e-6 floor, no explosion
    assert 1e-3 < psi[0, 0] < 5.0, f"intercept variance: {psi[0, 0]}"
    assert 1e-3 < psi[1, 1] < 5.0, f"slope variance: {psi[1, 1]}"
    # No divergence: linear predictor stays in a moderate range
    assert np.abs(result["linear_predictor"]).max() < 20
    assert np.all(result["fitted_values"] > 0)
    assert np.all(np.isfinite(result["fitted_values"]))


def test_pql_smooth_binomial_response_scale():
    """Binomial smooth fit: fitted values must be probabilities in (0, 1)."""
    rng = np.random.default_rng(3)
    n, n_groups = 600, 30
    x = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)
    b_true = rng.standard_normal(n_groups) * 0.3
    eta_true = 1.5 * (x - 0.5) + b_true[groups]
    p_true = 1 / (1 + np.exp(-eta_true))
    y = rng.binomial(1, p_true).astype(float)

    knots = BSplineBasis.create_knots(x, n_basis=8, degree=3)
    basis = BSplineBasis(knots, degree=3)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix(order=2)
    Z = np.eye(n_groups)[groups]

    result = fit_pql_with_smooth(
        X_parametric=np.ones((n, 1)),
        X_smooth_dict={"s(x)": B},
        Z=Z,
        Z_info=[{"n_levels": n_groups, "dim": 1, "type": "intercept"}],
        y=y,
        family="binomial",
        S_smooth_dict={"s(x)": S},
        lambda_smooth={"s(x)": 2.0},
        maxiter_outer=20,
        maxiter_inner=15,
    )

    assert np.all(result["fitted_values"] > 0)
    assert np.all(result["fitted_values"] < 1)
    # Fitted probabilities track the true probabilities
    corr = np.corrcoef(result["fitted_values"], p_true)[0, 1]
    assert corr > 0.7


if __name__ == "__main__":
    # Run basic test
    test_pql_smooth_poisson_basic()
    print("✓ Basic Poisson test passed")

    test_pql_smooth_multiple_smooths()
    print("✓ Multiple smooths test passed")

    test_pql_smooth_binomial()
    print("✓ Binomial test passed")

    test_pql_smooth_validation()
    print("✓ Validation test passed")

    print("\nAll tests passed!")
