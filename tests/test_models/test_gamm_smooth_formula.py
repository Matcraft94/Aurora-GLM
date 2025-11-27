"""Integration tests for GAMM with smooth terms via formula interface (Phase 5.1.2)."""
import numpy as np
import pandas as pd
import pytest


def test_gamm_poisson_smooth_formula():
    """Test GAMM with Poisson family and smooth term via formula."""
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

    # Create DataFrame
    data = pd.DataFrame({
        'y': y,
        'x': x,
        'subject': groups
    })

    # Fit model with formula
    from aurora.models.gamm import fit_gamm

    result = fit_gamm(
        formula='y ~ s(x) + (1 | subject)',
        data=data,
        family='poisson',
        maxiter=10,
    )

    # Check results
    assert result.converged, "Model should converge"
    assert result.family == 'poisson'
    assert 's(x)' in result.beta_smooth
    assert result.beta_smooth['s(x)'].shape[0] == 10  # Default n_basis
    assert len(result.variance_components) > 0

    # Check EDF
    assert 's(x)' in result.edf_smooth
    edf = result.edf_smooth['s(x)']
    assert 1.0 < edf < 10.0, f"EDF should be between 1 and 10, got {edf}"

    # Check fitted values
    assert result.fitted_values.shape == (n,)
    assert np.all(np.isfinite(result.fitted_values))


def test_gamm_binomial_smooth_formula():
    """Test GAMM with binomial family and smooth term via formula."""
    np.random.seed(123)
    n = 150
    n_groups = 15

    # Generate data
    x = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)

    # True smooth function
    f_true = 3 * (x - 0.5)

    # Random effects
    b_true = np.random.randn(n_groups) * 0.5

    # Linear predictor
    eta_true = f_true + b_true[groups]
    prob_true = 1 / (1 + np.exp(-eta_true))

    # Generate binomial response
    y = np.random.binomial(1, prob_true)

    # Create DataFrame
    data = pd.DataFrame({
        'y': y,
        'x': x,
        'group': groups
    })

    # Fit model
    from aurora.models.gamm import fit_gamm

    result = fit_gamm(
        formula='y ~ s(x) + (1 | group)',
        data=data,
        family='binomial',
        maxiter=8,
    )

    # Check results
    assert result.converged
    assert result.family == 'binomial'
    assert 's(x)' in result.beta_smooth
    assert 's(x)' in result.edf_smooth


def test_gamm_multiple_smooths_formula():
    """Test GAMM with multiple smooth terms via formula."""
    np.random.seed(456)
    n = 200
    n_groups = 20

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

    # Create DataFrame
    data = pd.DataFrame({
        'count': y,
        'time': x1,
        'temperature': x2,
        'subject': groups
    })

    # Fit model with multiple smooths
    from aurora.models.gamm import fit_gamm

    result = fit_gamm(
        formula='count ~ s(time) + s(temperature) + (1 | subject)',
        data=data,
        family='poisson',
        maxiter=10,
    )

    # Check results
    assert result.converged
    assert result.family == 'poisson'
    assert 's(time)' in result.beta_smooth
    assert 's(temperature)' in result.beta_smooth
    assert 's(time)' in result.edf_smooth
    assert 's(temperature)' in result.edf_smooth


def test_gamm_smooth_with_parametric():
    """Test GAMM with both smooth and parametric terms."""
    np.random.seed(789)
    n = 150
    n_groups = 15

    # Generate data
    x_smooth = np.linspace(0, 1, n)
    x_para = np.random.randn(n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)

    # True functions
    f_smooth = np.sin(2 * np.pi * x_smooth)
    beta_para = 0.5

    # Random effects
    b_true = np.random.randn(n_groups) * 0.3

    # Linear predictor
    eta_true = 1.0 + f_smooth + beta_para * x_para + b_true[groups]
    mu_true = np.exp(eta_true)

    # Generate Poisson response
    y = np.random.poisson(mu_true)

    # Create DataFrame
    data = pd.DataFrame({
        'y': y,
        'x_smooth': x_smooth,
        'x_para': x_para,
        'subject': groups
    })

    # Fit model
    from aurora.models.gamm import fit_gamm

    result = fit_gamm(
        formula='y ~ x_para + s(x_smooth) + (1 | subject)',
        data=data,
        family='poisson',
        maxiter=10,
    )

    # Check results
    assert result.converged
    assert result.beta_parametric.shape[0] == 2  # Intercept + x_para
    assert 's(x_smooth)' in result.beta_smooth


def test_gamm_smooth_custom_parameters():
    """Test GAMM with custom smooth parameters via formula."""
    np.random.seed(101)
    n = 100
    n_groups = 10

    x = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)

    f_true = 2 * np.sin(2 * np.pi * x)
    b_true = np.random.randn(n_groups) * 0.4

    eta_true = f_true + b_true[groups]
    mu_true = np.exp(eta_true)
    y = np.random.poisson(mu_true)

    data = pd.DataFrame({
        'y': y,
        'x': x,
        'subject': groups
    })

    # Fit with custom basis size
    from aurora.models.gamm import fit_gamm

    result = fit_gamm(
        formula='y ~ s(x, k=8) + (1 | subject)',  # 8 basis functions instead of default 10
        data=data,
        family='poisson',
        maxiter=10,
    )

    # Check custom basis size was used
    assert result.beta_smooth['s(x)'].shape[0] == 8


def test_gamm_smooth_validation():
    """Test validation for smooth terms in formula."""
    n = 50
    x = np.linspace(0, 1, n)
    groups = np.repeat(np.arange(5), 10)
    y = np.random.poisson(1, size=n)

    data = pd.DataFrame({
        'y': y,
        'x': x,
        'subject': groups
    })

    from aurora.models.gamm import fit_gamm

    # Variable not in data
    with pytest.raises(ValueError, match="not found in data"):
        fit_gamm(
            formula='y ~ s(unknown_var) + (1 | subject)',
            data=data,
            family='poisson',
        )


if __name__ == '__main__':
    # Run tests
    test_gamm_poisson_smooth_formula()
    print("✓ Poisson smooth formula test passed")

    test_gamm_binomial_smooth_formula()
    print("✓ Binomial smooth formula test passed")

    test_gamm_multiple_smooths_formula()
    print("✓ Multiple smooths formula test passed")

    test_gamm_smooth_with_parametric()
    print("✓ Smooth + parametric formula test passed")

    test_gamm_smooth_custom_parameters()
    print("✓ Custom parameters formula test passed")

    test_gamm_smooth_validation()
    print("✓ Validation test passed")

    print("\n🎉 All GAMM smooth formula integration tests passed!")
