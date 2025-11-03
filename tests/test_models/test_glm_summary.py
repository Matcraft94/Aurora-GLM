"""Tests for GLMResult.summary() method."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models.glm import fit_glm


def test_summary_includes_coefficient_table_with_all_columns():
    """Summary should include coefficient table with estimates, std err, z, p-values."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(150, 2))
    y = rng.poisson(np.exp(X[:, 0] * 0.5 - 0.3))

    result = fit_glm(X, y, family="poisson", link="log")
    summary = result.summary()

    # Check that summary is a string
    assert isinstance(summary, str)

    # Check header elements
    assert "Generalized Linear Model Results" in summary
    assert "Poisson" in summary
    assert "Log" in summary

    # Check coefficient table columns
    assert "coef" in summary
    assert "std err" in summary
    assert "z" in summary
    assert "P>|z|" in summary
    assert "[0.025" in summary
    assert "0.975]" in summary

    # Check coefficient rows
    assert "intercept" in summary
    assert "X0" in summary
    assert "X1" in summary

    # Check significance codes
    assert "Significance codes" in summary


def test_summary_shows_convergence_status():
    """Summary should display convergence status and iteration count."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 3))
    y = rng.normal(X @ np.array([1.0, -0.5, 0.3]) + 0.5, scale=0.5)

    result = fit_glm(X, y, family="gaussian", link=None)
    summary = result.summary()

    assert "Converged:" in summary
    assert "Yes" in summary or "No" in summary
    assert "No. Iterations:" in summary
    assert str(result.n_iter_) in summary


def test_summary_includes_model_metrics_aic_bic_deviance():
    """Summary should include AIC, BIC, deviance, and pseudo R²."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(200, 2))
    linear = X @ np.array([0.7, -1.2]) + 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-linear)))

    result = fit_glm(X, y, family="binomial", link="logit")
    summary = result.summary()

    # Check goodness-of-fit metrics
    assert "Deviance:" in summary
    assert "Null Deviance:" in summary
    assert "AIC:" in summary
    assert "BIC:" in summary
    assert "Pseudo R-squared:" in summary

    # Check that numerical values are present
    assert f"{result.deviance_:.2f}" in summary
    assert f"{result.aic_:.2f}" in summary


def test_summary_works_without_intercept():
    """Summary should handle models fitted without an intercept."""
    rng = np.random.default_rng(99)
    X = rng.normal(size=(100, 3))
    y = rng.poisson(np.exp(X @ np.array([0.5, -0.3, 0.2])))

    result = fit_glm(X, y, family="poisson", link="log", fit_intercept=False)
    summary = result.summary()

    # Should NOT include intercept row
    assert "intercept" not in summary

    # Should include X0, X1, X2
    assert "X0" in summary
    assert "X1" in summary
    assert "X2" in summary

    # Should still show model info
    assert "Converged:" in summary
    assert "Deviance:" in summary


def test_summary_handles_non_converged_models():
    """Summary should gracefully handle non-converged models."""
    rng = np.random.default_rng(456)
    X = rng.normal(size=(50, 2))
    y = rng.poisson(np.exp(X[:, 0] * 2.0))

    # Force non-convergence with very few iterations
    result = fit_glm(X, y, family="poisson", link="log", max_iter=1, tol=1e-12)
    summary = result.summary()

    # Should still generate a summary
    assert isinstance(summary, str)
    assert "Generalized Linear Model Results" in summary

    # Should indicate convergence status
    if not result.converged_:
        # Note: summary() doesn't explicitly flag non-convergence beyond "Converged: No"
        # but it should still work
        assert "Converged:" in summary


def test_summary_detailed_parameter():
    """Test that detailed parameter controls output verbosity."""
    rng = np.random.default_rng(789)
    X = rng.normal(size=(100, 2))
    y = rng.normal(X @ np.array([1.0, -0.5]) + 2.0, scale=1.0)

    result = fit_glm(X, y, family="gaussian")

    # Detailed summary (default)
    detailed_summary = result.summary(detailed=True)
    assert "coef" in detailed_summary
    assert "std err" in detailed_summary
    assert "Significance codes" in detailed_summary

    # Non-detailed summary
    brief_summary = result.summary(detailed=False)
    assert "Deviance:" in brief_summary
    assert "AIC:" in brief_summary
    # Should not include coefficient table
    assert "coef" not in brief_summary or "std err" not in brief_summary


def test_summary_with_gamma_family():
    """Summary should work with Gamma family (different variance structure)."""
    rng = np.random.default_rng(321)
    X = rng.normal(size=(150, 2))
    shape = 2.0
    mu = np.exp(X @ np.array([0.5, -0.3]) + 1.0)
    scale = mu / shape
    y = rng.gamma(shape, scale)

    result = fit_glm(X, y, family="gamma", link="log")
    summary = result.summary()

    assert "Gamma" in summary
    assert "Log" in summary
    assert "intercept" in summary
    assert "Pseudo R-squared:" in summary
